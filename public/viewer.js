import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GUI } from "dat.gui";

// Configuration
const DEFAULT_SSE_REFINE = 18.0;   // pixels
const DEFAULT_SSE_COARSEN = DEFAULT_SSE_REFINE * 0.5;
const ENABLE_SSE_AUTO_CALIBRATION = false;
let SSE_THRESHOLD_REFINE = DEFAULT_SSE_REFINE;
let SSE_THRESHOLD_COARSEN = DEFAULT_SSE_COARSEN;   // hysteresis (auto-updated)
const MAX_CONCURRENT = 6;
const MAX_CACHE_BYTES = 600 * 1024 * 1024; // ~600 MB budget
const MAX_TILES = 500;
const TILE_DEBUG_COLORS = [
  0x4e79a7, 0xa0cbe8, 0xf28e2b, 0xffbe7a, 0xe15759,
  0xff9d9a, 0x76b7b2, 0x86bcb6, 0x59a14f, 0x8cd17d,
  0xedc948, 0xb6992d, 0xb07aa1, 0xd4a6c8, 0xff9da7,
  0x9c755f, 0xd7b5a6, 0xbab0ac, 0xd1d1d1, 0x7f7f7f
];
const REPO_URL = "https://github.com/grahampullan/extract-tiles";

function pickDebugColor() {
  const idx = Math.floor(Math.random() * TILE_DEBUG_COLORS.length);
  return new THREE.Color(TILE_DEBUG_COLORS[idx]);
}

function deduceTileBaseUrl(manifestUrl, manifest) {
  if (!manifestUrl) return null;
  try {
    const absoluteManifestUrl = new URL(manifestUrl, window.location.href);

    if (manifest && typeof manifest.tilesBasePath === "string" && manifest.tilesBasePath.length) {
      return new URL(manifest.tilesBasePath, absoluteManifestUrl).href;
    }

    const manifestDirUrl = new URL("./", absoluteManifestUrl);
    const dirParts = manifestDirUrl.pathname.split("/").filter(Boolean);
    const manifestIdx = dirParts.indexOf("manifest");

    if (manifestIdx !== -1) {
      const swapped = [...dirParts];
      swapped[manifestIdx] = "tiles";
      const path = `/${swapped.join("/")}/`;
      return new URL(path, absoluteManifestUrl).href;
    }

    return manifestDirUrl.href;
  } catch (err) {
    console.warn("Failed to deduce tile base URL", manifestUrl, err);
    return null;
  }
}

function resolveTileUrl(rawUrl, tileBaseUrl) {
  if (!rawUrl) return null;
  if (/^https?:\/\//i.test(rawUrl)) {
    return rawUrl;
  }
  if (rawUrl.startsWith("//")) {
    return `${window.location.protocol}${rawUrl}`;
  }
  if (rawUrl.startsWith("/")) {
    return rawUrl;
  }
  if (!tileBaseUrl) {
    return rawUrl;
  }
  try {
    return new URL(rawUrl, tileBaseUrl).href;
  } catch (err) {
    console.warn("Failed to resolve tile URL", rawUrl, tileBaseUrl, err);
    return rawUrl;
  }
}

function resolveAbsoluteUrl(pathOrUrl) {
  if (typeof pathOrUrl !== "string" || !pathOrUrl.length) {
    return null;
  }
  try {
    return new URL(pathOrUrl, window.location.href).href;
  } catch (err) {
    console.warn("Failed to resolve absolute URL", pathOrUrl, err);
    return pathOrUrl;
  }
}

function ensureTrailingSlash(url) {
  if (typeof url !== "string" || !url.length) {
    return url;
  }
  return url.endsWith("/") ? url : `${url}/`;
}

function parseVector3(arr) {
  if (!Array.isArray(arr) || arr.length !== 3) {
    return null;
  }
  const [x, y, z] = arr.map(Number);
  if ([x, y, z].some((v) => !Number.isFinite(v))) {
    return null;
  }
  return [x, y, z];
}

function formatPattern(pattern, extract, time) {
  return pattern
    .replace(/\{extract\}/g, extract)
    .replace(/\{time\}/g, String(time));
}

function normalizeConfigDatasets(rawDatasets) {
  if (!Array.isArray(rawDatasets) || rawDatasets.length === 0) {
    return null;
  }
  const normalized = [];
  for (const entry of rawDatasets) {
    if (!entry || typeof entry.name !== "string" || !entry.name.length) continue;
    const manifestEntries = new Map();
    if (Array.isArray(entry.manifests)) {
      for (const descriptor of entry.manifests) {
        if (!descriptor) continue;
        const time = Number(descriptor.time);
        if (!Number.isFinite(time)) continue;
        manifestEntries.set(time, {
          manifestUrl: descriptor.manifest ? resolveAbsoluteUrl(descriptor.manifest) : null,
          payloadUrl: descriptor.payload ? resolveAbsoluteUrl(descriptor.payload) : null,
        });
      }
    }
    const timesSet = new Set();
    if (Array.isArray(entry.times)) {
      for (const t of entry.times) {
        const n = Number(t);
        if (Number.isFinite(n)) timesSet.add(n);
      }
    }
    for (const t of manifestEntries.keys()) {
      timesSet.add(t);
    }
    const times = Array.from(timesSet).filter((t) => Number.isFinite(t)).sort((a, b) => a - b);
    const defaultTime = Number.isFinite(entry.defaultTime) ? Number(entry.defaultTime) : (times[0] ?? 0);
    normalized.push({
      name: entry.name,
      label: entry.label || entry.name,
      times: times.length ? times : [defaultTime],
      defaultTime,
      manifestPattern: typeof entry.manifestPattern === "string" ? entry.manifestPattern : null,
      payloadPattern: typeof entry.payloadPattern === "string" ? entry.payloadPattern : null,
      tilesBasePath: typeof entry.tilesBasePath === "string"
        ? ensureTrailingSlash(resolveAbsoluteUrl(entry.tilesBasePath))
        : null,
      manifestEntries,
    });
  }
  return normalized.length ? normalized : null;
}

class TileManager {
  constructor(scene, camera, renderer) {
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.loader = new GLTFLoader();
    this.manifest = null;
    this.byId = new Map();
    this.tiles = new Map();   // loaded: id -> {obj3d, meta}
    this.cache = new Map();   // LRU: id -> rec
    this.cacheBytes = 0;
    this.queue = [];
    this.inflight = 0;
    this.frustum = new THREE.Frustum();
    this.projScreenMatrix = new THREE.Matrix4();
    this.hudTiles = document.getElementById("tiles") || { textContent: "" };
    this.hudCache = document.getElementById("cache") || { textContent: "" };
    this.loadingIndicator = document.getElementById("loading");
    this.wireframeMode = false; // Track wireframe state
    this.showBoundingBoxes = false; // Diagnostic overlay toggle
    this.tileColorMode = false; // Diagnostic colouring toggle
    this.displayMode = "shaded";
    this.levelGeMedian = new Map(); // Depth -> geometric error median
    this.requestPriority = new Map(); // TileId -> SSE-based priority
    this.rootTileIds = new Set();
    this._tickLock = false;
    this._tickPending = false;
    this.manifestVersion = 0;
    this._queueSeq = 0;
    this.manifestUrl = null;
    this.tileBaseUrl = null;
    this.wireOverlayEnabled = false;
    this.wireOverlayAngle = 0;
    this.wireOverlayOpacity = 1.0;
    this.datasetKey = null;
    this.staticLayoutType = null;
    this.staticLayoutActive = false;
    this.staticReuseDisabled = false;
    this.currentTimeIndex = null;
    this.targetTimeIndex = null;
    this.staticRefreshPending = false;
    this._visibleMin = new THREE.Vector3();
    this._visibleMax = new THREE.Vector3();
    this._visibleBox = new THREE.Box3(this._visibleMin, this._visibleMax);
  }

  _useStaticLayout() {
    return !!this.staticLayoutActive && !this.staticReuseDisabled;
  }

  async init(manifestUrl, preloadedManifest = null, options = {}) {
    const version = ++this.manifestVersion;
    this.manifestUrl = manifestUrl;
    this.rootTileIds.clear();
    const { datasetKey = null, timeIndex = null, tileBaseOverride = null } = options;
    try {
      let manifest = preloadedManifest;
      if (!manifest) {
        const response = await fetch(manifestUrl);
        manifest = await response.json();
      }
      if (version !== this.manifestVersion) {
        return;
      }
      this.manifest = manifest;
      this.datasetKey = datasetKey ?? this.datasetKey;
      this.staticLayoutType = (manifest.layout && manifest.layout.type) || null;
      this.staticLayoutActive = (this.staticLayoutType === "static-octree");
      const resolvedBase = tileBaseOverride ?? deduceTileBaseUrl(manifestUrl, manifest);
      this.tileBaseUrl = resolvedBase;
      this.targetTimeIndex = timeIndex != null ? timeIndex : (manifest.time ?? null);
      if (!this._useStaticLayout() || this.tiles.size === 0) {
        this.currentTimeIndex = this.targetTimeIndex;
      }
      this._computeLevelGeStats();
      for (const t of this.manifest.tiles) {
        this.byId.set(t.tileId, t);
        if (t.parent == null) {
          this.rootTileIds.add(t.tileId);
        }
      }
      // Load root tiles
      for (const t of this.manifest.tiles) {
        if (t.z === 0) {
          this._enqueue(t.tileId, null, { forceReload: this._useStaticLayout() });
        }
      }
    } catch (err) {
      console.error("Failed to load manifest:", err);
    }
  }

  _updateFrustum() {
    this.camera.updateMatrixWorld();
    this.projScreenMatrix.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.projScreenMatrix);
  }

  _visible(meta) {
    if (!meta || !meta.aabbWorld) return false;
    this._visibleMin.fromArray(meta.aabbWorld[0]);
    this._visibleMax.fromArray(meta.aabbWorld[1]);
    return this.frustum.intersectsBox(this._visibleBox);
  }

  _allChildrenLoaded(meta) {
    const kids = (meta.children || []);
    if (!kids.length) return true;
    for (const cid of kids) {
      if (!this.tiles.has(cid)) return false;
    }
    return true;
  }

  _hasPendingDescendants(tileId, wantSet) {
    for (const candidateId of wantSet) {
      if (candidateId === tileId) continue;
      let currentId = candidateId;
      while (currentId) {
        if (currentId === tileId) {
          if (!this.tiles.has(candidateId)) {
            return true;
          }
          break;
        }
        const meta = this.byId.get(currentId);
        if (!meta || !meta.parent) {
          currentId = null;
        } else {
          currentId = meta.parent;
        }
      }
    }
    return false;
  }

  _sse(meta) {
    const c = new THREE.Vector3(...meta.aabbWorld[0])
      .add(new THREE.Vector3(...meta.aabbWorld[1]))
      .multiplyScalar(0.5);
    const dist = c.distanceTo(this.camera.position) + 1e-6;
    const ge = meta.geometricError || 0.01;
    const h = this.renderer.domElement.clientHeight;
    const fov = this.camera.fov * Math.PI/180;
    return (ge / (dist * Math.tan(fov/2))) * h;
  }

  _children(meta) {
    return (meta.children || []).map(id => this.byId.get(id)).filter(Boolean);
  }

  _disposeWireOverlay(mesh) {
    if (mesh.userData?.wireOverlay) {
      const overlay = mesh.userData.wireOverlay;
      mesh.remove(overlay);
      overlay.geometry?.dispose();
      overlay.material?.dispose?.();
      mesh.userData.wireOverlay = null;
    }
  }

  _disposeMaterial(material) {
    if (!material) return;
    if (Array.isArray(material)) {
      material.forEach(m => this._disposeMaterial(m));
      return;
    }
    const keys = Object.keys(material);
    for (const key of keys) {
      const value = material[key];
      if (value && typeof value === "object" && value.isTexture) {
        value.dispose?.();
      }
    }
    material.dispose?.();
  }

  _disposeRecord(rec) {
    if (!rec) return;
    if (rec.obj3d) {
      rec.obj3d.traverse(o => {
        if (o.isMesh || o.isLine) {
          o.geometry?.dispose();
          if (o.material) {
            this._disposeMaterial(o.material);
          }
        }
        if (o.userData?.wireOverlay) {
          this._disposeWireOverlay(o);
        }
      });
    }
    if (rec.bboxHelper) {
      rec.bboxHelper.geometry?.dispose();
      rec.bboxHelper.material?.dispose?.();
    }
  }

  _createWireOverlay(mesh) {
    if (!mesh.isMesh || !mesh.geometry) return;
    if (mesh.userData?.wireOverlay) return;
    try {
      const threshold = Number.isFinite(this.wireOverlayAngle) ? this.wireOverlayAngle : 0;
      const edgesGeo = new THREE.EdgesGeometry(mesh.geometry, threshold);
      const edgesMat = new THREE.LineBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: this.wireOverlayOpacity ?? 0.9,
        depthTest: true,
        depthWrite: false,
        polygonOffset: true,
        polygonOffsetFactor: -1,
        polygonOffsetUnits: -1
      });
      const edges = new THREE.LineSegments(edgesGeo, edgesMat);
      edges.renderOrder = 1;
      mesh.add(edges);
      this._applyWireOverlayMaterial(edges);
      mesh.userData.wireOverlay = edges;
    } catch (err) {
      console.warn("Failed to create wire overlay", err);
    }
  }

  _applyWireOverlayMaterial(overlay) {
    if (!overlay || !overlay.material) return;
    overlay.material.transparent = true;
    overlay.material.opacity = this.wireOverlayOpacity ?? 0.9;
    overlay.material.needsUpdate = true;
  }

  _setWireOverlayVisible(obj, visible) {
    if (!obj) return;
    obj.traverse(o => {
      if (!o.isMesh) return;
      if (visible) {
        if (!o.userData?.wireOverlay) {
          this._createWireOverlay(o);
        }
        if (o.userData?.wireOverlay) {
          o.userData.wireOverlay.visible = true;
          this._applyWireOverlayMaterial(o.userData.wireOverlay);
        }
      } else if (o.userData?.wireOverlay) {
        o.userData.wireOverlay.visible = false;
      }
    });
  }

  applyWireOverlayState(value) {
    this.wireOverlayEnabled = value;
    for (const [, rec] of this.tiles) {
      this._setWireOverlayVisible(rec.obj3d, value);
    }
    for (const [id, rec] of this.cache) {
      if (this.tiles.has(id)) continue;
      this._setWireOverlayVisible(rec.obj3d, value);
    }
  }

  rebuildWireOverlayGeometry() {
    const clear = (obj) => {
      if (!obj) return;
      obj.traverse(o => {
        if (o.isMesh) {
          this._disposeWireOverlay(o);
        }
      });
    };
    for (const [, rec] of this.tiles) {
      clear(rec.obj3d);
    }
    for (const [id, rec] of this.cache) {
      if (this.tiles.has(id)) continue;
      clear(rec.obj3d);
    }
    if (this.wireOverlayEnabled) {
      this.applyWireOverlayState(true);
    }
  }


  _decide() {
    const replaceParents = new Set();
    const roots = [...this.byId.values()].filter(t => t.z === 0);
    const want = new Set();

    this.requestPriority.clear();

    const visit = (meta) => {
      if (!this._visible(meta)) return;

      const sse = this._sse(meta);
      const medianGe = this.levelGeMedian.get(meta.z);
      const geoError = meta.geometricError;
      const normalizedSse = (medianGe && geoError)
        ? sse * (medianGe / Math.max(geoError, 1e-9))
        : sse;
      const hasChildren = (meta.children || []).length > 0;

      this.requestPriority.set(meta.tileId, normalizedSse);

      if (hasChildren && normalizedSse > SSE_THRESHOLD_REFINE) {
        replaceParents.add(meta.tileId);
        // Need more detail - recurse to children
        for (const c of this._children(meta)) {
          visit(c);
        }
      } else if (normalizedSse < SSE_THRESHOLD_COARSEN || !hasChildren) {
        // Use this tile
        want.add(meta.tileId);
      } else {
        // In hysteresis zone - keep current tile
        want.add(meta.tileId);
      }
    };

    for (const r of roots) {
      visit(r);
    }

    return { want, replaceParents };
  }

  _computeLevelGeStats() {
    this.levelGeMedian.clear();
    const perLevel = new Map();
    for (const tile of this.manifest?.tiles || []) {
      const ge = tile?.geometricError;
      if (!(typeof ge === 'number') || ge <= 0) continue;
      if (!perLevel.has(tile.z)) {
        perLevel.set(tile.z, []);
      }
      perLevel.get(tile.z).push(ge);
    }
    for (const [depth, values] of perLevel) {
      values.sort((a, b) => a - b);
      const mid = Math.floor(values.length / 2);
      const median = values.length % 2
        ? values[mid]
        : 0.5 * (values[mid - 1] + values[mid]);
      this.levelGeMedian.set(depth, median || values[mid] || 0);
    }
  }

  async _tickOnce() {
    if (!this.manifest) return;
    const versionAtStart = this.manifestVersion;

    this._updateFrustum();
    const { want, replaceParents } = this._decide();
    this._refreshQueuePriorities();
    if (versionAtStart !== this.manifestVersion) return;

    // Ensure roots are always requested
    // Update visibility of already loaded tiles before any load/unload
    for (const [id, rec] of this.tiles) {
      const keepAsFallback = replaceParents.has(id) && this._hasPendingDescendants(id, want);
      const needed = want.has(id) || keepAsFallback;
      rec.obj3d.visible = needed;
      if (rec.bboxHelper) {
        rec.bboxHelper.visible = this.showBoundingBoxes && needed;
      }
    }

    // Queue tiles we need but don't have
    const rootIds = [...this.rootTileIds];
    for (const id of rootIds) {
      if (!want.has(id)) continue;
      const rec = this.tiles.get(id);
      if (!rec) {
        this._enqueue(id, this.requestPriority.get(id));
      } else if (this._useStaticLayout() && this.targetTimeIndex != null && rec.timeIndex !== this.targetTimeIndex) {
        this._enqueue(id, this.requestPriority.get(id), { forceReload: true });
      }
    }
    for (const id of want) {
      if (this.rootTileIds.has(id)) continue;
      const rec = this.tiles.get(id);
      if (!rec) {
        this._enqueue(id, this.requestPriority.get(id));
      } else if (this._useStaticLayout() && this.targetTimeIndex != null && rec.timeIndex !== this.targetTimeIndex) {
        this._enqueue(id, this.requestPriority.get(id), { forceReload: true });
      }
    }

    // Drop any queued requests we no longer need
    if (this.queue.length) {
      this.queue = this.queue.filter(entry => want.has(entry.id));
    }
    if (versionAtStart !== this.manifestVersion) return;

    // Unload tiles we have but don't need
    for (const [id, rec] of [...this.tiles]) {
      if (!want.has(id)) {
        const keepAsFallback = replaceParents.has(id) && this._hasPendingDescendants(id, want);
        if (!keepAsFallback) {
          this._unload(id);
        }
      }
    }

    // Process queue until we either run out of tiles or hit the concurrency cap
    let launched;
    do {
      launched = [];
      while (this.queue.length && this.inflight < MAX_CONCURRENT) {
        const nextEntry = this.queue.shift();
        launched.push(this._load(nextEntry));
      }
      if (launched.length) {
        this._setLoadingIndicator(true);
        const loaded = await Promise.all(launched);
        if (versionAtStart !== this.manifestVersion) return;
        // Hide freshly loaded tiles that are no longer needed
        for (const result of loaded) {
          if (!result || !result.rec) continue;
          const { rec, entry } = result;
          const id = rec.meta.tileId;
          const keepAsFallback = replaceParents.has(id) && this._hasPendingDescendants(id, want);
          const needed = want.has(id) || keepAsFallback;
          this._installTile(rec, { needed });
        }
      }
    } while (this.queue.length && this.inflight < MAX_CONCURRENT);

    this.hudTiles.textContent = `${this.tiles.size}`;
    this.hudCache.textContent = `${this.cache.size}`;

    this.updateBoundingBoxVisibility();
    if (versionAtStart !== this.manifestVersion) return;

    this._setLoadingIndicator(this.inflight > 0 || this.queue.length > 0);
    if (this._useStaticLayout() && this.staticRefreshPending && this._staticRootsAtTargetTime()) {
      this.staticRefreshPending = false;
      this.currentTimeIndex = this.targetTimeIndex;
      this._tickPending = true;
    }
  }

  async tick() {
    if (this._tickLock) {
      this._tickPending = true;
      return;
    }
    this._tickLock = true;
    do {
      this._tickPending = false;
      await this._tickOnce();
    } while (this._tickPending);
    this._tickLock = false;
  }

  _setLoadingIndicator(active) {
    if (!this.loadingIndicator) return;
    if (active) {
      this.loadingIndicator.classList.add('active');
    } else {
      this.loadingIndicator.classList.remove('active');
    }
  }

  _tileMatchesTime(rec, desiredTime) {
    if (!rec) return false;
    if (!this._useStaticLayout()) return true;
    if (desiredTime == null) return true;
    return rec.timeIndex === desiredTime;
  }

  _staticRootsAtTargetTime() {
    if (!this._useStaticLayout()) return false;
    if (this.targetTimeIndex == null) return false;
    for (const id of this.rootTileIds) {
      const rec = this.tiles.get(id);
      if (!rec) return false;
      if (rec.timeIndex !== this.targetTimeIndex) {
        return false;
      }
    }
    return true;
  }

  _enqueue(id, priority = null, options = {}) {
    const priorityHint = (priority ?? this.requestPriority.get(id) ?? 0);
    const { forceReload = false, timeIndex = this.targetTimeIndex } = options;
    const desiredTime = timeIndex != null ? timeIndex : this.targetTimeIndex;
    const wantStatic = this._useStaticLayout();
    const activeRec = this.tiles.get(id);
    const recIsFresh = this._tileMatchesTime(activeRec, desiredTime);
    let needsReload = forceReload || (wantStatic && activeRec && !recIsFresh);
    if (activeRec && recIsFresh && !forceReload) {
      return;
    }
    // Check cache for reusable record
    if (this.cache.has(id)) {
      const rec = this.cache.get(id);
      const cacheFresh = this._tileMatchesTime(rec, desiredTime);
      if (cacheFresh && !forceReload) {
        this.cache.delete(id);
        this.cache.set(id, rec); // Move to end (LRU)
        this.scene.add(rec.obj3d);
        if (rec.bboxHelper) {
          if (this.showBoundingBoxes) {
            if (rec.bboxHelper.parent !== this.scene) {
              this.scene.add(rec.bboxHelper);
            }
            rec.bboxHelper.visible = true;
          } else {
            rec.bboxHelper.visible = false;
            if (rec.bboxHelper.parent === this.scene) {
              this.scene.remove(rec.bboxHelper);
            }
          }
        }
        this.tiles.set(id, rec);
        this._applyActiveMaterial(rec);
        this._setWireOverlayVisible(rec.obj3d, this.wireOverlayEnabled);
        return;
      }
    }

    // Avoid duplicate queue entries
    const existing = this.queue.find(entry => entry.id === id);
    if (existing) {
      if (priorityHint !== existing.priority) {
        existing.priority = priorityHint;
        this._sortQueue();
      }
      existing.forceReload = existing.forceReload || needsReload;
      if (desiredTime != null) {
        existing.timeIndex = desiredTime;
      }
      return;
    }

    this.queue.push({
      id,
      priority: priorityHint,
      seq: this._queueSeq++,
      forceReload: needsReload,
      timeIndex: desiredTime
    });
    this._sortQueue();
  }

  _refreshQueuePriorities() {
    let needsSort = false;
    for (const entry of this.queue) {
      const updated = this.requestPriority.get(entry.id);
      if (updated != null && updated !== entry.priority) {
        entry.priority = updated;
        needsSort = true;
      }
    }
    if (needsSort) {
      this._sortQueue();
    }
  }

  _sortQueue() {
    this.queue.sort((a, b) => {
      if (b.priority !== a.priority) {
        return b.priority - a.priority;
      }
      return a.seq - b.seq;
    });
  }

  async _load(entry) {
    const { id, timeIndex } = entry;
    const meta = this.byId.get(id);
    if (!meta) return null;

    const tileUrl = resolveTileUrl(meta.url, this.tileBaseUrl);
    const desiredTime = timeIndex != null ? timeIndex : this.targetTimeIndex;
    const versionAtStart = this.manifestVersion;
    this.inflight++;
    let rec = null;
    try {
      if (!tileUrl) {
        const emptyGroup = new THREE.Group();
        const bboxHelper = this._createBoundingBoxHelper(meta);
        bboxHelper.userData.tileId = id;
        rec = { obj3d: emptyGroup, meta, bboxHelper, timeIndex: desiredTime, empty: true };
      } else {
        let glb;
        const prefetched = prefetchedTileBuffers.get(tileUrl);
        if (prefetched) {
          prefetchedTileBuffers.delete(tileUrl);
          glb = await new Promise((resolve, reject) => {
            this.loader.parse(prefetched.slice(0), '', resolve, reject);
          });
        } else {
          glb = await this.loader.loadAsync(tileUrl);
        }
        if (versionAtStart !== this.manifestVersion) {
        glb.scene.traverse(o => {
          if (o.isMesh || o.isLine) {
            o.geometry?.dispose();
            if (o.material) {
              this._disposeMaterial(o.material);
            }
          }
        });
        return null;
      }
        const obj = glb.scene;
        obj.userData.tileId = id;

        obj.traverse(o => {
          if (o.isMesh && o.material) {
            const geom = o.geometry;
            const hasVertexColors = !!(geom && geom.attributes && geom.attributes.color);
            if (geom && (!geom.attributes || !geom.attributes.normal)) {
              geom.computeVertexNormals?.();
            }
            const baseColor = (o.material.color && o.material.color.isColor)
              ? o.material.color.clone()
              : new THREE.Color(0.85, 0.85, 0.85);
            const newMaterial = new THREE.MeshPhongMaterial({
              color: baseColor,
              vertexColors: hasVertexColors,
              transparent: false,
              opacity: 1.0,
              wireframe: this.wireframeMode,
              side: THREE.DoubleSide,
              shininess: 60,
              specular: new THREE.Color(0.35, 0.35, 0.35)
            });

            if (o.material.map) {
              newMaterial.map = o.material.map;
            }

            o.material = newMaterial;
            o.userData.debugInfo = {
              originalMaterial: newMaterial,
              debugColor: pickDebugColor(),
              originalColorAttr: hasVertexColors && geom?.attributes?.color
                ? geom.attributes.color.clone()
                : null,
              overrideMaterial: null
            };
          }
        });

        const bboxHelper = this._createBoundingBoxHelper(meta);
        bboxHelper.userData.tileId = id;
        rec = { obj3d: obj, meta, bboxHelper, timeIndex: desiredTime };
      }
    } catch (e) {
      console.error("Tile load failed", id, e);
    } finally {
      this.inflight--;
    }

    if (versionAtStart !== this.manifestVersion) {
      if (rec) {
        this._disposeRecord(rec);
      }
      return null;
    }

    return { rec, entry };
  }

  _createBoundingBoxHelper(meta) {
    const min = new THREE.Vector3(...meta.aabbWorld[0]);
    const max = new THREE.Vector3(...meta.aabbWorld[1]);
    const box = new THREE.Box3(min, max);
    const hue = ((meta.z || 0) % 6) / 6;
    const color = new THREE.Color().setHSL(hue, 0.85, 0.30);
    const helper = new THREE.Box3Helper(box, color);
    if (helper.material) {
      helper.material.depthTest = false;
      helper.material.transparent = true;
      helper.material.opacity = 0.9;
      helper.material.needsUpdate = true;
    }
    return helper;
  }

  _installTile(rec, options = {}) {
    const id = rec.meta.tileId;
    const needed = options.needed !== undefined ? options.needed : true;
    const existing = this.tiles.get(id);
    if (existing) {
      this._cacheRemove(id);
      this.scene.remove(existing.obj3d);
      if (existing.bboxHelper) {
        this.scene.remove(existing.bboxHelper);
      }
      this._disposeRecord(existing);
      this.tiles.delete(id);
    }

    this.scene.add(rec.obj3d);
    if (rec.bboxHelper) {
      if (this.showBoundingBoxes) {
        rec.bboxHelper.visible = true;
        this.scene.add(rec.bboxHelper);
      } else {
        rec.bboxHelper.visible = false;
      }
    }

    this._applyActiveMaterial(rec);
    this._setWireOverlayVisible(rec.obj3d, this.wireOverlayEnabled);

    this.tiles.set(id, rec);
    this._cacheInsert(id, rec);

    if (!needed) {
      this._unload(id);
    } else {
      rec.obj3d.visible = true;
      if (rec.bboxHelper) {
        rec.bboxHelper.visible = this.showBoundingBoxes;
      }
    }
  }

  applyStaticPayload(payload) {
    if (!payload || !Array.isArray(payload.fields) || !Array.isArray(payload.tiles)) {
      return false;
    }
    if (!this.manifest || !Array.isArray(this.manifest.tiles)) {
      return false;
    }
    const fields = payload.fields;
    if (!fields.length || fields[0] !== "tileId") {
      console.warn("Payload fields must start with tileId");
      return false;
    }
    const tileIndex = new Map();
    for (const tile of this.manifest.tiles) {
      if (tile?.tileId != null) {
        tileIndex.set(tile.tileId, tile);
      }
    }
    for (const row of payload.tiles) {
      if (!Array.isArray(row) || row.length < 1) continue;
      const tileId = row[0];
      const meta = tileIndex.get(tileId);
      if (!meta) continue;
      for (let i = 1; i < fields.length && i < row.length; i++) {
        meta[fields[i]] = row[i];
      }
    }
    if (typeof payload.time === "number") {
      this.manifest.time = payload.time;
    }
    return true;
  }

  _unload(id) {
    const rec = this.tiles.get(id);
    if (!rec) return;

    this.scene.remove(rec.obj3d);
    if (rec.bboxHelper) {
      this.scene.remove(rec.bboxHelper);
    }

    // Keep in cache, but dispose if too many
    // Note: We're not disposing here to keep in cache
    this.tiles.delete(id);
  }

  _cacheInsert(id, rec) {
    const bytes = (rec.meta && rec.meta.approxBytes) ? rec.meta.approxBytes : 0;
    this.cache.set(id, rec);
    this.cacheBytes += bytes;

    // Evict oldest if cache too large
    while (this.cache.size > MAX_TILES || this.cacheBytes > MAX_CACHE_BYTES) {
      let evictCandidate = null;
      for (const key of this.cache.keys()) {
        if (this.rootTileIds.has(key)) {
          continue;
        }
        if (!this.tiles.has(key)) {
          evictCandidate = key;
          break;
        }
      }

      if (!evictCandidate) {
        // Every cached entry is currently active; stop trying to evict.
        break;
      }

      const evictRec = this.cache.get(evictCandidate);
      if (evictRec) {
        const eb = (evictRec.meta && evictRec.meta.approxBytes) ? evictRec.meta.approxBytes : 0;
        this.cacheBytes = Math.max(0, this.cacheBytes - eb);
        evictRec.obj3d.traverse(o => {
          if (o.isMesh || o.isLine) {
            o.geometry?.dispose();
            if (o.material) {
              this._disposeMaterial(o.material);
            }
          }
        });
        if (evictRec.bboxHelper) {
          this.scene.remove(evictRec.bboxHelper);
          evictRec.bboxHelper.geometry?.dispose();
          evictRec.bboxHelper.material?.dispose();
        }
      }
      this.cache.delete(evictCandidate);
    }
  }

  _cacheRemove(id) {
    if (!this.cache.has(id)) return;
    const rec = this.cache.get(id);
    const bytes = (rec.meta && rec.meta.approxBytes) ? rec.meta.approxBytes : 0;
    this.cache.delete(id);
    this.cacheBytes = Math.max(0, this.cacheBytes - bytes);
  }

  _applyActiveMaterial(rec) {
    if (this.tileColorMode) {
      this._applyTileColor(rec);
    } else if (this.wireOverlayEnabled) {
      this._applyOverlayBaseShading(rec);
    } else {
      this._restoreTileMaterial(rec);
    }
  }

  _applyTileColor(rec) {
    rec.obj3d.traverse(obj => {
      if (obj.isMesh && obj.userData && obj.userData.debugInfo) {
        const info = obj.userData.debugInfo;
        if (!info.overrideMaterial) {
          info.overrideMaterial = new THREE.MeshBasicMaterial({
            color: info.debugColor,
            side: THREE.DoubleSide,
            fog: false
          });
        }
        obj.material = info.overrideMaterial;
        const geom = obj.geometry;
        if (geom && info.originalColorAttr) {
          geom.deleteAttribute('color');
        }
      }
    });
  }

  _applyOverlayBaseShading(rec) {
    rec.obj3d.traverse(obj => {
      if (obj.isMesh && obj.userData && obj.userData.debugInfo) {
        const info = obj.userData.debugInfo;
        if (!info.overlayMaterial) {
          info.overlayMaterial = new THREE.MeshBasicMaterial({
            color: new THREE.Color(0.1, 0.1, 0.1),
            vertexColors: false,
            side: THREE.DoubleSide,
            fog: false
          });
        }
        obj.material = info.overlayMaterial;
      }
    });
  }

  _restoreTileMaterial(rec) {
    rec.obj3d.traverse(obj => {
      if (obj.isMesh && obj.userData && obj.userData.debugInfo) {
        const info = obj.userData.debugInfo;
        if (info.originalMaterial) {
          obj.material = info.originalMaterial;
        }
        const geom = obj.geometry;
        if (geom) {
          if (info.originalColorAttr) {
            const restored = info.originalColorAttr.clone();
            restored.needsUpdate = true;
            geom.setAttribute('color', restored);
          } else {
            geom.deleteAttribute('color');
          }
        }
      }
    });
  }

  updateBoundingBoxVisibility() {
    const activeHelpers = new Set();

    for (const [, rec] of this.tiles) {
      if (!rec.bboxHelper) continue;
      if (this.showBoundingBoxes) {
        rec.bboxHelper.visible = true;
        if (rec.bboxHelper.parent !== this.scene) {
          this.scene.add(rec.bboxHelper);
        }
        activeHelpers.add(rec.bboxHelper);
      } else {
        rec.bboxHelper.visible = false;
        if (rec.bboxHelper.parent === this.scene) {
          this.scene.remove(rec.bboxHelper);
        }
      }
    }

    for (const [id, rec] of this.cache) {
      if (this.tiles.has(id)) continue;
      if (!rec.bboxHelper) continue;
      rec.bboxHelper.visible = false;
      if (rec.bboxHelper.parent === this.scene) {
        this.scene.remove(rec.bboxHelper);
      }
    }

    // Remove any lingering helpers that no longer correspond to active tiles
    const strayHelpers = [];
    for (const child of this.scene.children) {
      if (child.isBox3Helper && !activeHelpers.has(child)) {
        strayHelpers.push(child);
      }
    }
    for (const helper of strayHelpers) {
      this.scene.remove(helper);
    }
  }

  clear() {
    this.manifestVersion += 1;
    this.rootTileIds.clear();
    this.levelGeMedian.clear();
    this.staticLayoutType = null;
    this.staticLayoutActive = false;
    this.datasetKey = null;
    this.currentTimeIndex = null;
    this.targetTimeIndex = null;
    this.staticRefreshPending = false;
    // Unload all tiles
    for (const [, rec] of this.tiles) {
      this.scene.remove(rec.obj3d);
      if (rec.bboxHelper) {
        this.scene.remove(rec.bboxHelper);
        rec.bboxHelper.geometry?.dispose();
        rec.bboxHelper.material?.dispose();
      }
    }
    this.tiles.clear();

    // Clear cache
    for (const [, rec] of this.cache) {
      rec.obj3d.traverse(o => {
        if (o.isMesh || o.isLine) {
          o.geometry?.dispose();
          if (o.material) {
            this._disposeMaterial(o.material);
          }
        }
      });
      if (rec.bboxHelper) {
        this.scene.remove(rec.bboxHelper);
        rec.bboxHelper.geometry?.dispose();
        rec.bboxHelper.material?.dispose();
      }
    }
    this.cache.clear();
    this.cacheBytes = 0;

    // Clear queue
    this.queue = [];
    this._queueSeq = 0;
    this.manifest = null;
    this.byId.clear();
  }
}

const prefetchedTileBuffers = new Map();
async function fetchPayloadJson(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`payload fetch failed: ${response.status}`);
    }
    return await response.json();
  } catch (err) {
    console.warn("Failed to load payload", url, err);
    return null;
  }
}
export async function initViewer(userConfig = {}) {
  const configDatasets = normalizeConfigDatasets(userConfig?.datasets);
  const configDefaultSse = Number(userConfig?.defaultSseRefine);
  if (Number.isFinite(configDefaultSse) && configDefaultSse > 0) {
    SSE_THRESHOLD_REFINE = configDefaultSse;
    SSE_THRESHOLD_COARSEN = Math.max(0.05, configDefaultSse * 0.5);
  }
  const datasetConfigMeta = new Map();
  if (configDatasets) {
    for (const ds of configDatasets) {
      datasetConfigMeta.set(ds.name, ds);
    }
  }
  const DEFAULT_EXTRACT_FALLBACK = 'dns-rough-2';
  let defaultExtract = userConfig?.defaultExtract
    ?? configDatasets?.[0]?.name
    ?? DEFAULT_EXTRACT_FALLBACK;
  if (configDatasets && configDatasets.length && !configDatasets.some(ds => ds.name === defaultExtract)) {
    defaultExtract = configDatasets[0].name;
  }
  let defaultTimeValue = userConfig?.defaultTime;
  if ((defaultTimeValue === undefined || defaultTimeValue === null) && defaultExtract && datasetConfigMeta.has(defaultExtract)) {
    defaultTimeValue = datasetConfigMeta.get(defaultExtract).defaultTime;
  }
  if (defaultTimeValue === undefined || defaultTimeValue === null) {
    defaultTimeValue = configDatasets?.[0]?.defaultTime ?? 0;
  }
  const CAMERA_DISTANCE_MULTIPLIER = 2.0;
  // Setup renderer
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0xe0e0e0, 1);
  document.getElementById('app').appendChild(renderer.domElement);
  const hudContainer = document.getElementById('hud');
  if (hudContainer && !hudContainer.querySelector('.repo-link')) {
    const link = document.createElement('a');
    link.href = REPO_URL;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = "Extract-Tiles GitHub";
    link.className = "repo-link";
    link.style.marginLeft = "8px";
    link.style.color = "#9ad8ff";
    link.style.textDecoration = "none";
    const sep = document.createElement('span');
    sep.textContent = "|";
    sep.style.margin = "0 6px";
    hudContainer.appendChild(sep);
    hudContainer.appendChild(link);
  }

  // Setup scene
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xe0e0e0); 

  // Setup camera
  const camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.01,
    10000
  );
  camera.position.set(0, 0, 2);
  scene.add(camera);

  // Setup controls
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.update();
  const mgr = new TileManager(scene, camera, renderer);

  // Setup lighting - ambient fill plus camera-attached key light for lambert shading
  scene.add(new THREE.AmbientLight(0xffffff, 0.35));
  const cameraLight = new THREE.DirectionalLight(0xffffff, 1.0);
  cameraLight.position.set(0, 0, 1);
  camera.add(cameraLight);

  const DEFAULT_EXTRACT = defaultExtract || DEFAULT_EXTRACT_FALLBACK;
  const DEFAULT_TIME_STRING = String(defaultTimeValue ?? 0);
  const DISPLAY_MODES = {
    SHADED: 'shaded',
    WIREFRAME: 'wireframe',
    TILE_COLOURS: 'tileColors',
    WIRE_OVERLAY: 'wireOverlay'
  };
  const DISPLAY_MODE_OPTIONS = {
    'Shaded (Phong)': DISPLAY_MODES.SHADED,
    'Wireframe': DISPLAY_MODES.WIREFRAME,
    'Tile Colours': DISPLAY_MODES.TILE_COLOURS,
    'Wireframe Overlay': DISPLAY_MODES.WIRE_OVERLAY
  };

  const settings = {
    extract: DEFAULT_EXTRACT,
    time: DEFAULT_TIME_STRING,
    sseRefine: SSE_THRESHOLD_REFINE,
    preserveCameraView: false,
    boundingBoxes: true,
    displayMode: DISPLAY_MODES.SHADED,
    wireframeOverlayOpacity: 1.0,
    showAxes: false,
  };

  mgr.showBoundingBoxes = settings.boundingBoxes;
  mgr.displayMode = settings.displayMode;
  mgr.wireOverlayOpacity = settings.wireframeOverlayOpacity;
  const axesHelper = new THREE.AxesHelper(0.5);
  axesHelper.visible = settings.showAxes;
  scene.add(axesHelper);

  const gui = new GUI({ width: 300 });
  let datasetFolder = null;
  function ensureDatasetFolder() {
    if (!datasetFolder) {
      datasetFolder = gui.addFolder('Dataset');
    }
    datasetFolder.domElement.style.display = '';
    return datasetFolder;
  }
  function hideDatasetFolder() {
    if (datasetFolder) {
      datasetFolder.domElement.style.display = 'none';
    }
  }
  const lodFolder = gui.addFolder('LOD');
  const diagnosticsFolder = gui.addFolder('Diagnostics');

  let extractController = null;
  let timeController = null;
  let displayModeController = null;
  let sseRefineController = null;
  let preserveCameraController = null;
  let suppressSseHandler = false;
  let userAdjustedSSE = false;
  let timeIsSlider = false;
  let extractsCache = [];
  let hasLoadedDataset = false;
  let currentManifestKey = null;
  let currentExtract = null;
  let userToggledPreserveCamera = false;
  let currentDatasetHasMultipleTimes = false;
  let autoEnabledPreserveCamera = false;
  let suppressPreserveHandler = false;
  let hasMultipleExtracts = false;
  let skipInitialRecenter = false;

  function applyWireframe(wireframe) {
    const apply = (obj) => {
      if (!obj) return;
      obj.traverse(o => {
        if (o.isMesh && o.material) {
          if ('wireframe' in o.material) {
            o.material.wireframe = wireframe;
          }
          o.material.transparent = false;
          o.material.opacity = 1.0;
        }
      });
    };

    for (const [, rec] of mgr.tiles) {
      apply(rec.obj3d);
    }
    for (const [, rec] of mgr.cache) {
      apply(rec.obj3d);
    }
  }

  function applyDisplayMode(mode) {
    if (!Object.values(DISPLAY_MODES).includes(mode)) {
      mode = DISPLAY_MODES.SHADED;
    }
    settings.displayMode = mode;
    mgr.displayMode = mode;
    mgr.tileColorMode = (mode === DISPLAY_MODES.TILE_COLOURS);

    const wantWireframe = (mode === DISPLAY_MODES.WIREFRAME);
    mgr.wireframeMode = wantWireframe;
    applyWireframe(wantWireframe);

    const wantOverlay = (mode === DISPLAY_MODES.WIRE_OVERLAY);
    mgr.applyWireOverlayState(wantOverlay);

    const applyState = (rec) => {
      if (mgr.tileColorMode) {
        mgr._applyTileColor(rec);
      } else if (wantOverlay) {
        mgr._applyOverlayBaseShading(rec);
      } else {
        mgr._restoreTileMaterial(rec);
      }
    };

    for (const [, rec] of mgr.tiles) {
      applyState(rec);
    }
    for (const [id, rec] of mgr.cache) {
      if (mgr.tiles.has(id)) continue;
      applyState(rec);
    }
  }

  function resolveManifestTargets(extractName, timeValue) {
    const t = Number(timeValue);
    const descriptor = datasetConfigMeta.get(extractName);
    if (!descriptor) {
      return {
        manifestUrl: `/manifest/${extractName}/${t}.json`,
        payloadUrl: `/payload/${extractName}/${t}.json`,
        tilesBaseOverride: null,
      };
    }
    const entry = descriptor.manifestEntries.get(t);
    const manifestUrl = (entry && entry.manifestUrl)
      ? entry.manifestUrl
      : (descriptor.manifestPattern
        ? resolveAbsoluteUrl(formatPattern(descriptor.manifestPattern, extractName, t))
        : `/manifest/${extractName}/${t}.json`);
    let payloadUrl;
    if (entry && Object.prototype.hasOwnProperty.call(entry, 'payloadUrl')) {
      payloadUrl = entry.payloadUrl;
    } else if (descriptor.payloadPattern) {
      payloadUrl = resolveAbsoluteUrl(formatPattern(descriptor.payloadPattern, extractName, t));
    } else {
      payloadUrl = `/payload/${extractName}/${t}.json`;
    }
    return {
      manifestUrl,
      payloadUrl,
      tilesBaseOverride: descriptor.tilesBasePath || null,
    };
  }

  function calibrateSSEThreshold() {
    if (userAdjustedSSE) {
      return false;
    }

    if (!mgr || !mgr.rootTileIds || !mgr.rootTileIds.size) {
      return false;
    }

    let maxSSE = 0;
    for (const id of mgr.rootTileIds) {
      const meta = mgr.byId.get(id);
      if (!meta) continue;
      const sse = mgr._sse(meta);
      if (!Number.isFinite(sse)) continue;
      if (sse > maxSSE) {
        maxSSE = sse;
      }
    }

    if (!Number.isFinite(maxSSE) || maxSSE <= 0) {
      return false;
    }

    const target = Math.max(maxSSE * 1.1, 30.0);
    const clamped = Math.min(target, 120.0);
    SSE_THRESHOLD_REFINE = clamped;
    SSE_THRESHOLD_COARSEN = Math.max(0.05, clamped * 0.5);
    settings.sseRefine = clamped;
    if (sseRefineController) {
      suppressSseHandler = true;
      sseRefineController.setValue(clamped);
      suppressSseHandler = false;
    }
    return true;
  }

  function rebuildExtractController() {
    if (extractController) {
      if (datasetFolder) {
        datasetFolder.remove(extractController);
      } else {
        gui.remove(extractController);
      }
    }

    const names = extractsCache.length ? extractsCache.map(e => e.name) : [DEFAULT_EXTRACT];
    const showExtractDropdown = hasMultipleExtracts && names.length > 1;
    if (!names.includes(settings.extract)) {
      settings.extract = names[0];
    }

    if (!showExtractDropdown) {
      extractController = null;
      return;
    }

    const folder = ensureDatasetFolder();
    extractController = folder.add(settings, 'extract', names);
    extractController.name('Extract');
    extractController.onChange((value) => {
      settings.extract = value;
      rebuildTimeController();
      loadManifest();
    });
    extractController.updateDisplay?.();
  }

  function rebuildTimeController() {
    if (timeController) {
      if (timeController.__useDatasetFolder && datasetFolder) {
        datasetFolder.remove(timeController);
      } else {
        gui.remove(timeController);
      }
      timeController = null;
    }

    const selected = extractsCache.find(e => e.name === settings.extract);
    const times = selected && selected.times && selected.times.length ? selected.times : [0];
    currentDatasetHasMultipleTimes = times.length > 1;

    if (!times.includes(Number(settings.time))) {
      settings.time = String(times[0]);
    }

    if (times.length <= 1) {
      timeIsSlider = false;
      settings.time = String(times[0] ?? 0);
      return;
    }

    const useSlider = times.length <= 100;
    timeIsSlider = useSlider;
    const attachToFolder = hasMultipleExtracts;
    const targetFolder = attachToFolder ? ensureDatasetFolder() : gui;
    if (useSlider) {
      const min = Math.min(...times);
      const max = Math.max(...times);
      settings.timeIndex = Number(settings.time);
      timeController = targetFolder.add(settings, 'timeIndex', min, max, 1).name('Time');
      timeController.__useDatasetFolder = attachToFolder;
      timeController.onChange((value) => {
        const rounded = Math.round(value);
        const clamped = Math.min(Math.max(rounded, min), max);
        settings.timeIndex = clamped;
        const newTime = String(clamped);
        const manifestKey = `${settings.extract}:${newTime}`;
        if (currentManifestKey === manifestKey) {
          settings.time = newTime;
          return;
        }
        settings.time = newTime;
        loadManifest();
      });
      timeController.updateDisplay?.();
    } else {
      const timeStrings = times.map(t => t.toString());
      timeController = targetFolder.add(settings, 'time', timeStrings).name('Time');
      timeController.__useDatasetFolder = attachToFolder;
      timeController.onChange((value) => {
        const manifestKey = `${settings.extract}:${value}`;
        if (currentManifestKey === manifestKey) {
          settings.time = value;
          return;
        }
        settings.time = value;
        loadManifest();
      });
      timeController.updateDisplay?.();
    }
  }

  function recenterCamera() {
    const manifest = mgr.manifest;
    if (!manifest || !manifest.tiles || !manifest.tiles.length) {
      return;
    }

    const roots = manifest.tiles.filter(t => t.parent == null && t.aabbWorld);
    const candidates = roots.length ? roots : manifest.tiles;
    const bbox = new THREE.Box3();

    for (const tile of candidates) {
      if (!tile.aabbWorld || tile.aabbWorld.length !== 2) continue;
      const lo = tile.aabbWorld[0];
      const hi = tile.aabbWorld[1];
      if (!Array.isArray(lo) || !Array.isArray(hi) || lo.length !== 3 || hi.length !== 3) continue;
      bbox.expandByPoint(new THREE.Vector3().fromArray(lo));
      bbox.expandByPoint(new THREE.Vector3().fromArray(hi));
    }

    if (!isFinite(bbox.min.x) || !isFinite(bbox.max.x)) {
      return;
    }

    const center = bbox.getCenter(new THREE.Vector3());
    const size = bbox.getSize(new THREE.Vector3());
    const radius = Math.max(size.length() * 0.5, 0.05);
    const baseDistance = radius * CAMERA_DISTANCE_MULTIPLIER;
    const distance = Math.max(baseDistance, 0.05);
    const direction = new THREE.Vector3(0, 0, 1);

    camera.position.copy(center.clone().add(direction.multiplyScalar(distance)));
    camera.lookAt(center);
    controls.target.copy(center);
    controls.update();
  }

  async function loadManifest() {
    const manifestTargets = resolveManifestTargets(settings.extract, settings.time);
    const manifestUrl = manifestTargets.manifestUrl;
    const payloadUrl = manifestTargets.payloadUrl;
    const tileBaseOverride = manifestTargets.tilesBaseOverride;
    const extractChanged = currentExtract !== settings.extract;
    prefetchedTileBuffers.clear();
    let manifest = null;
    const datasetKey = settings.extract;
    const canUsePayload = !!payloadUrl && !extractChanged
      && !mgr.staticReuseDisabled
      && mgr.staticLayoutActive
      && mgr.datasetKey === datasetKey
      && mgr.manifest
      && mgr.manifest.layout?.type === "static-octree";

    if (canUsePayload) {
      const payload = await fetchPayloadJson(payloadUrl);
      if (payload && mgr.applyStaticPayload(payload)) {
        manifest = mgr.manifest;
      } else {
        console.warn("Failed to apply payload; falling back to full manifest");
      }
    }

    if (!manifest) {
      try {
        const response = await fetch(manifestUrl);
        manifest = await response.json();
      } catch (err) {
        console.error('Failed to load manifest JSON:', err);
        return;
      }
    }
    if (extractChanged) {
      currentExtract = settings.extract;
      hasLoadedDataset = false;
      autoEnabledPreserveCamera = false;
      if (!userToggledPreserveCamera && settings.preserveCameraView) {
        settings.preserveCameraView = false;
        if (preserveCameraController) {
          suppressPreserveHandler = true;
          preserveCameraController.setValue(false);
          suppressPreserveHandler = false;
        }
      }
    }
    mgr.showBoundingBoxes = settings.boundingBoxes;
    mgr.wireframeMode = (settings.displayMode === DISPLAY_MODES.WIREFRAME);
    const manifestLayout = manifest?.layout?.type || null;
    const manifestTime = manifest?.time != null ? manifest.time : settings.time;
    const wantsStatic = (manifestLayout === "static-octree");
    const canReuse = wantsStatic && !mgr.staticReuseDisabled && mgr.staticLayoutActive && mgr.datasetKey === datasetKey && mgr.tiles.size > 0;
    if (!canReuse) {
      mgr.clear();
    } else {
      mgr.staticRefreshPending = true;
    }
    SSE_THRESHOLD_REFINE = 1e6;
    SSE_THRESHOLD_COARSEN = 5e5;
    await mgr.init(manifestUrl, manifest, {
      datasetKey,
      timeIndex: manifestTime,
      reuseTiles: canReuse,
      tileBaseOverride,
    });
    let shouldRecenter = (!settings.preserveCameraView || !hasLoadedDataset);
    if (skipInitialRecenter && !hasLoadedDataset) {
      shouldRecenter = false;
      skipInitialRecenter = false;
    }
    if (shouldRecenter) {
      recenterCamera();
    }
    if (!hasLoadedDataset) {
      hasLoadedDataset = true;
      if (!settings.preserveCameraView && currentDatasetHasMultipleTimes && !userToggledPreserveCamera) {
        settings.preserveCameraView = true;
        if (preserveCameraController) {
          suppressPreserveHandler = true;
          preserveCameraController.setValue(true);
          suppressPreserveHandler = false;
        }
        autoEnabledPreserveCamera = true;
      }
    } else if (currentDatasetHasMultipleTimes && autoEnabledPreserveCamera && !settings.preserveCameraView) {
      settings.preserveCameraView = true;
      if (preserveCameraController) {
        suppressPreserveHandler = true;
        preserveCameraController.setValue(true);
        suppressPreserveHandler = false;
      }
    }
    const calibrated = ENABLE_SSE_AUTO_CALIBRATION ? calibrateSSEThreshold() : false;
    if (!calibrated) {
      SSE_THRESHOLD_REFINE = settings.sseRefine;
      SSE_THRESHOLD_COARSEN = Math.max(0.05, settings.sseRefine * 0.5);
      sseRefineController?.updateDisplay?.();
    }
    mgr.updateBoundingBoxVisibility();
    await mgr.tick();
    applyDisplayMode(settings.displayMode);
    currentManifestKey = `${settings.extract}:${settings.time}`;
  }

  async function loadExtractsList() {
    if (configDatasets && configDatasets.length) {
      extractsCache = configDatasets.map(ds => ({
        name: ds.name,
        label: ds.label,
        times: ds.times,
      }));
      hasMultipleExtracts = extractsCache.length > 1;
      if (!extractsCache.some(e => e.name === settings.extract)) {
        settings.extract = extractsCache[0].name;
        const meta = datasetConfigMeta.get(settings.extract);
        const fallbackTime = meta?.defaultTime ?? meta?.times?.[0] ?? 0;
        settings.time = String(fallbackTime);
      }
      rebuildExtractController();
      rebuildTimeController();
      refreshPreserveCameraController();
      if (hasMultipleExtracts) {
        ensureDatasetFolder().open();
      } else {
        hideDatasetFolder();
      }
      await loadManifest();
      return;
    }
    try {
      const response = await fetch('/api/extracts');
      extractsCache = await response.json();
      hasMultipleExtracts = extractsCache.length > 1;
      rebuildExtractController();
      rebuildTimeController();
      refreshPreserveCameraController();
      if (hasMultipleExtracts) {
        ensureDatasetFolder().open();
      } else {
        hideDatasetFolder();
      }
      await loadManifest();
    } catch (err) {
      console.error('Failed to load extracts list:', err);
      extractsCache = [];
      hasMultipleExtracts = false;
      settings.extract = DEFAULT_EXTRACT;
      settings.time = DEFAULT_TIME_STRING;
      rebuildExtractController();
      rebuildTimeController();
      refreshPreserveCameraController();
      await loadManifest();
    }
  }

  sseRefineController = lodFolder.add(settings, 'sseRefine', 0.1, 200, 0.1).name('SSE Refine').onChange((value) => {
    if (suppressSseHandler) {
      settings.sseRefine = value;
      return;
    }
    userAdjustedSSE = true;
    SSE_THRESHOLD_REFINE = value;
    SSE_THRESHOLD_COARSEN = Math.max(0.05, value * 0.5);
    mgr.tick();
  });

  diagnosticsFolder.add(settings, 'boundingBoxes').name('Bounding Boxes').onChange((value) => {
    settings.boundingBoxes = value;
    mgr.showBoundingBoxes = value;
    mgr.updateBoundingBoxVisibility();
  });

  diagnosticsFolder.add(settings, 'showAxes').name('Axes helper').onChange((value) => {
    settings.showAxes = value;
    axesHelper.visible = value;
  });

  displayModeController = diagnosticsFolder.add(settings, 'displayMode', DISPLAY_MODE_OPTIONS).name('Display mode').onChange((value) => {
    applyDisplayMode(value);
    mgr.tick();
  });


  diagnosticsFolder.open();
  lodFolder.open();

  function refreshPreserveCameraController() {
    if (preserveCameraController) {
      datasetFolder.remove(preserveCameraController);
      preserveCameraController = null;
    }
    if (!hasMultipleExtracts) {
      settings.preserveCameraView = false;
      return;
    }
    const folder = ensureDatasetFolder();
    preserveCameraController = folder.add(settings, 'preserveCameraView').name('Preserve camera view').onChange((value) => {
      settings.preserveCameraView = value;
      if (!suppressPreserveHandler) {
        userToggledPreserveCamera = true;
      }
      if (!value) {
        autoEnabledPreserveCamera = false;
      }
    });
  }

  // Update on controls change
  controls.addEventListener('change', () => mgr.tick());

  // Handle resize
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    mgr.tick();
  });

  // Animation loop
  function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
  }
  animate();

  // Initialize
  await loadExtractsList();
}

export default initViewer;
