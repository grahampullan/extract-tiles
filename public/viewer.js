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
    this.hudLOD = document.getElementById("lod") || { textContent: "" };
    this.hudTiles = document.getElementById("tiles") || { textContent: "" };
    this.hudCache = document.getElementById("cache") || { textContent: "" };
    this.loadingIndicator = document.getElementById("loading");
    this.wireframeMode = false; // Track wireframe state
    this.showBoundingBoxes = false; // Diagnostic overlay toggle
    this.tileColorMode = false; // Diagnostic colouring toggle
    this.simpleShadingMode = false; // Optional lambert shading toggle
    this.levelGeMedian = new Map(); // Depth -> geometric error median
    this.requestPriority = new Map(); // TileId -> SSE-based priority
    this.rootTileIds = new Set();
    this._tickLock = false;
    this._tickPending = false;
    this.manifestVersion = 0;
    this._queueSeq = 0;
    this.manifestUrl = null;
    this.tileBaseUrl = null;
  }

  async init(manifestUrl) {
    const version = ++this.manifestVersion;
    this.manifestUrl = manifestUrl;
    this.rootTileIds.clear();
    try {
      const manifest = await (await fetch(manifestUrl)).json();
      if (version !== this.manifestVersion) {
        return;
      }
      this.manifest = manifest;
      this.tileBaseUrl = deduceTileBaseUrl(manifestUrl, manifest);
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
          this._enqueue(t.tileId);
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
    const min = new THREE.Vector3(...meta.aabbWorld[0]);
    const max = new THREE.Vector3(...meta.aabbWorld[1]);
    return this.frustum.intersectsBox(new THREE.Box3(min, max));
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

    this.hudLOD.textContent = `τ=${SSE_THRESHOLD_REFINE.toFixed(1)}px / ${SSE_THRESHOLD_COARSEN.toFixed(1)}px`;
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
      if (want.has(id) && !this.tiles.has(id)) {
        this._enqueue(id, this.requestPriority.get(id));
      }
    }
    for (const id of want) {
      if (this.rootTileIds.has(id)) continue;
      if (!this.tiles.has(id)) {
        this._enqueue(id, this.requestPriority.get(id));
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
        const nextId = nextEntry.id;
        launched.push(this._load(nextId));
      }
      if (launched.length) {
        this._setLoadingIndicator(true);
        const loaded = await Promise.all(launched);
        if (versionAtStart !== this.manifestVersion) return;
        // Hide freshly loaded tiles that are no longer needed
        for (const rec of loaded) {
          if (!rec) continue;
          const id = rec.meta.tileId;
          const keepAsFallback = replaceParents.has(id) && this._hasPendingDescendants(id, want);
          const needed = want.has(id) || keepAsFallback;
          if (!needed) {
            this._unload(id);
          } else {
            rec.obj3d.visible = true;
            if (rec.bboxHelper) {
              rec.bboxHelper.visible = this.showBoundingBoxes;
            }
          }
        }
      }
    } while (this.queue.length && this.inflight < MAX_CONCURRENT);

    this.hudTiles.textContent = `${this.tiles.size}`;
    this.hudCache.textContent = `${this.cache.size}`;

    this.updateBoundingBoxVisibility();
    if (versionAtStart !== this.manifestVersion) return;

    this._setLoadingIndicator(this.inflight > 0 || this.queue.length > 0);
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

  _enqueue(id, priority = null) {
    const priorityHint = (priority ?? this.requestPriority.get(id) ?? 0);
    // Check cache first
    if (this.cache.has(id)) {
      const rec = this.cache.get(id);
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
      if (this.tileColorMode) {
        this._applyTileColor(rec);
      } else if (this.simpleShadingMode) {
        this._applySimpleShading(rec);
      } else {
        this._restoreTileMaterial(rec);
      }
      return;
    }

    // Avoid duplicate queue entries
    const existing = this.queue.find(entry => entry.id === id);
    if (existing) {
      if (priorityHint !== existing.priority) {
        existing.priority = priorityHint;
        this._sortQueue();
      }
      return;
    }

    this.queue.push({ id, priority: priorityHint, seq: this._queueSeq++ });
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

  async _load(id) {
    const meta = this.byId.get(id);
    if (!meta) return null;

    const tileUrl = resolveTileUrl(meta.url, this.tileBaseUrl);
    if (!tileUrl) return null;

    const versionAtStart = this.manifestVersion;
    this.inflight++;
    let rec = null;
    try {
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
          if (o.isMesh) {
            o.geometry?.dispose();
            if (o.material) {
              if (o.material.map) o.material.map.dispose();
              o.material.dispose?.();
            }
          }
        });
        return null;
      }
      const obj = glb.scene;
      obj.userData.tileId = id;

      obj.traverse(o => {
        if (o.isMesh && o.material) {
          const newMaterial = new THREE.MeshStandardMaterial({
            vertexColors: true,
            transparent: false,
            opacity: 1.0,
            wireframe: this.wireframeMode,
            side: THREE.DoubleSide
          });

          if (o.material.map) {
            newMaterial.map = o.material.map;
          }

          o.material = newMaterial;
          o.userData.debugInfo = {
            originalMaterial: newMaterial,
            debugColor: new THREE.Color().setHSL(Math.random(), 0.7, 0.5),
            originalColorAttr: o.geometry && o.geometry.attributes && o.geometry.attributes.color
              ? o.geometry.attributes.color.clone()
              : null,
            overrideMaterial: null
          };

          if (!(o.geometry && o.geometry.attributes.color)) {
            o.material = new THREE.MeshStandardMaterial({
              color: new THREE.Color().setHSL(Math.random(), 0.8, 0.6),
              vertexColors: false,
              transparent: false,
              opacity: 1.0,
              wireframe: this.wireframeMode,
              side: THREE.DoubleSide
            });
            o.userData.debugInfo.originalMaterial = o.material;
          }
        }
      });

      this.scene.add(obj);

      const bboxHelper = this._createBoundingBoxHelper(meta);
      bboxHelper.userData.tileId = id;
      if (this.showBoundingBoxes) {
        bboxHelper.visible = true;
        this.scene.add(bboxHelper);
      } else {
        bboxHelper.visible = false;
      }

      rec = { obj3d: obj, meta, bboxHelper };
      this.tiles.set(id, rec);
      this._cacheInsert(id, rec);

      if (this.tileColorMode) {
        this._applyTileColor(rec);
      } else if (this.simpleShadingMode) {
        this._applySimpleShading(rec);
      }
    } catch (e) {
      console.error("Tile load failed", id, e);
    } finally {
      this.inflight--;
    }

    if (versionAtStart !== this.manifestVersion) {
      if (rec) {
        this._unload(id);
      }
      return null;
    }

    return rec;
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
              if (o.material.map) o.material.map.dispose();
              o.material.dispose?.();
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

  _applySimpleShading(rec) {
    rec.obj3d.traverse(obj => {
      if (obj.isMesh && obj.userData && obj.userData.debugInfo) {
        const info = obj.userData.debugInfo;
        if (!info.simpleMaterial) {
          const hasVertexColors = !!(info.originalMaterial && info.originalMaterial.vertexColors);
          const baseColor = info.originalMaterial && info.originalMaterial.color
            ? info.originalMaterial.color.clone()
            : new THREE.Color(0.8, 0.8, 0.8);
          const specularColor = baseColor.clone().lerp(new THREE.Color(1, 1, 1), 0.5);
          info.simpleMaterial = new THREE.MeshPhongMaterial({
            color: baseColor,
            vertexColors: hasVertexColors,
            side: THREE.DoubleSide,
            shininess: 60,
            specular: specularColor
          });
        }
        const geom = obj.geometry;
        if (geom && !geom.attributes?.normal) {
          geom.computeVertexNormals?.();
          info.generatedNormals = true;
        } else if (geom) {
          info.generatedNormals = false;
        }
        obj.material = info.simpleMaterial;
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
          if (info.generatedNormals) {
            geom.deleteAttribute('normal');
            info.generatedNormals = false;
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
            if (o.material.map) o.material.map.dispose();
            o.material.dispose?.();
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
const prefetchInFlight = new Set();

async function prefetchRootTiles(extract, time) {
  const key = `${extract}:${time}`;
  if (prefetchInFlight.has(key)) return;
  prefetchInFlight.add(key);
  try {
    const manifestUrl = `/manifest/${extract}/${time}.json`;
    const resp = await fetch(manifestUrl);
    if (!resp.ok) return;
    const manifest = await resp.json();
    const tileBaseUrl = deduceTileBaseUrl(manifestUrl, manifest);
    const roots = (manifest.tiles || []).filter(t => t.z === 0);
    await Promise.all(roots.map(async (tile) => {
      const resolvedUrl = resolveTileUrl(tile.url, tileBaseUrl);
      if (!resolvedUrl || prefetchedTileBuffers.has(resolvedUrl)) return;
      try {
        const res = await fetch(resolvedUrl);
        if (!res.ok) return;
        const buffer = await res.arrayBuffer();
        prefetchedTileBuffers.set(resolvedUrl, buffer);
      } catch (err) {
        console.warn('Prefetch tile failed', tile.url, err);
      }
    }));
  } catch (err) {
    console.warn('Prefetch manifest failed', extract, time, err);
  } finally {
    prefetchInFlight.delete(key);
  }
}

function schedulePrefetchNeighbours(settings, extractsCache) {
  const selected = extractsCache.find(e => e.name === settings.extract);
  if (!selected || !selected.times || !selected.times.length) return;
  const times = selected.times.slice().sort((a, b) => a - b);
  const current = Number(settings.time);
  const offsets = [1, 2, -1, -2];
  for (const offset of offsets) {
    const target = current + offset;
    if (times.includes(target)) {
      prefetchRootTiles(settings.extract, target);
    }
  }
}

// Main application
(async function main() {
  // Setup renderer
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  document.getElementById('app').appendChild(renderer.domElement);

  // Setup scene
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000); 

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
  const mgr = new TileManager(scene, camera, renderer);

  // Setup lighting - ambient fill plus camera-attached key light for lambert shading
  scene.add(new THREE.AmbientLight(0xffffff, 0.35));
  const cameraLight = new THREE.DirectionalLight(0xffffff, 1.0);
  cameraLight.position.set(0, 0, 1);
  camera.add(cameraLight);

  const DEFAULT_EXTRACT = 'dns-rough-2';

  const settings = {
    extract: DEFAULT_EXTRACT,
    time: '0',
    sseRefine: SSE_THRESHOLD_REFINE,
    wireframe: false,
    boundingBoxes: true,
    tileColorMode: false,
    simpleShading: true
  };

  mgr.showBoundingBoxes = settings.boundingBoxes;
  mgr.simpleShadingMode = settings.simpleShading;

  const gui = new GUI({ width: 300 });
  const datasetFolder = gui.addFolder('Dataset');
  const lodFolder = gui.addFolder('LOD');
  const diagnosticsFolder = gui.addFolder('Diagnostics');

  let extractController = null;
  let timeController = null;
  let tileColorController = null;
  let simpleShadingController = null;
  let sseRefineController = null;
  let suppressTileColorHandler = false;
  let suppressSimpleShadingHandler = false;
  let suppressSseHandler = false;
  let userAdjustedSSE = false;
  let timeIsSlider = false;
  let extractsCache = [];

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

  function applyTileColorState(value) {
    settings.tileColorMode = value;
    mgr.tileColorMode = value;

    if (value) {
      for (const [, rec] of mgr.tiles) {
        mgr._applyTileColor(rec);
      }
      for (const [id, rec] of mgr.cache) {
        if (mgr.tiles.has(id)) continue;
        mgr._applyTileColor(rec);
      }
    } else {
      for (const [, rec] of mgr.tiles) {
        if (mgr.simpleShadingMode) {
          mgr._applySimpleShading(rec);
        } else {
          mgr._restoreTileMaterial(rec);
        }
      }
      for (const [id, rec] of mgr.cache) {
        if (mgr.tiles.has(id)) continue;
        if (mgr.simpleShadingMode) {
          mgr._applySimpleShading(rec);
        } else {
          mgr._restoreTileMaterial(rec);
        }
      }
    }
  }

  function applySimpleShadingState(value) {
    settings.simpleShading = value;
    mgr.simpleShadingMode = value;

    if (value) {
      for (const [, rec] of mgr.tiles) {
        mgr._applySimpleShading(rec);
      }
      for (const [id, rec] of mgr.cache) {
        if (mgr.tiles.has(id)) continue;
        mgr._applySimpleShading(rec);
      }
    } else {
      for (const [, rec] of mgr.tiles) {
        if (mgr.tileColorMode) {
          mgr._applyTileColor(rec);
        } else {
          mgr._restoreTileMaterial(rec);
        }
      }
      for (const [id, rec] of mgr.cache) {
        if (mgr.tiles.has(id)) continue;
        if (mgr.tileColorMode) {
          mgr._applyTileColor(rec);
        } else {
          mgr._restoreTileMaterial(rec);
        }
      }
    }
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
      datasetFolder.remove(extractController);
    }

    const names = extractsCache.length ? extractsCache.map(e => e.name) : [DEFAULT_EXTRACT];
    if (!names.includes(settings.extract)) {
      settings.extract = names[0];
    }

    extractController = datasetFolder.add(settings, 'extract', names);
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
      datasetFolder.remove(timeController);
      timeController = null;
    }

    const selected = extractsCache.find(e => e.name === settings.extract);
    const times = selected && selected.times && selected.times.length ? selected.times : [0];

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
    if (useSlider) {
      const min = Math.min(...times);
      const max = Math.max(...times);
      settings.timeIndex = Number(settings.time);
      timeController = datasetFolder.add(settings, 'timeIndex', min, max, 1).name('Time');
      timeController.onChange((value) => {
        settings.time = String(Math.round(value));
        loadManifest();
      });
      timeController.updateDisplay?.();
    } else {
      const timeStrings = times.map(t => t.toString());
      timeController = datasetFolder.add(settings, 'time', timeStrings).name('Time');
      timeController.onChange((value) => {
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
    const radius = Math.max(size.length() * 0.5, 0.5);
    const distance = Math.max(radius * 2.5, 1.0);
    const direction = new THREE.Vector3(0, 0, 1);

    camera.position.copy(center.clone().add(direction.multiplyScalar(distance)));
    camera.lookAt(center);
    controls.target.copy(center);
    controls.update();
  }

  async function loadManifest() {
    const manifestUrl = `/manifest/${settings.extract}/${settings.time}.json`;
    mgr.showBoundingBoxes = settings.boundingBoxes;
    mgr.simpleShadingMode = settings.simpleShading;
    mgr.wireframeMode = settings.wireframe;
    mgr.clear();
    SSE_THRESHOLD_REFINE = 1e6;
    SSE_THRESHOLD_COARSEN = 5e5;
    await mgr.init(manifestUrl);
    recenterCamera();
    const calibrated = ENABLE_SSE_AUTO_CALIBRATION ? calibrateSSEThreshold() : false;
    if (!calibrated) {
      SSE_THRESHOLD_REFINE = settings.sseRefine;
      SSE_THRESHOLD_COARSEN = Math.max(0.05, settings.sseRefine * 0.5);
      sseRefineController?.updateDisplay?.();
    }
    applyWireframe(settings.wireframe);
    mgr.updateBoundingBoxVisibility();
    await mgr.tick();
    if (settings.simpleShading) {
      applySimpleShadingState(true);
    }
    schedulePrefetchNeighbours(settings, extractsCache);
  }

  async function loadExtractsList() {
    try {
      const response = await fetch('/api/extracts');
      extractsCache = await response.json();
      rebuildExtractController();
      rebuildTimeController();
      datasetFolder.open();
      await loadManifest();
    } catch (err) {
      console.error('Failed to load extracts list:', err);
      extractsCache = [];
      settings.extract = DEFAULT_EXTRACT;
      settings.time = '0';
      rebuildExtractController();
      rebuildTimeController();
      await loadManifest();
    }
  }

  sseRefineController = lodFolder.add(settings, 'sseRefine', 0.1, 120, 0.1).name('SSE Refine').onChange((value) => {
    if (suppressSseHandler) {
      settings.sseRefine = value;
      return;
    }
    userAdjustedSSE = true;
    SSE_THRESHOLD_REFINE = value;
    SSE_THRESHOLD_COARSEN = Math.max(0.05, value * 0.5);
    mgr.tick();
  });

  diagnosticsFolder.add(settings, 'wireframe').name('Wireframe').onChange((value) => {
    settings.wireframe = value;
    mgr.wireframeMode = value;
    applyWireframe(value);
    mgr.tick();
  });

  diagnosticsFolder.add(settings, 'boundingBoxes').name('Bounding Boxes').onChange((value) => {
    settings.boundingBoxes = value;
    mgr.showBoundingBoxes = value;
    mgr.updateBoundingBoxVisibility();
  });

  tileColorController = diagnosticsFolder.add(settings, 'tileColorMode').name('Tile Colours').onChange((value) => {
    if (suppressTileColorHandler) {
      settings.tileColorMode = value;
      return;
    }

    if (value && settings.simpleShading) {
      if (simpleShadingController) {
        suppressSimpleShadingHandler = true;
        simpleShadingController.setValue(false);
        suppressSimpleShadingHandler = false;
      }
      applySimpleShadingState(false);
    }

    applyTileColorState(value);
  });

  simpleShadingController = diagnosticsFolder.add(settings, 'simpleShading').name('Simple Shading').onChange((value) => {
    if (suppressSimpleShadingHandler) {
      settings.simpleShading = value;
      return;
    }

    if (value && settings.tileColorMode) {
      if (tileColorController) {
        suppressTileColorHandler = true;
        tileColorController.setValue(false);
        suppressTileColorHandler = false;
      }
      applyTileColorState(false);
    }

    applySimpleShadingState(value);
  });

  diagnosticsFolder.open();
  lodFolder.open();

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
})();
    this.rootTileIds = new Set();
