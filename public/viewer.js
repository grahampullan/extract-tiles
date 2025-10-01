import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GUI } from "dat.gui";

// Configuration
let SSE_THRESHOLD_REFINE = 3.0;   // pixels
let SSE_THRESHOLD_COARSEN = 1.5;  // hysteresis (auto-updated)
const MAX_CONCURRENT = 6;
const MAX_CACHE_BYTES = 600 * 1024 * 1024; // ~600 MB budget
const MAX_TILES = 200;

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
    this._tickLock = false;
    this._tickPending = false;
  }

  async init(manifestUrl) {
    try {
      this.manifest = await (await fetch(manifestUrl)).json();
      for (const t of this.manifest.tiles) {
        this.byId.set(t.tileId, t);
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

    const visit = (meta) => {
      if (!this._visible(meta)) return;

      const sse = this._sse(meta);
      const hasChildren = (meta.children || []).length > 0;

      if (hasChildren && sse > SSE_THRESHOLD_REFINE) {
        replaceParents.add(meta.tileId);
        // Need more detail - recurse to children
        for (const c of this._children(meta)) {
          visit(c);
        }
      } else if (sse < SSE_THRESHOLD_COARSEN || !hasChildren) {
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

  async _tickOnce() {
    if (!this.manifest) return;

    this._updateFrustum();
    const { want, replaceParents } = this._decide();

    // Update visibility of already loaded tiles before any load/unload
    for (const [id, rec] of this.tiles) {
      const needed = want.has(id);
      rec.obj3d.visible = needed;
      if (rec.bboxHelper) {
        rec.bboxHelper.visible = this.showBoundingBoxes && needed;
      }
    }

    // Queue tiles we need but don't have
    for (const id of want) {
      if (!this.tiles.has(id)) {
        this._enqueue(id);
      }
    }

    // Drop any queued requests we no longer need
    if (this.queue.length) {
      this.queue = this.queue.filter(id => want.has(id));
    }

    // Unload tiles we have but don't need
    for (const [id, rec] of [...this.tiles]) {
      if (!want.has(id)) {
        // Keep parent until all children are resident
        const meta = rec.meta || this.byId.get(id);
        this._unload(id);
      }
    }

    // Process queue until we either run out of tiles or hit the concurrency cap
    let launched;
    do {
      launched = [];
      while (this.queue.length && this.inflight < MAX_CONCURRENT) {
        const nextId = this.queue.shift();
        launched.push(this._load(nextId));
      }
      if (launched.length) {
        this._setLoadingIndicator(true);
        const loaded = await Promise.all(launched);
        // Hide freshly loaded tiles that are no longer needed
        for (const rec of loaded) {
          if (!rec) continue;
          const id = rec.meta.tileId;
          const needed = want.has(id);
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

  _enqueue(id) {
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
      } else {
        this._restoreTileMaterial(rec);
      }
      return;
    }

    // Avoid duplicate queue entries
    if (!this.queue.includes(id)) {
      this.queue.push(id);
    }
  }

  async _load(id) {
    const meta = this.byId.get(id);
    if (!meta) return null;

    this.inflight++;
    let rec = null;
    try {
      const glb = await this.loader.loadAsync(meta.url);
      const obj = glb.scene;
      obj.userData.tileId = id;

      const fadeInEnabled = !this.wireframeMode;

      obj.traverse(o => {
        if (o.isMesh && o.material) {
          const newMaterial = new THREE.MeshStandardMaterial({
            vertexColors: true,
            transparent: fadeInEnabled,
            opacity: fadeInEnabled ? 0.0 : 1.0,
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
              transparent: fadeInEnabled,
              opacity: fadeInEnabled ? 0.0 : 1.0,
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

      if (fadeInEnabled) {
        const start = performance.now();
        const dur = 180;
        const animate = () => {
          const t = Math.min(1, (performance.now() - start) / dur);
          obj.traverse(o => {
            if (o.isMesh && o.material) {
              o.material.opacity = 0.2 + 0.8 * t;
            }
          });
          if (t < 1) {
            requestAnimationFrame(animate);
          } else {
            obj.traverse(o => {
              if (o.isMesh && o.material) {
                o.material.transparent = false;
                o.material.opacity = 1.0;
              }
            });
          }
        };
        animate();
      } else {
        obj.traverse(o => {
          if (o.isMesh && o.material) {
            o.material.transparent = false;
            o.material.opacity = 1.0;
          }
        });
      }

      rec = { obj3d: obj, meta, bboxHelper };
      this.tiles.set(id, rec);
      this._cacheInsert(id, rec);

      if (this.tileColorMode) {
        this._applyTileColor(rec);
      }
    } catch (e) {
      console.error("Tile load failed", id, e);
    }

    this.inflight--;
    return rec;
  }

  _createBoundingBoxHelper(meta) {
    const min = new THREE.Vector3(...meta.aabbWorld[0]);
    const max = new THREE.Vector3(...meta.aabbWorld[1]);
    const box = new THREE.Box3(min, max);
    const hue = ((meta.z || 0) % 6) / 6;
    const color = new THREE.Color().setHSL(hue, 0.65, 0.55);
    return new THREE.Box3Helper(box, color);
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
    this.manifest = null;
    this.byId.clear();
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
  scene.background = new THREE.Color(0x404040); // Lighter gray background

  // Setup camera
  const camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.01,
    10000
  );
  camera.position.set(0, 0, 2);

  // Setup controls
  const controls = new OrbitControls(camera, renderer.domElement);
  const mgr = new TileManager(scene, camera, renderer);

  // Setup lighting - brighter for vertex colors
  scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const dir = new THREE.DirectionalLight(0xffffff, 0.6);
  dir.position.set(1, 1, 1);
  scene.add(dir);

  const settings = {
    extract: 'default',
    time: '0',
    sseRefine: SSE_THRESHOLD_REFINE,
    wireframe: false,
    boundingBoxes: false,
    tileColorMode: false
  };

  const gui = new GUI({ width: 300 });
  const datasetFolder = gui.addFolder('Dataset');
  const lodFolder = gui.addFolder('LOD');
  const diagnosticsFolder = gui.addFolder('Diagnostics');

  let extractController = null;
  let timeController = null;
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

  function rebuildExtractController() {
    if (extractController) {
      datasetFolder.remove(extractController);
    }

    const names = extractsCache.length ? extractsCache.map(e => e.name) : ['default'];
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
    }

    const selected = extractsCache.find(e => e.name === settings.extract);
    const times = selected && selected.times && selected.times.length
      ? selected.times.map(t => t.toString())
      : ['0'];

    if (!times.includes(settings.time)) {
      settings.time = times[0];
    }

    timeController = datasetFolder.add(settings, 'time', times);
    timeController.name('Time');
    timeController.onChange((value) => {
      settings.time = value;
      loadManifest();
    });
    timeController.updateDisplay?.();
  }

  async function loadManifest() {
    const manifestUrl = `/manifest/${settings.extract}/${settings.time}.json`;
    mgr.showBoundingBoxes = settings.boundingBoxes;
    mgr.wireframeMode = settings.wireframe;
    mgr.clear();
    await mgr.init(manifestUrl);
    applyWireframe(settings.wireframe);
    mgr.updateBoundingBoxVisibility();
    await mgr.tick();
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
      settings.extract = 'default';
      settings.time = '0';
      rebuildExtractController();
      rebuildTimeController();
      await loadManifest();
    }
  }

  lodFolder.add(settings, 'sseRefine', 0.1, 120, 0.1).name('SSE Refine').onChange((value) => {
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

  diagnosticsFolder.add(settings, 'tileColorMode').name('Tile Colours').onChange((value) => {
    mgr.tileColorMode = value;
    if (value) {
      for (const [, rec] of mgr.tiles) {
        mgr._applyTileColor(rec);
      }
    } else {
      for (const [, rec] of mgr.tiles) {
        mgr._restoreTileMaterial(rec);
      }
    }
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
