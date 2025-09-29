import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// Configuration
let SSE_THRESHOLD_REFINE = 3.0;
let SSE_THRESHOLD_COARSEN = 1.5;
const GLOBAL_MAX_TILES = 400;
const GLOBAL_MAX_CONCURRENT = 10;

class TileManager {
  constructor(scene, camera, renderer, opts = {}) {
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.loader = opts.sharedLoader || new GLTFLoader();
    this.sharedLRU = opts.sharedLRU || new Map();
    this.sharedInflight = opts.sharedInflight || { n: 0 };
    this.manifest = null;
    this.byId = new Map();
    this.tiles = new Map();
    this.queue = [];
    this.frustum = new THREE.Frustum();
    this.projScreenMatrix = new THREE.Matrix4();
    this.extractName = opts.extractName || 'unknown';
  }

  async initFromURL(url) {
    try {
      this.manifest = await (await fetch(url)).json();
      for (const t of this.manifest.tiles) {
        // Prefix tile IDs with extract name to avoid collisions
        const globalId = `${this.extractName}:${t.tileId}`;
        this.byId.set(t.tileId, { ...t, globalId });
      }
      // Load root tiles
      for (const t of this.manifest.tiles) {
        if (t.z === 0) {
          this._enqueue(t.tileId);
        }
      }
      console.log(`[${this.extractName}] Loaded manifest: ${this.manifest.tiles.length} tiles`);
    } catch (err) {
      console.error(`[${this.extractName}] Failed to load manifest:`, err);
    }
  }

  _updateFrustum() {
    this.camera.updateMatrixWorld();
    this.projScreenMatrix.multiplyMatrices(
      this.camera.projectionMatrix,
      this.camera.matrixWorldInverse
    );
    this.frustum.setFromProjectionMatrix(this.projScreenMatrix);
  }

  _visible(meta) {
    const min = new THREE.Vector3(...meta.aabbWorld[0]);
    const max = new THREE.Vector3(...meta.aabbWorld[1]);
    return this.frustum.intersectsBox(new THREE.Box3(min, max));
  }

  _sse(meta) {
    const c = new THREE.Vector3(...meta.aabbWorld[0])
      .add(new THREE.Vector3(...meta.aabbWorld[1]))
      .multiplyScalar(0.5);
    const dist = c.distanceTo(this.camera.position) + 1e-6;
    const ge = meta.geometricError || 0.01;
    const h = this.renderer.domElement.clientHeight;
    const fov = (this.camera.fov * Math.PI) / 180;
    return (ge / (dist * Math.tan(fov / 2))) * h;
  }

  _children(meta) {
    return (meta.children || []).map(id => this.byId.get(id)).filter(Boolean);
  }

  hasTile(id) {
    return this.tiles.has(id);
  }

  forceUnload(id) {
    this._unload(id);
  }

  _decide() {
    const roots = [...this.byId.values()].filter(t => t.z === 0);
    const want = new Set();

    const visit = m => {
      if (!this._visible(m)) return;
      const sse = this._sse(m);
      if ((m.children || []).length && sse > SSE_THRESHOLD_REFINE) {
        this._children(m).forEach(visit);
      } else if (sse < SSE_THRESHOLD_COARSEN || !(m.children || []).length) {
        want.add(m.tileId);
      } else {
        want.add(m.tileId);
      }
    };

    roots.forEach(visit);
    return want;
  }

  async tick() {
    if (!this.manifest) return;

    this._updateFrustum();
    const want = this._decide();

    for (const id of want) {
      if (!this.tiles.has(id)) {
        this._enqueue(id);
      }
    }

    for (const [id] of [...this.tiles]) {
      if (!want.has(id)) {
        this._unload(id);
      }
    }

    while (this.sharedInflight.n < GLOBAL_MAX_CONCURRENT && this.queue.length) {
      await this._load(this.queue.shift());
    }
  }

  _enqueue(id) {
    const meta = this.byId.get(id);
    if (!meta) return;

    const globalId = meta.globalId;

    if (this.sharedLRU.has(globalId)) {
      const rec = this.sharedLRU.get(globalId);
      this.sharedLRU.delete(globalId);
      this.sharedLRU.set(globalId, rec); // Move to end (LRU)
      this.scene.add(rec.obj3d);
      this.tiles.set(id, rec);
      return;
    }

    if (!this.queue.includes(id)) {
      this.queue.push(id);
    }
  }

  async _load(id) {
    const meta = this.byId.get(id);
    if (!meta) return;

    const globalId = meta.globalId;
    this.sharedInflight.n++;

    try {
      const glb = await this.loader.loadAsync(meta.url);
      const obj = glb.scene;
      obj.userData.tileId = id;
      obj.userData.globalId = globalId;

      // Fade-in animation
      obj.traverse(o => {
        if (o.isMesh) {
          o.material.transparent = true;
          o.material.opacity = 0.0;
        }
      });

      this.scene.add(obj);

      const start = performance.now();
      const dur = 180;
      const animate = () => {
        const t = Math.min(1, (performance.now() - start) / dur);
        obj.traverse(o => {
          if (o.isMesh) {
            o.material.opacity = 0.2 + 0.8 * t;
          }
        });
        if (t < 1) {
          requestAnimationFrame(animate);
        }
      };
      animate();

      const rec = { obj3d: obj, meta };
      this.tiles.set(id, rec);
      this.sharedLRU.set(globalId, rec);

      // Evict oldest from cache if too large
      while (this.sharedLRU.size > GLOBAL_MAX_TILES) {
        const evictId = this.sharedLRU.keys().next().value;
        // Check if any manager is using this tile
        let inUse = false;
        for (const mgr of window.allManagers || []) {
          for (const [tileId, rec] of mgr.tiles) {
            if (rec.obj3d.userData.globalId === evictId) {
              inUse = true;
              break;
            }
          }
          if (inUse) break;
        }

        if (!inUse) {
          const evictRec = this.sharedLRU.get(evictId);
          if (evictRec) {
            evictRec.obj3d.traverse(o => {
              if (o.isMesh) {
                o.geometry?.dispose();
                if (o.material) {
                  if (o.material.map) o.material.map.dispose();
                  o.material.dispose?.();
                }
              }
            });
          }
          this.sharedLRU.delete(evictId);
        } else {
          break; // Can't evict, stop trying
        }
      }
    } catch (e) {
      console.error(`[${this.extractName}] Tile load failed`, id, e);
    } finally {
      this.sharedInflight.n--;
    }
  }

  _unload(id) {
    const rec = this.tiles.get(id);
    if (!rec) return;

    this.scene.remove(rec.obj3d);
    this.tiles.delete(id);
    // Keep in shared cache
  }

  clear() {
    // Unload all tiles from this manager
    for (const [id, rec] of this.tiles) {
      this.scene.remove(rec.obj3d);
    }
    this.tiles.clear();
    this.queue = [];
    this.manifest = null;
    this.byId.clear();
  }

  getTileCount() {
    return this.tiles.size;
  }
}

// Main application
(async function main() {
  // Setup renderer
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  document.getElementById("app").appendChild(renderer.domElement);

  // Setup scene
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111111);

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

  // Setup lighting
  scene.add(new THREE.AmbientLight(0xffffff, 0.35));
  const dl = new THREE.DirectionalLight(0xffffff, 0.9);
  dl.position.set(1, 1, 1);
  scene.add(dl);

  // Shared resources for all managers
  const sharedLRU = new Map();
  const sharedInflight = { n: 0 };
  const loader = new GLTFLoader();
  const managers = [];

  // Store managers globally for cache management
  window.allManagers = managers;

  // HUD elements
  const tilesHud = document.getElementById("tiles");
  const reqsHud = document.getElementById("reqs");
  const cacheHud = document.getElementById("cache");
  const loadingIndicator = document.getElementById("loading");
  const extractList = document.getElementById("extractList");

  // Load available extracts
  async function loadExtractsList() {
    try {
      const response = await fetch("/api/extracts");
      const extracts = await response.json();

      extractList.innerHTML = "";

      extracts.forEach(extract => {
        const item = document.createElement("div");
        item.className = "extract-item";

        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.id = `extract-${extract.name}`;
        checkbox.dataset.extractName = extract.name;

        const label = document.createElement("label");
        label.textContent = extract.name;
        label.htmlFor = checkbox.id;

        const timeSelect = document.createElement("select");
        timeSelect.id = `time-${extract.name}`;
        extract.times.forEach(t => {
          const option = document.createElement("option");
          option.value = t;
          option.textContent = `T${t}`;
          timeSelect.appendChild(option);
        });

        item.appendChild(checkbox);
        item.appendChild(label);
        item.appendChild(timeSelect);
        extractList.appendChild(item);

        // Handle extract toggle
        checkbox.addEventListener("change", async () => {
          if (checkbox.checked) {
            await loadExtract(extract.name, timeSelect.value);
          } else {
            unloadExtract(extract.name);
          }
        });

        // Handle time change
        timeSelect.addEventListener("change", async () => {
          if (checkbox.checked) {
            unloadExtract(extract.name);
            await loadExtract(extract.name, timeSelect.value);
          }
        });
      });
    } catch (err) {
      console.error("Failed to load extracts list:", err);
      extractList.innerHTML = "<div>Failed to load extracts</div>";
    }
  }

  async function loadExtract(name, time) {
    const url = `/manifest/${name}/${time}.json`;
    const mgr = new TileManager(scene, camera, renderer, {
      sharedLRU,
      sharedInflight,
      sharedLoader: loader,
      extractName: name
    });

    await mgr.initFromURL(url);
    managers.push(mgr);
    await tick();
  }

  function unloadExtract(name) {
    const index = managers.findIndex(m => m.extractName === name);
    if (index !== -1) {
      managers[index].clear();
      managers.splice(index, 1);
      tick();
    }
  }

  async function tick() {
    for (const m of managers) {
      await m.tick();
    }

    // Update HUD
    const totalTiles = managers.reduce((sum, m) => sum + m.getTileCount(), 0);
    tilesHud.textContent = totalTiles;
    reqsHud.textContent = sharedInflight.n;
    cacheHud.textContent = sharedLRU.size;

    if (sharedInflight.n > 0) {
      loadingIndicator.classList.add("active");
    } else {
      loadingIndicator.classList.remove("active");
    }
  }

  // SSE controls
  const sseRefineSlider = document.getElementById("sseRefine");
  const sseRefineValue = document.getElementById("sseRefineValue");
  const sseCoarsenSlider = document.getElementById("sseCoarsen");
  const sseCoarsenValue = document.getElementById("sseCoarsenValue");

  sseRefineSlider.addEventListener("input", e => {
    SSE_THRESHOLD_REFINE = parseFloat(e.target.value);
    sseRefineValue.textContent = SSE_THRESHOLD_REFINE.toFixed(1);
    tick();
  });

  sseCoarsenSlider.addEventListener("input", e => {
    SSE_THRESHOLD_COARSEN = parseFloat(e.target.value);
    sseCoarsenValue.textContent = SSE_THRESHOLD_COARSEN.toFixed(1);
    tick();
  });

  // Clear all button
  document.getElementById("clearAll").addEventListener("click", () => {
    managers.forEach(m => m.clear());
    managers.length = 0;
    sharedLRU.forEach(rec => {
      rec.obj3d.traverse(o => {
        if (o.isMesh) {
          o.geometry?.dispose();
          if (o.material) {
            if (o.material.map) o.material.map.dispose();
            o.material.dispose?.();
          }
        }
      });
    });
    sharedLRU.clear();

    // Uncheck all checkboxes
    extractList.querySelectorAll('input[type="checkbox"]').forEach(cb => {
      cb.checked = false;
    });

    tick();
  });

  // Update on controls change
  controls.addEventListener("change", tick);

  // Handle resize
  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    tick();
  });

  // Animation loop
  function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
  }
  animate();

  // Initialize
  await loadExtractsList();
  await tick();
})();