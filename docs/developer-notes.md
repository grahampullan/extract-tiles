# Developer Notes: Tiling Pipeline and Viewer

This document captures how the Python preprocessor (`build_tiles.py`) and the WebGL viewer (`public/viewer.js`) work together to generate, serve, and visualise multi-resolution tiles.

---

## 1. build_tiles.py (Preprocessor)

### 1.1. High-Level Flow

1. **Load GLB**
   - `load_glb_mesh_primitives` or `load_glb_arrays` pulls vertex positions, indices, UVs, and vertex colours for each mesh/primitive.
   - Zero-triangle primitives are skipped early.

2. **Per-Mesh Chart Loop** (UV mode)
   - When `--split_meshes` is set each primitive becomes its own "chart" producing a disjoint quadtree. Tile IDs are namespaced as `meshIndex/z/x/y` and files are written under `mesh_<index>/`.
   - Without `--split_meshes` everything is flattened into a single quadtree.

3. **Tile Generation**
   - Iterate depths from `max_depth` down to 0. This makes sure coarser tiles are available when finer ones are skipped.
   - For each tile cell, build a triangle mask (`uv_in_tile` or `in_aabb`) from centroid tests and assemble a `trimesh` subset.
   - Optionally compute border samples when `--preserve_borders` is enabled.
   - Decimate via Open3D (`decimate_to_target`) until the estimated size hits `target_kb` ± tolerance.
   - If border preservation is on, snap the simplified border back onto the captured outline. With absolute tolerance use `--snap_radius` (world units). With relative tolerance use `--snap_ratio` (fraction of the tile’s world-space diagonal). Skirts are **only** generated when borders are not preserved.

4. **Tile Export**
   - `write_glb_from_trimesh` emits a minimal GLB: position/UV/colour/index buffer views and a single primitive. Indices downgrade to 16-bit when possible.
   - Tile metadata (AABB, UV bounds, children ids, geometric error, approx bytes, etc.) is stored in `mesh.extras` and recorded in the manifest.

5. **Manifest**
   - A single JSON file per extract/time step. Each record carries `mesh`, `tileId`, hierarchy links, SSE data, and the URL served by Fastify (`/tiles/<extract>/<time>/<...>.glb`).
   - When `--split_meshes` is used the manifest `charts` count equals the number of primitives seen in the GLB.

### 1.2. CLI Flags (selected)

| Flag | Purpose |
| --- | --- |
| `--tiling_space {uv,world}` | UV quadtree (requires TEXCOORD_0) or world octree. |
| `--split_meshes` | Build a separate quadtree per primitive (UV mode only). |
| `--max_depth` / `--target_kb` | Control LOD depth and per-tile budget. |
| `--preserve_borders` | Disable skirts and snap simplified border vertices back to the original outline. |
| `--snap_radius` / `--snap_ratio` | Absolute or relative tolerance for border snapping. |

### 1.3. Performance Notes

* Decimation is the dominant cost; enable `--preserve_borders` only when the extra seam accuracy is required.
* Tile generation is embarrassingly parallel—each tile is independent once its mask is known. If throughput becomes an issue, consider a `multiprocessing` executor around the tile loop.

---

## 2. Fastify Server (`server.mjs`)

* Serves static viewer assets from `public/` and tiles from `tiles_out/`.
* HTTP API:
  - `GET /manifest/:extract/:time.json`
  - `GET /tiles/*`
  - `GET /api/extracts` (enumerates extracts and available time steps).

---

## 3. public/viewer.js (Client Viewer)

The viewer is a single-page Three.js application. `TileManager` owns all loading/caching logic.

### 3.1. TileManager Core State

| Field | Description |
| --- | --- |
| `manifest` | Parsed JSON manifest for the current extract/time. |
| `byId` | `Map<tileId, meta>`. Full metadata lookup table. |
| `tiles` | Active tiles currently in the scene (`tileId → { obj3d, meta, bboxHelper }`). |
| `cache` | In-memory LRU cache (same payload as `tiles`). Active tiles are *also* stored here for reuse, so cache size ≥ live tile count. Eviction now scans for the oldest **inactive** entry to honour `MAX_TILES`/`MAX_CACHE_BYTES`. |
| `queue` / `inflight` | Pending tile requests and outstanding network loads. |

### 3.2. LOD Loop (`tick` → `_tickOnce`)

1. **Frustum + SSE pass**: `_decide()` traverses the quadtree, computing projected SSE per tile. It returns a `want` set and (for diagnostics) a `replaceParents` set.

2. **Visibility update**: All tiles already in `this.tiles` are immediately shown/hidden according to `want`. Helpers mirror the tile’s visibility.

3. **Queue management**:
   - Enqueue new tiles in `want` that are not already loaded.
   - Drop stale items from the queue (only keep IDs still in `want`).

4. **Unload**: `this._unload(id)` removes any active tile not in `want` from the scene but leaves it in the cache.

5. **Async loads**: Fire at most `MAX_CONCURRENT` requests at a time. After `Promise.all()` completes, compare each returned tile with the current `want` set—unload immediately if it’s no longer needed (this prevents “ghost” tiles when the camera moves back out).

6. **Diagnostics / HUD**: Update tile/cache counts, bounding boxes, and the loading indicator.

`tick` calls are serialised (`_tickLock`/`_tickPending`) so the loop cannot overlap and reintroduce stale tiles.

### 3.3. Diagnostics & GUI Controls

The dat.GUI panel exposes:

* **Extract / Time** – selects manifest via `/api/extracts`.
* **SSE Refine** – continuous slider (0.1–120 px). Internally the coarsen threshold tracks half this value.
* **Wireframe** – toggles `material.wireframe` on active tiles.
* **Bounding Boxes** – adds/removes `Box3Helper`s for active tiles only.
* **Tile Colours** – swaps the mesh material to a flat `MeshBasicMaterial` (lighting-independent), storing/restoring the original material/colour attributes.

### 3.4. Tile Lifetime & Cache

1. Future requests always check `this.cache` first. Cache hits reattach the existing mesh and (if diagnostics are on) reapply colours/helpers.
2. When a tile leaves `this.tiles`, it remains cached until an eviction finds an inactive entry to reclaim.
3. Cache eviction walks the map in insertion order to find the oldest entry not currently displayed; if every cached entry is still active the cache is temporarily allowed to exceed its limits.

### 3.5. Miscellaneous

* The loading badge is driven by `_setLoadingIndicator(active)` which watches both `inflight` and the queue.
* Bounding boxes survive view changes because `updateBoundingBoxVisibility` tracks helpers tied to `this.tiles` and removes only true stragglers from the scene graph.

---

## 4. Rebuilding the Pipeline Locally

1. Create GLB data (e.g. the cylinder generator):

   ```bash
   python3 examples/cylinders/cylinders.py \
     --spokes 10 --wheel-count 1 --r-hub 0.75 --r-tip 1.0 \
     --tube-radius 0.1 --wave-amplitude 0.15 \
     --segments-axis 64 --segments-circ 96 \
     -o examples/cylinders/cylinders.glb
   ```

2. Run the tiler (UV mode with separate charts and preserved borders):

   ```bash
   python3 build_tiles.py \
     --in_glb examples/cylinders/cylinders.glb \
     --out_dir tiles_out \
     --extract cylinders_uv_preserve \
     --tiling_space uv \
     --split_meshes \
     --max_depth 5 \
     --target_kb 200 \
     --preserve_borders \
     --snap_ratio 0.001
   ```

3. Start the server: `npm run dev` and browse to `http://localhost:8080/`.

---

## 5. Troubleshooting Cheatsheet

| Symptom | Likely Cause | Suggested Fix |
| --- | --- | --- |
| Cracks between tiles | Skirts disabled without border snapping | Enable `--preserve_borders` or keep skirts on. |
| Tile colours look washed out | Legacy diagnostic using vertex colours | Current implementation swaps to `MeshBasicMaterial` for solid colours. Reload after pulling latest changes. |
| Stale tiles/overdraw after moving camera | Overlapping `tick()` calls or cached tiles not evicted | Serialization plus queue filtering fixes this; ensure viewer.js is up to date. |
| Cache HUD grows without bound | Old eviction logic stopped at first active tile | Updated cache scan now evicts oldest *inactive* entry first. |

---

This overview should give new contributors enough context to reason about the pipeline end-to-end. For deeper dives, the inline comments in `build_tiles.py` and `public/viewer.js` call out the most nuanced sections (border snapping, cache eviction, etc.).
