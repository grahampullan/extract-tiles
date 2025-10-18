# Developer Notes: `build_tiles.py` & `public/viewer.js`

This document explains how the tiler and viewer work under the hood. It dives into key functions, their dependencies, and how data flows from the source GLB into the on-screen scenegraph.

---

## 1. `build_tiles.py`

### 1.1. Module Overview

`build_tiles.py` is the command-line preprocessor that turns GLB meshes into multi-resolution tile hierarchies plus manifests consumed by the viewer. Entry point is `main()` → `build_uv_quadtree` **or** `build_world_octree`.

### 1.2. Shared Utilities (selected)

| Function | Purpose |
| --- | --- |
| `load_glb_arrays(path)` | Bulk loader returning global position, UV, colour, index arrays across all primitives. Used when `--split_meshes` is false. |
| `load_glb_mesh_primitives(path)` | Returns a list of `dict`s (positions/indices/UVs/colours) per primitive. Stays close to the glTF structure and keeps meshes separate for `--split_meshes`. |
| `tri_areas`, `tri_centroids_world`, `uv_centroids` | Vectorised helpers to compute per-triangle areas and centroids (drives UV/world partitioning). |
| `decimate_to_target(trimesh, target_bytes)` | Converts a `trimesh.Trimesh` into Open3D, runs quadric decimation until estimated size ≤ target × tolerance, and re-projects UV/colour attributes. |
| `add_skirts(tri_mesh, skirt_h_ratio)` | Optional seam guard: duplicates boundary vertices, extrudes along normals, and rebuilds UV/colour arrays. |
| `snap_decimated_border(decimated, original_border_pts, snap_radius)` | After decimation, snaps border vertices back onto pre-captured positions (used with `--preserve_borders`). |
| `write_glb_from_trimesh(tri_mesh, meta, out_path)` | Minimal glTF writer using pygltflib. Creates buffer views/accessors for POSITION, TEXCOORD_0, COLOR_0, indices, and attaches `meta` as mesh extras. |

### 1.3. `build_uv_quadtree`

**Signature**: `build_uv_quadtree(src_glb, out_dir, extract="default", time_index=0, max_depth=5, target_bytes=200_000, split_meshes=False, preserve_borders=False, snap_radius=None, snap_ratio=None)`

| Step | Details |
| --- | --- |
| 1. Mesh prep | Depending on `split_meshes`, call `load_glb_mesh_primitives` or flatten via `load_glb_arrays`. Each entry tracks name, positions, UVs, colours, and indices. |
| 2. Stats | Compute global triangle stats (total count, average/min area) for manifest metadata. Filter out primitives with zero triangles. |
| 3. Manifest stub | Initialise manifest header with `charts` = number of primitives. |
| 4. Per-mesh tile loop | For each mesh: <br> • Precompute triangle areas (`tri_areas`) and UV centroids (`uv_centroids`). <br> • Iterate `z` from `max_depth` down to 0. For each (x, y) tile compute bounding box `b` = `uv_tile_bounds`. <br> • Build triangle mask (`uv_in_tile`) and create subset via `subset_trimesh`. <br> • If `preserve_borders`, capture boundary vertex positions (`compute_border_edges`). |
| 5. Decimation & seams | • Run `decimate_to_target`. <br> • If `preserve_borders`, call `snap_decimated_border` with absolute (`snap_radius`) or relative (`snap_ratio * diag`) tolerance. <br> • Else, add skirts via `add_skirts` (10% of mean edge length). |
| 6. Output | Write tile GLB under `mesh_<idx>/<z>/<x>/<y>.glb`. Record both `approxBytes` (estimate) and the real file size (`actualBytes`) in manifest entries. |
| 7. Finalise manifest & stats | Sort tiles by `(mesh, z, x, y)` and write `manifest_<time>.json`. Afterwards `summarize_tile_sizes` prints min/mean/median/max, percentiles, and a KB histogram. `summarize_tile_sizes_by_depth` also reports per-depth stats and highlights the fraction of tiles under 25 KB so you can detect overly deep refinements. |

### 1.4. `build_world_octree`

Similar structure but partitions using world-space AABBs instead of UV quads. Key differences:

* Use `child_bounds(scene_aabb, z, i, j, k)` to divide the global bounding box.
* Triangle filtering uses `in_aabb` on world centroids.
* UVs are optional; emitted GLBs may carry only positions/colours.

### 1.5. CLI (`main`)

* Builds the `argparse` parser with shared flags. Notable options now include `--tiling_space`, `--split_meshes`, `--preserve_borders`, `--snap_radius`, `--snap_ratio`, plus snapshot helpers `--snapshots` and `--input_dir` (process sequential time steps) alongside the direct `--in_glb` path.
* Normalises sizes (`target_bytes` = KB × 1024) and validates snap tolerances (`--snap_ratio`, `--snap_radius`). If `--preserve_borders` is enabled without an explicit tolerance the code falls back to a default `snap_ratio` of `1e-3`.
* Dispatches to UV or world builder through an inner `process_single`. When `--snapshots` is set it sorts all `*.glb` files in the input directory, increments `time` per file, and invokes `process_single` for each. World-space tiling still rejects `--split_meshes`. The flag `--skip_leaf_decimation` skips `decimate_to_target` for tiles at `max_depth` so leaves retain the original geometry. Additional decimation knobs expose the reduction floor: `--min_ratio` (default 0.02), `--min_tris` (default 32), and `--max_iter` (default 6) let you tighten those limits. A root fallback (`--root_voxel_ratio` + `--root_voxel_trigger`) runs voxel clustering on z=0 tiles when the quadric decimator can’t hit the byte target, producing a lightweight preview without affecting deeper levels.
* Prior to extracting arrays we bake any `baseColorTexture` images into per-vertex `COLOR_0` attributes (via Pillow + PyGLTFLib). That way world-space tiles carry colours even though we drop UVs/textures downstream.

---

## 2. `server.mjs`

Quick summary (mostly unchanged):

* Serves static assets from `public/` and tiles from `tiles_out/` under `/tiles/*`.
* Exposes `{extract, time}` manifests and `/api/extracts` listing detectible extracts.

---

## 3. `public/viewer.js`

### 3.1. Class Structure

`TileManager` is the core orchestrator. Important members:

* `loader` – `GLTFLoader` instance.
* `byId`, `tiles`, `cache` – manifest lookup, active tiles, and cached tiles (LRU style, default cap 500 tiles / ~600 MB).
* `queue`, `inflight` – track pending/active loads.
* Diagnostics toggles: `wireframeMode`, `showBoundingBoxes`, `tileColorMode`, `simpleShadingMode`.
* Lighting: ambient fill plus a directional light parented to the camera so the simple-shading option stays evenly lit as the user orbits.
* `rootTileIds` – set of `tileId`s whose `parent` is `null`. Roots are kept in the cache (eviction skips them) so they can be reinstated quickly, but they are only visible when the LOD logic needs them as a fallback.
* `_hasPendingDescendants(tileId, wantSet)` – helper used during LOD transitions to keep a parent visible while any wanted descendants are still loading.
* `_tickLock`, `_tickPending` – ensure the async tick loop doesn’t overlap.
* HUD wiring: `hudLOD`, `hudTiles`, `hudCache`, and `loadingIndicator` update the overlay counters/spinner each tick.
* Threshold/budget globals: `SSE_THRESHOLD_REFINE`/`COARSEN`, `MAX_CONCURRENT`, `MAX_TILES`, `MAX_CACHE_BYTES` govern LOD, request concurrency, and cache eviction.

### 3.2. Lifecycle

#### `init(manifestUrl)`

* Fetch manifest JSON, populate `byId`.
* Enqueue root tiles (`z === 0`) via `_enqueue`.

#### `_enqueue(tileId)`

* If tile exists in `cache`, detach it from cache, attach to scene, update diagnostics, add to `tiles`, and return.
* Otherwise push `tileId` into the queue if it’s not already present.

#### `_load(tileId)`

* Asynchronously load GLB with `GLTFLoader`. Each mesh gets:
  - A fresh `MeshStandardMaterial` (keeps vertex colours) storing debug info for diagnostics.
  - Cached vertex colour attributes so diagnostics can restore them later.
* Consume any prefetched binary payload from `prefetchedTileBuffers` before falling back to `loader.loadAsync` (saves a round-trip when neighbour prefetch succeeds).
* Create a `Box3Helper` for bounding boxes.
* Construct `{ obj3d, meta, bboxHelper }` record, add to `tiles`, and call `_cacheInsert`.
* Apply tile colour override if diagnostic is active.
* returns the tile record.

#### `_cacheInsert(tileId, record)`

* Insert into `cache` and bump `cacheBytes`.
* Evict oldest inactive entries while the cache exceeds `MAX_TILES` or `MAX_CACHE_BYTES`.
  - The loop now scans keys until it finds one **not** present in `this.tiles`.
  - Evicted mesh resources (geometry/material/helper) are disposed.

#### `clear()`

* Bumps `manifestVersion`, clears `rootTileIds`, detaches every active tile/helper from the scene, and traverses cached meshes to dispose GPU resources.
* Empties `tiles`, `cache`, and `queue`, zeroes the tracked byte budget, and resets the manifest lookup.

#### `_unload(tileId)`

* Remove tile’s `obj3d` and helper from the scene.
* Delete entry from `tiles` (record persists in `cache`).


### 3.3. LOD Loop (`tick` / `_tickOnce`)

1. **Prevent overlap**: `tick` sets `_tickLock` to true while `_tickOnce` runs. If another tick is requested, set `_tickPending` and rerun once current pass completes.

2. **Frustum & SSE**: `_decide()` calculates which tiles are needed (`want`). It precomputes median geometric error per depth, normalises each tile's SSE (`sse * medianDepthGE / tile.geometricError`), and recurses into children when that scaled value exceeds `SSE_THRESHOLD_REFINE` so varying mesh density doesn't skip intermediate levels.

3. **Visibility pre-pass**: Toggle visibility (and helper visibility) for all `this.tiles` based on membership in `want`. Parents that appear in `replaceParents` stay visible while their descendants are still downloading (`_hasPendingDescendants`).

4. **Queue**: Enqueue missing tiles with their current `normalizedSse` as a priority; pending entries are sorted so higher-error (closer) tiles are requested first while still filtering duplicate/stale IDs.

5. **Removal**: Call `_unload` for any active tile not in `want`. Parents remain visible while any of their wanted descendants are still loading (`_hasPendingDescendants`). Root tiles are always retained as a fallback.

6. **Load burst**: While there are queued IDs and `inflight < MAX_CONCURRENT`, shift IDs into `_load`. After `Promise.all()` resolves, inspect each returned tile record:
   * If tile is still wanted (or acting as a fallback) keep it visible and sync helper state.
   * Otherwise, `_unload(id)` immediately to guard against late arrivals.

7. **HUD update**: Update tile/cache counts, bounding boxes, and loading indicator via `_setLoadingIndicator(active)`.

### 3.4. Diagnostics & GUI

`dat.gui` initialises settings object:

```js
const settings = {
  extract: 'default',
  time: '0',
  sseRefine: SSE_THRESHOLD_REFINE,
  wireframe: false,
  boundingBoxes: true,
  tileColorMode: false,
  simpleShading: true
};
```

#### Controls

* **Dataset folder** – repopulated from `/api/extracts`, dispatches manifest loads on change.
* **LOD folder** – exposes `sseRefine` (0.1–120 px, default 18 px). Whenever the slider moves, `SSE_THRESHOLD_COARSEN` auto-updates to half of `sseRefine` and `tick()` reruns.
* **Diagnostics** – toggles wireframe, bounding boxes (default on), simple shading (default on), and tile colours. Each handler updates active tiles immediately:
  - Wireframe: sets `material.wireframe` for all meshes in `tiles` and `cache`.
  - Bounding boxes: adds/removes helpers only for active tiles. `updateBoundingBoxVisibility` ensures cached helpers are detached.
  - Tile colours: `_applyTileColor` swaps shading for active tiles using a per-mesh random HSL tint; `_restoreTileMaterial` returns to the stored material when turned off.
  - Simple shading: `_applySimpleShading` switches meshes to a cached `MeshLambertMaterial` (respecting vertex colours) for softer lighting; turning it off restores or reapplies tile colours as needed.

### 3.5. Extract Selection & Prefetching

* `loadExtractsList` hits `/api/extracts`, caches the response, rebuilds the GUI controls, and triggers an initial manifest load. Errors fall back to the default extract/time pair.
* `rebuildTimeController` chooses between a discrete dropdown (many timesteps) and a slider (<=100). The slider stores a numeric `timeIndex` so dat.gui can emit integers even though `settings.time` remains stringified for URL construction.
* `schedulePrefetchNeighbours` determines adjacent timesteps (+/-1, +/-2) and calls `prefetchRootTiles`, which fetches the neighbour manifest and root tile GLBs ahead of time. Prefetched ArrayBuffers live in `prefetchedTileBuffers`, letting `_load` short circuit straight into `GLTFLoader.parse` without another HTTP transfer.
* Diagnostics interplay: enabling tile colours or simple shading automatically disables the other mode and re-applies the relevant material override to all loaded tiles.

---

## 4. End-to-End Flow Example

1. **Generate GLB** using `examples/cylinders/cylinders.py` (supports sinusoidal radius modulation via `--wave-amplitude`/`--wave-cycles-…`).
2. **Tile** with `build_tiles.py` setting `--split_meshes` and `--preserve_borders`.
3. **Serve** with `npm run dev` → open `http://localhost:8080`.
4. Viewer loads `manifest`, spawns root tiles, then refines/coarsens as you orbit. Diagnostics highlight tiles and bounding boxes without doubling geometry.

---

## 5. Troubleshooting Quick Reference

| Issue | Cause | Mitigation |
| --- | --- | --- |
| Coarse/fine tiles overlap after zooming out | Late-arriving child tiles | `_load` returns record & immediate `_unload` if tile no longer in `want` (already implemented). |
| Debug colours look washed out | Lighting interacting with materials | Latest build swaps to `MeshBasicMaterial` for diagnostics. |
| Bounding boxes flash | Helpers removed when tile still active | `updateBoundingBoxVisibility` now ignores cached-but-active helpers. |
| Cache HUD keeps growing | Eviction stops at first active entry | Loop now scans for oldest inactive entry before breaking. |
| Cracks along seams | Skirts disabled without snapping | Use `--preserve_borders` (with `--snap_ratio` or `--snap_radius`) or keep skirts on. |

---

For more context, the codebase has descriptive comments around the trickier areas (border snapping, cache management). Workflows are captured by the sample commands above.
