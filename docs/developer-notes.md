# Developer Notes: `build_tiles.py` & `public/viewer.js`

This document summarises the current tiler/viewer pipeline, with emphasis on the new static-octree workflow that keeps the scene visible while timesteps swap in. Use it as a field guide when tweaking the builder CLI or hacking on the viewer.

---

## 1. `build_tiles.py`

### 1.1 Overview
`build_tiles.py` ingests GLB snapshots and emits:
- A manifest (`manifest_<time>.json`) describing every tile (AABB, children, `approxBytes`, etc.).
- A directory of GLB tiles organised by depth and Morton coordinates.
- Optional Cesium 3D Tiles 1.1 tileset metadata (`--write_tileset`).

The script supports two partitioning modes:
- **UV quadtree** (`--tiling_space uv`): splits meshes per primitive, driven by UV bounds.
- **World octree** (`--tiling_space world`): splits using spatial AABBs and triangle centroids.

Shared helper functions handle GLB loading (`load_glb_arrays` / `load_glb_mesh_primitives`), triangle metrics (`tri_areas`, `tri_centroids_world`), decimation (`decimate_to_target`), skirt generation, and seam snapping. All GLBs are created via `write_glb_from_trimesh`.

### 1.2 Static Octree Mode (`--static_octree`)
World-space tiling can now pre-align all timesteps so the viewer can reuse the spatial hierarchy:
1. **Global extent scan** – when `--static_octree` is set, `compute_global_world_bounds()` runs before tiling. Each snapshot prints its min/max world coordinates and a final union summary. Only one GLB is loaded at a time, so memory usage stays flat.
2. **Fixed scene bounds** – `build_world_octree` receives `forced_scene_aabb`. Every tile derives its child bounds from this shared AABB, ensuring identical `tileId`, `z/x/y/k`, and `aabbWorld` per manifest.
3. **Manifest metadata** – manifests get `layout: { "type": "static-octree" }` plus `global.aabbWorld=[min,max]`. Viewers can detect this to enable mesh reuse.
4. **Empty tiles** – if a tile’s region has no triangles at a timestep, the manifest still includes the entry with `triCount: 0`, `approxBytes: 0`, and `url: null`. This keeps IDs aligned without writing placeholder GLBs.
5. **Caching semantics** – `actualBytes` is `0` for empty tiles; downstream consumers should treat `url: null` as “skip loading but leave the node alive”.

When `--static_octree` is absent, the legacy per-timestep behaviour remains: each GLB defines its own scene bounds and only existing tiles are written to the manifest.

### 1.3 UV vs. World Key Differences
| Aspect | UV Quadtree | World Octree |
| --- | --- | --- |
| Partition | `uv_tile_bounds(z,x,y)` within [0,1] UV domain; optional `--split_meshes` per primitive | `child_bounds(scene_aabb,z,i,j,k)` within world AABB |
| Triangle selection | `uv_in_tile` mask on UV centroids | `in_aabb` mask on world centroids |
| Border handling | `preserve_borders` snaps UV seams; can project onto UV tile planes | `preserve_borders` reprojects to analytic cell planes or snaps with `snap_radius` |
| Output path | `mesh_<idx>/<z>/<x>/<y>.glb` | `<z>/<x>/<y>/<k>.glb` |

### 1.4 CLI Highlights
```bash
python3 build_tiles.py \
  --snapshots --input_dir examples/multi-stage-comp/all_snapshots \
  --tiling_space world \
  --out_dir tiles_out \
  --extract multi-stage-comp-unst \
  --max_depth 4 \
  --target_kb 300 \
  --preserve_borders --snap_ratio 0.001 \
  --skip_leaf_decimation --min_ratio 0.05 --min_tris 500 --max_iter 5 \
  --root_voxel_ratio 0.02 \
  --static_octree
```
Key flags:
- `--snapshots` + `--input_dir` – batch process sorted GLBs; `--time` is used as an offset.
- `--tiling_space` – world/uv selection.
- `--static_octree` – world mode only; runs the global scan/logging, forces shared bounds, emits `layout.type`.
- `--preserve_borders`, `--snap_radius`, `--snap_ratio`, `--border_projection` – seam integrity controls.
- `--root_voxel_ratio` / `--root_voxel_trigger` – fallback voxel clustering when root tiles blow past target size.
- `--write_tileset`, `--tileset-origin`, `--tileset-scale` – optional 3D Tiles output.

---

## 2. `public/viewer.js`

### 2.1 Architecture Recap
`TileManager` manages GLB streaming and scene graph nodes:
- Maintains `tiles` (active) + `cache` (LRU) keyed by `tileId`.
- Computes screen-space error (SSE) thresholds (`SSE_THRESHOLD_REFINE/COARSEN`) to decide which tiles stay loaded.
- Applies diagnostics (wireframe, bounding boxes, tile colours) and overlays (wireframe overlay, simple shading).
- HUD shows current thresholds, tile counts, and cache usage.

### 2.2 Static Layout Reuse
Static manifests (those declaring `layout.type = "static-octree"`) can skip the clear/reload cycle when the timestep changes:
1. **Detection** – `mgr.init()` inspects `manifest.layout.type`. If it’s `static-octree`, `staticLayoutActive` becomes `true`. A Diagnostics toggle (`Static layout reuse`) lets developers disable reuse at runtime.
2. **Time tracking** – each tile record now stores `timeIndex`. Cache entries carry the same field. `_tileMatchesTime()` ensures we only reuse meshes that reflect the target timestep.
3. **Queue behaviour** – `_enqueue()` accepts `{timeIndex, forceReload}`. When the viewer already has a tile but at an old timestep, the entry is re-queued with `forceReload=true` instead of unloading the mesh.
4. **Loading** – `_load()` consumes queue entries (rather than tile IDs alone) so it knows the desired timestep. If `url` is null, it synthesises an empty `THREE.Group` and just marks the tile as refreshed. Otherwise it loads and processes the GLB as before.
5. **Swap install** – `_installTile()` replaces the previous mesh/bbox helper in-place, reapplies diagnostics, and inserts the record into both `tiles` and `cache`. Old meshes are disposed via `_disposeRecord`.
6. **SSE refresh** – once every root tile reports `timeIndex === targetTimeIndex`, `staticRefreshPending` triggers another tick so SSE refinement/coarsening honours the new timestep immediately.
7. **Fallback** – if the dataset changes (different extract or a missing tile), `loadManifest()` falls back to `mgr.clear()` and the legacy behaviour resumes automatically.

Legacy datasets operate exactly as before: the scene clears, caches reset, and tiles reload per manifest.

### 2.3 Diagnostics & Settings
`settings` now includes `staticReuse` (default `true`). The Diagnostics GUI exposes a “Static layout reuse” checkbox that flips `mgr.staticReuseDisabled`. Other controls remain:
- Wireframe & wireframe overlay toggles
- Bounding boxes and axes helper
- Tile colour / simple shading modes
- SSE refine slider (with optional auto-calibration)

When static reuse is enabled, the HUD never flashes blank while scrubbing time; the previous timestep stays visible until the new GLBs arrive.

### 2.4 Time-Switch Flow Summary
1. `loadManifest()` fetches JSON first, figures out whether reuse is safe (same extract, static layout, reuse toggle on).
2. If reuse is disabled, it calls `mgr.clear()` as before. Otherwise it sets `staticRefreshPending=true`, records the target time, and calls `mgr.init()` with `reuseTiles=true` so root nodes stay attached.
3. `_tickOnce()` keeps enqueuing tiles until each required `tileId` has data for the new timestep. Queue entries carry the desired time so stale GLBs are ignored.
4. After all root tiles refresh, `_staticRootsAtTargetTime()` triggers another tick -> SSE refine -> final view matches the new timestep without a flash.

---

## 3. End-to-End Flow Example

1. **Generate GLBs** – e.g. `examples/multi-stage-comp/convert_bunny.py` or your CFD snapshots.
2. **Tile** – run `build_tiles.py` with `--tiling_space world --static_octree` so every manifest advertises `layout.type = "static-octree"`.
3. **Serve** – `npm install` (once) then `npm run dev`; open `http://localhost:8080`.
4. **View** – select your extract/time. Scrub between timesteps: the previous mesh persists while request queue refreshes tiles; once completed, SSE refines the new timestep.

---

## 4. Troubleshooting Quick Reference

| Issue | Cause | Fix |
| --- | --- | --- |
| Static dataset still flashes blank | Manifest missing `layout.type = "static-octree"` or GUI toggle off | Regenerate tiles with `--static_octree`, ensure Diagnostics → Static layout reuse is enabled |
| Tile never updates to new timestep | Entry stuck with `url: null` or network error logged | Check manifest `url` for that tile; rebuild or inspect server response |
| Cache grows without bound | All entries active (static swap keeps them alive) | Increase `MAX_TILES`/`MAX_CACHE_BYTES` or lower `max_depth`/`target_kb` |
| Bounding boxes disappear after reuse | Viewer toggles bounding boxes per tile install | Toggle “Bounding Boxes” off/on to reattach helpers if needed |
| Static scan seems slow | `compute_global_world_bounds` logs every GLB | Normal; use smaller `--input_dir` during tests or run once and reuse the output |

For deeper debugging, inspect `dev_notes/viewer-static-tiles-plan.md` (action plan for static swaps) and `dev_notes/aligned-tiles-plan.md` (builder requirements).
