# Extract-Tiles

A multi-resolution tiling system for visualizing large triangulated mesh surfaces from CFD simulations.

The two core entry points are:
- `build_tiles.py` – Python preprocessor that turns GLB meshes into manifests + multi-resolution GLB tiles.
- `public/viewer.js` – Three.js client that streams manifests/tiles, performs LOD selection, and exposes the diagnostics HUD.

## Features

- **Multi-resolution tiles**: Automatically generates LOD tiles from GLB files.
- **Multiple tiling modes**:
  - UV-Quadtree: For meshes with texture coordinates.
  - World-Space Octree: For meshes without UVs (e.g., isosurfaces).
- **Crack prevention**: Includes overlap margins, optional skirt generation, and border snapping with configurable tolerances.
- **Dynamic LOD**: Screen-space-error refinement with hysteresis, per-depth SSE normalisation to avoid skipping intermediate levels, and queue throttling to keep loads responsive.
- **Time-series aware**: Discovers available extracts/timesteps, prefetches neighbouring manifests/tiles, and auto-picks a slider or dropdown based on timestep count.
- **HUD & diagnostics**: Live overlay for SSE thresholds/tile counts/cache size, wireframe toggle, per-tile bounding boxes, tile-colour debug mode, and a camera-aligned Phong shading option with specular highlights.
- **Efficient streaming**: Fastify server with compression and caching.
- **Static-octree payloads**: World-space tiler can emit deterministic static layouts plus flattened payloads for lightweight viewers or offline workflows.
- **3D Tiles export**: Optional Cesium-compatible `tileset.json` generation using the `3DTILES_content_gltf` extension referencing the GLB tiles.
- **Interactive viewers**:
  - Single-extract viewer with dat.GUI controls for dataset/time selection, LOD tuning, diagnostics, and HUD indicators.
  - Multi-extract viewer for comparing multiple datasets.
- **Precompressed manifests**: Fastify serves `.json.gz` manifests automatically when present (keep the `.json` fallback alongside it).

## Installation

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm

### Setup

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install Node.js dependencies:**
```bash
npm install
```

## Usage

### 1. Generate Tiles

Use the Python preprocessor to generate multi-resolution tiles from your GLB files:

#### UV-Quadtree mode (requires TEXCOORD_0):
```bash
python3 build_tiles.py \
  --in_glb path/to/mesh.glb \
  --out_dir tiles_out \
  --extract myextract \
  --time 0 \
  --tiling_space uv \
  --max_depth 5 \
  --target_kb 200 \
  --split_meshes
```

Each run prints a size histogram (min/median/max, percentiles, KB buckets) plus a per-depth breakdown highlighting the share of tiles under 25 KB - handy for catching overly aggressive depth settings.

#### World-Space Octree mode (no UVs required):
```bash
python3 build_tiles.py \
  --in_glb path/to/isosurface.glb \
  --out_dir tiles_out \
  --extract isosurface \
  --time 0 \
  --tiling_space world \
  --max_depth 5 \
  --target_kb 200
```

#### Static world-octree payloads

For world-space tiling you can request a deterministic static octree layout (the full octree is fixed in space for all snapshots) plus a flattened payload JSON used by lightweight viewers. Add `--static_octree` to any world-tiling command:

```bash
python3 build_tiles.py \
  --in_glb path/to/isosurface.glb \
  --out_dir tiles_out \
  --extract isosurface \
  --tiling_space world \
  --max_depth 5 \
  --target_kb 200 \
  --static_octree
```

This writes the usual manifest/tiles plus `payload_<time>.json` describing the ordered field list and per-tile metadata compatible with the static viewer flow.

Example command we use for an unsteady world-space dataset (snapshots not in this repo, but illustrates the full flag set):

```bash
python3 build_tiles.py \
  --snapshots \
  --input_dir examples/multi-stage-comp/all_snapshots \
  --tiling_space world \
  --out_dir tiles_out \
  --extract multi-stage-comp-unst-static-octree-tmp \
  --max_depth 4 \
  --target_kb 300 \
  --preserve_borders \
  --snap_ratio 0.001 \
  --skip_leaf_decimation \
  --min_ratio 0.05 \
  --min_tris 500 \
  --max_iter 5 \
  --root_voxel_ratio 0.02 \
  --static_octree
```

### Parameters:
- `--in_glb`: Input GLB file path
- `--out_dir`: Output directory for tiles (default: tiles_out)
- `--extract`: Extract name (used for organizing multiple datasets)
- `--time`: Time index for unsteady simulations (default: 0)
- `--tiling_space`: Either 'uv' or 'world'
- `--max_depth`: Maximum tile depth/zoom level (default: 5)
- `--target_kb`: Target tile size in KB (default: 200)
- `--split_meshes`: (UV mode) build a separate quadtree for each mesh/primitive inside the GLB; tile outputs are stored under `mesh_<index>/...`
- `--preserve_borders`: Snap simplified tile boundaries back to their original vertex positions and skip skirt geometry (helps on closed seams)
- `--snap_radius`: Absolute world-space snapping tolerance (used with `--preserve_borders`)
- `--snap_ratio`: Relative snapping tolerance expressed as a fraction of the tile’s bounding-box diagonal (default 1e-3, set to 0 to disable when using `--snap_radius` instead)
- `--world_eps_ratio`: Override the world-space overlap margin per tile (default 0.01). Increase when you need thicker skirts around world tiles.
- `--uv_eps_ratio`: Override the *relative* UV overlap margin per tile (default 0.01 = 1% of each tile’s span)
- `--border_projection`: When used with `--preserve_borders` (world tiler), project boundary vertices to the analytic cell planes instead of snapping to nearest original seam points.
- `--input_dir` + `--snapshots`: Treat every `.glb` in a directory as a successive time step (use with `--time` to set the starting index)
- `--skip_leaf_decimation`: Keep tiles at the deepest level at full resolution while still simplifying parent levels
- `--min_ratio`: Minimum fraction of faces retained per decimation pass (default 0.02 = 2%)
- `--min_tris`: Minimum triangle count allowed during decimation (default 32)
- `--max_iter`: Maximum number of decimation iterations per tile (default 6)
- `--root_voxel_ratio`: (optional) fraction of the root tile bounding-box diagonal to use for voxel clustering fallback when decimation cannot hit the byte target
- `--root_voxel_trigger`: Multiple of the byte target that triggers the voxel clustering fallback (default 4x)
- `--static_octree`: Enable deterministic static world-space layouts that emit placeholder tiles at every node plus a flattened payload (pairs with `--tiling_space world`)
- `--write_tileset`: Emit a 3D Tiles 1.1 `tileset_<time>.json` alongside the manifest (uses the `3DTILES_content_gltf` extension to point at the existing GLB tiles)
- `--tileset-origin`: When used with `--write_tileset`, place the tileset root in an ENU frame centred on the supplied WGS84 `lat,lon[,height]`
- `--tileset-scale`: Optional uniform scale factor applied at the tileset root (default 1.0)
- `--debug_scene`: Print the glTF scene graph and per-primitive attribute summary while loading (handy when validating GLB exports)

During import, any PBR base-colour textures are baked into per-vertex `COLOR_0` attributes so that world-space tiles can render with colour without shipping textures.

#### Quick-start example

If you simply want to reproduce the single-cylinder dataset we ship in `examples/`, the exact command we use during development is:

```bash
python3 build_tiles.py \
  --in_glb examples/single_cylinder/single_cylinder.glb \
  --tiling_space uv \
  --out_dir tiles_out \
  --extract single-cylinder-uv \
  --max_depth 4 \
  --target_kb 100 \
  --preserve_borders \
  --snap_ratio 0.001 \
  --skip_leaf_decimation \
  --min_ratio 0.002 \
  --min_tris 8 \
  --max_iter 6 \
  --root_voxel_ratio 0.02 \
  --root_voxel_trigger 4 
```

Those switches keep the deepest tiles at full fidelity, aggressively snap borders, and emit both the manifest and a 3D Tiles tileset. Drop the flags you don’t need or tweak the depth/target size to suit your dataset.

### Unsteady datasets

`examples/cylinders_unsteady/cylinders_unsteady.py` creates a series of rotating-cylinder snapshots:

```bash
# Generate 20 GLBs, rotating 5° per frame
python3 examples/cylinders_unsteady/cylinders_unsteady.py \
  --snapshots 20 --delta-degrees 5 \
  --output_dir examples/cylinders_unsteady/output

# Tile every snapshot (world-space octree example)
python3 build_tiles.py --snapshots \
  --input_dir examples/cylinders_unsteady/output \
  --out_dir tiles_out \
  --extract cylinders_unsteady_world \
  --tiling_space world --max_depth 2 --target_kb 500 \
  --preserve_borders --snap_ratio 0.001
```

Run the viewer and select `cylinders_unsteady_world`; the time slider scrubs between the manifests, and the viewer prefetches +/-1 and +/-2 timesteps to hide network latency.

### 2. Start the Server

```bash
npm start
# or for development with auto-reload:
npm run dev
```

The server will start on http://localhost:8080

### Alternate: Static Viewer With Config

If you want to host the viewer from a plain static server (no Fastify, no `/api/extracts`), drop a `viewer-config.json` alongside `public/index.html` describing the datasets you want to expose:

```json
{
  "datasets": [
    {
      "name": "cylinders_world_2",
      "label": "Multi-stage cylinders",
      "times": [0, 1, 2],
      "defaultTime": 0,
      "tilesBasePath": "/no_api_test/cylinders_world_2/",
      "manifests": [
        { "time": 0, "manifest": "/no_api_test/cylinders_world_2/manifest_0.json" },
        { "time": 1, "manifest": "/no_api_test/cylinders_world_2/manifest_1.json" },
        { "time": 2, "manifest": "/no_api_test/cylinders_world_2/manifest_2.json" }
      ]
    }
  ],
  "defaultExtract": "cylinders_world_2",
  "defaultTime": 0,
  "defaultSseRefine": 25
}
```

Top-level fields such as `defaultSseRefine` (pixels) let you customise the initial SSE threshold without touching the code. Then update `public/index.html` to fetch that config on startup (see inline comments in the file). In this mode the viewer never calls `/api/extracts`—it just loads the manifests/payloads referenced in the config and points tiles at the static `tilesBasePath`.

### 3. View the Tiles

- **Single-extract viewer**: http://localhost:8080/
  - dat.GUI panel (top-right) lets you pick extract/time (slider when <=100 timesteps, dropdown otherwise), adjust SSE thresholds, toggle wireframe, overlay per-tile bounding boxes, and enable tile colours
  - HUD (top-left) shows current SSE thresholds, active tile count, cache population, and a loading spinner
- **Multi-extract viewer**: http://localhost:8080/multi_index.html

## Directory Structure

```
extract-tiles/
├── build_tiles.py          # Python preprocessor
├── server.mjs              # Fastify tile server
├── package.json
├── requirements.txt
├── public/
│   ├── index.html          # Single-extract viewer
│   ├── viewer.js
│   ├── multi_index.html   # Multi-extract viewer
│   └── multi_viewer.js
└── tiles_out/              # Generated tiles (created by preprocessor)
    └── <extract>/
        ├── manifest_<time>.json
        ├── manifest_<time>.json.gz  # Optional precompressed manifest (keep alongside .json)
        └── <time>/
            └── z/x/y.glb   # Tile files
```

## Manifest Format

Each extract generates a manifest JSON file with metadata (see `tiles_out/multi-stage-comp-unst-static-octree/manifest_0.json` for a full example):

```json
{
  "extract": "name",
  "time": 0,
  "source": "input.glb",
  "tilingSpace": "uv",
  "grid": "quadtree",
  "maxDepth": 5,
  "targetTileBytes": 204800,
  "global": {
    "triCount": 999999,
    "avgTriArea": 0.0003,
    "minTriArea": 1.1e-7
  },
  "tiles": [...]
}
```

If you have large manifests, you can precompress them with `gzip -k manifest_<time>.json`. Keep both the `.json` and `.json.gz` files in place—the Fastify server automatically serves the precompressed payload when the client requests gzip while falling back to the plain JSON when needed.

## Tile Metadata

Each GLB tile includes metadata in `mesh.extras`: 

```json
{
  "tileId": "z/x/y",
  "z": 3, "x": 5, "y": 6,
  "k": 2,
  "parent": "2/2/3/1",
  "children": ["4/10/12", "4/11/12", "4/10/13", "4/11/13"],
  "aabbWorld": [[xmin,ymin,zmin], [xmax,ymax,zmax]],
  "aabbUV": [[u0,v0], [u1,v1]],
  "triCount": 1234,
  "geometricError": 0.002,
  "approxBytes": 204800,
  "time": 0
}
```

`k` encodes the octant index for world-space octrees (0-7) and stays 0 for UV quadtrees. When `--static_octree` is enabled every node is emitted (even empty leaves) so `children` may include placeholders even when `triCount` is zero.

## Static Payload Format (`--static_octree`)

When `--static_octree` is used the tiler also emits `payload_<time>.json`. This is a flattened table of selected tile fields for lightweight viewers:

```json
{
  "extract": "name",
  "time": 0,
  "baseManifest": "manifest_0.json",
  "fields": ["tileId", "actualBytes", "approxBytes", ...],
  "tiles": [
    ["0/0/0/0", 883232, 865968, 5.27e-7, ...],
    ["1/0/0/0", 288404, 347556, 5.27e-7, ...]
  ]
}
```

`fields` defines the column order, and each entry under `tiles` is a row matching that order (always starting with `tileId`). The payload doesn't duplicate geometry; it just mirrors manifest metadata for consumers that want deterministic tree layouts without walking the full manifest.

## Viewer Controls

- **Orbit**: Left mouse drag
- **Zoom**: Scroll wheel or right mouse drag
- **Pan**: Middle mouse drag
- **dat.GUI panel**:
  - **Extract / Time**: Switch between datasets and time indices
  - **SSE Refine**: Set the maximum screen-space error before refinement (0.1-120 px; default 18 px). The coarsen threshold automatically tracks at half this value for hysteresis.
  - **Wireframe**: Toggle mesh rendering between solid and wireframe
  - **Bounding Boxes**: Overlay each tile's world-space bounding box while keeping the surface visible (default: on)
  - **Tile Colours**: Replace surface shading with a unique colour per tile to visualise current LOD coverage
  - **Simple Shading**: Swap to a Phong material (optionally using vertex colours) for soft lighting plus specular highlights; combines the viewer’s ambient fill with a directional light mounted to the camera so highlights track your orbit (default: on)

## Performance Tuning

### Tile Size
- Adjust `--target_kb` to balance between tile count and download size
- Typical values: 50-500 KB per tile

### LOD Depth
- `--max_depth` controls maximum refinement level
- Higher values = more detail but more tiles
- Typical values: 4-7

### SSE Thresholds
- **Refine threshold**: Pixels of error before loading finer tiles (default: 6.0; adjustable via GUI 0.1-50 px)
- **Coarsen threshold**: Automatically set to half the refine value to provide hysteresis and reduce LOD thrashing

### Cache Settings
- Adjust `MAX_TILES` and `MAX_CACHE_BYTES` in `public/viewer.js` for memory usage
- Defaults: 500 tiles / ~600 MB budget; root tiles stay resident for fast fallback

## API Endpoints

- `GET /tiles/*` - Serve tile GLB files
- `GET /manifest/:extract/:time.json` - Get manifest for an extract
- `GET /api/extracts` - List available extracts and time steps

## Testing

Create a sample GLB file with UV coordinates and test the system:

```bash
# Generate tiles
python3 build_tiles.py --in_glb sample.glb --tiling_space uv

# Start server
npm start

# Open browser to http://localhost:8080
```

## Example Generators

- `examples/helical/helical.py`: builds an annular helical strip with UVs and colors.
- `examples/cylinders/cylinders.py`: builds spokes of open cylinders (multiple meshes) between hub and tip radii; supports stacking multiple wheels along the Z-axis and optional sinusoidal radius modulation per cylinder.
- `examples/cylinders_unsteady/cylinders_unsteady.py`: produces a time series of rotating cylinder spokes suitable for testing unsteady tiling.

## License

MIT
