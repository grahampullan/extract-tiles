# Extract-Tiles System

A multi-resolution tiling system for visualizing large triangulated mesh surfaces from CFD simulations. This implementation follows the concepts from Graham Pullan's AIAA paper on visualization of aerospace simulations using a navigation approach.

## Features

- **Multi-resolution tiles**: Automatically generates LOD tiles from GLB files.
- **Multiple tiling modes**:
  - UV-Quadtree: For meshes with texture coordinates.
  - World-Space Octree: For meshes without UVs (e.g., isosurfaces).
- **Crack prevention**: Includes overlap margins, optional skirt generation, and border snapping with configurable tolerances.
- **Dynamic LOD**: Screen-space-error refinement with hysteresis plus queue throttling to keep loads responsive.
- **Time-series aware**: Discovers available extracts/timesteps, prefetches neighbouring manifests/tiles, and auto-picks a slider or dropdown based on timestep count.
- **HUD & diagnostics**: Live overlay for SSE thresholds/tile counts/cache size, wireframe toggle, per-tile bounding boxes, and colour-debug mode.
- **Efficient streaming**: Fastify server with compression and caching.
- **Interactive viewers**:
  - Single-extract viewer with dat.GUI controls for dataset/time selection, LOD tuning, diagnostics, and HUD indicators.
  - Multi-extract viewer for comparing multiple datasets.

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
- `--input_dir` + `--snapshots`: Treat every `.glb` in a directory as a successive time step (use with `--time` to set the starting index)

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
        └── <time>/
            └── z/x/y.glb   # Tile files
```

## Manifest Format

Each extract generates a manifest JSON file with metadata:

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

## Tile Metadata

Each GLB tile includes metadata in `mesh.extras`:

```json
{
  "tileId": "z/x/y",
  "z": 3, "x": 5, "y": 6,
  "parent": "2/2/3",
  "children": ["4/10/12", "4/11/12", "4/10/13", "4/11/13"],
  "aabbWorld": [[xmin,ymin,zmin], [xmax,ymax,zmax]],
  "aabbUV": [[u0,v0], [u1,v1]],
  "triCount": 1234,
  "geometricError": 0.002,
  "approxBytes": 204800,
  "time": 0
}
```

## Viewer Controls

- **Orbit**: Left mouse drag
- **Zoom**: Scroll wheel or right mouse drag
- **Pan**: Middle mouse drag
- **dat.GUI panel**:
  - **Extract / Time**: Switch between datasets and time indices
  - **SSE Refine**: Set the maximum screen-space error before refinement (0.1-50 px; default 6 px). The coarsen threshold automatically tracks at half this value for hysteresis.
  - **Wireframe**: Toggle mesh rendering between solid and wireframe
  - **Bounding Boxes**: Overlay each tile's world-space bounding box while keeping the surface visible
  - **Tile Colours**: Replace surface shading with a unique colour per tile to visualise current LOD coverage

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

## Troubleshooting

### "UV mode requires TEXCOORD_0" error
- Your GLB file doesn't have UV coordinates
- Use `--tiling_space world` instead

### Tiles not loading
- Check browser console for errors
- Ensure server is running
- Verify manifest path is correct

### Poor performance
- Reduce `--max_depth` for fewer tiles
- Increase `--target_kb` for larger tiles
- Adjust SSE thresholds in viewer

## License

MIT

## Credits

Based on concepts from "Visualisation of aerospace simulations - a navigation approach" by Graham Pullan, AIAA 2026.
