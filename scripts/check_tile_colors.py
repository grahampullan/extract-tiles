#!/usr/bin/env python3
"""Quick helper to inspect COLOR_0 values in a tiled GLB."""

import argparse
import numpy as np
from pathlib import Path
from pygltflib import GLTF2


def read_color_accessor(tile_path: Path, max_rows: int = 10) -> np.ndarray:
    gltf = GLTF2().load_binary(str(tile_path))
    if not gltf.meshes:
        raise ValueError("No meshes found in tile")

    prim = gltf.meshes[0].primitives[0]
    color_idx = prim.attributes.COLOR_0
    if color_idx is None:
        raise ValueError("Tile has no COLOR_0 accessor")

    accessor = gltf.accessors[color_idx]
    if accessor.bufferView is None:
        raise ValueError("COLOR_0 accessor missing bufferView")

    view = gltf.bufferViews[accessor.bufferView]
    blob = gltf.binary_blob()
    start = (view.byteOffset or 0)
    end = start + (view.byteLength or 0)
    raw = blob[start:end]

    colors = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4)
    return colors[:max_rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect COLOR_0 values inside a tile GLB.")
    parser.add_argument("tile", type=Path, help="Path to the tile GLB file")
    parser.add_argument("--rows", type=int, default=10, help="Number of rows to print (default: 10)")
    args = parser.parse_args()

    colors = read_color_accessor(args.tile, max_rows=args.rows)
    print(f"Tile: {args.tile}")
    print(colors)


if __name__ == "__main__":
    main()
