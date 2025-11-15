#!/usr/bin/env python3
"""Report which meshes/primitives in a glTF/GLB contain COLOR_0 attributes."""

import argparse
from pathlib import Path
from pygltflib import GLTF2

def main() -> None:
    parser = argparse.ArgumentParser(description="Check vertex-color presence in a glTF/GLB")
    parser.add_argument("input", type=Path, help="Path to .gltf or .glb file")
    args = parser.parse_args()

    gltf = GLTF2().load_binary(str(args.input)) if args.input.suffix.lower() == '.glb' else GLTF2().load(str(args.input))

    if not gltf.meshes:
        print("No meshes found")
        return

    total = 0
    with_color = 0
    for mesh_idx, mesh in enumerate(gltf.meshes):
        for prim_idx, prim in enumerate(mesh.primitives or []):
            total += 1
            has_color = getattr(prim.attributes, 'COLOR_0', None) is not None
            if has_color:
                with_color += 1
            print(f"Mesh {mesh_idx}, primitive {prim_idx}: {'has COLOR_0' if has_color else 'NO COLOR_0'}")

    print(f"Summary: {with_color}/{total} primitives have vertex colors")

if __name__ == "__main__":
    main()
