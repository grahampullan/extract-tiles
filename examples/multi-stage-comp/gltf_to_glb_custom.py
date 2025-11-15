#!/usr/bin/env python3
"""Custom GLB converter for multi-stage compressor (color overrides only)."""

import argparse
import base64
from pathlib import Path
from collections import Counter
import numpy as np
from pygltflib import Accessor, Buffer, BufferView, GLTF2

BRIGHT_BLUE = np.array([20, 160, 255, 255], dtype=np.uint8)
LIGHT_GREY = np.array([210, 210, 210, 255], dtype=np.uint8)
BLUE_RANGE = range(0, 12)


def _append_color_accessor(gltf: GLTF2, color_bytes: bytes, count: int) -> int:
    uri = "data:application/octet-stream;base64," + base64.b64encode(color_bytes).decode("ascii")
    gltf.buffers = list(gltf.buffers or [])
    gltf.bufferViews = list(gltf.bufferViews or [])
    gltf.accessors = list(gltf.accessors or [])

    gltf.buffers.append(Buffer(uri=uri, byteLength=len(color_bytes)))
    buffer_idx = len(gltf.buffers) - 1

    gltf.bufferViews.append(BufferView(buffer=buffer_idx, byteOffset=0, byteLength=len(color_bytes), target=34962))
    view_idx = len(gltf.bufferViews) - 1

    accessor = Accessor(bufferView=view_idx, byteOffset=0, componentType=5121, count=count, type="VEC4", normalized=True)
    gltf.accessors.append(accessor)
    return len(gltf.accessors) - 1


def apply_color_overrides(gltf: GLTF2) -> Counter:
    counter: Counter[tuple[int, int, int, int]] = Counter()
    if not getattr(gltf, "meshes", None):
        return counter

    for mesh_idx, mesh in enumerate(gltf.meshes):
        target_color = BRIGHT_BLUE if mesh_idx in BLUE_RANGE else LIGHT_GREY
        for prim in mesh.primitives or []:
            pos_idx = getattr(prim.attributes, "POSITION", None)
            if pos_idx is None:
                continue
            accessor = gltf.accessors[pos_idx]
            count = accessor.count or 0
            if count <= 0:
                continue
            color_array = np.tile(target_color, (count, 1)).astype(np.uint8)
            accessor_idx = _append_color_accessor(gltf, color_array.tobytes(), count)
            prim.attributes.COLOR_0 = accessor_idx
            counter[tuple(int(c) for c in target_color)] += 1
    return counter


def convert(src: Path, dst: Path) -> None:
    gltf = GLTF2().load(str(src))
    color_counter = apply_color_overrides(gltf)
    if color_counter:
        for rgba, cnt in color_counter.items():
            print(f"Applied color {rgba} to {cnt} primitives")
    gltf.convert_buffers(buffer_format="glb")
    dst.parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(str(dst))


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom color GLB packer for multi-stage compressor")
    parser.add_argument("input", type=Path, help="Path to scene.gltf")
    parser.add_argument("output", type=Path, nargs="?", help="Output .glb path")
    args = parser.parse_args()

    dst = args.output or args.input.with_suffix("_custom.glb")
    convert(args.input, dst)
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
