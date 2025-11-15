#!/usr/bin/env python3
"""Convert a .gltf (plus external .bin/.image files) into a single .glb, adding base-color vertex data when needed."""

import argparse
import base64
import numpy as np
from pathlib import Path
from collections import Counter
from pygltflib import Accessor, Buffer, BufferView, GLTF2


def _material_base_color_rgba(gltf: GLTF2, material_index: int | None) -> np.ndarray:
    """Return RGBA color (uint8) derived from baseColorFactor or fallback to white."""
    default = np.array([255, 255, 255, 255], dtype=np.uint8)
    if material_index is None:
        return default
    materials = getattr(gltf, "materials", None) or []
    if material_index < 0 or material_index >= len(materials):
        return default
    mat = materials[material_index]
    pbr = getattr(mat, "pbrMetallicRoughness", None)
    if pbr and isinstance(pbr.baseColorFactor, list) and len(pbr.baseColorFactor) == 4:
        return np.array([min(255, max(0, int(round(c * 255)))) for c in pbr.baseColorFactor], dtype=np.uint8)
    emissive = getattr(mat, "emissiveFactor", None)
    if isinstance(emissive, list) and len(emissive) == 3:
        vals = [min(255, max(0, int(round(c * 255)))) for c in emissive]
        return np.array([*vals, 255], dtype=np.uint8)
    return default


def _append_color_accessor(gltf: GLTF2, color_bytes: bytes, count: int) -> int:
    """Create Buffer/BufferView/Accessor for COLOR_0 data and return accessor index."""
    uri = "data:application/octet-stream;base64," + base64.b64encode(color_bytes).decode("ascii")
    gltf.buffers = list(gltf.buffers or [])
    gltf.bufferViews = list(gltf.bufferViews or [])
    gltf.accessors = list(gltf.accessors or [])

    gltf.buffers.append(Buffer(uri=uri, byteLength=len(color_bytes)))
    buffer_idx = len(gltf.buffers) - 1

    gltf.bufferViews.append(BufferView(buffer=buffer_idx, byteOffset=0, byteLength=len(color_bytes), target=34962))
    view_idx = len(gltf.bufferViews) - 1

    accessor = Accessor(
        bufferView=view_idx,
        byteOffset=0,
        componentType=5121,  # UNSIGNED_BYTE
        count=count,
        type="VEC4",
        normalized=True,
    )
    gltf.accessors.append(accessor)
    return len(gltf.accessors) - 1


def ensure_vertex_colors(gltf: GLTF2):
    """Add COLOR_0 attributes derived from baseColorFactor where missing. Returns (count, color counter)."""
    if not getattr(gltf, "meshes", None):
        return 0, Counter()
    updated = 0
    color_counter: Counter[tuple[int, int, int, int]] = Counter()
    for mesh in gltf.meshes or []:
        for prim in mesh.primitives or []:
            attrs = prim.attributes
            if getattr(attrs, "COLOR_0", None) is not None:
                continue
            pos_idx = getattr(attrs, "POSITION", None)
            if pos_idx is None:
                continue
            accessor = gltf.accessors[pos_idx]
            count = accessor.count or 0
            if count <= 0:
                continue
            rgba = _material_base_color_rgba(gltf, prim.material)
            color_counter[tuple(int(c) for c in rgba)] += 1
            color_array = np.tile(rgba, (count, 1)).astype(np.uint8)
            accessor_idx = _append_color_accessor(gltf, color_array.tobytes(), count)
            attrs.COLOR_0 = accessor_idx
            updated += 1
    return updated, color_counter


def convert(src: Path, dst: Path) -> None:
    gltf = GLTF2().load(str(src))
    added, color_counter = ensure_vertex_colors(gltf)
    if added:
        print(f"Added COLOR_0 to {added} primitives lacking vertex colors")
        top_colors = color_counter.most_common(5)
        for rgba, count in top_colors:
            print(f"  color {rgba} used in {count} primitives")
    gltf.convert_buffers(buffer_format="glb")
    dst.parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(str(dst))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack a glTF + external buffers into one GLB (adds base-color vertex data when missing)")
    parser.add_argument("input", type=Path, help="Path to scene.gltf")
    parser.add_argument("output", type=Path, nargs="?", help="Destination .glb path")
    args = parser.parse_args()

    dst = args.output or args.input.with_suffix(".glb")
    convert(args.input, dst)
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
