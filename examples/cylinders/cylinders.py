#!/usr/bin/env python3
"""Generate a GLB containing multiple cylindrical surface meshes.

Each cylinder is represented only by its curved surface (no end caps) and includes
UV coordinates so that `u` wraps around the circumference [0,1] and `v` runs along
its axis [0,1]. Cylinders are arranged like the spokes of a wheel between a hub
radius and a tip radius. Multiple wheels can optionally be stacked along the Z
axis for testing multi-mesh datasets.
"""

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from pygltflib import (
    GLTF2,
    Asset,
    Buffer,
    BufferView,
    Accessor,
    Mesh,
    Node,
    Scene,
    Attributes,
    Primitive,
)


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def make_cylinder_mesh(
    axis_start: np.ndarray,
    axis_end: np.ndarray,
    radius: float,
    segments_axis: int,
    segments_circ: int,
    color_rgba: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (positions, indices, uvs, colors) for a single open cylinder."""
    axis_vec = axis_end - axis_start
    length = np.linalg.norm(axis_vec)
    if length <= 0:
        raise ValueError("Cylinder axis length must be positive")

    tangent = axis_vec / length

    # Build an orthonormal frame (normal, binormal, tangent)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(up, tangent)) > 0.999:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    binormal = np.cross(up, tangent)
    binorm_len = np.linalg.norm(binormal)
    if binorm_len < 1e-8:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        binormal = np.cross(up, tangent)
        binorm_len = np.linalg.norm(binormal)
        if binorm_len < 1e-8:
            raise ValueError("Failed to build orthonormal basis for cylinder")
    binormal /= binorm_len
    normal = np.cross(tangent, binormal)

    # Parameter grids
    v_vals = np.linspace(0.0, 1.0, segments_axis + 1)
    u_vals = np.linspace(0.0, 1.0, segments_circ + 1)

    verts = []
    uvs = []
    colors = []

    for v in v_vals:
        center = axis_start + tangent * (v * length)
        for u in u_vals:
            theta = u * 2.0 * math.pi
            offset = radius * (math.cos(theta) * normal + math.sin(theta) * binormal)
            verts.append(center + offset)
            uvs.append([u % 1.0, v])  # wrap u in [0,1)
            colors.append(color_rgba)

    verts = np.asarray(verts, dtype=np.float32)
    uvs = np.asarray(uvs, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)

    # Triangulate quads
    verts_per_ring = segments_circ + 1
    faces = []
    for i in range(segments_axis):
        for j in range(segments_circ):
            v00 = i * verts_per_ring + j
            v01 = v00 + 1
            v10 = (i + 1) * verts_per_ring + j
            v11 = v10 + 1
            faces.append([v00, v01, v11])
            faces.append([v00, v11, v10])

    faces = np.asarray(faces, dtype=np.uint32)
    return verts, faces.reshape(-1), uvs, colors


def hue_to_rgba(h: float, alpha: int = 255) -> Tuple[int, int, int, int]:
    """Simple HSV->RGBA helper (s=v=1)."""
    h = h % 1.0
    i = int(h * 6)
    f = h * 6 - i
    p = 0.0
    q = 1.0 - f
    t = f
    choices = [
        (1.0, t, p),
        (q, 1.0, p),
        (p, 1.0, t),
        (p, q, 1.0),
        (t, p, 1.0),
        (1.0, p, q),
    ]
    r, g, b = choices[i % 6]
    return (
        int(round(r * 255)),
        int(round(g * 255)),
        int(round(b * 255)),
        alpha,
    )


# -----------------------------------------------------------------------------
# glTF writer (multi-mesh)
# -----------------------------------------------------------------------------

def write_multi_mesh_glb(mesh_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], out_path: Path) -> None:
    gltf = GLTF2(asset=Asset(version="2.0"))
    buffer_views: List[BufferView] = []
    accessors: List[Accessor] = []
    meshes: List[Mesh] = []
    nodes: List[Node] = []

    blob = bytearray()

    def push_bytes(data: bytes, target: int) -> int:
        # Ensure 4-byte alignment
        pad = (4 - (len(blob) % 4)) % 4
        if pad:
            blob.extend(b"\x00" * pad)
        offset = len(blob)
        blob.extend(data)
        buffer_views.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(data), target=target))
        return len(buffer_views) - 1

    def make_accessor(view_idx: int, count: int, ctype: int, a_type: str, minmax=None, normalized=False) -> int:
        accessor = Accessor(
            bufferView=view_idx,
            componentType=ctype,
            count=count,
            type=a_type,
            normalized=normalized,
        )
        if minmax is not None:
            accessor.min = list(map(float, minmax[0]))
            accessor.max = list(map(float, minmax[1]))
        accessors.append(accessor)
        return len(accessors) - 1

    for verts, indices, uvs, colors in mesh_data:
        v_bytes = verts.tobytes()
        uv_bytes = uvs.tobytes()
        c_bytes = colors.tobytes()
        idx_bytes = indices.tobytes()

        v_view = push_bytes(v_bytes, 34962)  # ARRAY_BUFFER
        uv_view = push_bytes(uv_bytes, 34962)
        c_view = push_bytes(c_bytes, 34962)
        i_view = push_bytes(idx_bytes, 34963)  # ELEMENT_ARRAY_BUFFER

        pos_accessor = make_accessor(v_view, len(verts), 5126, "VEC3", (verts.min(axis=0), verts.max(axis=0)))
        uv_accessor = make_accessor(uv_view, len(uvs), 5126, "VEC2")
        col_accessor = make_accessor(c_view, len(colors), 5121, "VEC4", normalized=True)
        idx_accessor = make_accessor(i_view, len(indices), 5125, "SCALAR")

        attrs = Attributes(POSITION=pos_accessor, TEXCOORD_0=uv_accessor, COLOR_0=col_accessor)
        prim = Primitive(attributes=attrs, indices=idx_accessor, mode=4)  # TRIANGLES
        mesh = Mesh(primitives=[prim])
        meshes.append(mesh)
        nodes.append(Node(mesh=len(meshes) - 1))

    gltf.bufferViews = buffer_views
    gltf.accessors = accessors
    gltf.meshes = meshes
    gltf.nodes = nodes
    gltf.scenes = [Scene(nodes=list(range(len(nodes))))]
    gltf.scene = 0

    gltf.buffers = [Buffer(byteLength=len(blob))]
    gltf.set_binary_blob(bytes(blob))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(str(out_path))


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def build_spoke_wheels(
    spokes: int,
    wheels: int,
    wheel_spacing: float,
    r_hub: float,
    r_tip: float,
    tube_radius: float,
    segments_axis: int,
    segments_circ: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if spokes < 1:
        raise ValueError("spokes must be >= 1")
    if wheels < 1:
        raise ValueError("wheel-count must be >= 1")
    if r_tip <= r_hub:
        raise ValueError("r_tip must be greater than r_hub")
    if tube_radius <= 0:
        raise ValueError("tube radius must be positive")
    if segments_axis < 1 or segments_circ < 3:
        raise ValueError("Need at least 1 axis segment and 3 circumferential segments")

    meshes = []
    axis_length = r_tip - r_hub

    # Center wheels around the origin along Z
    z_offsets = np.linspace(
        -0.5 * wheel_spacing * (wheels - 1),
        0.5 * wheel_spacing * (wheels - 1),
        wheels,
    )

    for w, z_off in enumerate(z_offsets):
        for s in range(spokes):
            angle = (2.0 * math.pi * s) / spokes
            dir_xy = np.array([math.cos(angle), math.sin(angle), 0.0], dtype=np.float64)
            start = dir_xy * r_hub + np.array([0.0, 0.0, z_off], dtype=np.float64)
            end = dir_xy * r_tip + np.array([0.0, 0.0, z_off], dtype=np.float64)
            color = hue_to_rgba((s / max(1, spokes)) + 0.17 * w)
            mesh = make_cylinder_mesh(start, end, tube_radius, segments_axis, segments_circ, color)
            meshes.append(mesh)

    return meshes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate spoke-aligned cylinder meshes and export to GLB")
    parser.add_argument("-o", "--output", type=Path, default=Path("cylinders.glb"), help="Output GLB path")
    parser.add_argument("--spokes", type=int, default=6, help="Number of cylinders per wheel")
    parser.add_argument("--wheel-count", type=int, default=1, help="Number of wheel stacks along Z")
    parser.add_argument("--wheel-spacing", type=float, default=4.0, help="Spacing between wheels along Z")
    parser.add_argument("--r-hub", type=float, default=4.0, help="Inner radius for cylinder axes")
    parser.add_argument("--r-tip", type=float, default=12.0, help="Outer radius for cylinder axes")
    parser.add_argument("--tube-radius", type=float, default=0.4, help="Radius of each cylinder surface")
    parser.add_argument("--segments-axis", type=int, default=32, help="Segments along cylinder axis (v direction)")
    parser.add_argument("--segments-circ", type=int, default=48, help="Segments around circumference (u direction)")
    return parser.parse_args()


def main():
    args = parse_args()
    meshes = build_spoke_wheels(
        spokes=args.spokes,
        wheels=args.wheel_count,
        wheel_spacing=args.wheel_spacing,
        r_hub=args.r_hub,
        r_tip=args.r_tip,
        tube_radius=args.tube_radius,
        segments_axis=args.segments_axis,
        segments_circ=args.segments_circ,
    )
    write_multi_mesh_glb(meshes, args.output)
    print(f"Wrote {len(meshes)} cylinder meshes to {args.output}")


if __name__ == "__main__":
    main()
