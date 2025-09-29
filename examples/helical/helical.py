#!/usr/bin/env python3
"""
Helical strip surface with annular cross-section -> GLB with vertex colors.

Creates a single triangulated surface that follows a helical path with radial extent.
When viewed from the front, you see an annular (ring-shaped) cross-section.

Parameters:
  --inner-radius    : inner radius of the annular strip
  --outer-radius    : outer radius of the annular strip
  --pitch           : axial advance per full revolution
  --turns           : number of full revolutions
  --segments-per-turn: tessellation along the helix
  --radial-segments : tessellation across radial width (inner to outer)
  --color-repeat    : color cycles per revolution (θ)
  --alpha           : vertex color alpha (0..255)

Example:
  python helical.py -o helical_strip.glb \
    --inner-radius 8 --outer-radius 12 --pitch 10 --turns 3 \
    --segments-per-turn 256 --radial-segments 64 \
    --color-repeat 2
"""

import argparse
import numpy as np
import trimesh
from pathlib import Path
from pygltflib import (
    GLTF2, Buffer, BufferView, Accessor, Mesh, Node, Scene,
    Asset, Attributes, Primitive
)

# --- HSV->RGB helper (0..1 in, 0..255 out) ---
def hsv_to_rgb_u8(h, s=1.0, v=1.0):
    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = np.choose(i % 6, [v, q, p, p, t, v])
    g = np.choose(i % 6, [t, v, v, q, p, p])
    b = np.choose(i % 6, [p, p, t, v, v, q])
    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb * 255, 0, 255)).astype(np.uint8)

def write_glb_with_uv(vertices, faces, uvs, vertex_colors, out_path):
    """Write GLB file with proper UV coordinates"""
    positions = np.asarray(vertices, dtype=np.float32)
    indices = np.asarray(faces, dtype=np.uint32).reshape(-1)
    uv = np.asarray(uvs, dtype=np.float32)
    colors = np.asarray(vertex_colors, dtype=np.uint8)

    gltf = GLTF2(asset=Asset(version="2.0"))
    chunks, offsets = [], {}

    def push(name, data_bytes):
        offsets[name] = sum(len(c) for c in chunks)
        chunks.append(data_bytes)

    push("pos", positions.tobytes())
    push("idx", indices.tobytes())
    push("uv", uv.tobytes())
    push("col", colors.tobytes())

    blob = b"".join(chunks)
    gltf.buffers = [Buffer(byteLength=len(blob))]
    gltf.set_binary_blob(blob)

    views, accs = [], []

    def bv(name, target):
        start = offsets[name]
        if name == "pos":
            size = positions.nbytes
        elif name == "idx":
            size = indices.nbytes
        elif name == "uv":
            size = uv.nbytes
        else:  # col
            size = colors.nbytes
        views.append(BufferView(buffer=0, byteOffset=start, byteLength=size, target=target))
        return len(views)-1

    def acc(bvi, ct, count, typ, normalized=False):
        a = Accessor(bufferView=bvi, componentType=ct, count=count, type=typ, normalized=normalized)
        accs.append(a)
        return len(accs)-1

    gltf.bufferViews = views
    gltf.accessors = accs

    ai_pos = acc(bv("pos", 34962), 5126, len(positions), "VEC3")
    ai_idx = acc(bv("idx", 34963), 5125, len(indices), "SCALAR")
    ai_uv = acc(bv("uv", 34962), 5126, len(uv), "VEC2")
    ai_col = acc(bv("col", 34962), 5121, len(colors), "VEC4", normalized=True)

    attrs = Attributes(POSITION=ai_pos, TEXCOORD_0=ai_uv, COLOR_0=ai_col)
    prim = Primitive(attributes=attrs, indices=ai_idx, mode=4)
    mesh = Mesh(primitives=[prim])

    gltf.meshes = [mesh]
    gltf.nodes = [Node(mesh=0)]
    gltf.scenes = [Scene(nodes=[0])]
    gltf.scene = 0

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(out_path)

def build_helical_strip(
    inner_radius: float,
    outer_radius: float,
    pitch: float,
    turns: float,
    segments_per_turn: int = 128,
    radial_segments: int = 32,
    color_repeat: float = 1.0,
    alpha: int = 255,
) -> trimesh.Trimesh:
    """
    Build a single-surface helical strip with radial extent (annular cross-section).

    Parameters:
      inner_radius: inner radius of the annular strip
      outer_radius: outer radius of the annular strip
      pitch: axial advance per full revolution
      turns: number of full revolutions
      segments_per_turn: tessellation along the helix
      radial_segments: tessellation across radial width (inner to outer)
      color_repeat: color cycles per revolution
      alpha: vertex color alpha (0..255)

    Creates a 2D grid of vertices that forms a single surface strip with annular cross-section.
    """
    if inner_radius <= 0 or outer_radius <= inner_radius or pitch == 0 or turns <= 0:
        raise ValueError("Must have 0 < inner_radius < outer_radius, and positive pitch/turns.")
    if segments_per_turn < 3 or radial_segments < 1:
        raise ValueError("segments_per_turn must be >= 3, radial_segments >= 1.")
    if not (0 <= alpha <= 255):
        raise ValueError("alpha must be 0..255")

    # Discretization
    I = int(max(3, round(segments_per_turn * turns)))  # along helix
    J = int(radial_segments)                           # across radial width

    # Parameter arrays
    # s: parameter along helix (0 to 1)
    # r: parameter across radial width (0 to 1, inner to outer)
    s_vals = np.linspace(0.0, 1.0, I + 1)
    r_vals = np.linspace(0.0, 1.0, J + 1)

    # Create vertex grid
    verts = []
    uvs = []
    thetas_for_color = []

    for j in range(J + 1):  # across radial width (inner to outer)
        r_param = r_vals[j]
        current_radius = inner_radius + r_param * (outer_radius - inner_radius)

        for i in range(I + 1):  # along helix
            s = s_vals[i]

            # Helix parameters
            theta = s * 2.0 * np.pi * turns
            z = s * pitch * turns

            # Position on helical surface
            x = current_radius * np.cos(theta)
            y = current_radius * np.sin(theta)

            vertex = np.array([x, y, z])
            verts.append(vertex)

            # UV coordinates
            # u: along spiral (0 to 1 from start to end)
            # v: radial direction (0 to 1 from inner to outer)
            u = s  # goes from 0 to 1 along the spiral
            v = r_param  # goes from 0 to 1 from inner to outer radius
            uvs.append([u, v])

            thetas_for_color.append(theta)

    V = np.array(verts, dtype=np.float64)
    UV = np.array(uvs, dtype=np.float32)

    # Create faces (quads made of two triangles)
    faces = []
    for j in range(J):      # across radial width
        for i in range(I):  # along helix
            # Vertex indices for current quad
            v00 = j * (I + 1) + i       # (j, i)
            v01 = j * (I + 1) + i + 1   # (j, i+1)
            v10 = (j + 1) * (I + 1) + i # (j+1, i)
            v11 = (j + 1) * (I + 1) + i + 1 # (j+1, i+1)

            # Two triangles per quad
            faces.append([v00, v01, v11])
            faces.append([v00, v11, v10])

    F = np.array(faces, dtype=np.int32)

    # Vertex colors based on angular position
    hue = ((np.array(thetas_for_color) / (2.0 * np.pi)) * float(color_repeat)) % 1.0
    rgb = hsv_to_rgb_u8(hue, 1.0, 1.0)
    a_chan = np.full((rgb.shape[0], 1), int(alpha), dtype=np.uint8)
    vertex_colors = np.hstack([rgb, a_chan])  # RGBA

    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    mesh.visual.vertex_colors = vertex_colors
    mesh.visual.uv = UV

    # Clean up the mesh
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    return mesh

def main():
    ap = argparse.ArgumentParser(description="Generate a colored helical strip surface with annular cross-section and export as GLB.")
    ap.add_argument("-o", "--output", default="helical_strip.glb", help="Output GLB path.")
    ap.add_argument("--stl", action="store_true", help="Also export STL (no vertex colors in STL).")
    ap.add_argument("--inner-radius", type=float, default=8.0, help="Inner radius of the annular strip.")
    ap.add_argument("--outer-radius", type=float, default=12.0, help="Outer radius of the annular strip.")
    ap.add_argument("--pitch", type=float, default=8.0, help="Axial advance per full revolution.")
    ap.add_argument("--turns", type=float, default=5.0, help="Number of full revolutions.")
    ap.add_argument("--segments-per-turn", type=int, default=128, help="Tessellation along the helix.")
    ap.add_argument("--radial-segments", type=int, default=32, help="Tessellation across radial width (inner to outer).")
    ap.add_argument("--color-repeat", type=float, default=1.0, help="Color cycles per full 360° revolution.")
    ap.add_argument("--alpha", type=int, default=255, help="Vertex color alpha (0..255).")
    args = ap.parse_args()

    mesh = build_helical_strip(
        inner_radius=args.inner_radius,
        outer_radius=args.outer_radius,
        pitch=args.pitch,
        turns=args.turns,
        segments_per_turn=args.segments_per_turn,
        radial_segments=args.radial_segments,
        color_repeat=args.color_repeat,
        alpha=args.alpha,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Use custom GLB writer to ensure UV coordinates are properly saved
    write_glb_with_uv(
        vertices=mesh.vertices,
        faces=mesh.faces,
        uvs=mesh.visual.uv,
        vertex_colors=mesh.visual.vertex_colors,
        out_path=out.as_posix()
    )
    print(f"Saved GLB: {out.resolve()} (with vertex colors and UV coordinates)")

    if args.stl:
        stl_path = out.with_suffix(".stl")
        mesh.export(stl_path.as_posix())
        print(f"Saved STL: {stl_path.resolve()}")

if __name__ == "__main__":
    main()
