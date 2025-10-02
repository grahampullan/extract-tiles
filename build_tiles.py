#!/usr/bin/env python3
"""
Multi-Resolution Extract-Tiling for Triangulated Mesh Surfaces
Preprocessor for generating multi-resolution tiles from GLB files
"""

import json
import os
import pathlib
import math
import base64
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import trimesh as tm
import open3d as o3d
from pygltflib import (
    GLTF2, Buffer, BufferView, Accessor, Mesh, Node, Scene,
    Asset, Attributes, Primitive
)

# Configuration constants
TARGET_TILE_BYTES = 200_000     # ~200 KB per tile
SIZE_TOLERANCE    = 0.15        # ±15%
UV_EPS            = 0.005       # UV overlap margin (~0.5%)
WORLD_EPS_RATIO   = 0.01        # world overlap margin (~1% of node size)
MAX_DEPTH         = 5

# ============================================================================
# Shared Utilities
# ============================================================================

def _load_gltf_with_buffers(path: str):
    """Load a glTF/GLB file and return (GLTF2, list[bytes])."""
    src = Path(path)
    suffix = src.suffix.lower()

    if suffix == ".glb":
        gltf = GLTF2().load_binary(path)
        blob = gltf.binary_blob()
        buffers: List[bytes] = []
        cursor = 0
        if gltf.buffers:
            for buf in gltf.buffers:
                length = buf.byteLength or 0
                buffers.append(blob[cursor:cursor+length])
                cursor += length
        else:
            buffers.append(blob)
        return gltf, buffers

    if suffix == ".gltf":
        gltf = GLTF2().load(path)
        base_dir = src.parent
        buffers = []
        for idx, buf in enumerate(gltf.buffers or []):
            uri = buf.uri
            if not uri:
                raise ValueError(f"Buffer {idx} in '{path}' has no URI; embed data or convert to GLB.")
            if uri.startswith("data:"):
                _, data = uri.split(",", 1)
                buffers.append(base64.b64decode(data))
            else:
                buf_path = base_dir / uri
                if not buf_path.exists():
                    raise FileNotFoundError(f"Referenced buffer '{uri}' not found relative to '{path}'.")
                buffers.append(buf_path.read_bytes())
        return gltf, buffers

    raise ValueError(f"Unsupported file extension '{suffix}' for '{path}'. Expected .gltf or .glb")


def _make_accessor_reader(gltf: GLTF2, buffers: List[bytes]):
    comps = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT3": 9, "MAT4": 16}
    types = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }

    def to_np(acc_idx):
        if acc_idx is None:
            return None
        acc = gltf.accessors[acc_idx]
        if acc.bufferView is None:
            raise ValueError("Sparse accessors are not supported.")
        view = gltf.bufferViews[acc.bufferView]
        buf_idx = view.buffer
        if buf_idx is None or buf_idx >= len(buffers):
            raise ValueError(f"Buffer index {buf_idx} out of range for accessor {acc_idx}.")
        buf_bytes = buffers[buf_idx]
        if buf_bytes is None:
            raise ValueError(f"Buffer {buf_idx} has no data loaded.")

        ncomp = comps[acc.type]
        dtype = types[acc.componentType]
        comp_size = np.dtype(dtype).itemsize
        offset = (view.byteOffset or 0) + (acc.byteOffset or 0)
        stride = view.byteStride or 0
        mv = memoryview(buf_bytes)

        if stride and stride != comp_size * ncomp:
            span = stride * acc.count
            raw = mv[offset:offset + span]
            arr = np.ndarray((acc.count, ncomp), dtype=dtype, buffer=raw, strides=(stride, comp_size))
            return np.array(arr, copy=True)

        length = acc.count * ncomp
        raw = mv[offset:offset + length * comp_size]
        arr = np.frombuffer(raw, dtype=dtype, count=length)
        return arr.reshape(acc.count, ncomp).copy()

    return to_np


def _triangulate_indices(raw_idx: Optional[np.ndarray], mode: Optional[int], vertex_count: int) -> np.ndarray:
    """Return flattened triangle indices for the given primitive."""
    mode = 4 if mode is None else mode

    def ensure_base() -> np.ndarray:
        if raw_idx is not None:
            return raw_idx.astype(np.uint32).reshape(-1)
        base = np.arange(vertex_count, dtype=np.uint32)
        return base

    if mode == 4:  # TRIANGLES
        base = ensure_base()
        if base.size % 3 != 0:
            raise ValueError(f"Triangle primitive has {base.size} indices (not divisible by 3).")
        return base

    if mode == 5:  # TRIANGLE_STRIP
        base = ensure_base()
        if base.size < 3:
            return np.empty(0, dtype=np.uint32)
        tris: List[int] = []
        flip = False
        for i in range(base.size - 2):
            a, b, c = base[i], base[i + 1], base[i + 2]
            if a == b or b == c or a == c:
                flip = not flip
                continue
            if flip:
                tris.extend([b, a, c])
            else:
                tris.extend([a, b, c])
            flip = not flip
        return np.asarray(tris, dtype=np.uint32)

    if mode == 6:  # TRIANGLE_FAN
        base = ensure_base()
        if base.size < 3:
            return np.empty(0, dtype=np.uint32)
        origin = base[0]
        tris: List[int] = []
        for i in range(1, base.size - 1):
            a, b = base[i], base[i + 1]
            if origin == a or a == b or origin == b:
                continue
            tris.extend([origin, a, b])
        return np.asarray(tris, dtype=np.uint32)

    # Unsupported primitive type (points/lines). Skip.
    return np.empty(0, dtype=np.uint32)


def load_glb_arrays(path):
    """Load GLB/GLTF file and extract arrays for vertices, UVs, colors, and indices."""
    gltf, buffers = _load_gltf_with_buffers(path)
    to_np = _make_accessor_reader(gltf, buffers)

    all_pos = []
    all_uv  = []
    all_col = []
    all_idx = []
    v_offset = 0

    # Iterate all meshes/primitives
    for mesh in (gltf.meshes or []):
        for prim in (mesh.primitives or []):
            pos = to_np(prim.attributes.POSITION)
            if pos is None or pos.size == 0:
                continue
            pos = pos.astype(np.float32)

            tri_idx = _triangulate_indices(to_np(prim.indices), prim.mode, len(pos))
            if tri_idx.size == 0:
                continue
            idx = tri_idx + np.uint32(v_offset)

            # Optional attributes
            uv = None
            if hasattr(prim.attributes, 'TEXCOORD_0') and prim.attributes.TEXCOORD_0 is not None:
                uv = to_np(prim.attributes.TEXCOORD_0).astype(np.float32)

            col = None
            if hasattr(prim.attributes, 'COLOR_0') and prim.attributes.COLOR_0 is not None:
                col_raw = to_np(prim.attributes.COLOR_0)
                # Normalize to uint8 RGBA for consistency
                if col_raw.dtype == np.uint8:
                    if col_raw.shape[1] == 3:
                        col = np.hstack([col_raw, np.full((col_raw.shape[0],1), 255, dtype=np.uint8)])
                    else:
                        col = col_raw.astype(np.uint8)
                else:
                    # float colors in [0,1]
                    c = np.clip(col_raw.astype(np.float32), 0.0, 1.0)
                    if c.shape[1] == 3:
                        c = np.hstack([c, np.ones((c.shape[0],1), np.float32)])
                    col = (c * 255.0 + 0.5).astype(np.uint8)

            all_pos.append(pos)
            if uv is not None:   all_uv.append(uv)
            if col is not None:  all_col.append(col)
            all_idx.append(idx)
            v_offset += pos.shape[0]

    if not all_pos:
        raise RuntimeError("No meshes/primitives found in GLB/GLTF source.")

    pos = np.vstack(all_pos).astype(np.float32)
    idx = np.concatenate(all_idx).astype(np.uint32)

    # If any primitive had UV or COLOR, concatenate; else set to None
    uv = np.vstack(all_uv).astype(np.float32) if all_uv else None
    col = np.vstack(all_col).astype(np.uint8) if all_col else None

    return pos, uv, col, idx


def load_glb_mesh_primitives(path):
    """Load GLB/GLTF and return per-primitive arrays (positions, uv, colors, indices)."""
    gltf, buffers = _load_gltf_with_buffers(path)
    to_np = _make_accessor_reader(gltf, buffers)

    meshes = []
    for mesh_idx, mesh in enumerate(gltf.meshes or []):
        for prim_idx, prim in enumerate(mesh.primitives or []):
            pos = to_np(prim.attributes.POSITION)
            if pos is None:
                continue
            pos = pos.astype(np.float32)

            tri_idx = _triangulate_indices(to_np(prim.indices), prim.mode, len(pos))
            if tri_idx.size == 0:
                continue

            uv = None
            if hasattr(prim.attributes, 'TEXCOORD_0') and prim.attributes.TEXCOORD_0 is not None:
                uv = to_np(prim.attributes.TEXCOORD_0)
                if uv is not None:
                    uv = uv.astype(np.float32)

            col = None
            if hasattr(prim.attributes, 'COLOR_0') and prim.attributes.COLOR_0 is not None:
                col_raw = to_np(prim.attributes.COLOR_0)
                if col_raw.dtype == np.uint8:
                    if col_raw.shape[1] == 3:
                        col = np.hstack([col_raw, np.full((col_raw.shape[0],1), 255, dtype=np.uint8)])
                    else:
                        col = col_raw.astype(np.uint8)
                else:
                    c = np.clip(col_raw.astype(np.float32), 0.0, 1.0)
                    if c.shape[1] == 3:
                        c = np.hstack([c, np.ones((c.shape[0],1), np.float32)])
                    col = (c * 255.0 + 0.5).astype(np.uint8)

            meshes.append({
                "name": mesh.name or f"mesh{mesh_idx}",
                "primitive": prim_idx,
                "positions": pos,
                "indices": tri_idx.astype(np.uint32),
                "uv": uv,
                "colors": col,
            })

    return meshes


def summarize_tile_sizes(sizes, heading="Tile size summary"):
    if not sizes:
        print(f"{heading}: no tiles generated")
        return

    arr = np.asarray(sizes, dtype=np.float64)
    total_mb = arr.sum() / (1024.0 * 1024.0)
    min_kb = arr.min() / 1024.0
    max_kb = arr.max() / 1024.0
    mean_kb = arr.mean() / 1024.0
    median_kb = np.median(arr) / 1024.0

    print(f"{heading}: {len(arr)} tiles, total {total_mb:.2f} MB")
    print(f"  Size (KB) -> min {min_kb:.1f} | median {median_kb:.1f} | mean {mean_kb:.1f} | max {max_kb:.1f}")

    percentiles = [50, 75, 90, 95, 99]
    pct_values = np.percentile(arr, percentiles)
    pct_str = ", ".join(f"p{p}: {v/1024.0:.1f} KB" for p, v in zip(percentiles, pct_values))
    print(f"  Percentiles -> {pct_str}")

    bins_kb = [0, 25, 50, 75, 100, 150, 200, 300, 500, np.inf]
    labels = ["<25", "25-50", "50-75", "75-100", "100-150", "150-200", "200-300", "300-500", ">500"]
    hist, _ = np.histogram(arr / 1024.0, bins=bins_kb)
    print("  Histogram (KB):")
    for label, count in zip(labels, hist):
        print(f"    {label}: {int(count)}")


def summarize_tile_sizes_by_depth(depth_sizes, heading="Per-depth tile stats", small_threshold_kb=25.0):
    if not depth_sizes:
        return

    buckets = {}
    for z, size in depth_sizes:
        buckets.setdefault(z, []).append(size)

    print(heading + ":")
    for z in sorted(buckets.keys()):
        arr = np.asarray(buckets[z], dtype=np.float64)
        count = len(arr)
        mean_kb = arr.mean() / 1024.0
        min_kb = arr.min() / 1024.0
        max_kb = arr.max() / 1024.0
        small_frac = (arr / 1024.0 < small_threshold_kb).sum() / count * 100.0
        print(
            f"  z={z}: count {count:5d}, mean {mean_kb:6.1f} KB, min {min_kb:5.1f} KB, max {max_kb:6.1f} KB, "
            f"small (<{small_threshold_kb:.0f} KB): {small_frac:5.1f}%"
        )


def tri_areas(verts, idx):
    """Calculate area of each triangle"""
    a = verts[idx[0::3]]
    b = verts[idx[1::3]]
    c = verts[idx[2::3]]
    return 0.5 * np.linalg.norm(np.cross(b-a, c-a), axis=1)

def o3d_from_trimesh(m):
    """Convert trimesh to Open3D mesh"""
    g = o3d.geometry.TriangleMesh()
    g.vertices = o3d.utility.Vector3dVector(np.asarray(m.vertices))
    g.triangles = o3d.utility.Vector3iVector(np.asarray(m.faces))
    g.compute_vertex_normals()
    return g

def transfer_attrs_nn(orig_pos, uv, col, simp_pos):
    """Transfer attributes using interpolation for colors, nearest-neighbor for UV"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(orig_pos.astype(np.float64))
    kdt = o3d.geometry.KDTreeFlann(pcd)

    simp_uv = np.zeros((simp_pos.shape[0],2), np.float32) if uv is not None else None
    simp_col = np.zeros((simp_pos.shape[0],4), np.uint8) if col is not None else None

    for i, p in enumerate(simp_pos):
        if simp_uv is not None:
            # Use nearest neighbor for UV (discrete mapping)
            _, idx, _ = kdt.search_knn_vector_3d(p.astype(np.float64), 1)
            j = idx[0]
            simp_uv[i] = uv[j]

        if simp_col is not None:
            # Use k-nearest neighbors with distance-weighted interpolation for colors
            k = min(4, len(orig_pos))  # Use up to 4 neighbors
            _, idx, dists = kdt.search_knn_vector_3d(p.astype(np.float64), k)

            # Convert distances to weights (closer = higher weight)
            weights = 1.0 / (np.array(dists) + 1e-8)  # Add small epsilon to avoid division by zero
            weights = weights / weights.sum()  # Normalize

            # Weighted average of colors
            if col.dtype==np.uint8 and col.shape[1]==4:
                # For RGBA colors, interpolate each channel
                interp_color = np.zeros(4, dtype=np.float32)
                for w, j in zip(weights, idx):
                    interp_color += w * col[j].astype(np.float32)
                simp_col[i] = np.clip(interp_color, 0, 255).astype(np.uint8)
            else:
                # For RGB colors
                interp_color = np.zeros(3, dtype=np.float32)
                for w, j in zip(weights, idx):
                    interp_color += w * np.clip(col[j], 0, 1)
                rgb = (interp_color * 255).astype(np.uint8)
                simp_col[i] = np.array([*rgb, 255], np.uint8)

    return simp_uv, simp_col

def estimate_bytes(nv, nf, has_uv=True, has_col=True):
    """Estimate GLB file size in bytes"""
    # POSITION(float32x3)=12B, UV(float32x2)=8B, COLOR_0(u8x4 normalized)=4B, INDEX(uint32x3)=12B
    per_v = 12 + (8 if has_uv else 0) + (4 if has_col else 0)
    per_f = 12
    return nv*per_v + nf*per_f + 1024  # small header fudge

def decimate_to_target(m: tm.Trimesh, target_bytes, min_ratio=0.02, min_tris=32, max_iter=6):
    """Iteratively decimate mesh until it meets the target byte budget."""
    src_uv = getattr(m.visual, 'uv', None)
    src_col = getattr(m.visual, 'vertex_colors', None)

    current = m.copy()
    for iteration in range(max_iter):
        cur_size = estimate_bytes(len(current.vertices), len(current.faces),
                                  src_uv is not None, src_col is not None)
        if cur_size <= target_bytes * (1 + SIZE_TOLERANCE):
            break
        if len(current.faces) <= min_tris:
            break

        ratio = max(min_ratio, target_bytes / cur_size)
        target_tris = max(min_tris, int(len(current.faces) * ratio))

        g = o3d_from_trimesh(current)
        g_s = g.simplify_quadric_decimation(target_number_of_triangles=target_tris)

        simp_pos = np.asarray(g_s.vertices, dtype=np.float32)
        simp_faces = np.asarray(g_s.triangles, dtype=np.uint32)

        if simp_faces.shape[0] == 0 or simp_pos.shape[0] == 0:
            break

        simp_uv, simp_col = transfer_attrs_nn(
            np.asarray(current.vertices),
            src_uv,
            src_col,
            simp_pos
        )

        next_mesh = tm.Trimesh(vertices=simp_pos, faces=simp_faces, process=False)
        if simp_uv is not None:
            next_mesh.visual.uv = simp_uv
        if simp_col is not None:
            next_mesh.visual.vertex_colors = simp_col

        # If simplification produced no change, stop.
        if len(next_mesh.faces) >= len(current.faces):
            current = next_mesh
            break

        current = next_mesh

    return current

def write_glb_from_trimesh(m: tm.Trimesh, meta: dict, out_path: str):
    """Write trimesh to GLB file with metadata.
    Now chooses 16-bit indices when possible for broader compatibility (and smaller files).
    """
    if m is None or len(m.faces)==0:
        return

    positions = np.asarray(m.vertices, dtype=np.float32)
    indices32 = np.asarray(m.faces, dtype=np.uint32).reshape(-1)

    # Ensure UV and color arrays match vertex count
    vertex_count = positions.shape[0]

    uv = None
    uv_attr = getattr(m.visual, 'uv', None)
    if uv_attr is not None:
        uv_raw = np.asarray(uv_attr, dtype=np.float32)
        if uv_raw.shape[0] == vertex_count:
            uv = uv_raw
        else:
            print(f"Warning: UV count {uv_raw.shape[0]} != vertex count {vertex_count}, skipping UVs")

    colors = None
    color_attr = getattr(m.visual, 'vertex_colors', None)
    if color_attr is not None:
        colors_raw = np.asarray(color_attr, dtype=np.uint8)
        if colors_raw.shape[0] == vertex_count:
            colors = colors_raw
        else:
            print(f"Warning: Color count {colors_raw.shape[0]} != vertex count {vertex_count}, skipping colors")

    use_u16 = positions.shape[0] <= 65535 and (indices32.max(initial=0) <= 65535)
    indices  = indices32.astype(np.uint16 if use_u16 else np.uint32, copy=False)

    gltf = GLTF2(asset=Asset(version="2.0"))
    chunks, offsets = [], {}

    def push(name, data_bytes):
        # Ensure 4-byte alignment for glTF compliance
        current_size = sum(len(c) for c in chunks)
        padding = (4 - (current_size % 4)) % 4
        if padding > 0:
            chunks.append(b'\x00' * padding)
        offsets[name] = sum(len(c) for c in chunks)
        chunks.append(data_bytes)

    # Buffer data
    push("POSITION", positions.tobytes())
    if uv is not None:
        push("TEXCOORD_0", uv.astype(np.float32).tobytes())
    if colors is not None:
        push("COLOR_0", colors.astype(np.uint8).tobytes())
    push("INDICES", indices.tobytes())

    # Create buffer and bufferViews
    total_len = sum(len(c) for c in chunks)
    gltf.buffers = [Buffer(byteLength=total_len)]

    views = {}
    def mkview(name, length, target=None):
        views[name] = len(gltf.bufferViews)
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offsets[name], byteLength=length, target=target))

    mkview("POSITION", positions.nbytes, 34962)   # ARRAY_BUFFER
    if uv is not None: mkview("TEXCOORD_0", uv.astype(np.float32).nbytes, 34962)
    if colors is not None: mkview("COLOR_0", colors.astype(np.uint8).nbytes, 34962)
    mkview("INDICES", indices.nbytes, 34963)     # ELEMENT_ARRAY_BUFFER

    # Accessors
    gltf.accessors = []

    def mkacc(name, count, acc_type, comp_type, view_name, minmax=None, normalized=False):
        gltf.accessors.append(Accessor(
            bufferView=views[view_name],
            componentType=comp_type,
            count=count,
            type=acc_type,
            normalized=normalized,
            min=list(map(float, minmax[0])) if minmax is not None else None,
            max=list(map(float, minmax[1])) if minmax is not None else None
        ))
        return len(gltf.accessors)-1

    # All vertex attributes must have the same count (number of vertices)
    vertex_count = positions.shape[0]

    a_pos = mkacc("POSITION", vertex_count, "VEC3", 5126, "POSITION", (positions.min(axis=0), positions.max(axis=0)))
    a_uv  = mkacc("TEXCOORD_0", uv.shape[0], "VEC2", 5126, "TEXCOORD_0") if uv is not None else None
    a_col = mkacc("COLOR_0", colors.shape[0], "VEC4", 5121, "COLOR_0", normalized=True) if colors is not None else None  # 5121 = UNSIGNED_BYTE, normalized
    a_idx = mkacc("INDICES", indices.shape[0], "SCALAR", 5123 if use_u16 else 5125, "INDICES")           # 5123=U16, 5125=U32

    # Primitive & mesh
    attrs = Attributes(POSITION=a_pos)
    if a_uv is not None:  attrs.TEXCOORD_0 = a_uv
    if a_col is not None: attrs.COLOR_0    = a_col

    prim = Primitive(attributes=attrs, indices=a_idx, mode=4)
    mesh = Mesh(primitives=[prim], extras=meta)

    gltf.meshes = [mesh]
    gltf.nodes = [Node(mesh=0)]
    gltf.scenes = [Scene(nodes=[0])]
    gltf.scene = 0

    # Assemble BIN
    glb_bin = bytearray(total_len)
    cursor = 0
    for c in chunks:
        glb_bin[cursor:cursor+len(c)] = c
        cursor += len(c)

    gltf.binary_blob = lambda: bytes(glb_bin)  # provide the blob to pygltflib
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(out_path)


# ============================================================================
# UV-Quadtree Mode
# ============================================================================

def uv_centroids(uv, idx):
    """Calculate UV centroids of triangles"""
    ua = uv[idx[0::3]]
    ub = uv[idx[1::3]]
    uc = uv[idx[2::3]]
    return (ua+ub+uc)/3.0

def uv_tile_bounds(z,x,y):
    """Get UV bounds for a tile"""
    s = 1.0/(1<<z)
    u0 = x*s
    v0 = y*s
    return u0, v0, u0+s, v0+s

def uv_in_tile(centroids, b):
    """Check which triangles are in a UV tile (with overlap)"""
    u0, v0, u1, v1 = b
    return (centroids[:,0]>=u0-UV_EPS) & (centroids[:,0]<=u1+UV_EPS) & \
           (centroids[:,1]>=v0-UV_EPS) & (centroids[:,1]<=v1+UV_EPS)

def subset_trimesh(pos, uv, col, idx, tri_mask):
    """Create subset mesh from triangle mask"""
    tri_idx = idx.reshape(-1,3)[tri_mask]
    if tri_idx.size==0:
        return None

    uniq, remap = np.unique(tri_idx.flatten(), return_inverse=True)
    new_pos = pos[uniq]
    new_uv = uv[uniq] if uv is not None else None
    new_col = col[uniq] if col is not None else None
    new_idx = remap.reshape(-1,3)

    m = tm.Trimesh(vertices=new_pos, faces=new_idx, process=False)
    if new_uv is not None:
        m.visual.uv = new_uv
    if new_col is not None:
        # Handle both normalized (0-1) and uint8 (0-255) color formats
        if new_col.dtype == np.float32:
            # Colors are normalized 0-1, convert to 0-255
            rgba = (np.clip(new_col,0,1)*255).astype(np.uint8)
        else:
            # Colors are already 0-255 uint8
            rgba = new_col.astype(np.uint8)

        if rgba.shape[1]==3:
            rgba = np.hstack([rgba, np.full((rgba.shape[0],1),255,np.uint8)])
        m.visual.vertex_colors = rgba

    return m

def build_uv_quadtree(src_glb, out_dir, extract="default", time_index=0,
                      max_depth=MAX_DEPTH, target_bytes=TARGET_TILE_BYTES,
                      split_meshes=False, preserve_borders=False,
                      snap_radius=None, snap_ratio=None):
    """Build UV-based quadtree tiles"""
    mesh_entries = []
    if split_meshes:
        mesh_entries = load_glb_mesh_primitives(src_glb)
        if not mesh_entries:
            raise RuntimeError("No meshes/primitives found in GLB.")
        for entry in mesh_entries:
            if entry["uv"] is None:
                raise ValueError("split_meshes requires TEXCOORD_0 on every primitive")
    else:
        pos, uv, col, idx = load_glb_arrays(src_glb)
        if uv is None:
            raise ValueError("UV mode requires TEXCOORD_0")
        mesh_entries = [{
            "name": os.path.splitext(os.path.basename(src_glb))[0],
            "primitive": 0,
            "positions": pos,
            "indices": idx,
            "uv": uv,
            "colors": col,
        }]

    mesh_entries = [entry for entry in mesh_entries if entry["indices"].size > 0]

    total_triangles = 0
    total_area = 0.0
    min_area = math.inf
    for entry in mesh_entries:
        triA = tri_areas(entry["positions"], entry["indices"])
        total_triangles += len(triA)
        total_area += float(triA.sum())
        if len(triA) > 0:
            min_area = min(min_area, float(triA.min()))

    if total_triangles == 0:
        raise RuntimeError("No triangles found in GLB.")

    avg_area = total_area / total_triangles
    if min_area is math.inf:
        min_area = 0.0

    manifest = {
        "extract": extract,
        "time": time_index,
        "source": os.path.basename(src_glb),
        "tilingSpace": "uv",
        "grid": "quadtree",
        "maxDepth": max_depth,
        "targetTileBytes": target_bytes,
        "global": {
            "triCount": int(total_triangles),
            "avgTriArea": float(avg_area),
            "minTriArea": float(min_area)
        },
        "charts": len(mesh_entries),
        "tiles": []
    }

    all_tiles = []
    depth_size_samples = []

    for mesh_idx, entry in enumerate(mesh_entries):
        pos = entry["positions"]
        uv = entry["uv"]
        col = entry["colors"]
        idx = entry["indices"]

        triA = tri_areas(pos, idx)
        triC = uv_centroids(uv, idx)

        if triA.size == 0:
            continue

        mesh_tiles = {}

        for z in range(max_depth, -1, -1):
            for x in range(1<<z):
                for y in range(1<<z):
                    b = uv_tile_bounds(z,x,y)
                    mask = uv_in_tile(triC, b)
                    m = subset_trimesh(pos, uv, col, idx, mask)

                    if m is None:
                        continue

                    orig_border_pts = None
                    snap_tol = None
                    if preserve_borders:
                        border_edges = compute_border_edges(m)
                        if border_edges:
                            unique_idx = np.unique(np.asarray(border_edges).flatten())
                            orig_border_pts = np.asarray(m.vertices)[unique_idx]
                            diag = float(np.linalg.norm(m.bounds[1] - m.bounds[0]))
                            if snap_radius is not None:
                                snap_tol = snap_radius
                            elif snap_ratio is not None and diag > 0:
                                snap_tol = snap_ratio * diag
                            elif snap_ratio is not None:
                                snap_tol = snap_ratio

                    m = decimate_to_target(m, target_bytes)

                    if preserve_borders and orig_border_pts is not None and snap_tol is not None:
                        snap_decimated_border(m, orig_border_pts, snap_radius=snap_tol)

                    aabb_min = m.bounds[0].tolist()
                    aabb_max = m.bounds[1].tolist()
                    tid = f"{mesh_idx}/{z}/{x}/{y}"

                    kids = [f"{mesh_idx}/{z+1}/{2*x+dx}/{2*y+dy}" for dx in (0,1) for dy in (0,1)] if z<max_depth else []

                    approx = estimate_bytes(len(m.vertices), len(m.faces),
                                           getattr(m.visual, 'uv', None) is not None,
                                           getattr(m.visual, 'vertex_colors', None) is not None)

                    meta = {
                        "tileId": tid,
                        "mesh": mesh_idx,
                        "meshName": entry["name"],
                        "z": z, "x": x, "y": y,
                        "parent": f"{mesh_idx}/{z-1}/{x>>1}/{y>>1}" if z>0 else None,
                        "children": kids,
                        "aabbWorld": [aabb_min, aabb_max],
                        "aabbUV": [[b[0],b[1]], [b[2],b[3]]],
                        "triCount": int(len(m.faces)),
                        "avgTriArea": float(triA.mean()) if len(triA) else 0.0,
                        "minTriArea": float(triA.min()) if len(triA) else 0.0,
                        "geometricError": float(np.mean(m.edges_unique_length)) if m.edges_unique_length.size>0 else 0.0,
                        "approxBytes": int(approx),
                        "time": time_index
                    }

                    out_path = os.path.join(out_dir, extract, str(time_index),
                                           f"mesh_{mesh_idx}", str(z), str(x), f"{y}.glb")
                    if not preserve_borders:
                        add_skirts(m, skirt_h_ratio=0.10)
                    write_glb_from_trimesh(m, meta, out_path)
                    actual_bytes = os.path.getsize(out_path)
                    depth_size_samples.append((z, actual_bytes))
                    mesh_tiles[tid] = {**meta, "actualBytes": int(actual_bytes),
                                       "url": f"/tiles/{extract}/{time_index}/mesh_{mesh_idx}/{z}/{x}/{y}.glb"}

        all_tiles.extend(mesh_tiles.values())

    manifest["tiles"] = sorted(all_tiles, key=lambda t: (t.get("mesh", 0), t["z"], t["x"], t["y"]))

    man_path = os.path.join(out_dir, extract, f"manifest_{time_index}.json")
    pathlib.Path(os.path.dirname(man_path)).mkdir(parents=True, exist_ok=True)

    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(all_tiles)} UV-quadtree tiles across {len(mesh_entries)} mesh charts")
    size_only = [size for (_, size) in depth_size_samples]
    summarize_tile_sizes(size_only, heading=f"Tile sizes for '{extract}' (uv)")
    summarize_tile_sizes_by_depth(depth_size_samples, heading="  Depth breakdown")

# ============================================================================
# World-Space Octree Mode
# ============================================================================

def world_aabb(pos):
    """Get world-space axis-aligned bounding box"""
    mn = pos.min(axis=0)
    mx = pos.max(axis=0)
    return mn, mx

def expand_aabb(aabb, ratio):
    """Expand AABB by ratio"""
    mn, mx = aabb
    size = mx - mn
    pad = size * ratio
    return (mn-pad, mx+pad)

def tri_centroids_world(pos, idx):
    """Calculate world-space centroids of triangles"""
    a = pos[idx[0::3]]
    b = pos[idx[1::3]]
    c = pos[idx[2::3]]
    return (a+b+c)/3.0

def child_bounds(aabb, z, i, j, k):
    """Get bounds for octree child node"""
    (mn, mx) = aabb
    size = (mx - mn) / (1<<z)
    base = mn + np.array([i,j,k]) * size
    return (base, base + size)

def in_aabb(points, aabb):
    """Check which points are inside AABB"""
    mn, mx = aabb
    return np.all((points>=mn-1e-9) & (points<=mx+1e-9), axis=1)

def subset_trimesh_by_mask(pos, idx, mask, uv=None, col=None):
    """Create subset mesh from triangle mask (world-space version)"""
    tri_idx = idx.reshape(-1,3)[mask]
    if tri_idx.size==0:
        return None

    uniq, remap = np.unique(tri_idx.flatten(), return_inverse=True)
    new_pos = pos[uniq]
    new_idx = remap.reshape(-1,3)

    m = tm.Trimesh(vertices=new_pos, faces=new_idx, process=False)

    if uv is not None:
        m.visual.uv = uv[uniq]
    if col is not None:
        c = col[uniq]
        if c.dtype != np.uint8:
            c = (np.clip(c,0,1)*255).astype(np.uint8)
        if c.shape[1]==3:
            c = np.hstack([c, np.full((c.shape[0],1),255,np.uint8)])
        m.visual.vertex_colors = c

    return m

def build_world_octree(src_glb, out_dir, extract="default", time_index=0,
                       max_depth=MAX_DEPTH, target_bytes=TARGET_TILE_BYTES,
                       preserve_borders=False, snap_radius=None, snap_ratio=None):
    """Build world-space octree tiles"""
    pos, uv, col, idx = load_glb_arrays(src_glb)

    triA = tri_areas(pos, idx)
    cent = tri_centroids_world(pos, idx)
    scene_aabb = world_aabb(pos)

    manifest = {
        "extract": extract,
        "time": time_index,
        "source": os.path.basename(src_glb),
        "tilingSpace": "world",
        "grid": "octree",
        "maxDepth": max_depth,
        "targetTileBytes": target_bytes,
        "global": {
            "triCount": int(len(triA)),
            "avgTriArea": float(triA.mean()),
            "minTriArea": float(triA.min())
        },
        "charts": 0,
        "tiles": []
    }

    tiles_meta = {}
    depth_size_samples = []

    for z in range(0, max_depth+1):
        div = 1<<z
        for i in range(div):
            for j in range(div):
                for k in range(div):
                    aabb = child_bounds(scene_aabb, z, i, j, k)
                    aabb_loose = expand_aabb(aabb, WORLD_EPS_RATIO)
                    mask = in_aabb(cent, aabb_loose)

                    m = subset_trimesh_by_mask(pos, idx, mask, uv=None, col=col)
                    if m is None:
                        continue

                    orig_border_pts = None
                    snap_tol = None
                    if preserve_borders:
                        border_edges = compute_border_edges(m)
                        if border_edges:
                            unique_idx = np.unique(np.asarray(border_edges).flatten())
                            orig_border_pts = np.asarray(m.vertices)[unique_idx]
                            diag = float(np.linalg.norm(m.bounds[1] - m.bounds[0]))
                            if snap_radius is not None:
                                snap_tol = snap_radius
                            elif snap_ratio is not None and diag > 0:
                                snap_tol = snap_ratio * diag
                            elif snap_ratio is not None:
                                snap_tol = snap_ratio

                    m = decimate_to_target(m, target_bytes)

                    if preserve_borders and orig_border_pts is not None and snap_tol is not None:
                        snap_decimated_border(m, orig_border_pts, snap_radius=snap_tol)

                    aabb_min = m.bounds[0].tolist()
                    aabb_max = m.bounds[1].tolist()

                    approx = estimate_bytes(len(m.vertices), len(m.faces),
                                           False, getattr(m.visual, 'vertex_colors', None) is not None)

                    tid = f"{z}/{i}/{j}/{k}"
                    kids = [f"{z+1}/{2*i+di}/{2*j+dj}/{2*k+dk}"
                           for di in (0,1) for dj in (0,1) for dk in (0,1)] if z<max_depth else []

                    meta = {
                        "tileId": tid,
                        "z": z, "x": i, "y": j, "k": k,
                        "parent": f"{z-1}/{i>>1}/{j>>1}/{k>>1}" if z>0 else None,
                        "children": kids,
                        "aabbWorld": [aabb_min, aabb_max],
                        "triCount": int(len(m.faces)),
                        "avgTriArea": float(triA.mean()),
                        "minTriArea": float(triA.min()),
                        "geometricError": float(np.mean(m.edges_unique_length)) if m.edges_unique_length.size>0 else 0.0,
                        "approxBytes": int(approx),
                        "time": time_index
                    }

                    out_path = os.path.join(out_dir, extract, str(time_index),
                                           str(z), str(i), str(j), f"{k}.glb")
                    if not preserve_borders:
                        add_skirts(m, skirt_h_ratio=0.10)
                    write_glb_from_trimesh(m, meta, out_path)
                    actual_bytes = os.path.getsize(out_path)
                    depth_size_samples.append((z, actual_bytes))
                    tiles_meta[tid] = {**meta, "actualBytes": int(actual_bytes),
                                       "url": f"/tiles/{extract}/{time_index}/{z}/{i}/{j}/{k}.glb"}

    def keyz(tid):
        return tuple(map(int, tid.split('/')))

    manifest["tiles"] = [tiles_meta[k] for k in sorted(tiles_meta.keys(), key=keyz)]

    man_path = os.path.join(out_dir, extract, f"manifest_{time_index}.json")
    pathlib.Path(os.path.dirname(man_path)).mkdir(parents=True, exist_ok=True)

    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(tiles_meta)} world-octree tiles")
    size_only = [size for (_, size) in depth_size_samples]
    summarize_tile_sizes(size_only, heading=f"Tile sizes for '{extract}' (world)")
    summarize_tile_sizes_by_depth(depth_size_samples, heading="  Depth breakdown")

# ============================================================================
# Crack Prevention: Skirts & Border Snapping
# ============================================================================

def compute_border_edges(tri_mesh: tm.Trimesh):
    """Returns list of (v0, v1) edges that belong to exactly one face"""
    edges = tri_mesh.edges_sorted
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    border = unique[counts==1]
    return [tuple(e) for e in border]

def add_skirts(tri_mesh: tm.Trimesh, skirt_h_ratio=0.1):
    """Offset duplicated boundary vertices along vertex normals by skirt_h
    Also properly extends UV and color arrays to match new vertices"""
    if tri_mesh.vertices.shape[0]==0 or tri_mesh.faces.shape[0]==0:
        return

    if tri_mesh.vertex_normals is None or len(tri_mesh.vertex_normals)==0:
        _ = tri_mesh.vertex_normals  # force compute in trimesh

    border_edges = compute_border_edges(tri_mesh)
    if len(border_edges)==0:
        return

    V = np.asarray(tri_mesh.vertices)
    F = np.asarray(tri_mesh.faces)
    N = np.asarray(tri_mesh.vertex_normals)

    # Get original UV and color data
    orig_uv = None
    orig_colors = None
    if tri_mesh.visual.uv is not None:
        orig_uv = np.asarray(tri_mesh.visual.uv)
    if tri_mesh.visual.vertex_colors is not None:
        orig_colors = np.asarray(tri_mesh.visual.vertex_colors)

    mean_edge = float(np.mean(tri_mesh.edges_unique_length)) if tri_mesh.edges_unique_length.size else 0.0
    skirt_h = max(1e-6, skirt_h_ratio * mean_edge)

    newV = V.tolist()
    newF = F.tolist()
    newUV = orig_uv.tolist() if orig_uv is not None else None
    newColors = orig_colors.tolist() if orig_colors is not None else None
    dup = {}

    for (a,b) in border_edges:
        for v in (a,b):
            if v not in dup:
                dup[v] = len(newV)
                # Add new vertex position
                newV.append(V[v] - N[v] * skirt_h)
                # Add corresponding UV and color data
                if newUV is not None:
                    newUV.append(orig_uv[v].tolist())
                if newColors is not None:
                    newColors.append(orig_colors[v].tolist())
        a2, b2 = dup[a], dup[b]
        newF.append([a, b, b2])
        newF.append([a, b2, a2])

    tri_mesh.vertices = np.asarray(newV)
    tri_mesh.faces = np.asarray(newF)

    # Update UV and color arrays if they exist
    if newUV is not None:
        tri_mesh.visual.uv = np.asarray(newUV)
    if newColors is not None:
        tri_mesh.visual.vertex_colors = np.asarray(newColors)

def snap_decimated_border(decimated: tm.Trimesh, original_border_pts: np.ndarray, snap_radius: float=1e-3):
    """Snap decimated border vertices back to original border polyline"""
    # Sample the original border polyline and build KDTree
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(original_border_pts.astype(np.float64))
    kdt = o3d.geometry.KDTreeFlann(pcd)

    V = np.asarray(decimated.vertices)
    for i in range(V.shape[0]):
        # Heuristic: only snap vertices close to the border plane/box
        _, idx, _ = kdt.search_knn_vector_3d(V[i].astype(np.float64), 1)
        nearest = np.asarray(pcd.points)[idx[0]]
        if np.linalg.norm(nearest - V[i]) <= snap_radius:
            V[i] = nearest

    decimated.vertices = V
    return decimated

# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Build multi-resolution tiles from GLB')
    parser.add_argument('--in_glb', help='Input GLB file path')
    parser.add_argument('--input_dir', help='Directory containing GLB snapshots (time steps)')
    parser.add_argument('--out_dir', default='tiles_out', help='Output directory')
    parser.add_argument('--extract', default='default', help='Extract name')
    parser.add_argument('--time', type=int, default=0, help='Time index')
    parser.add_argument('--tiling_space', choices=['uv', 'world'], default='uv',
                       help='Tiling space: uv (requires TEXCOORD_0) or world')
    parser.add_argument('--max_depth', type=int, default=5, help='Maximum tile depth')
    parser.add_argument('--target_kb', type=float, default=200,
                       help='Target tile size in KB')
    parser.add_argument('--split_meshes', action='store_true',
                       help='UV mode only: build a separate quadtree for each mesh primitive')
    parser.add_argument('--preserve_borders', action='store_true',
                       help='Snap decimated tile borders back to original positions and disable skirts')
    parser.add_argument('--snap_radius', type=float, default=None,
                       help='Absolute snap radius (world units) when --preserve_borders is enabled')
    parser.add_argument('--snap_ratio', type=float, default=1e-3,
                       help='Relative snap tolerance as a fraction of tile diagonal (set 0 to disable)')

    parser.add_argument('--snapshots', action='store_true',
                        help='Process all GLB files in --input_dir as sequential time steps')

    args = parser.parse_args()

    if args.snapshots:
        if args.input_dir is None:
            raise ValueError('--snapshots requires --input_dir')
        if args.in_glb:
            raise ValueError('Specify either --in_glb or --snapshots/--input_dir, not both.')
    else:
        if not args.in_glb:
            raise ValueError('Must provide --in_glb unless using --snapshots with --input_dir')

    target_bytes = int(args.target_kb * 1024)

    if args.snap_radius is not None and args.snap_radius <= 0:
        raise ValueError('snap_radius must be positive if specified')

    snap_ratio = args.snap_ratio if args.snap_ratio is not None and args.snap_ratio > 0 else None
    snap_radius = args.snap_radius

    if args.preserve_borders and snap_radius is None and snap_ratio is None:
        snap_ratio = 1e-3  # default relative tolerance

    def process_single(glb_path: str, time_idx: int):
        if args.tiling_space == 'uv':
            build_uv_quadtree(
                glb_path, args.out_dir, args.extract, time_idx,
                args.max_depth, target_bytes,
                split_meshes=args.split_meshes,
                preserve_borders=args.preserve_borders,
                snap_radius=snap_radius,
                snap_ratio=snap_ratio
            )
        else:
            if args.split_meshes:
                raise ValueError('--split_meshes is only supported for uv tiling space')
            build_world_octree(
                glb_path, args.out_dir, args.extract, time_idx,
                args.max_depth, target_bytes,
                preserve_borders=args.preserve_borders,
                snap_radius=snap_radius,
                snap_ratio=snap_ratio
            )

    if args.snapshots:
        input_dir = Path(args.input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"Input directory '{input_dir}' does not exist or is not a directory")
        glb_files = sorted(p for p in input_dir.glob('*.glb'))
        if not glb_files:
            raise ValueError(f"No .glb files found in '{input_dir}'")
        for offset, glb_path in enumerate(glb_files):
            time_idx = args.time + offset
            print(f"Processing snapshot {glb_path.name} (time={time_idx})")
            process_single(str(glb_path), time_idx)
    else:
        process_single(args.in_glb, args.time)

if __name__ == '__main__':
    main()
