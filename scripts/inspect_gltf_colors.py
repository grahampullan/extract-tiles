#!/usr/bin/env python3
"""Inspect where colour information lives inside a glTF/GLB file."""

import argparse
import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image
from pygltflib import GLTF2


def load_buffers(gltf: GLTF2, source: Path):
    buffers = []
    for buf in gltf.buffers or []:
        uri = buf.uri
        if uri is None:
            buffers.append(bytes())
        elif uri.startswith('data:'):
            _, data = uri.split(',', 1)
            buffers.append(base64.b64decode(data))
        else:
            buffers.append((source.parent / uri).read_bytes())
    if getattr(gltf, 'is_glb', False):
        buffers = [gltf.binary_blob()]
    return buffers


def get_image_bytes(gltf: GLTF2, buffers, image_index: int, source: Path):
    images = gltf.images or []
    if image_index is None or image_index < 0 or image_index >= len(images):
        return None
    image = images[image_index]
    if image.uri:
        if image.uri.startswith('data:'):
            _, data = image.uri.split(',', 1)
            return base64.b64decode(data)
        return (source.parent / image.uri).read_bytes()
    if image.bufferView is not None:
        view = gltf.bufferViews[image.bufferView]
        buf = buffers[view.buffer]
        start = view.byteOffset or 0
        end = start + (view.byteLength or 0)
        return bytes(memoryview(buf)[start:end])
    return None


def read_accessor(gltf: GLTF2, buffers, accessor_index):
    if accessor_index is None:
        return None
    accessor = gltf.accessors[accessor_index]
    if accessor.bufferView is None:
        return None
    view = gltf.bufferViews[accessor.bufferView]
    buf = buffers[view.buffer]
    offset = (view.byteOffset or 0) + (accessor.byteOffset or 0)
    ncomp = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}.get(accessor.type, 1)
    ct2dt = {5126:np.float32,5125:np.uint32,5123:np.uint16,5121:np.uint8}
    dtype = ct2dt.get(accessor.componentType)
    if dtype is None:
        return None
    itemsize = np.dtype(dtype).itemsize
    length = accessor.count * ncomp * itemsize
    raw = memoryview(buf)[offset:offset+length]
    arr = np.frombuffer(raw, dtype=dtype, count=accessor.count * ncomp)
    return arr.reshape(accessor.count, ncomp)


def sample_texture_colors(gltf: GLTF2, buffers, tex_accessor_idx, tex_info, source: Path):
    if tex_accessor_idx is None:
        return None
    uv = read_accessor(gltf, buffers, tex_accessor_idx)
    if uv is None or uv.size == 0:
        return None
    textures = gltf.textures or []
    if tex_info.index is None or tex_info.index >= len(textures):
        return None
    image_bytes = get_image_bytes(gltf, buffers, textures[tex_info.index].source, source)
    if not image_bytes:
        return None
    img = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
    tex = np.array(img)
    if tex.size == 0:
        return None
    h, w, _ = tex.shape
    uv = np.clip(np.nan_to_num(uv[:, :2]), 0.0, 1.0)
    xs = np.clip(np.round(uv[:, 0] * (w - 1)).astype(int), 0, w - 1)
    ys = np.clip(np.round((1.0 - uv[:, 1]) * (h - 1)).astype(int), 0, h - 1)
    return tex[ys, xs]


def summarize_colors(path: Path):
    gltf = GLTF2().load_binary(str(path)) if path.suffix.lower() == '.glb' else GLTF2().load(str(path))
    gltf.is_glb = (path.suffix.lower() == '.glb')
    buffers = load_buffers(gltf, path)

    print(f"File: {path}")
    print(f"Meshes: {len(gltf.meshes or [])}, Materials: {len(gltf.materials or [])}, Textures: {len(gltf.textures or [])}")

    for mi, mesh in enumerate(gltf.meshes or []):
        for pi, prim in enumerate(mesh.primitives or []):
            attrs = getattr(prim, 'attributes', None)
            attr_dict = dict(getattr(attrs, '__dict__', {})) if attrs else {}
            mat_idx = prim.material
            print(f"  Mesh {mi} Prim {pi}: material={mat_idx}, attributes={list(attr_dict.keys())}")

            colour_data = None
            if attr_dict.get('COLOR_0') is not None:
                colour_data = read_accessor(gltf, buffers, attr_dict['COLOR_0'])
                if colour_data is not None:
                    print(f"    COLOR_0 count={len(colour_data)}, dtype={colour_data.dtype}")
                    print(f"    COLOR_0 sample={colour_data[:5]}")
            elif attr_dict.get('TEXCOORD_0') is not None:
                tex_info = None
                if mat_idx is not None and gltf.materials:
                    mat = gltf.materials[mat_idx]
                    pbr = getattr(mat, 'pbrMetallicRoughness', None)
                    tex_info = getattr(pbr, 'baseColorTexture', None) if pbr else None
                    print(f"    baseColorFactor={getattr(pbr, 'baseColorFactor', None)} texture={tex_info}")
                if tex_info and tex_info.index is not None:
                    colour_data = sample_texture_colors(gltf, buffers, attr_dict['TEXCOORD_0'], tex_info, path)
                    if colour_data is not None:
                        print(f"    Sampled texture colours: count={len(colour_data)}, sample={colour_data[:5]}")
            else:
                if mat_idx is not None and gltf.materials:
                    mat = gltf.materials[mat_idx]
                    pbr = getattr(mat, 'pbrMetallicRoughness', None) if mat else None
                    print(f"    baseColorFactor={getattr(pbr, 'baseColorFactor', None)} texture=None")
            print()


def main():
    parser = argparse.ArgumentParser(description="Report colour sources in a glTF/GLB.")
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    summarize_colors(args.path)


if __name__ == '__main__':
    main()
