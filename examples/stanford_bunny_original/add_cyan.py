#!/usr/bin/env python3
"""Add uniform bright cyan vertex colours to an existing GLB."""

import sys
import numpy as np
import trimesh

if len(sys.argv) < 3:
    print("Usage: add_cyan.py input.glb output.glb")
    sys.exit(1)

src, dst = sys.argv[1], sys.argv[2]
mesh = trimesh.load(src, force='mesh', skip_materials=True)
c = np.array([0, 255, 255, 255], dtype=np.uint8)
mesh.visual.vertex_colors = np.tile(c, (mesh.vertices.shape[0], 1))
mesh.export(dst)
print(f"Wrote {dst}")
