#!/usr/bin/env python3
"""Generate rotating cylinder spokes (unsteady dataset)."""

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

import sys
from pathlib import Path as _Path

_repo_root = _Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from examples.cylinders.cylinders import build_spoke_wheels, write_multi_mesh_glb


def resolve_output_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    repo_candidate = (_repo_root / path).resolve()
    if repo_candidate.parent.exists():
        return repo_candidate
    script_dir = _Path(__file__).resolve().parent
    script_candidate = (script_dir / path).resolve()
    if script_candidate.parent.exists():
        return script_candidate
    cwd_candidate = (Path.cwd() / path).resolve()
    return cwd_candidate


def rotate_points(points: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return (rot @ points.T).T


def make_rotated_meshes(meshes, angle: float):
    rotated = []
    for verts, indices, uvs, colors in meshes:
        verts_rot = rotate_points(verts, angle).astype(np.float32)
        rotated.append((
            verts_rot,
            indices.copy(),
            uvs.copy() if uvs is not None else None,
            colors.copy() if colors is not None else None,
        ))
    return rotated


def parse_args():
    parser = argparse.ArgumentParser(description="Generate rotating cylinder spokes GLBs for multiple time steps")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("examples/cylinders_unsteady/output"),
        help="Directory for output GLBs (relative paths resolved against repo root or script location)."
    )
    parser.add_argument("--snapshots", type=int, default=10, help="Number of time steps")
    parser.add_argument("--delta-degrees", type=float, default=10.0, help="Rotation per snapshot in degrees")
    parser.add_argument("--spokes", type=int, default=6, help="Cylinders per wheel")
    parser.add_argument("--wheel-count", type=int, default=1, help="Number of wheels")
    parser.add_argument("--wheel-spacing", type=float, default=4.0, help="Wheel spacing along Z")
    parser.add_argument("--r-hub", type=float, default=4.0, help="Inner radius")
    parser.add_argument("--r-tip", type=float, default=12.0, help="Outer radius")
    parser.add_argument("--tube-radius", type=float, default=0.4, help="Cylinder radius")
    parser.add_argument("--segments-axis", type=int, default=32, help="Segments along axis")
    parser.add_argument("--segments-circ", type=int, default=48, help="Segments around circumference")
    parser.add_argument("--wave-amplitude", type=float, default=0.0, help="Sinusoidal radius modulation amplitude")
    parser.add_argument("--wave-cycles-min", type=float, default=0.5, help="Minimum wave cycles")
    parser.add_argument("--wave-cycles-max", type=float, default=2.0, help="Maximum wave cycles")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = resolve_output_dir(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_meshes = build_spoke_wheels(
        spokes=args.spokes,
        wheels=args.wheel_count,
        wheel_spacing=args.wheel_spacing,
        r_hub=args.r_hub,
        r_tip=args.r_tip,
        tube_radius=args.tube_radius,
        segments_axis=args.segments_axis,
        segments_circ=args.segments_circ,
        wave_amplitude=max(0.0, args.wave_amplitude),
        wave_cycles_min=max(0.0, args.wave_cycles_min),
        wave_cycles_max=max(args.wave_cycles_min, args.wave_cycles_max),
    )

    delta_rad = math.radians(args.delta_degrees)

    for t in range(args.snapshots):
        angle = t * delta_rad
        meshes = make_rotated_meshes(base_meshes, angle)
        out_path = out_dir / f"cylinders_{t:04d}.glb"
        write_multi_mesh_glb(meshes, out_path)
        print(f"Wrote {out_path} (angle {math.degrees(angle):.2f}°)")


if __name__ == "__main__":
    main()
