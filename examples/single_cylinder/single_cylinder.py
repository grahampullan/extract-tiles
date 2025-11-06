#!/usr/bin/env python3
"""Generate a GLB containing a single oblique cylinder surface mesh.

The cylinder is intentionally oriented off all world axes so the difference
between UV-space tiling and world-space projected tiling is immediately obvious.
Additional sinusoidal perturbations along the axis and around the circumference
highlight how tiling reacts to surface undulations.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

# Reuse the mesh construction helpers from the multi-cylinder example.
EXAMPLES_ROOT = Path(__file__).resolve().parent
CYLINDER_LIB = EXAMPLES_ROOT.parent / "cylinders"
if str(CYLINDER_LIB) not in sys.path:
    sys.path.insert(0, str(CYLINDER_LIB))

from cylinders import make_cylinder_mesh, write_multi_mesh_glb  # type: ignore

Color = Tuple[int, int, int, int]


def parse_vec3(text: str) -> np.ndarray:
    """Parse an "x,y,z" string into a float64 vector."""
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values for a vector")
    return np.asarray(parts, dtype=np.float64)


def _orthonormal_frame(axis_start: np.ndarray, axis_end: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    axis_vec = axis_end - axis_start
    length = np.linalg.norm(axis_vec)
    if length <= 0.0:
        raise ValueError("Cylinder axis length must be positive")

    tangent = axis_vec / length
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
    return tangent, normal, binormal, length


def apply_sinusoidal_perturbation(
    verts: np.ndarray,
    uvs: np.ndarray,
    axis_start: np.ndarray,
    axis_end: np.ndarray,
    base_radius: float,
    axial_amplitude: float,
    axial_wavelength: float,
    circum_amplitude: float,
    circum_wavelength: float,
) -> None:
    tangent, normal, binormal, length = _orthonormal_frame(axis_start, axis_end)
    if base_radius <= 0.0:
        raise ValueError("Radius must be positive")

    axis_cycles = 0.0
    if axial_wavelength > 0.0:
        axis_cycles = length / axial_wavelength
    circumference = 2.0 * math.pi * base_radius
    circum_cycles = 0.0
    if circum_wavelength > 0.0:
        circum_cycles = circumference / circum_wavelength

    for idx, (u, v) in enumerate(uvs):
        center = axis_start + tangent * (float(v) * length)
        theta = float(u) * 2.0 * math.pi
        basis_dir = math.cos(theta) * normal + math.sin(theta) * binormal

        perturb = 0.0
        if axial_amplitude != 0.0 and axis_cycles != 0.0:
            perturb += axial_amplitude * math.sin(2.0 * math.pi * axis_cycles * float(v))
        if circum_amplitude != 0.0 and circum_cycles != 0.0:
            perturb += circum_amplitude * math.sin(2.0 * math.pi * circum_cycles * float(u))

        effective_radius = max(base_radius + perturb, 1e-6)
        verts[idx] = center + basis_dir * effective_radius


def build_single_oblique_cylinder(
    axis_start: np.ndarray,
    axis_end: np.ndarray,
    radius: float,
    segments_axis: int,
    segments_circ: int,
    color: Color,
    axial_wave_amplitude: float,
    axial_wave_wavelength: float,
    circum_wave_amplitude: float,
    circum_wave_wavelength: float,
) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    verts, indices, uvs, colors = make_cylinder_mesh(
        axis_start,
        axis_end,
        radius,
        segments_axis,
        segments_circ,
        color,
    )
    apply_sinusoidal_perturbation(
        verts,
        uvs,
        axis_start,
        axis_end,
        radius,
        axial_wave_amplitude,
        axial_wave_wavelength,
        circum_wave_amplitude,
        circum_wave_wavelength,
    )
    return [(
        verts.astype(np.float32, copy=False),
        indices,
        uvs,
        colors,
    )]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a single oblique cylinder glTF asset")
    parser.add_argument("-o", "--output", type=Path, default=Path("single_cylinder.glb"), help="Output GLB path")
    parser.add_argument("--axis-start", type=parse_vec3,
                        default=np.array([-1.5, -0.7, -1.2], dtype=np.float64),
                        help="Cylinder axis start position as x,y,z")
    parser.add_argument("--axis-end", type=parse_vec3,
                        default=np.array([1.2, 1.8, 1.4], dtype=np.float64),
                        help="Cylinder axis end position as x,y,z")
    parser.add_argument("--radius", type=float, default=0.7, help="Cylinder radius (world units)")
    parser.add_argument("--segments-axis", type=int, default=48, help="Segments along the cylinder axis")
    parser.add_argument("--segments-circ", type=int, default=96, help="Segments around the circumference")
    parser.add_argument("--axial-wave-amplitude", type=float, default=0.12,
                        help="Radial perturbation amplitude along the cylinder axis (world units)")
    parser.add_argument("--axial-wave-wavelength", type=float, default=0.8,
                        help="Wavelength for the axial perturbation measured along the axis length")
    parser.add_argument("--circ-wave-amplitude", type=float, default=0.06,
                        help="Radial perturbation amplitude around the circumference (world units)")
    parser.add_argument("--circ-wave-wavelength", type=float, default=0.35,
                        help="Wavelength for the circumferential perturbation measured along the wrapped surface")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    color = (64, 240, 255, 255)
    meshes = build_single_oblique_cylinder(
        axis_start=np.asarray(args.axis_start, dtype=np.float64),
        axis_end=np.asarray(args.axis_end, dtype=np.float64),
        radius=args.radius,
        segments_axis=args.segments_axis,
        segments_circ=args.segments_circ,
        color=color,
        axial_wave_amplitude=args.axial_wave_amplitude,
        axial_wave_wavelength=args.axial_wave_wavelength,
        circum_wave_amplitude=args.circ_wave_amplitude,
        circum_wave_wavelength=args.circ_wave_wavelength,
    )
    write_multi_mesh_glb(list(meshes), args.output)
    print(f"Wrote oblique cylinder mesh to {args.output}")


if __name__ == "__main__":
    main()
