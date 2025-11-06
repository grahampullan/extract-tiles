# single_cylinder example

This example generates a single oblique cylinder mesh to showcase how UV tiling
differs from world-space projected tiling when the geometry is not aligned with
the primary axes. Sinusoidal perturbations along the axis and around the
circumference exaggerate how each tiling strategy responds to surface detail.

Run the generator to produce `single_cylinder.glb`:

```bash
python3 single_cylinder.py --output single_cylinder.glb
```

## CLI arguments

- `-o, --output`: Output GLB path (default `single_cylinder.glb`).
- `--axis-start`: Cylinder axis start as `x,y,z` (default `-1.5,-0.7,-1.2`).
- `--axis-end`: Cylinder axis end as `x,y,z` (default `1.2,1.8,1.4`).
- `--radius`: Cylinder radius in world units (default `0.7`).
- `--segments-axis`: Segments along the cylinder axis (default `48`).
- `--segments-circ`: Segments around the circumference (default `96`).
- `--axial-wave-amplitude`: Radial perturbation amplitude along the axis (default `0.12`).
- `--axial-wave-wavelength`: Wavelength for the axial perturbation (default `0.8`).
- `--circ-wave-amplitude`: Radial perturbation amplitude around the circumference (default `0.06`).
- `--circ-wave-wavelength`: Wavelength for the circumferential perturbation (default `0.35`).

Set either wavelength to a non-positive value to disable that perturbation, or
adjust amplitudes to tune how aggressively the surface deviates from the base
cylinder.
