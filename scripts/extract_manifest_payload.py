#!/usr/bin/env python3
"""Extract per-timestep payload fields from a static-octree manifest.

Given a full manifest (as produced by build_tiles.py --static_octree), this
script emits a compact JSON file that retains only the fields that typically
vary between timesteps (URLs, byte counts, geometric error, etc.). The goal is
to experiment with lighter-weight per-time payloads while reusing the static
tile hierarchy.

Usage:
  python3 scripts/extract_manifest_payload.py \
      --manifest tiles_out/.../manifest_0.json \
      --out payload_0.json

The output schema:
{
  "extract": "...",
  "time": 0,
  "baseManifest": "manifest_0.json",
  "fields": ["url", "approxBytes", ...],
  "tiles": [
    {"tileId": "0/0/0/0", "url": "...", ...},
    ...
  ]
}

Only non-structural tile keys are copied. Structural keys (coordinates,
children, AABBs, etc.) are assumed to be identical across timesteps and are
omitted.
"""

import argparse
import json
import pathlib
from typing import Dict, Any, List, Set

# Fields that describe the hierarchy/geometry layout and should be skipped.
DEFAULT_STRUCTURAL_KEYS: Set[str] = {
    "tileId", "z", "x", "y", "k",
    "parent", "children",
    "aabbWorld", "aabbUV",
    "mesh", "meshName"
}


def load_manifest(path: pathlib.Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def extract_payload(manifest: Dict[str, Any], structural_keys: Set[str], manifest_name: str) -> Dict[str, Any]:
    tiles = manifest.get('tiles') or []
    observed_fields: Set[str] = set()

    for tile in tiles:
        for key, value in tile.items():
            if key in structural_keys:
                continue
            observed_fields.add(key)

    # Ensure tileId is always present and first
    ordered_fields: List[str] = ["tileId"] + sorted(observed_fields - {"tileId"})
    # numeric fields that benefit from truncation to keep payload compact
    float_fields = {"avgTriArea", "minTriArea", "geometricError"}
    def format_value(field: str, value: Any) -> Any:
        if field in float_fields and isinstance(value, (int, float)):
            return float(f"{value:.4e}")
        return value

    payload_rows: List[List[Any]] = []
    for tile in tiles:
        row: List[Any] = []
        for field in ordered_fields:
            if field == "tileId":
                row.append(tile.get("tileId"))
            else:
                row.append(format_value(field, tile.get(field)))
        payload_rows.append(row)

    return {
        "extract": manifest.get("extract"),
        "time": manifest.get("time"),
        "baseManifest": manifest_name,
        "fields": ordered_fields,
        "tiles": payload_rows
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-timestep payload fields from a static-octree manifest")
    parser.add_argument('--manifest', required=True, help='Path to the full manifest (manifest_<time>.json)')
    parser.add_argument('--out', help='Output path for the payload JSON (defaults to payload_<time>.json alongside the manifest)')
    parser.add_argument('--structural-keys', nargs='*', default=sorted(DEFAULT_STRUCTURAL_KEYS),
                        help='Tile keys to treat as structural (excluded from payload). Defaults cover common fields.')

    args = parser.parse_args()
    manifest_path = pathlib.Path(args.manifest)
    if not manifest_path.exists():
        parser.error(f"Manifest not found: {manifest_path}")

    manifest = load_manifest(manifest_path)
    structural = set(args.structural_keys)
    layout_type = (manifest.get("layout") or {}).get("type")
    if layout_type != "static-octree":
        print(f"Warning: manifest layout type is {layout_type!r}; results may not be reusable across timesteps.")

    payload = extract_payload(manifest, structural, manifest_path.name)

    out_path = pathlib.Path(args.out) if args.out else manifest_path.with_name(f"payload_{payload.get('time')}.json")
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote payload for time {payload.get('time')} to {out_path}")


if __name__ == "__main__":
    main()
