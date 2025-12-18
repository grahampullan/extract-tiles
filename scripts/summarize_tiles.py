#!/usr/bin/env python3
"""Summarize tile counts, triangle totals, and file sizes per level.

The script reads a tiles manifest JSON file (as emitted by build_tiles.py) and
looks up the referenced tile binaries on disk to report per-level aggregates.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize tileset statistics per octree level")
    parser.add_argument("manifest", type=Path, help="Path to manifest_X.json produced by build_tiles.py")
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Skip missing tile binaries instead of failing",
    )
    return parser.parse_args()


def resolve_tile_path(manifest_dir: Path, url: str) -> Path:
    """Convert a manifest tile URL to a local filesystem path."""
    rel = url.lstrip("/")
    parts = Path(rel).parts
    if parts and parts[0] == "tiles":
        parts = parts[1:]
    if parts and parts[0] == manifest_dir.name:
        parts = parts[1:]
    return manifest_dir.joinpath(*parts)


def summarize(manifest_path: Path, ignore_missing: bool) -> Dict[int, Dict[str, float]]:
    data = json.loads(manifest_path.read_text())
    manifest_dir = manifest_path.parent

    per_level: Dict[int, Dict[str, float]] = defaultdict(lambda: {"count": 0, "triangles": 0, "bytes": 0})

    for tile in data.get("tiles", []):
        level = int(tile.get("z", 0))
        tri_count = int(tile.get("triCount", 0))
        per_level[level]["count"] += 1
        per_level[level]["triangles"] += tri_count

        url = tile.get("url")
        path = resolve_tile_path(manifest_dir, url) if url else None
        if path and path.exists():
            per_level[level]["bytes"] += path.stat().st_size
        else:
            if "actualBytes" in tile:
                per_level[level]["bytes"] += int(tile["actualBytes"])
            elif not ignore_missing:
                missing = path if path else url
                raise FileNotFoundError(f"Tile binary not found for {missing}")

    return per_level


def format_bytes(value: float) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:,.1f} {unit}"
        size /= 1024.0
    return f"{value} B"


def main() -> None:
    args = parse_args()
    per_level = summarize(args.manifest, args.ignore_missing)
    total_levels = sorted(per_level)

    print(f"Manifest: {args.manifest}")
    print("| Level | Tiles | Total triangles | Total size | Mean size/tile |")
    print("|-------|-------|----------------|------------|----------------|")
    for level in total_levels:
        info = per_level[level]
        tiles = int(info["count"])
        tris = int(info["triangles"])
        total_bytes = info["bytes"]
        size = format_bytes(total_bytes)
        mean_size = format_bytes(total_bytes / max(tiles, 1))
        print(f"| {level:5d} | {tiles:5d} | {tris:14,d} | {size:>10} | {mean_size:>14} |")


if __name__ == "__main__":
    main()
