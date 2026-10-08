"""Occupancy grid loaded from a ROS map_server YAML/PGM pair."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import yaml

SOURCE_MAP = Path(__file__).resolve().parent.parent / 'maps' / 'map.yaml'


def default_map_path() -> Path:
    """Return the bundled map: installed share directory, else the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        installed = Path(get_package_share_directory('did_agent')) / 'maps' / 'map.yaml'
        if installed.exists():
            return installed
    except Exception:  # noqa: BLE001 - plain Python run without ROS installed
        pass
    return SOURCE_MAP


@dataclass
class GridMap:
    """Static map. Row 0 is the bottom row (smallest y), column 0 the left."""

    resolution: float
    origin_x: float
    origin_y: float
    occupied: np.ndarray
    unknown: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        """Return (rows, columns)."""
        return self.occupied.shape

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """Return the (row, col) of the cell that contains a world point."""
        col = int(np.floor((x - self.origin_x) / self.resolution))
        row = int(np.floor((y - self.origin_y) / self.resolution))
        return row, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Return the world coordinates of a cell centre."""
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        """Return whether a cell index lies inside the grid."""
        return 0 <= row < self.shape[0] and 0 <= col < self.shape[1]


def _read_pgm(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    header = re.match(rb'P5\s+(?:#[^\n]*\n\s*)*(\d+)\s+(\d+)\s+(\d+)\s', raw)
    if header is None:
        raise ValueError(f'{path} is not a binary PGM (P5) file')
    width, height, _ = map(int, header.groups())
    pixels = np.frombuffer(raw[header.end():], dtype=np.uint8)
    return pixels[: width * height].reshape(height, width)


def load_map(yaml_path: str | Path | None = None) -> GridMap:
    """Load a map described by a map_server YAML file."""
    yaml_path = Path(yaml_path) if yaml_path else default_map_path()
    meta = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
    pixels = _read_pgm(yaml_path.parent / meta['image'])
    probability = pixels / 255.0 if meta.get('negate', 0) else 1.0 - pixels / 255.0
    occupied = probability > float(meta['occupied_thresh'])
    free = probability < float(meta['free_thresh'])
    # Image row 0 is the top of the map; flip so that row 0 is the smallest y.
    return GridMap(
        resolution=float(meta['resolution']),
        origin_x=float(meta['origin'][0]),
        origin_y=float(meta['origin'][1]),
        occupied=occupied[::-1].copy(),
        unknown=(~occupied & ~free)[::-1].copy(),
    )
