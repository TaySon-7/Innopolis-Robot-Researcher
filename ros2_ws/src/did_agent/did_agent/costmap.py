"""Cost grid for planning: walls are forbidden, floor types have a price."""

from __future__ import annotations

from math import hypot
from typing import Any

import numpy as np

from did_agent.grid import GridMap
from did_agent.grid import load_map


# How long a lidar-seen obstacle stays blocked without a fresh sighting: a
# false positive or a moved object must not poison the map forever.
OBSTACLE_TTL = 30.0


def _disk_offsets(radius_cells: float) -> list[tuple[int, int, float]]:
    reach = int(np.ceil(radius_cells))
    return [
        (dr, dc, hypot(dr, dc))
        for dr in range(-reach, reach + 1)
        for dc in range(-reach, reach + 1)
        if hypot(dr, dc) <= radius_cells
    ]


def _shifted_or(mask: np.ndarray, offsets: list[tuple[int, int, float]]) -> np.ndarray:
    """Dilate a boolean mask by a set of (row, col) offsets."""
    rows, cols = mask.shape
    result = np.zeros_like(mask)
    for dr, dc, _ in offsets:
        r0, r1 = max(0, dr), min(rows, rows + dr)
        c0, c1 = max(0, dc), min(cols, cols + dc)
        result[r0:r1, c0:c1] |= mask[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return result


def _distance_to(mask: np.ndarray, limit_cells: float) -> np.ndarray:
    """Distance in cells to the nearest True cell, capped at the limit."""
    rows, cols = mask.shape
    distance = np.full(mask.shape, limit_cells, dtype=np.float32)
    for dr, dc, length in _disk_offsets(limit_cells):
        r0, r1 = max(0, dr), min(rows, rows + dr)
        c0, c1 = max(0, dc), min(cols, cols + dc)
        hit = mask[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
        window = distance[r0:r1, c0:c1]
        np.minimum(window, np.where(hit, length, limit_cells), out=window)
    return distance


class CostMap:
    """Planning layers on top of the static map.

    * ``blocked``   - cells the robot centre must never enter (walls plus a
      safety margin, unknown space and obstacles seen by the lidar);
    * ``wall_cost`` - extra price per metre close to walls, to keep a clearance;
    * ``terrain``   - price multiplier of the floor, learned by the agent
      (1.0 = ordinary floor). It defaults to what the agent knows: nothing.
    """

    def __init__(
        self,
        grid: GridMap | None = None,
        *,
        robot_radius: float = 0.20,
        inflation_radius: float = 0.45,
        wall_weight: float = 4.0,
    ) -> None:
        self.grid = grid or load_map()
        self.resolution = self.grid.resolution
        self.robot_radius = robot_radius
        solid = self.grid.occupied | self.grid.unknown
        self.static_blocked = _shifted_or(
            solid,
            _disk_offsets(robot_radius / self.resolution),
        )
        limit = inflation_radius / self.resolution
        self.wall_distance = _distance_to(solid, limit) * self.resolution
        wall_distance = self.wall_distance
        self.wall_cost = (
            wall_weight * np.clip(1.0 - wall_distance / inflation_radius, 0.0, 1.0)
        ).astype(np.float32)
        self.terrain = np.ones(self.grid.shape, dtype=np.float32)
        self.last_seen = np.full(self.grid.shape, -1e9, dtype=np.float32)  # when evidence last arrived
        self._dynamic = np.zeros(self.grid.shape, dtype=bool)
        # When a dynamic block was last confirmed: -inf for an unused cell,
        # +inf for a block given without an evidence time (never expires).
        self._dynamic_stamp = np.full(self.grid.shape, -np.inf, dtype=np.float32)
        self.blocked = self.static_blocked.copy()
        self.version = 0

    # --- queries ---------------------------------------------------------

    def is_free(self, row: int, col: int) -> bool:
        """Return whether the robot centre may be in this cell."""
        return self.grid.in_bounds(row, col) and not self.blocked[row, col]

    def cost_per_metre(self, row: int, col: int) -> float:
        """Return the planning price of travelling one metre through a cell."""
        return float(self.terrain[row, col] + self.wall_cost[row, col])

    def nearest_free(
        self,
        row: int,
        col: int,
        max_radius: float = 0.5,
    ) -> tuple[int, int] | None:
        """Return the closest free cell within a radius, or None."""
        if self.is_free(row, col):
            return row, col
        best = None
        best_distance = max_radius / self.resolution
        for dr, dc, length in _disk_offsets(best_distance):
            if length < best_distance and self.is_free(row + dr, col + dc):
                best, best_distance = (row + dr, col + dc), length
        return best

    def near_known_solid(self, points: np.ndarray, tolerance: float = 0.15) -> np.ndarray:
        """For an (n, 2) array of world points: True where the map explains a hit."""
        cols = np.floor((points[:, 0] - self.grid.origin_x) / self.resolution).astype(int)
        rows = np.floor((points[:, 1] - self.grid.origin_y) / self.resolution).astype(int)
        inside = (
            (rows >= 0) & (rows < self.grid.shape[0])
            & (cols >= 0) & (cols < self.grid.shape[1])
        )
        known = np.ones(len(points), dtype=bool)  # outside the map: nothing new to learn
        known[inside] = self.wall_distance[rows[inside], cols[inside]] <= tolerance
        return known

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """Return the cell that contains a world point."""
        return self.grid.world_to_cell(x, y)

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Return the world coordinates of a cell centre."""
        return self.grid.cell_to_world(row, col)

    # --- learned floor prices -------------------------------------------------

    def cells_in_region(self, region: dict[str, Any]) -> np.ndarray:
        """Return a boolean mask of the cells covered by a region description."""
        mask = np.zeros(self.grid.shape, dtype=bool)
        rows, cols = self.grid.shape
        res = self.resolution
        row_y = self.grid.origin_y + (np.arange(rows) + 0.5) * res
        col_x = self.grid.origin_x + (np.arange(cols) + 0.5) * res
        grid_x, grid_y = np.meshgrid(col_x, row_y)
        if 'circle' in region:
            circle = region['circle']
            radius = float(circle.get('r', circle.get('radius')))
            mask = np.hypot(grid_x - circle['x'], grid_y - circle['y']) <= radius
        elif 'rect' in region:
            rect = region['rect']
            mask = (
                (grid_x >= rect['x_min']) & (grid_x <= rect['x_max'])
                & (grid_y >= rect['y_min']) & (grid_y <= rect['y_max'])
            )
        elif 'cells' in region:
            for row, col in region['cells']:
                if self.grid.in_bounds(int(row), int(col)):
                    mask[int(row), int(col)] = True
        else:
            raise ValueError('region needs one of: circle, rect, cells')
        return mask

    def update(self, region: dict[str, Any], cost: float) -> int:
        """Set the floor price multiplier in a region; return changed cells."""
        if cost <= 0.0:
            raise ValueError('cost multiplier must be positive')
        mask = self.cells_in_region(region)
        changed = int(np.count_nonzero(self.terrain[mask] != cost))
        if changed:
            self.terrain[mask] = cost
            self.version += 1
        return changed

    def set_terrain(
        self,
        row0: int,
        col0: int,
        values: np.ndarray,
        threshold: float = 0.2,
    ) -> bool:
        """Overwrite a block of floor prices; bump the version on a material change.

        Small drifts are stored but do not bump the version, so the navigator is
        not forced to re-plan every second while the estimate settles.
        """
        rows, cols = values.shape
        block = self.terrain[row0:row0 + rows, col0:col0 + cols]
        material = float(np.abs(values - block).max()) > threshold
        block[...] = values
        if material:
            self.version += 1
        return material

    # --- obstacles seen by the lidar -------------------------------------------

    def add_obstacles(
        self,
        points: list[tuple[float, float]],
        radius: float = 0.03,
        now: float | None = None,
    ) -> None:
        """Block the area around world points that the lidar saw as obstacles.

        ``now`` is the sighting time read by :meth:`expire_dynamic`; without it
        the block is kept until something else clears the cell.
        """
        offsets = _disk_offsets((radius + self.robot_radius) / self.resolution)
        stamp = float('inf') if now is None else float(now)
        for x, y in points:
            row, col = self.world_to_cell(x, y)
            for dr, dc, _ in offsets:
                if self.grid.in_bounds(row + dr, col + dc):
                    self._dynamic[row + dr, col + dc] = True
                    seen = float(self._dynamic_stamp[row + dr, col + dc])
                    self._dynamic_stamp[row + dr, col + dc] = max(seen, stamp)
        self.blocked = self.static_blocked | self._dynamic
        self.version += 1

    def block_disk(self, x: float, y: float, radius: float) -> None:
        """Forbid a round area for planning (a place that hurt the robot).

        Hazard knowledge does not expire: the judge never removes a hazard, so
        neither does the agent's map of one.
        """
        mask = self.cells_in_region({'circle': {'x': x, 'y': y, 'r': radius}})
        self._dynamic |= mask
        self._dynamic_stamp[mask] = np.inf
        self.blocked = self.static_blocked | self._dynamic
        self.version += 1

    def expire_dynamic(self, now: float, ttl: float = OBSTACLE_TTL) -> int:
        """Free lidar blocks that were not re-confirmed for ``ttl`` seconds.

        Blocks stamped without an evidence time (``block_disk``,
        ``add_obstacles`` without ``now``) never expire, and ``ttl <= 0``
        switches expiry off entirely.  Bumps the version when cells were freed
        so the navigator re-plans with the recovered space.  Returns the number
        of cells freed.
        """
        if ttl <= 0.0:
            return 0
        stale = self._dynamic & (self._dynamic_stamp < now - ttl)
        freed = int(np.count_nonzero(stale))
        if freed:
            self._dynamic[stale] = False
            self.blocked = self.static_blocked | self._dynamic
            self.version += 1
        return freed

    # --- path metrics ---------------------------------------------------------------

    def path_length(self, path: list[tuple[float, float]]) -> float:
        """Return the length of a world-coordinate polyline in metres."""
        return sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))

    def energy_cost(
        self,
        path: list[tuple[float, float]],
        step: float = 0.05,
        pessimism: float = 0.0,
        now: float | None = None,
    ) -> float:
        """Estimate the battery use of a polyline: distance times floor price.

        With ``pessimism`` > 0 the part of a price above the ordinary floor is
        exaggerated by that fraction: a dear floor may get dearer without notice.
        With ``now`` given, knowledge also loses trust with age: a price that was
        last confirmed long ago is exaggerated further (up to 1.5 more).
        """
        total = 0.0
        for a, b in zip(path, path[1:]):
            length = hypot(b[0] - a[0], b[1] - a[1])
            count = max(1, int(np.ceil(length / step)))
            for i in range(count):
                t = (i + 0.5) / count
                row, col = self.world_to_cell(
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                )
                if self.grid.in_bounds(row, col):
                    price = float(self.terrain[row, col])
                    if price > 1.0:
                        doubt = pessimism
                        if now is not None:
                            doubt += min(1.5, max(0.0, (now - float(self.last_seen[row, col])) / 90.0))
                        price = 1.0 + (price - 1.0) * (1.0 + doubt)
                    total += length / count * price
        return total
