"""A* on the cost grid and conversion of cell paths to waypoints."""

from __future__ import annotations

import heapq
from math import hypot
from math import sqrt

from did_agent.costmap import CostMap

Cell = tuple[int, int]

_NEIGHBOURS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, sqrt(2)), (-1, 1, sqrt(2)), (1, -1, sqrt(2)), (1, 1, sqrt(2)),
]


def plan_cells(costmap: CostMap, start: Cell, goal: Cell) -> list[Cell] | None:
    """Return the cheapest 8-connected cell path from start to goal, or None."""
    if not costmap.is_free(*start) or not costmap.is_free(*goal):
        return None
    resolution = costmap.resolution
    rows, cols = costmap.grid.shape
    cheapest = costmap.terrain.min()  # keeps the heuristic admissible
    price = costmap.terrain + costmap.wall_cost
    blocked = costmap.blocked

    best = {start: 0.0}
    parent: dict[Cell, Cell] = {}
    heap = [(0.0, 0.0, start)]
    closed: set[Cell] = set()
    while heap:
        _, cost, cell = heapq.heappop(heap)
        if cell in closed:
            continue
        if cell == goal:
            path = [cell]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return path[::-1]
        closed.add(cell)
        row, col = cell
        for dr, dc, length in _NEIGHBOURS:
            nrow, ncol = row + dr, col + dc
            if not (0 <= nrow < rows and 0 <= ncol < cols) or blocked[nrow, ncol]:
                continue
            if dr and dc and (blocked[row + dr, col] or blocked[row, col + dc]):
                continue  # do not cut corners
            neighbour = (nrow, ncol)
            if neighbour in closed:
                continue
            step = length * resolution * 0.5 * (price[row, col] + price[nrow, ncol])
            new_cost = cost + float(step)
            if new_cost < best.get(neighbour, float('inf')):
                best[neighbour] = new_cost
                parent[neighbour] = cell
                heuristic = hypot(goal[0] - nrow, goal[1] - ncol) * resolution * cheapest
                heapq.heappush(heap, (new_cost + float(heuristic), new_cost, neighbour))
    return None


def _line_cells(a: Cell, b: Cell) -> list[Cell]:
    """Return the cells on the straight line between two cells (Bresenham)."""
    (r0, c0), (r1, c1) = a, b
    cells = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    error = dc - dr
    while True:
        cells.append((r0, c0))
        if (r0, c0) == (r1, c1):
            return cells
        doubled = 2 * error
        if doubled > -dr:
            error -= dr
            c0 += sc
        if doubled < dc:
            error += dc
            r0 += sr


def _segment_price(costmap: CostMap, a: Cell, b: Cell) -> float | None:
    """Price of the straight line a->b, or None if it touches a blocked cell."""
    cells = _line_cells(a, b)
    total = 0.0
    for first, second in zip(cells, cells[1:]):
        if costmap.blocked[second]:
            return None
        if first[0] != second[0] and first[1] != second[1]:
            if costmap.blocked[first[0], second[1]] or costmap.blocked[second[0], first[1]]:
                return None
        length = hypot(second[0] - first[0], second[1] - first[1]) * costmap.resolution
        total += length * 0.5 * (
            costmap.cost_per_metre(*first) + costmap.cost_per_metre(*second)
        )
    return total


def smooth(costmap: CostMap, path: list[Cell], tolerance: float = 1.02) -> list[Cell]:
    """Drop intermediate cells when a straight shortcut is free and not dearer."""
    if len(path) < 3:
        return path
    kept = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        reach = anchor + 1
        for candidate in range(len(path) - 1, anchor, -1):
            price = _segment_price(costmap, path[anchor], path[candidate])
            if price is None:
                continue
            along = sum(
                _segment_price(costmap, path[i], path[i + 1]) or 0.0
                for i in range(anchor, candidate)
            )
            if price <= along * tolerance:
                reach = candidate
                break
        kept.append(path[reach])
        anchor = reach
    return kept


def plan_waypoints(
    costmap: CostMap,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    snap_radius: float = 0.5,
) -> list[tuple[float, float]] | None:
    """Plan from a world point to a world point; return world waypoints.

    Both ends are snapped to the nearest free cell (the robot may stand close
    to a wall, a sample may lie inside the safety margin).
    """
    start = costmap.nearest_free(*costmap.world_to_cell(*start_xy), snap_radius)
    goal = costmap.nearest_free(*costmap.world_to_cell(*goal_xy), snap_radius)
    if start is None or goal is None:
        return None
    cells = plan_cells(costmap, start, goal)
    if cells is None:
        return None
    return [costmap.cell_to_world(*cell) for cell in smooth(costmap, cells)]
