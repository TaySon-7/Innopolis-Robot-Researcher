"""ROS-independent translation and command checks for the Nav2 adapter.

Both navigators consume the same observed CostMap. Nav2 applies its own robot
footprint inflation to the raw static geometry; terrain remains a graded cost.
"""

from math import isfinite, pi

import numpy as np

from did_agent.controller import Command
from did_agent.costmap import CostMap
from did_agent.navigator_core import Scan


def knowledge_grid(costmap: CostMap) -> np.ndarray:
    """OccupancyGrid values, bottom row first, with no scenario truth input."""
    terrain = np.asarray(costmap.terrain, dtype=float)
    if not np.isfinite(terrain).all() or (terrain <= 0).any():
        raise ValueError('invalid observed terrain costs')
    values = np.rint(np.clip((terrain - 1.0) * 12.0, 0, 90)).astype(np.int8)
    values[costmap.grid.unknown] = -1
    # static_blocked already includes robot clearance, unlike the raw grid.
    # Observed dynamic exclusions remain conservative rather than shrinking
    # a learned forbidden region back into traversable space.
    dynamic = costmap.blocked & ~costmap.static_blocked
    values[costmap.grid.occupied | dynamic] = 100
    return values


def checked_velocity(command: Command, scan: Scan | None, now: float) -> tuple[Command, str]:
    """Bound Nav2 speeds and stop on blind, stale or imminently unsafe input."""
    if not all(isfinite(v) for v in (command.linear, command.angular, now)):
        return Command(), 'invalid Nav2 velocity'
    if scan is None or scan.stamp is None:
        return Command(), 'missing lidar scan'
    if not -0.1 <= now - scan.stamp <= 0.75:
        return Command(), 'stale lidar scan'
    if (not all(isfinite(v) for v in (scan.angle_min, scan.angle_increment, scan.range_min))
            or scan.angle_increment == 0 or scan.range_min < 0):
        return Command(), 'invalid lidar scan geometry'
    ranges = np.asarray(scan.ranges, dtype=float)
    usable = np.isposinf(ranges) | (np.isfinite(ranges) & (ranges >= scan.range_min))
    if not usable.any():
        return Command(), 'lidar scan has no usable ranges'
    angles = (scan.angles() + pi) % (2 * pi) - pi
    if command.linear != 0:
        direction = np.abs(angles) <= 0.6 if command.linear > 0 else np.abs(angles) >= pi - 0.6
        if not direction.any() or not usable[direction].all():
            return Command(), 'lidar scan has missing movement coverage'
        if (ranges[direction] <= 0.16).any():
            return Command(), 'obstacle inside emergency stopping distance'
    if command.angular != 0:
        if not usable.all() or abs(scan.angle_increment) * len(ranges) < 2 * pi - 0.15:
            return Command(), 'lidar scan has missing coverage for turning'
        if (ranges <= 0.14).any():
            return Command(), 'obstacle inside turning clearance'
    return Command(max(-0.18, min(0.18, command.linear)),
                   max(-1.2, min(1.2, command.angular))), ''
