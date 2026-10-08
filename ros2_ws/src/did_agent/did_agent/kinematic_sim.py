"""Tiny kinematic simulator: unicycle robot and a ray-cast lidar on the real map."""

from __future__ import annotations

from math import cos
from math import pi
from math import sin

import numpy as np

from did_agent.controller import Pose
from did_agent.grid import GridMap
from did_agent.navigator_core import DONE
from did_agent.navigator_core import FAILED
from did_agent.navigator_core import NavigatorCore
from did_agent.navigator_core import Scan

ROBOT_RADIUS = 0.105
BASE = (-2.0, -0.5)


class KinematicSim:
    """Robot on a map. ``hidden`` obstacles are circles the planner knows nothing of."""

    def __init__(
        self,
        grid: GridMap,
        pose: Pose = Pose(BASE[0], BASE[1], 0.0),
        hidden: list[tuple[float, float, float]] | None = None,
        beams: int = 72,
        yaw_noise: float = 0.0,
        seed: int = 7,
    ) -> None:
        self.grid = grid
        self.pose = Pose(pose.x, pose.y, pose.yaw)
        self.hidden = hidden or []
        self.beams = beams
        self.yaw_noise = yaw_noise
        self._rng = np.random.default_rng(seed)
        self.t = 0.0
        self.collisions = 0
        self.distance = 0.0
        self._touching = False
        self._ring = np.array([
            (ROBOT_RADIUS * cos(a), ROBOT_RADIUS * sin(a))
            for a in np.linspace(0, 2 * pi, 16, endpoint=False)
        ] + [(0.0, 0.0)])

    def _solid(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        cols = np.floor((xs - self.grid.origin_x) / self.grid.resolution).astype(int)
        rows = np.floor((ys - self.grid.origin_y) / self.grid.resolution).astype(int)
        inside = (
            (rows >= 0) & (rows < self.grid.shape[0])
            & (cols >= 0) & (cols < self.grid.shape[1])
        )
        hit = np.ones(xs.shape, dtype=bool)  # outside the map counts as solid
        hit[inside] = (
            self.grid.occupied[rows[inside], cols[inside]]
            | self.grid.unknown[rows[inside], cols[inside]]
        )
        for cx, cy, radius in self.hidden:
            hit |= np.hypot(xs - cx, ys - cy) <= radius
        return hit

    def scan(self) -> Scan:
        angles = np.linspace(-pi, pi, self.beams, endpoint=False)
        steps = np.arange(0.12, 3.5, 0.025)
        direction = self.pose.yaw + angles
        xs = self.pose.x + np.outer(np.cos(direction), steps)
        ys = self.pose.y + np.outer(np.sin(direction), steps)
        hit = self._solid(xs, ys)
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        ranges = np.where(first >= 0, steps[np.maximum(first, 0)], np.inf)
        return Scan(-pi, 2 * pi / self.beams, ranges)

    def step(self, linear: float, angular: float, dt: float = 0.05) -> None:
        linear = max(-0.22, min(0.22, linear))
        angular = max(-2.84, min(2.84, angular))
        self.pose.yaw += angular * dt
        if self.yaw_noise and linear:
            self.pose.yaw += float(self._rng.normal(0.0, self.yaw_noise))  # wheel slip
        dx, dy = linear * cos(self.pose.yaw) * dt, linear * sin(self.pose.yaw) * dt
        self.pose.x += dx
        self.pose.y += dy
        self.distance += abs(linear) * dt
        self.t += dt
        xs = self.pose.x + self._ring[:, 0]
        ys = self.pose.y + self._ring[:, 1]
        touching = bool(self._solid(xs, ys).any())
        if touching and not self._touching:
            self.collisions += 1
        self._touching = touching


def drive(
    nav: NavigatorCore,
    sim: KinematicSim,
    goal: tuple[float, float],
    timeout: float = 240.0,
) -> str:
    """Run the navigator in the simulator until it finishes; return its status."""
    if not nav.set_goal(goal, sim.pose, sim.t):
        return nav.status
    start = sim.t
    while sim.t - start < timeout:
        command = nav.update(sim.pose, sim.scan(), sim.t)
        sim.step(command.linear, command.angular)
        if nav.status in (DONE, FAILED):
            break
    return nav.status
