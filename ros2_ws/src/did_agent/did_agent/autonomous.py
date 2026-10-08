"""Autonomous agent without an LLM: sweep the arena, climb the signal, collect.

This is the fallback planner. It uses the same skills as the LLM plan executor,
so the demo works even when the language model is unavailable.
"""

from __future__ import annotations

from math import hypot
from typing import Any
from typing import Callable

from did_agent.planner import plan_cells
from did_agent.robot import BASE
from did_agent.robot import Robot
from did_agent.skills import Skills


class AutonomousAgent:
    """Visit sweep points, react to the signal, stop in time to get home."""

    def __init__(
        self,
        robot: Robot,
        skills: Skills | None = None,
        *,
        spacing: float = 1.3,
        signal: float = 0.08,
        log: Callable[[str], None] = lambda message: None,
    ) -> None:
        self.robot = robot
        self.skills = skills or Skills(robot)
        self.spacing = spacing
        self.signal = signal
        self.log = log
        self.attempts = 0

    def sweep_points(self) -> list[tuple[float, float]]:
        """Reachable free points on a regular grid covering the arena."""
        costmap = self.robot.costmap
        rows, cols = costmap.grid.shape
        start = costmap.nearest_free(*costmap.world_to_cell(*BASE), 0.5)
        points = []
        x_min, y_min = -2.8, -2.4
        count_x, count_y = int(5.4 / self.spacing) + 1, int(5.0 / self.spacing) + 1
        for i in range(count_x):
            for j in range(count_y):
                x, y = x_min + i * self.spacing, y_min + j * self.spacing
                cell = costmap.nearest_free(*costmap.world_to_cell(x, y), 0.3)
                if cell is None:
                    continue
                if plan_cells(costmap, start, cell) is not None:
                    points.append(costmap.cell_to_world(*cell))
        return points

    def _hear_and_collect(self) -> int:
        """Collect everything we can hear from the current spot; return the count."""
        gained, failures = 0, 0
        while failures < 2 and self.skills.battery_allows_more():
            reading = self.robot.read_sensor(15 if self.robot.noise_level() > 0.06 else 5)
            if reading.value < self.signal:
                break
            pose = self.robot.pose()
            self.log(f'Слышу сигнал {reading.value:.2f} в ({pose.x:.2f}; {pose.y:.2f}), ищу образец')
            found = self.skills.search_around(pose.x, pose.y, 0.5)
            if self.robot.preempted():
                break
            if found.ok and self.skills.collect().ok:
                gained += 1
                self.log(f'Образец собран: {self.robot.collected()}/{self.robot.samples_total()}')
                failures = 0
            else:
                failures += 1
        return gained

    def run(self) -> dict[str, Any]:
        """Run until all samples are collected or the battery says go home."""
        pending = self.sweep_points()
        self.log(f'Обход арены: {len(pending)} точек обзора')
        while pending and self.robot.collected() < self.robot.samples_total():
            pose = self.robot.pose()
            pending.sort(key=lambda p: hypot(p[0] - pose.x, p[1] - pose.y))
            target = pending.pop(0)
            leg = self._leg_cost(target)
            if self.robot.battery() < leg + self.skills.reserve:
                continue
            result = self.skills.goto(*target, guarded=True)
            if self.robot.preempted():
                break
            if not result.ok:
                self.log(f'Пропускаю точку ({target[0]:.1f}; {target[1]:.1f}): {result.reason}')
                continue
            here = self.robot.pose()
            pending = [p for p in pending if hypot(p[0] - here.x, p[1] - here.y) > 0.7]
            self._hear_and_collect()
            if not self.skills.battery_allows_more():
                self.log('Запас батареи на пределе, еду домой')
                break
        returned = self.skills.return_to_base()
        return {
            'collected': self.robot.collected(),
            'samples_total': self.robot.samples_total(),
            'battery': round(self.robot.battery(), 2),
            'returned_to_base': returned.ok,
            'reason': returned.reason,
        }

    def _leg_cost(self, target: tuple[float, float]) -> float:
        """Battery for the leg to a sweep point and then home from there."""
        costmap = self.robot.costmap
        pose = self.robot.pose()
        a = costmap.nearest_free(*costmap.world_to_cell(pose.x, pose.y), 0.5)
        b = costmap.world_to_cell(*target)
        c = costmap.nearest_free(*costmap.world_to_cell(*BASE), 0.5)
        there, back = plan_cells(costmap, a, b), plan_cells(costmap, b, c)
        if there is None or back is None:
            return float('inf')

        def energy(path):
            return costmap.energy_cost([costmap.cell_to_world(*cell) for cell in path])

        return energy(there) * 1.2 + energy(back) * 1.3
