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
from did_agent.robot import Reading
from did_agent.robot import Robot
from did_agent.search import SAMPLE_SENSOR_RANGE
from did_agent.skills import Skills


MIN_LOCAL_SEARCH_RADIUS = 0.45
MAX_LOCAL_SEARCH_RADIUS = 1.20
FAILED_SEARCH_COOLDOWN_RADIUS = 0.70
FAILED_SEARCH_RETRY_GAIN = 0.12


def local_search_radius(reading: Reading) -> float:
    """Size a local search from the observed distance, including noise margin.

    The sensor is approximately ``1 - distance / 1.5``.  A fixed 0.5 m
    search was therefore too small for a perfectly valid 0.08--0.4 reading:
    the robot knew a sample was nearby but never swept far enough to reach it.
    Subtracting two noise estimates is conservative; the final 0.15 m allowance
    leaves the refinement comfortably inside the 0.30 m collection radius.
    """
    reliable = max(0.0, min(1.0, reading.value - 2.0 * max(0.0, reading.noise)))
    estimated_distance = SAMPLE_SENSOR_RANGE * (1.0 - reliable)
    return round(max(MIN_LOCAL_SEARCH_RADIUS,
                     min(MAX_LOCAL_SEARCH_RADIUS, estimated_distance - 0.15)), 2)


class AutonomousAgent:
    """Visit sweep points, react to the signal, stop in time to get home."""

    def __init__(
        self,
        robot: Robot,
        skills: Skills | None = None,
        *,
        # A 1.3 m lattice leaves blind pockets once pillars push a nominal
        # point to the nearest free cell.  A 1.0 m lattice reaches those pockets
        # sooner and, because the run ends as soon as every sample is collected,
        # used less battery than the coarser alternatives in the seeded
        # regression scenarios.
        spacing: float = 1.0,
        # Below 0.20, a sample can still be inside the nominal 1.5 m sensor
        # range, but pillars make directionless localization disproportionately
        # expensive.  The 1.0 m coverage lattice reaches a stronger viewpoint;
        # seeded hard runs collected more while using much less battery at 0.20.
        signal: float = 0.20,
        log: Callable[[str], None] = lambda message: None,
    ) -> None:
        self.robot = robot
        self.skills = skills or Skills(robot)
        self.spacing = spacing
        self.signal = signal
        self.log = log
        self.attempts = 0
        self._failed_search: tuple[float, float, float] | None = None

    def sweep_points(self) -> list[tuple[float, float]]:
        """Reachable free points on a regular grid covering the arena."""
        costmap = self.robot.costmap
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
        # A failed directionless search should buy a different viewpoint, not
        # immediately repeat the same expensive trajectory from the same pose.
        # Successful collections reset the counter so clustered samples are
        # still collected in one visit.
        while failures < 1 and self.skills.battery_allows_more():
            reading = self.robot.read_sensor(15 if self.robot.noise_level() > 0.06 else 5)
            reliable_signal = reading.value - 2.0 * max(0.0, reading.noise)
            if reliable_signal < self.signal:
                break
            pose = self.robot.pose()
            if self._failed_search is not None:
                failed_x, failed_y, failed_peak = self._failed_search
                same_viewpoint = hypot(pose.x - failed_x, pose.y - failed_y) < (
                    FAILED_SEARCH_COOLDOWN_RADIUS
                )
                if same_viewpoint and reliable_signal < failed_peak + FAILED_SEARCH_RETRY_GAIN:
                    self.log(
                        f'Сигнал {reading.value:.2f} не сильнее прошлого пика '
                        f'{failed_peak:.2f}; меняю точку обзора'
                    )
                    break
            radius = local_search_radius(reading)
            self.log(
                f'Слышу сигнал {reading.value:.2f} в '
                f'({pose.x:.2f}; {pose.y:.2f}), ищу в радиусе {radius:.2f} м'
            )
            found = self.skills.search_around(pose.x, pose.y, radius)
            if self.robot.preempted():
                break
            if found.ok and self.skills.collect().ok:
                gained += 1
                self._failed_search = None
                self.log(f'Образец собран: {self.robot.collected()}/{self.robot.samples_total()}')
                failures = 0
            else:
                self._failed_search = (
                    float(found.data.get('x', pose.x)),
                    float(found.data.get('y', pose.y)),
                    max(reliable_signal, float(found.data.get('peak', reading.value))),
                )
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
            result = self.skills.goto(*target, guarded=True, stop_on_signal=True)
            if self.robot.preempted():
                break
            if not result.ok:
                if result.reason == 'sample signal nearby':
                    gained = self._hear_and_collect()
                    if self.robot.preempted():
                        break
                    if gained:
                        # The interrupted coverage point was never reached.  A
                        # successful collection removes the signal that caused
                        # the stop, so it is safe to schedule that point again.
                        pending.append(target)
                        continue
                    # A single moving reading may be a transient/noisy peak.
                    # Resume this leg once without another signal interruption
                    # so a false positive cannot trap the sweep in place.
                    result = self.skills.goto(*target, guarded=True)
                    if self.robot.preempted():
                        break
                    if result.ok:
                        here = self.robot.pose()
                        pending = [
                            p for p in pending
                            if hypot(p[0] - here.x, p[1] - here.y) > 0.7
                        ]
                        self._hear_and_collect()
                        continue
                self.log(f'Пропускаю точку ({target[0]:.1f}; {target[1]:.1f}): {result.reason}')
                continue
            here = self.robot.pose()
            pending = [p for p in pending if hypot(p[0] - here.x, p[1] - here.y) > 0.7]
            self._hear_and_collect()
            if not self.skills.battery_allows_more():
                self.log('Запас батареи на пределе, еду домой')
                break
        if self.robot.preempted():
            return {
                'collected': self.robot.collected(),
                'samples_total': self.robot.samples_total(),
                'battery': round(self.robot.battery(), 2),
                'returned_to_base': False,
                'reason': 'preempted',
            }
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
