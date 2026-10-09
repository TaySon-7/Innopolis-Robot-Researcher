"""Find a hidden sample from the directionless /did/sample_sensor signal.

The only assumption is that the signal grows as the robot gets closer, so the
search is model-free: sweep a spiral until something is heard, jump to the
signal-weighted centroid, then climb the signal with a shrinking pattern search.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from math import cos
from math import hypot
from math import pi
from math import sin
from math import sqrt
from typing import Callable

from did_agent.robot import Reading
from did_agent.robot import Robot

GOLDEN_ANGLE = 2.399963
SAMPLE_SENSOR_RANGE = 1.5
SAMPLE_COLLECTION_RADIUS = 0.30
# The judge uses signal = 1 - distance / range.  A search must not report
# success outside the radius in which the following collect call is legal.
COLLECTION_SIGNAL_THRESHOLD = 1.0 - SAMPLE_COLLECTION_RADIUS / SAMPLE_SENSOR_RANGE
MAX_LOCAL_SEARCH_TRAVEL = 4.5


class Preempted(Exception):
    """A new plan or a stop request arrived while searching."""


@dataclass
class SearchResult:
    """Where the signal peaked and how strong it was."""

    found: bool
    peak: float
    x: float
    y: float
    reason: str = ''
    trace: list[tuple[float, float, float]] = field(default_factory=list)
    gradient: tuple[float, float] | None = None


class SampleSearch:
    """Spiral sweep, centroid jump, pattern-search refinement."""

    def __init__(
        self,
        robot: Robot,
        *,
        budget_ok: Callable[[], bool] = lambda: True,
        signal: float = 0.08,
        target: float = 0.86,
        found_level: float = COLLECTION_SIGNAL_THRESHOLD,
        spiral_points: int = 12,
        step: float = 0.4,
        min_step: float = 0.1,
        max_refine: int = 18,
        max_travel: float = MAX_LOCAL_SEARCH_TRAVEL,
    ) -> None:
        self.robot = robot
        self.budget_ok = budget_ok
        self.signal = signal
        self.target = target
        self.found_level = found_level
        self.spiral_points = spiral_points
        self.step = step
        self.min_step = min_step
        self.max_refine = max_refine
        self.max_travel = max_travel
        self.trace: list[tuple[float, float, float, float]] = []  # x, y, value, noise
        self.travelled = 0.0

    def _free_point(self, x: float, y: float, snap: float) -> tuple[float, float] | None:
        costmap = self.robot.costmap
        cell = costmap.nearest_free(*costmap.world_to_cell(x, y), snap)
        return None if cell is None else costmap.cell_to_world(*cell)

    def _visit(self, x: float, y: float) -> tuple[float, float, float, float] | None:
        """Drive to a point and read the sensor; None if the point is unreachable."""
        if self.robot.preempted():
            raise Preempted('preempted')
        if self.travelled >= self.max_travel:
            raise Preempted('local search travel budget reached')
        if not self.budget_ok():
            raise Preempted('battery reserve reached')
        before = self.robot.pose()
        result = self.robot.goto(x, y, guard=lambda: not self.budget_ok())
        if self.robot.preempted():
            raise Preempted('preempted')
        if not result.ok:
            return None
        reading: Reading = self.robot.read_sensor(self._readings())
        if self.robot.preempted():
            raise Preempted('preempted')
        pose = self.robot.pose()
        if pose is None:
            raise Preempted('physical world pose unavailable')
        if before is not None:
            self.travelled += hypot(pose.x - before.x, pose.y - before.y)
        entry = (pose.x, pose.y, reading.value, reading.noise)
        self.trace.append(entry)
        return entry

    def _readings(self) -> int:
        """Average more readings when the sensor is noisy (a fault, or bad luck)."""
        return 15 if self.robot.noise_level() > 0.06 else 5

    def _strongest(self) -> tuple[float, float, float, float]:
        return max(self.trace, key=lambda item: item[2])

    def _spiral(self, cx: float, cy: float, radius: float) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for k in range(self.spiral_points):
            r = radius * sqrt((k + 0.5) / self.spiral_points)
            angle = k * GOLDEN_ANGLE
            point = self._free_point(cx + r * cos(angle), cy + r * sin(angle), 0.4)
            if point and all(hypot(point[0] - p[0], point[1] - p[1]) > 0.3 for p in points):
                points.append(point)
        return points

    def _gradient(self, best: tuple[float, float, float, float]) -> tuple[float, float] | None:
        """Unit direction of increasing signal from the readings near the best one."""
        near = [
            item for item in self.trace
            if item[2] >= self.signal and hypot(item[0] - best[0], item[1] - best[1]) <= 1.5
        ]
        if len(near) < 2:
            return None
        n = len(near)
        mx = sum(item[0] for item in near) / n
        my = sum(item[1] for item in near) / n
        mv = sum(item[2] for item in near) / n
        # Ridge regression of v on (x, y): stable even when the points are collinear.
        sxx = sum((i[0] - mx) ** 2 for i in near) + 0.02
        syy = sum((i[1] - my) ** 2 for i in near) + 0.02
        sxy = sum((i[0] - mx) * (i[1] - my) for i in near)
        bx = sum((i[0] - mx) * (i[2] - mv) for i in near)
        by = sum((i[1] - my) * (i[2] - mv) for i in near)
        det = sxx * syy - sxy * sxy
        gx = (syy * bx - sxy * by) / det
        gy = (sxx * by - sxy * bx) / det
        length = hypot(gx, gy)
        if length < 1e-6:
            return None
        return gx / length, gy / length

    def _candidate(
        self,
        best: tuple[float, float, float, float],
        step: float,
        turn: float,
    ) -> tuple[float, float] | None:
        """A free point a step away from the best spot, along the signal gradient."""
        direction = self._gradient(best) or (1.0, 0.0)
        dx = direction[0] * cos(turn) - direction[1] * sin(turn)
        dy = direction[0] * sin(turn) + direction[1] * cos(turn)
        point = self._free_point(best[0] + dx * step, best[1] + dy * step, 0.1)
        if point is None or hypot(point[0] - best[0], point[1] - best[1]) < 0.5 * step:
            return None
        return point

    def run(self, cx: float, cy: float, radius: float) -> SearchResult:
        """Search around a point; the robot ends at the strongest spot found."""
        self.trace = []
        self.travelled = 0.0
        try:
            return self._run(cx, cy, radius)
        except Preempted as stop:
            if self.trace:
                best = self._strongest()
                return SearchResult(
                    False, best[2], best[0], best[1], str(stop),
                    self._plain(), self._gradient(best),
                )
            pose = self.robot.pose()
            return SearchResult(False, 0.0, pose.x if pose else cx, pose.y if pose else cy, str(stop))

    def _plain(self) -> list[tuple[float, float, float]]:
        return [(x, y, v) for x, y, v, _ in self.trace]

    def _run(self, cx: float, cy: float, radius: float) -> SearchResult:
        if self.robot.preempted():
            raise Preempted('preempted')
        pose = self.robot.pose()
        if pose is None:
            raise Preempted('physical world pose unavailable')
        first = self.robot.read_sensor(self._readings())
        if self.robot.preempted():
            raise Preempted('preempted')
        self.trace.append((pose.x, pose.y, first.value, first.noise))

        for x, y in self._spiral(cx, cy, radius):
            if any(item[2] >= self.signal for item in self.trace):
                break
            self._visit(x, y)

        if not any(item[2] >= self.signal for item in self.trace):
            return SearchResult(False, self._strongest()[2], pose.x, pose.y,
                                'no signal', self._plain())

        best = self._strongest()
        step, probes = self.step, 0
        while step >= self.min_step and probes < self.max_refine and best[2] < self.target:
            improved = False
            # Probe the full circle.  In particular, the very first reading has
            # no gradient yet; the former forward-only fan could shrink the step
            # forever while a valid sample sat behind the robot.  Keeping the
            # +/-60 degree probes early also lets the search skirt a pillar
            # without paying for the whole ring in the common case.
            for turn in (
                0.0, pi / 3.0, -pi / 3.0,
                2.0 * pi / 3.0, -2.0 * pi / 3.0, pi,
            ):
                point = self._candidate(best, step, turn)
                if point is None:
                    continue
                entry = self._visit(*point)
                probes += 1
                if entry is not None and entry[2] > best[2] + max(
                    0.02, 2.0 * max(entry[3], best[3])
                ):
                    best = entry
                    # The public sensor model is linear within 1.5 m.  Growing
                    # every successful step to 0.6 m made the robot repeatedly
                    # overshoot a sample after it was already only ~0.3 m away.
                    # Keep fast progress in weak signal, then shrink smoothly
                    # as the estimated remaining distance falls.
                    estimated_distance = SAMPLE_SENSOR_RANGE * (
                        1.0 - max(0.0, min(1.0, best[2]))
                    )
                    step = min(
                        step * 1.3,
                        0.6,
                        max(self.min_step, 0.8 * estimated_distance),
                    )
                    improved = True
                    break
                if probes >= self.max_refine:
                    break
            if not improved:
                step /= 2.0

        pose = self.robot.pose()
        if hypot(pose.x - best[0], pose.y - best[1]) > 0.1:
            self.robot.goto(best[0], best[1])
        found = best[2] >= self.found_level
        return SearchResult(found, best[2], best[0], best[1],
                            '' if found else 'signal too weak', self._plain(),
                            self._gradient(best))
