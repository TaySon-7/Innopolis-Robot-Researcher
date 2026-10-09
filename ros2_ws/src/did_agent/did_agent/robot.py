"""What the skills need from a robot, independent of ROS or simulation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable
from typing import Protocol

from did_agent.controller import Pose
from did_agent.costmap import CostMap

BASE = (-2.0, -0.5)
OPPORTUNISTIC_SIGNAL_MARGIN = 0.20


def sample_signal_near(value: float, noise: float) -> bool:
    """Whether a live reading is strong enough to interrupt a transit leg.

    This is deliberately stricter than the 0.08 threshold used after the robot
    has stopped and averaged a batch.  A moving robot acts on one live sample,
    so it subtracts two noise estimates and waits for a 0.20 margin before
    trading the current route for a local search.
    """
    return (isfinite(value) and isfinite(noise)
            and value - 2.0 * max(0.0, noise) >= OPPORTUNISTIC_SIGNAL_MARGIN)


@dataclass
class NavResult:
    """Outcome of a blocking drive."""

    status: str
    reason: str = ''
    distance_to_goal: float = 0.0
    replans: int = 0

    @property
    def ok(self) -> bool:
        return self.status == 'done'


@dataclass
class Reading:
    """A sample-sensor reading taken while standing still."""

    value: float
    noise: float = 0.0


class Robot(Protocol):
    """Blocking robot interface used by skills, the executor and the agent."""

    costmap: CostMap

    def pose(self) -> Pose: ...

    def now(self) -> float: ...

    def preempted(self) -> bool: ...

    def goto(
        self,
        x: float,
        y: float,
        timeout: float = 180.0,
        guard: Callable[[], bool] | None = None,
        stop_on_signal: bool = False,
    ) -> NavResult: ...

    def read_sensor(self, count: int = 5) -> Reading: ...

    def noise_level(self) -> float: ...

    def anomaly(self) -> dict[str, bool]: ...

    def collect(self) -> tuple[bool, str]: ...

    def finish(self) -> tuple[bool, str]: ...

    def battery(self) -> float: ...

    def samples_total(self) -> int: ...

    def collected(self) -> int: ...
