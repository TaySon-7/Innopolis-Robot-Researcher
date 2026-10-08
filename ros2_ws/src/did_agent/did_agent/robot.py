"""What the skills need from a robot, independent of ROS or simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Protocol

from did_agent.controller import Pose
from did_agent.costmap import CostMap

BASE = (-2.0, -0.5)


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
    ) -> NavResult: ...

    def read_sensor(self, count: int = 5) -> Reading: ...

    def noise_level(self) -> float: ...

    def anomaly(self) -> dict[str, bool]: ...

    def collect(self) -> tuple[bool, str]: ...

    def finish(self) -> tuple[bool, str]: ...

    def battery(self) -> float: ...

    def samples_total(self) -> int: ...

    def collected(self) -> int: ...
