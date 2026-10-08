"""Waypoint follower for a differential-drive robot."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2
from math import cos
from math import hypot
from math import pi


def wrap_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (angle + pi) % (2.0 * pi) - pi


@dataclass
class Pose:
    """Robot pose in world coordinates."""

    x: float
    y: float
    yaw: float


@dataclass
class Command:
    """Velocity command: forward speed in m/s, turn rate in rad/s."""

    linear: float = 0.0
    angular: float = 0.0


class WaypointFollower:
    """Drive through world waypoints: turn in place, then go forward."""

    def __init__(
        self,
        waypoints: list[tuple[float, float]],
        *,
        v_max: float = 0.18,
        w_max: float = 1.2,
        waypoint_tolerance: float = 0.12,
        goal_tolerance: float = 0.08,
        turn_in_place_angle: float = 0.5,
    ) -> None:
        self.waypoints = list(waypoints)
        self.v_max = v_max
        self.w_max = w_max
        self.waypoint_tolerance = waypoint_tolerance
        self.goal_tolerance = goal_tolerance
        self.turn_in_place_angle = turn_in_place_angle
        self.index = 0

    @property
    def goal(self) -> tuple[float, float]:
        """Return the last waypoint."""
        return self.waypoints[-1]

    def distance_to_goal(self, pose: Pose) -> float:
        """Return the straight-line distance to the last waypoint."""
        return hypot(self.goal[0] - pose.x, self.goal[1] - pose.y)

    def update(self, pose: Pose) -> tuple[Command, bool]:
        """Return the next command and whether the goal has been reached."""
        while self.index < len(self.waypoints):
            target = self.waypoints[self.index]
            distance = hypot(target[0] - pose.x, target[1] - pose.y)
            last = self.index == len(self.waypoints) - 1
            tolerance = self.goal_tolerance if last else self.waypoint_tolerance
            if distance > tolerance:
                break
            self.index += 1
        else:
            return Command(), True

        error = wrap_angle(atan2(target[1] - pose.y, target[0] - pose.x) - pose.yaw)
        angular = max(-self.w_max, min(self.w_max, 2.5 * error))
        if abs(error) > self.turn_in_place_angle:
            return Command(0.0, angular), False
        speed = self.v_max
        if self.index == len(self.waypoints) - 1:
            speed = min(speed, max(0.04, 0.8 * distance))
        return Command(speed * max(0.3, cos(error)), angular), False
