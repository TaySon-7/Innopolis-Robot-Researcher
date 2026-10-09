"""Navigation logic without ROS: plan, follow, react to obstacles, replan."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos
from math import hypot
from math import sin

import numpy as np

from did_agent.controller import Command
from did_agent.controller import Pose
from did_agent.controller import WaypointFollower
from did_agent.controller import wrap_angle
from did_agent.costmap import CostMap
from did_agent.planner import plan_waypoints

IDLE = 'idle'
RUNNING = 'running'
DONE = 'done'
FAILED = 'failed'


@dataclass
class Scan:
    """A planar lidar scan."""

    angle_min: float
    angle_increment: float
    ranges: np.ndarray
    range_min: float = 0.12

    def angles(self) -> np.ndarray:
        """Return the beam angles in the robot frame."""
        return self.angle_min + self.angle_increment * np.arange(len(self.ranges))


class NavigatorCore:
    """Drive the robot to a world point along the cheapest safe path."""

    def __init__(
        self,
        costmap: CostMap,
        *,
        stop_distance: float = 0.20,
        side_stop_distance: float = 0.17,
        front_half_angle: float = 0.6,
        side_half_angle: float = 1.7,
        imminent_front: float = 0.16,
        imminent_side: float = 0.14,
        max_replans: int = 4,
        stuck_window: float = 10.0,
        stuck_progress: float = 0.05,
        backoff_time: float = 1.0,
    ) -> None:
        self.costmap = costmap
        self.stop_distance = stop_distance
        self.side_stop_distance = side_stop_distance
        self.front_half_angle = front_half_angle
        self.side_half_angle = side_half_angle
        self.imminent_front = imminent_front
        self.imminent_side = imminent_side
        self._last_imminent_replan = -float('inf')
        self._last_cost_replan = -float('inf')
        self._last_expiry = -float('inf')
        self.max_replans = max_replans
        self.stuck_window = stuck_window
        self.stuck_progress = stuck_progress
        self.backoff_time = backoff_time
        self.status = IDLE
        self.reason = ''
        self.goal: tuple[float, float] | None = None
        self.waypoints: list[tuple[float, float]] = []
        self.replans = 0
        self._follower: WaypointFollower | None = None
        self._planned_version = -1
        self.last_stop: tuple[float, float] | None = None
        self._backoff_until: float | None = None
        self._progress_best = float('inf')
        self._progress_time = 0.0

    # --- goal handling -------------------------------------------------------------

    def set_goal(self, goal: tuple[float, float], pose: Pose, t: float) -> bool:
        """Plan to a new goal; return False (status failed) if there is no path."""
        self.goal = goal
        self.replans = 0
        self._backoff_until = None
        return self._plan(pose, t)

    def cancel(self) -> None:
        """Stop navigating."""
        self.status = IDLE
        self.reason = ''
        self._follower = None

    def _plan(self, pose: Pose, t: float) -> bool:
        waypoints = plan_waypoints(self.costmap, (pose.x, pose.y), self.goal)
        if waypoints is None:
            self.status, self.reason = FAILED, 'no path to goal'
            self._follower = None
            return False
        # The first waypoint is the cell we stand in; start towards the second.
        self.waypoints = waypoints[1:] or waypoints
        self._follower = WaypointFollower(self.waypoints)
        self._planned_version = self.costmap.version
        self._progress_best = self._follower.distance_to_goal(pose)
        self._progress_time = t
        self.status, self.reason = RUNNING, ''
        return True

    def _replan(self, pose: Pose, t: float, reason: str) -> bool:
        self.replans += 1
        if self.replans > self.max_replans:
            detail = ''
            if self.last_stop is not None:
                detail = f' ({self.last_stop[0]:.2f} m at {self.last_stop[1]:.2f} rad)'
            self.status, self.reason = FAILED, f'{reason}{detail}; gave up after replans'
            self._follower = None
            return False
        return self._plan(pose, t)

    # --- control loop -----------------------------------------------------------------

    def obstacles_ahead(
        self,
        pose: Pose,
        scan: Scan,
    ) -> tuple[str | None, list[tuple[float, float]]]:
        """Judge what the lidar sees ahead.

        Returns ('unknown', points) for an obstacle that the map does not
        explain (stop early and remember it), ('imminent', []) when even a known
        wall is about to be touched, else (None, []).
        """
        wrapped = np.array([wrap_angle(float(a)) for a in scan.angles()])
        ranges = np.asarray(scan.ranges, dtype=float)
        valid = np.isfinite(ranges) & (ranges > scan.range_min)
        ahead = valid & (np.abs(wrapped) <= self.side_half_angle)
        if not ahead.any():
            return None, []
        front = np.abs(wrapped) <= self.front_half_angle

        imminent_range = np.where(front, self.imminent_front, self.imminent_side)
        if (ahead & (ranges < imminent_range)).any():
            return 'imminent', []

        stop_range = np.where(front, self.stop_distance, self.side_stop_distance)
        candidates = ahead & (ranges < stop_range)
        if not candidates.any():
            return None, []
        angle = pose.yaw + wrapped
        safe = np.where(valid, ranges, 0.0)  # inf would poison the map lookup
        points = np.stack([pose.x + safe * np.cos(angle), pose.y + safe * np.sin(angle)], 1)
        unexplained = candidates & ~self.costmap.near_known_solid(points)
        if not unexplained.any():
            return None, []
        close = ahead & (ranges < max(self.stop_distance * 2.5, 0.5))
        close &= ~self.costmap.near_known_solid(points)
        self.last_stop = (
            float(ranges[unexplained].min()),
            float(wrapped[unexplained][np.argmin(ranges[unexplained])]),
        )
        return 'unknown', [(float(x), float(y)) for x, y in points[close]]

    def update(self, pose: Pose, scan: Scan | None, t: float) -> Command:
        """Return the velocity command for the current state."""
        if t - self._last_expiry >= 1.0:
            # Lidar sightings go stale: without this a false positive would
            # poison the map forever.  About once a second is enough; freeing
            # cells bumps the version, which can trigger a re-plan below.
            self._last_expiry = t
            self.costmap.expire_dynamic(t)
        if self.status != RUNNING or self._follower is None:
            return Command()

        if self._backoff_until is not None:
            if t < self._backoff_until:
                return Command(-0.08, 0.0)
            self._backoff_until = None
            if not self._replan(pose, t, 'obstacle'):
                return Command()

        if self.costmap.version != self._planned_version and t - self._last_cost_replan > 1.5:
            self._last_cost_replan = t
            if not self._plan(pose, t):
                return Command()

        command, reached = self._follower.update(pose)
        if reached:
            self.status, self.reason = DONE, ''
            return Command()

        distance = self._follower.distance_to_goal(pose)
        if distance < self._progress_best - self.stuck_progress:
            self._progress_best, self._progress_time = distance, t
        elif t - self._progress_time > self.stuck_window:
            if not self._replan(pose, t, 'no progress'):
                return Command()
            return Command()

        if scan is not None and command.linear > 0.0:
            kind, points = self.obstacles_ahead(pose, scan)
            if kind == 'unknown':
                self.costmap.add_obstacles(points, now=t)
                self._backoff_until = t + self.backoff_time
                return Command()
            if kind == 'imminent':
                # A known wall is about to be touched: do not drive forward, but
                # turning in place is fine. Re-plan (at most once a second).
                if t - self._last_imminent_replan > 1.0:
                    self._last_imminent_replan = t
                    if not self._replan(pose, t, 'too close to a wall'):
                        return Command()
                return Command(0.0, command.angular)
        return command

    def distance_to_goal(self, pose: Pose) -> float:
        """Return the straight-line distance to the requested goal."""
        if self.goal is None:
            return float('inf')
        return hypot(self.goal[0] - pose.x, self.goal[1] - pose.y)
