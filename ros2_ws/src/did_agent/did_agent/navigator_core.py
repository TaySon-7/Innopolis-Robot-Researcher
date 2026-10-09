"""Navigation logic without ROS: plan, follow, react to obstacles, replan."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from math import pi

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
    stamp: float | None = None
    x_offset: float = 0.0

    def angles(self) -> np.ndarray:
        """Return the beam angles in the robot frame."""
        return self.angle_min + self.angle_increment * np.arange(len(self.ranges))


class NavigatorCore:
    """Drive the robot to a world point along the cheapest safe path."""

    def __init__(
        self,
        costmap: CostMap,
        *,
        stop_distance: float = 0.24,
        side_stop_distance: float = 0.20,
        front_half_angle: float = 0.6,
        side_half_angle: float = 1.7,
        imminent_front: float = 0.16,
        imminent_side: float = 0.14,
        max_replans: int = 4,
        stuck_window: float = 4.0,
        stuck_progress: float = 0.05,
        backoff_time: float = 1.0,
        max_scan_age: float = 0.75,
    ) -> None:
        self.costmap = costmap
        self.stop_distance = stop_distance
        self.side_stop_distance = side_stop_distance
        self.front_half_angle = front_half_angle
        self.side_half_angle = side_half_angle
        self.imminent_front = imminent_front
        self.imminent_side = imminent_side
        self._last_cost_replan = -float('inf')
        self.max_replans = max_replans
        self.stuck_window = stuck_window
        self.stuck_progress = stuck_progress
        self.backoff_time = backoff_time
        self.max_scan_age = max_scan_age
        self.status = IDLE
        self.reason = ''
        self.goal: tuple[float, float] | None = None
        self.waypoints: list[tuple[float, float]] = []
        self.replans = 0
        self._follower: WaypointFollower | None = None
        self._planned_version = -1
        self.last_stop: tuple[float, float] | None = None
        self._backoff_until: float | None = None
        self._backoff_reason = 'obstacle'
        self._progress_pose = (0.0, 0.0)
        self._progress_time = 0.0

    # --- goal handling -------------------------------------------------------------

    def set_goal(self, goal: tuple[float, float], pose: Pose, t: float) -> bool:
        """Plan to a new goal; return False (status failed) if there is no path."""
        self.goal = goal
        self.replans = 0
        self._backoff_until = None
        self.last_stop = None
        self._last_cost_replan = -float('inf')
        self._progress_pose = (pose.x, pose.y)
        self._progress_time = t
        return self._plan(pose, t)

    def cancel(self) -> None:
        """Stop navigating."""
        self.status = IDLE
        self.reason = ''
        self._follower = None
        self._backoff_until = None

    def _fail(self, reason: str) -> Command:
        self.status, self.reason = FAILED, reason
        self._follower = None
        self._backoff_until = None
        return Command()

    def _plan(self, pose: Pose, t: float) -> bool:
        waypoints = plan_waypoints(self.costmap, (pose.x, pose.y), self.goal)
        if waypoints is None:
            self.status, self.reason = FAILED, 'no path to goal'
            self._follower = None
            return False
        # If the start was snapped out of an inflated obstacle, first reach
        # that safe cell instead of cutting straight to the following corner.
        free_start = self.costmap.is_free(*self.costmap.world_to_cell(pose.x, pose.y))
        self.waypoints = (waypoints[1:] or waypoints) if free_start else waypoints
        self._follower = WaypointFollower(self.waypoints)
        self._planned_version = self.costmap.version
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

    def _rear_clear(self, scan: Scan) -> bool:
        angles = (scan.angles() + pi) % (2.0 * pi) - pi
        rear = np.abs(angles) >= pi - self.side_half_angle
        ranges = np.asarray(scan.ranges, dtype=float)
        usable = np.isposinf(ranges) | (np.isfinite(ranges) & (ranges >= scan.range_min))
        # No rear coverage, a blind beam, or a close hit all forbid reversing.
        return bool(rear.any() and usable[rear].all()
                    and (ranges[rear] > self.stop_distance).all())

    # --- control loop -----------------------------------------------------------------

    def obstacles_ahead(
        self,
        pose: Pose,
        scan: Scan,
    ) -> tuple[str | None, list[tuple[float, float]]]:
        """Judge what the lidar sees ahead.

        Returns ('unknown', points) for an obstacle that the map does not
        explain (stop early and remember it), ('imminent', points) for a known
        wall inside the same safety clearance, else (None, []).
        """
        wrapped = np.array([wrap_angle(float(a)) for a in scan.angles()])
        ranges = np.asarray(scan.ranges, dtype=float)
        # Gazebo clips near hits to range_min. Dropping equality makes an
        # obstacle disappear exactly when Burger reaches it.
        valid = np.isfinite(ranges) & (ranges >= scan.range_min)
        ahead = valid & (np.abs(wrapped) <= self.side_half_angle)
        if not ahead.any():
            return None, []
        front = np.abs(wrapped) <= self.front_half_angle

        stop_range = np.where(front, self.stop_distance, self.side_stop_distance)
        candidates = ahead & (ranges <= stop_range)
        if not candidates.any():
            return None, []
        angle = pose.yaw + wrapped
        safe = np.where(valid, ranges, 0.0)  # inf would poison the map lookup
        origin_x = pose.x + scan.x_offset * np.cos(pose.yaw)
        origin_y = pose.y + scan.x_offset * np.sin(pose.yaw)
        points = np.stack([origin_x + safe * np.cos(angle), origin_y + safe * np.sin(angle)], 1)
        unexplained = candidates & ~self.costmap.near_known_solid(points)
        close = ahead & (ranges < max(self.stop_distance * 2.5, 0.5))
        self.last_stop = (
            float(ranges[candidates].min()),
            float(wrapped[candidates][np.argmin(ranges[candidates])]),
        )
        return ('unknown' if unexplained.any() else 'imminent',
                [(float(x), float(y)) for x, y in points[close]])

    def update(self, pose: Pose, scan: Scan | None, t: float) -> Command:
        """Return the velocity command for the current state."""
        if self.status != RUNNING or self._follower is None:
            return Command()

        if scan is None:
            return self._fail('missing lidar scan')
        # /clock and /scan callbacks can arrive in either order within a tick.
        if scan.stamp is not None and not -0.1 <= t - scan.stamp <= self.max_scan_age:
            return self._fail('stale lidar scan')
        ranges = np.asarray(scan.ranges, dtype=float)
        usable = np.isposinf(ranges) | (np.isfinite(ranges) & (ranges >= scan.range_min))
        if not usable.any():
            return self._fail('lidar scan has no usable ranges')

        if self._backoff_until is not None:
            if t < self._backoff_until:
                if not self._rear_clear(scan):
                    return self._fail('obstacle or missing scan coverage behind robot; cannot back off')
                return Command(-0.08, 0.0)
            self._backoff_until = None
            if not self._replan(pose, t, self._backoff_reason):
                return Command()

        if self.costmap.version != self._planned_version and t - self._last_cost_replan > 1.5:
            self._last_cost_replan = t
            if not self._plan(pose, t):
                return Command()

        command, reached = self._follower.update(pose)
        if reached:
            self.status, self.reason = DONE, ''
            return Command()

        # Use physical displacement: a valid detour may lead away from the
        # goal. Map updates and their replans must not reset this watchdog.
        distance = hypot(pose.x - self._progress_pose[0], pose.y - self._progress_pose[1])
        if distance >= self.stuck_progress:
            self._progress_pose, self._progress_time = (pose.x, pose.y), t
        elif t - self._progress_time > self.stuck_window:
            self._progress_time = t
            self._backoff_until = t + self.backoff_time
            self._backoff_reason = 'no progress'
            return Command()

        if command.linear > 0.0:
            angles = (scan.angles() + pi) % (2.0 * pi) - pi
            ahead = np.abs(angles) <= self.side_half_angle
            if not ahead.any() or not usable[ahead].all():
                return self._fail('lidar scan has missing forward coverage')
            kind, points = self.obstacles_ahead(pose, scan)
            if kind is not None:
                # Remember the measured surface even if the static map knows
                # it: the executed trajectory has exhausted its clearance.
                self.costmap.add_obstacles(points)
                self._backoff_until = t + self.backoff_time
                self._backoff_reason = 'obstacle'
                return Command()
        elif command.angular:
            if not usable.all():
                return self._fail('lidar scan has missing coverage for turning')
            if np.any(np.isfinite(ranges)
                      & (ranges <= max(self.imminent_front, self.imminent_side))):
                return self._fail('insufficient obstacle clearance to turn safely')
        return command

    def distance_to_goal(self, pose: Pose) -> float:
        """Return the straight-line distance to the requested goal."""
        if self.goal is None:
            return float('inf')
        return hypot(self.goal[0] - pose.x, self.goal[1] - pose.y)
