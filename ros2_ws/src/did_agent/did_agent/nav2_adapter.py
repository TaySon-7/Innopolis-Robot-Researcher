"""Nav2 action adapter, observed-map bridge and exclusive velocity gate.

Nav2 never writes the robot's /cmd_vel. Only a currently accepted action may
provide commands to Navigator.goto; stopping closes that gate immediately,
including while an action's acceptance or cancellation is still in flight.
"""

from copy import deepcopy
from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, sin
import time
from typing import Any

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import TransformStamped, TwistStamped
from lifecycle_msgs.srv import GetState
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
import rclpy
from rclpy.action import ActionClient
from rclpy.clock import Clock, ClockType
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from did_agent.controller import Command, wrap_angle
from did_agent.nav2_core import checked_velocity, knowledge_grid
from did_agent.navigator_core import DONE, FAILED, IDLE, RUNNING


@dataclass
class _Request:
    generation: int
    sent: Any = None
    handle: Any = None
    result: Any = None
    cancelled: bool = False

    def finished(self) -> bool:
        if self.sent is None or not self.sent.done():
            return False
        if self.sent.exception() is not None:
            return True
        handle = self.sent.result()
        return not handle.accepted or (self.result is not None and self.result.done())


class Nav2Adapter:
    def __init__(self, node) -> None:
        self.node = node
        self.client = ActionClient(node, NavigateToPose, '/nav2/navigate_to_pose')
        self.status, self.reason = IDLE, ''
        self.goal = None
        self.waypoints: list[tuple[float, float]] = []
        self.replans = 0
        self.recoveries = 0
        self._path_signature = None
        self._generation = 0
        self._accepted = False
        self._started = 0.0
        self._velocity: tuple[Command, float] | None = None
        self._requests: list[_Request] = []
        self._map_signature = None
        self._previous_pose = None
        self._clear_futures = []
        self._lifecycle_active = False
        self._lifecycle_checked = 0.0
        self._lifecycle_future = None
        self._lifecycle_requested = 0.0
        self._lifecycle_client = node.create_client(GetState, '/nav2/bt_navigator/get_state')
        self._map_pub = node.create_publisher(OccupancyGrid, '/nav2/knowledge_map', QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._scan_pub = node.create_publisher(LaserScan, '/nav2/scan', qos_profile_sensor_data)
        self._odom_pub = node.create_publisher(Odometry, '/nav2/odom', 10)
        self._tf = TransformBroadcaster(node)
        self._static_tf = StaticTransformBroadcaster(node)
        transform = TransformStamped()
        transform.header.stamp = node.get_clock().now().to_msg()
        transform.header.frame_id, transform.child_frame_id = 'nav2_base_link', 'nav2_scan'
        transform.transform.translation.x = -0.032
        transform.transform.translation.z = 0.172
        transform.transform.rotation.w = 1.0
        self._static_tf.sendTransform(transform)
        node.create_subscription(TwistStamped, '/nav2/cmd_vel', self._on_velocity, 10)
        node.create_subscription(Path, '/nav2/plan', self._on_path, 10)
        self._clear_clients = [node.create_client(ClearEntireCostmap, topic) for topic in (
            '/nav2/global_costmap/clear_entirely_global_costmap',
            '/nav2/local_costmap/clear_entirely_local_costmap')]
        node.create_timer(0.5, self.publish_map)
        # Service availability is a wall-clock health check, independent of
        # how quickly Gazebo advances (and of pauses in the simulation).
        self._health_clock = Clock(clock_type=ClockType.STEADY_TIME)
        node.create_timer(1.0, self._check_lifecycle, clock=self._health_clock)
        self.publish_map()

    def publish_map(self) -> None:
        costmap = self.node.costmap
        data = knowledge_grid(costmap)
        signature = (data.shape, data.tobytes(), costmap.resolution,
                     costmap.grid.origin_x, costmap.grid.origin_y)
        if signature == self._map_signature:
            return
        message = OccupancyGrid()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.header.frame_id = 'odom'
        message.info.resolution = float(costmap.resolution)
        message.info.height, message.info.width = data.shape
        message.info.origin.position.x = float(costmap.grid.origin_x)
        message.info.origin.position.y = float(costmap.grid.origin_y)
        message.info.origin.orientation.w = 1.0
        message.data = data.ravel().tolist()
        self._map_pub.publish(message)
        self._map_signature = signature

    def on_pose(self, message) -> None:
        """Separate TF branch prevents wheel drift from moving the world map."""
        transform = TransformStamped()
        transform.header.stamp = message.header.stamp
        transform.header.frame_id, transform.child_frame_id = 'odom', 'nav2_base_link'
        transform.transform.translation.x = message.pose.position.x
        transform.transform.translation.y = message.pose.position.y
        transform.transform.rotation = message.pose.orientation
        self._tf.sendTransform(transform)
        odometry = Odometry()
        odometry.header = deepcopy(transform.header)
        odometry.child_frame_id = 'nav2_base_link'
        odometry.pose.pose = deepcopy(message.pose)
        t = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        q = message.pose.orientation
        yaw = atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        x, y = message.pose.position.x, message.pose.position.y
        previous = self._previous_pose
        if previous is not None and 0 < t - previous[0] < 1.0:
            dt = t - previous[0]
            dx, dy = x - previous[1], y - previous[2]
            # Ignore a respawn jump; it is not a measured driving velocity.
            if hypot(dx, dy) < 0.2:
                odometry.twist.twist.linear.x = (dx * cos(yaw) + dy * sin(yaw)) / dt
                odometry.twist.twist.angular.z = wrap_angle(yaw - previous[3]) / dt
        self._previous_pose = (t, x, y, yaw)
        self._odom_pub.publish(odometry)

    def on_scan(self, message) -> None:
        converted = deepcopy(message)
        converted.header.frame_id = 'nav2_scan'
        self._scan_pub.publish(converted)

    def _on_velocity(self, message) -> None:
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        if (self.status == RUNNING and self._accepted and stamp >= self._started
                and stamp <= self.node.now() + 0.1
                and (self._velocity is None or stamp >= self._velocity[1])):
            self._velocity = (Command(message.twist.linear.x, message.twist.angular.z), stamp)

    def _on_path(self, message) -> None:
        if self.status != RUNNING or not self._accepted or message.header.frame_id != 'odom':
            return
        points = [(p.pose.position.x, p.pose.position.y) for p in message.poses]
        signature = tuple((round(x, 2), round(y, 2)) for x, y in points)
        if self._path_signature is not None and signature != self._path_signature:
            self.replans += 1
        self._path_signature = signature
        self.waypoints = points

    def _cancel_request(self, request: _Request) -> None:
        if request.handle is not None and request.handle.accepted and not request.cancelled:
            request.cancelled = True
            request.handle.cancel_goal_async()

    def cancel(self) -> None:
        self._generation += 1
        self._accepted = False
        self._velocity = None
        self.node.publish(Command())
        for request in self._requests:
            self._cancel_request(request)
        if self.status == RUNNING:
            self.status, self.reason = FAILED, 'preempted'

    def reset(self) -> None:
        self.cancel()
        self.status, self.reason = IDLE, ''
        self.waypoints = []
        self._map_signature = self._previous_pose = None
        self.publish_map()
        self._clear_futures = [client.call_async(ClearEntireCostmap.Request())
                               for client in self._clear_clients if client.service_is_ready()]

    def _accepted_goal(self, request: _Request, future) -> None:
        try:
            request.handle = future.result()
            if not request.handle.accepted:
                return
            request.result = request.handle.get_result_async()
            if request.generation != self._generation:
                self._cancel_request(request)
            else:
                self._accepted = True
        except Exception as error:
            if request.generation == self._generation:
                self.status, self.reason = FAILED, f'Nav2 goal acceptance failed: {error}'

    def _feedback(self, generation, message) -> None:
        if generation == self._generation:
            self.recoveries = int(message.feedback.number_of_recoveries)

    def ready(self) -> bool:
        return (self.client.server_is_ready() and self._lifecycle_active
                and time.monotonic() - self._lifecycle_checked < 5.0)

    def _check_lifecycle(self) -> None:
        now = time.monotonic()
        pending = self._lifecycle_future
        if pending is not None:
            if pending.done():
                self._lifecycle_result(pending)
            elif now - self._lifecycle_requested < 2.0:
                return
            else:
                # A response can disappear when Nav2 restarts. Do not let one
                # lost request block every subsequent health poll. Invalidate
                # its identity before cancellation schedules its callback.
                self._lifecycle_future = None
                self._lifecycle_active = False
                self._lifecycle_client.remove_pending_request(pending)
                pending.cancel()
        if not self._lifecycle_client.service_is_ready():
            self._lifecycle_active = False
            return
        try:
            self._lifecycle_future = self._lifecycle_client.call_async(GetState.Request())
        except Exception:
            self._lifecycle_active = False
            self._lifecycle_future = None
            return
        self._lifecycle_requested = now
        self._lifecycle_future.add_done_callback(self._lifecycle_result)

    def _lifecycle_result(self, future) -> None:
        if future is not self._lifecycle_future:
            return
        self._lifecycle_future = None
        try:
            self._lifecycle_active = future.result().current_state.id == 3
        except Exception:
            self._lifecycle_active = False
        self._lifecycle_checked = time.monotonic()

    def snapshot(self) -> dict:
        ready = self.ready() and self.node.ready()
        return {'status': self.status, 'replans': self.replans, 'recoveries': self.recoveries,
                'waypoints': self.waypoints[:80] if self.status == RUNNING else [],
                'ready': ready, 'reason': self.reason if ready else 'Nav2 or robot sensors not ready'}

    def _result(self) -> dict:
        pose = self.node.pose()
        distance = hypot(pose.x - self.goal[0], pose.y - self.goal[1]) if pose and self.goal else 0.0
        return {'status': self.status, 'reason': self.reason,
                'distance_to_goal': round(distance, 3), 'replans': self.replans,
                'backend': 'nav2', 'recoveries': self.recoveries}

    def _finish(self, status: str, reason: str = '') -> dict:
        self.cancel()
        self.status, self.reason = status, reason
        self.node.publish(Command())
        return self._result()

    def goto(self, x: float, y: float, timeout: float, guard=None) -> dict:
        self.cancel()
        self.goal = (x, y)
        self.replans = self.recoveries = 0
        self.waypoints, self._path_signature = [], None
        if not all(isfinite(v) for v in (x, y, timeout)) or timeout <= 0:
            return self._finish(FAILED, 'invalid navigation goal or timeout')
        if not self.node.wait_for_sensors():
            return self._finish(FAILED, 'no physical world pose or scan')
        if self.node.preempted():
            return self._finish(FAILED, 'preempted')
        if not self.node.costmap.is_free(*self.node.costmap.world_to_cell(x, y)):
            return self._finish(FAILED, 'goal is outside observed free space')
        # A late acceptance from an old action must be cancelled and finish
        # before another action can open the velocity gate.
        deadline = time.monotonic() + 10.0
        while rclpy.ok():
            self._requests = [request for request in self._requests if not request.finished()]
            if self.ready() and not self._requests and all(f.done() for f in self._clear_futures):
                break
            self.node.publish(Command())
            if self.node.preempted():
                return self._finish(FAILED, 'preempted')
            if time.monotonic() >= deadline:
                return self._finish(FAILED, 'Nav2 unavailable or previous goal cancellation not completed')
            rclpy.spin_once(self.node, timeout_sec=0.05)
        if not rclpy.ok():
            return self._finish(FAILED, 'ROS shutdown')
        self.publish_map()
        self._started = self.node.now()
        self.status, self.reason = RUNNING, ''
        generation = self._generation
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'odom'
        goal.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        goal.pose.pose.orientation.w = 1.0
        request = _Request(generation)
        self._requests.append(request)
        request.sent = self.client.send_goal_async(
            goal, feedback_callback=lambda message: self._feedback(generation, message))
        request.sent.add_done_callback(lambda future: self._accepted_goal(request, future))
        next_guard = self._started
        accepted_deadline = time.monotonic() + 10.0
        last_clock, last_clock_wall = self._started, time.monotonic()
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.05)
            now = self.node.now()
            if self.node.preempted() or generation != self._generation:
                return self._finish(FAILED, 'preempted')
            pose = self.node.pose()
            if pose is None or not all(isfinite(v) for v in (pose.x, pose.y, pose.yaw)):
                return self._finish(FAILED, 'physical world pose unavailable')
            pose_stamp = self.node.world_pose.header.stamp
            pose_age = now - (pose_stamp.sec + pose_stamp.nanosec * 1e-9)
            if not -0.1 <= pose_age <= 0.75:
                return self._finish(FAILED, 'stale physical world pose')
            if self.status == FAILED:
                return self._finish(FAILED, self.reason)
            if request.sent.done():
                if request.sent.exception() is not None:
                    return self._finish(FAILED, 'Nav2 action request failed')
                if not request.sent.result().accepted:
                    return self._finish(FAILED, 'Nav2 rejected the goal')
            if request.result is not None and request.result.done():
                if request.result.exception() is not None:
                    return self._finish(FAILED, 'Nav2 action result unavailable')
                response = request.result.result()
                if response.status == GoalStatus.STATUS_SUCCEEDED:
                    pose = self.node.pose()
                    distance = hypot(pose.x - x, pose.y - y)
                    if not isfinite(distance) or distance > 0.15:
                        return self._finish(FAILED, 'Nav2 reported success outside goal tolerance')
                    return self._finish(DONE)
                code = getattr(response.result, 'error_code', 0)
                detail = getattr(response.result, 'error_msg', '')
                return self._finish(FAILED, f'Nav2 status {response.status}, error {code}: {detail}'.rstrip(': '))
            if not self._accepted and time.monotonic() >= accepted_deadline:
                return self._finish(FAILED, 'Nav2 goal acceptance timeout')
            if now != last_clock:
                last_clock, last_clock_wall = now, time.monotonic()
            elif time.monotonic() - last_clock_wall > 10.0:
                return self._finish(FAILED, 'simulation clock stopped')
            if now - self._started > timeout:
                return self._finish(FAILED, 'timeout')
            if guard is not None and now >= next_guard:
                next_guard = now + 1.0
                if guard():
                    return self._finish(FAILED, 'battery reserve reached')
            command = Command()
            if self._velocity is not None and -0.1 <= now - self._velocity[1] <= 0.5:
                command = self._velocity[0]
            command, error = checked_velocity(command, self.node.scan, now)
            if error:
                return self._finish(FAILED, error)
            self.node.publish(command)
        return self._finish(FAILED, 'ROS shutdown')
