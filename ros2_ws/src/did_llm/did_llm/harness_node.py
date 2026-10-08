#!/usr/bin/env python3
"""Стенд вместо Gazebo: публикует те же топики, что и судья с симулятором.

Нужен для проверки слоя LLM без симулятора. Gazebo и Nav2 на этой машине не
ставить — зеркала не отдают крупные файлы, — а слой LLM от геометрии мира не
зависит: он работает с сообщениями ``/scan``, ``/odom``, ``/did/*``. Стенд
публикует ровно эти сообщения в том же формате, что и настоящий судья.

ЧТО ЭТО НЕ ЯВЛЯЕТСЯ
--------------------
Измерением. Все числа здесь заданы сценарием, а не получены из физики: лидар
рисует заранее известную комнату, расход батареи вычисляется по формуле, а
события выдаются по расписанию. Годится, чтобы проверить, что код правильно
принимает данные и строит план. Не годится, чтобы настроить пороги
проскальзывания, шум датчика или расход батареи — для этого нужен Gazebo.

Формат сообщений повторяет ``did_judge/judge_node.py``; при изменении судьи
менять и здесь.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Float32, String


# --------------------------------------------------------------------------
# Геометрия арены. Взята из turtlebot3_world: свободная область по map.pgm
# простирается по x от -2.85 до 2.55 и по y от -2.50 до 2.55, внутри стоят
# девять колонн 0.35 x 0.35 м на сетке с шагом 1.1 м.
# --------------------------------------------------------------------------
ARENA_X_MIN, ARENA_X_MAX = -2.85, 2.55
ARENA_Y_MIN, ARENA_Y_MAX = -2.50, 2.55
COLUMN_HALF = 0.175
COLUMN_POINTS = [(-1.1, -1.1), (-1.1, 0.0), (-1.1, 1.1),
                 (0.0, -1.1), (0.0, 0.0), (0.0, 1.1),
                 (1.1, -1.1), (1.1, 0.0), (1.1, 1.1)]

START_X, START_Y = -2.0, -0.5

#: Образцы сценария easy: три штуки в свободном месте.
SAMPLE_POINTS = [(-2.2, 1.9), (2.1, 1.9), (2.1, -1.9)]

#: Радиус, с которого датчик образцов начинает показывать сигнал.
SAMPLE_SIGNAL_RANGE = 0.6

#: Расход батареи, процент на метр, по зонам. Значения сценария из
#: scenarios.yaml намеренно НЕ используются: они не объявляются агенту и
#: нужны только судье. Здесь они заданы, чтобы стенд был предсказуем.
CHEAP_DRAIN_PCT_PER_M = 1.0
DEAR_DRAIN_PCT_PER_M = 3.0

#: Квадрат, в котором пол дорогой. Границы заданы явно, чтобы стенд давал
#: один и тот же результат при любом порядке обхода.
DEAR_ZONE_X_MIN, DEAR_ZONE_X_MAX = -0.6, 0.6
DEAR_ZONE_Y_MIN, DEAR_ZONE_Y_MAX = -2.2, 2.2

LIDAR_RANGE_MIN = 0.12
LIDAR_RANGE_MAX = 3.5
LIDAR_ANGLE_MIN = -1.0472        # TurtleBot3 LDS-01: +-60 градусов
LIDAR_ANGLE_MAX = 1.0472
LIDAR_ANGLE_INCREMENT = (LIDAR_ANGLE_MAX - LIDAR_ANGLE_MIN) / 359


def drain_at(x: float, y: float) -> float:
    """Батарея на метр в точке."""
    if DEAR_ZONE_X_MIN <= x <= DEAR_ZONE_X_MAX and \
            DEAR_ZONE_Y_MIN <= y <= DEAR_ZONE_Y_MAX:
        return DEAR_DRAIN_PCT_PER_M
    return CHEAP_DRAIN_PCT_PER_M


@dataclass
class Sample:
    x: float
    y: float
    collected: bool = False


@dataclass
class Harness:
    """Состояние стенда: поза, батарея, образцы, события."""

    x: float = START_X
    y: float = START_Y
    yaw: float = 0.0
    battery: float = 100.0
    distance: float = 0.0
    samples: list[Sample] = field(default_factory=lambda: [
        Sample(x, y) for x, y in SAMPLE_POINTS
    ])

    @property
    def collected(self) -> int:
        return sum(1 for s in self.samples if s.collected)

    def advance(self, vx: float, wz: float, dt: float) -> None:
        """Один шаг интегрирования: точка в подоле робота."""
        self.yaw += wz * dt
        step = vx * dt
        nx = self.x + step * math.cos(self.yaw)
        ny = self.y + step * math.sin(self.yaw)
        # Столкновение со стеной или колонной: робот остаётся на месте, как
        # в симуляторе при невозможности проехать.
        if not self._free(nx, ny):
            return
        self.x, self.y = nx, ny
        self.distance += abs(step)
        self.battery = max(0.0, self.battery - abs(step) * drain_at(nx, ny))

    @staticmethod
    def _free(x: float, y: float) -> bool:
        if not (ARENA_X_MIN <= x <= ARENA_X_MAX
                and ARENA_Y_MIN <= y <= ARENA_Y_MAX):
            return False
        for cx, cy in COLUMN_POINTS:
            if abs(x - cx) < COLUMN_HALF and abs(y - cy) < COLUMN_HALF:
                return False
        return True

    def sample_signal(self) -> float:
        """Сигнал датчика образцов: тем выше, чем ближе не взятый образец."""
        best = 0.0
        for sample in self.samples:
            if sample.collected:
                continue
            distance = math.hypot(self.x - sample.x, self.y - sample.y)
            if distance < SAMPLE_SIGNAL_RANGE:
                best = max(best, 1.0 - distance / SAMPLE_SIGNAL_RANGE)
        return round(best, 4)

    def try_collect(self) -> bool:
        for sample in self.samples:
            if not sample.collected and math.hypot(
                    self.x - sample.x, self.y - sample.y) < 0.3:
                sample.collected = True
                return True
        return False

    def score(self, scenario: str) -> dict:
        return {
            'scenario': scenario,
            'battery': round(self.battery, 3),
            'collected': self.collected,
            'samples_total': len(self.samples),
            'distance_travelled': round(self.distance, 3),
            'finished': self.collected >= len(self.samples),
            'world_pose': {'x': round(self.x, 3), 'y': round(self.y, 3)},
        }


class HarnessNode(Node):
    """Публикует топики стенда и слушает /cmd_vel."""

    def __init__(self) -> None:
        super().__init__('harness')
        self.harness = Harness()
        self.scenario = self.declare_parameter('scenario', 'easy').value

        qos = QoSProfile(depth=10)
        self.battery_pub = self.create_publisher(
            Float32, '/did/battery', qos)
        self.signal_pub = self.create_publisher(
            Float32, '/did/sample_sensor', qos)
        self.score_pub = self.create_publisher(String, '/did/score', qos)
        self.events_pub = self.create_publisher(String, '/did/events', qos)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', qos)
        self.odom_pub = self.create_publisher(Odometry, '/odom', qos)
        self.joints_pub = self.create_publisher(
            JointState, '/joint_states', qos)

        self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, qos)

        # События выдаются по расписанию, чтобы гипотеза Г4 проверялась
        # детерминированно: одно столкновение и одна опасная зона.
        self.tick_count = 0
        self.sent_collision = False
        self.sent_hazard = False

        self.create_timer(0.1, self._publish)

    # ------------------------------------------------------------------ вход
    def _on_cmd_vel(self, message: Twist) -> None:
        self.harness.advance(message.linear.x, message.angular.z, 0.1)

    # --------------------------------------------------------------- выходы
    def _publish(self) -> None:
        self.tick_count += 1
        h = self.harness

        battery = Float32()
        battery.data = round(h.battery, 4)
        self.battery_pub.publish(battery)

        signal = Float32()
        signal.data = h.sample_signal()
        self.signal_pub.publish(signal)

        score = String()
        score.data = json.dumps(h.score(self.scenario), sort_keys=True)
        self.score_pub.publish(score)

        self.scan_pub.publish(self._scan())
        self.odom_pub.publish(self._odom())
        self.joints_pub.publish(self._joints())

        if h.try_collect():
            self._event('sample_collected',
                        world_pose=h.score(self.scenario)['world_pose'])

        # При въезде в дорогую зону один раз за прогон — «опасная зона».
        if not self.sent_collision and h.distance > 3.0 and \
                abs(h.x) < 0.3 and h.battery < 100.0:
            self.sent_collision = True
            self._event('collision')

        if not self.sent_hazard and h.collected >= 1:
            self.sent_hazard = True
            self._event('hazard_hit')

    def _event(self, event: str, **details: object) -> None:
        message = String()
        message.data = json.dumps({'event': event, **details}, sort_keys=True)
        self.events_pub.publish(message)
        self.get_logger().info(f'событие: {event}')

    def _scan(self) -> LaserScan:
        """Лидар: лучи по углам до ближайшей стены или колонны."""
        h = self.harness
        message = LaserScan()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'base_scan'
        message.angle_min = LIDAR_ANGLE_MIN
        message.angle_max = LIDAR_ANGLE_MAX
        message.angle_increment = LIDAR_ANGLE_INCREMENT
        message.time_increment = 0.0
        message.scan_time = 0.1
        message.range_min = LIDAR_RANGE_MIN
        message.range_max = LIDAR_RANGE_MAX

        rays = []
        for index in range(360):
            angle = h.yaw + LIDAR_ANGLE_MIN + index * LIDAR_ANGLE_INCREMENT
            dx, dy = math.cos(angle), math.sin(angle)
            best = LIDAR_RANGE_MAX
            for cx, cy in COLUMN_POINTS:
                hit = _ray_box(h.x, h.y, dx, dy,
                               cx - COLUMN_HALF, cy - COLUMN_HALF,
                               cx + COLUMN_HALF, cy + COLUMN_HALF)
                if hit is not None and hit < best:
                    best = hit
            for wall in ('x', 'y'):
                hit = _ray_bounds(h.x, h.y, dx, dy, wall)
                if hit is not None and hit < best:
                    best = hit
            rays.append(round(best, 3))
        message.ranges = [float(value) for value in rays]
        return message

    def _odom(self) -> Odometry:
        h = self.harness
        message = Odometry()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'odom'
        message.child_frame_id = 'base_footprint'
        message.pose.pose.position.x = h.x - START_X
        message.pose.pose.position.y = h.y - START_Y
        half = h.yaw / 2.0
        message.pose.pose.orientation.z = math.sin(half)
        message.pose.pose.orientation.w = math.cos(half)
        return message

    def _joints(self) -> JointState:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = ['left_wheel_joint', 'right_wheel_joint']
        message.position = [0.0, 0.0]
        return message


def _ray_box(ox: float, oy: float, dx: float, dy: float,
             x_min: float, y_min: float, x_max: float, y_max: float):
    """Длина луча до прямоугольника или None.

    Классический slab-клиппинг: луч входит в прямоугольник, если отрезок
    пересечения по обеим осям не пуст. Важно не схлопывать отрицательное
    пересечение в ноль — прямоугольник за лучом не пересекается вовсе, и
    возвращать для него 0.0 значит «стена вплотную», что обнуляет весь скан.
    """
    t_enter, t_exit = 0.0, float('inf')
    for origin, direction, low, high in (
            (ox, dx, x_min, x_max), (oy, dy, y_min, y_max)):
        if abs(direction) < 1e-9:
            # Луч параллелен этой оси: попадание возможно, только если он уже
            # внутри полосы.
            if origin < low or origin > high:
                return None
            continue
        t1 = (low - origin) / direction
        t2 = (high - origin) / direction
        if t1 > t2:
            t1, t2 = t2, t1
        t_enter = max(t_enter, t1)
        t_exit = min(t_exit, t2)
        if t_enter > t_exit:
            return None
    return t_enter if t_exit >= t_enter else None


def _ray_bounds(ox: float, oy: float, dx: float, dy: float, wall: str):
    """Длина луча до стены арены по одной из осей, None если параллелен."""
    if wall == 'x':
        origin, direction = ox, dx
        low, high = ARENA_X_MIN, ARENA_X_MAX
    else:
        origin, direction = oy, dy
        low, high = ARENA_Y_MIN, ARENA_Y_MAX

    if direction > 1e-9:
        distance = (high - origin) / direction
    elif direction < -1e-9:
        distance = (low - origin) / direction
    else:
        return None
    return distance if distance > 0 else None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HarnessNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()