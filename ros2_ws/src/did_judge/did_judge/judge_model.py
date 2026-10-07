"""ROS-independent state model used by the level-zero judge."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
import random
from typing import Iterable


@dataclass
class Sample:
    """A collectable sample in Gazebo world coordinates."""

    x: float
    y: float
    collected: bool = False


class JudgeModel:
    """Track world position, battery use, and sample collection."""

    def __init__(
        self,
        *,
        base_x: float,
        base_y: float,
        initial_battery: float,
        battery_cost_per_meter: float,
        collection_radius: float,
        sensor_range: float,
        sensor_noise_stddev: float,
        random_seed: int,
        sample_positions: Iterable[float],
    ) -> None:
        positions = list(sample_positions)
        if len(positions) % 2:
            raise ValueError('sample_positions must contain x/y pairs')

        self.base_x = base_x
        self.base_y = base_y
        self.battery = initial_battery
        self.battery_cost_per_meter = battery_cost_per_meter
        self.collection_radius = collection_radius
        self.sensor_range = sensor_range
        self.sensor_noise_stddev = sensor_noise_stddev
        self.samples = [
            Sample(x, y) for x, y in zip(positions[0::2], positions[1::2])
        ]
        self.world_x = base_x
        self.world_y = base_y
        self.distance_travelled = 0.0
        self.finished = False
        self._last_odom: tuple[float, float] | None = None
        self._random = random.Random(random_seed)

    @property
    def collected_count(self) -> int:
        """Return the number of collected samples."""
        return sum(sample.collected for sample in self.samples)

    def update_odometry(self, odom_x: float, odom_y: float) -> None:
        """Update position and charge from odometry anchored at the start pose."""
        if self._last_odom is not None:
            step = hypot(odom_x - self._last_odom[0], odom_y - self._last_odom[1])
            self.distance_travelled += step
            self.battery = max(
                0.0,
                self.battery - step * self.battery_cost_per_meter,
            )
        self._last_odom = (odom_x, odom_y)
        self.world_x = self.base_x + odom_x
        self.world_y = self.base_y + odom_y

    def nearest_sample(self) -> tuple[Sample | None, float]:
        """Return the nearest uncollected sample and its distance."""
        available = [sample for sample in self.samples if not sample.collected]
        if not available:
            return None, float('inf')
        sample = min(
            available,
            key=lambda item: hypot(item.x - self.world_x, item.y - self.world_y),
        )
        distance = hypot(sample.x - self.world_x, sample.y - self.world_y)
        return sample, distance

    def sample_sensor(self) -> float:
        """Return a noisy, directionless proximity value in the [0, 1] range."""
        _, distance = self.nearest_sample()
        ideal = max(0.0, 1.0 - distance / self.sensor_range)
        noisy = ideal + self._random.gauss(0.0, self.sensor_noise_stddev)
        return min(1.0, max(0.0, noisy))

    def collect(self) -> bool:
        """Collect the nearest sample when it is within the allowed radius."""
        sample, distance = self.nearest_sample()
        if sample is None or distance > self.collection_radius:
            return False
        sample.collected = True
        return True

    def at_base(self) -> bool:
        """Return whether the robot is close enough to the start point."""
        return hypot(self.world_x - self.base_x, self.world_y - self.base_y) <= 0.30

