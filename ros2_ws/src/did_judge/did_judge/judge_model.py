"""ROS-independent state model of the DID judge."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos
from math import hypot
from math import isfinite
from math import sin
from math import sqrt
import random
from typing import Any

from did_judge.scenario import Scenario
from did_judge.scenario import ScenarioSample
from did_judge.scenario import Zone

SAMPLE_REWARD = 10.0
FINISH_REWARD = 5.0
COLLISION_PENALTY = 2.0
FALSE_COLLECT_PENALTY = 1.0
HAZARD_HIT_PENALTY = 3.0
BASE_RADIUS = 0.30

# Distance from the lidar to the robot body (TurtleBot3 Burger, lidar slightly
# behind the centre): ahead the body is closer to the lidar than at the sides.
BODY_AHEAD = 0.125
BODY_SIDE = 0.110
CONTACT_MARGIN = 0.015


def body_distance(angle: float) -> float:
    """Return how far the body surface is from the lidar along a beam angle."""
    return 1.0 / sqrt((cos(angle) / BODY_AHEAD) ** 2 + (sin(angle) / BODY_SIDE) ** 2)


def scan_clearance(angles, ranges, range_min: float = 0.0) -> float | None:
    """Smallest gap between the robot body and anything the lidar sees."""
    gaps = [
        value - body_distance(angle)
        for angle, value in zip(angles, ranges)
        if isfinite(value) and value > range_min
    ]
    return min(gaps) if gaps else None


@dataclass
class SampleState:
    """A sample and whether it has been collected."""

    sample: ScenarioSample
    collected: bool = False


class JudgeModel:
    """Track world position, battery use, penalties and hidden environment."""

    def __init__(
        self,
        scenario: Scenario,
        *,
        collision_clearance: float = CONTACT_MARGIN,
        collision_cooldown: float = 2.0,
    ) -> None:
        self.scenario = scenario
        self.base_x = scenario.base_x
        self.base_y = scenario.base_y
        self.battery = scenario.initial_battery
        self.collection_radius = scenario.collection_radius
        self.sensor_range = scenario.sensor_range
        self.collision_clearance = collision_clearance
        self.collision_cooldown = collision_cooldown

        self.samples = [SampleState(item) for item in scenario.samples]
        self._soil_multiplier = {
            zone.id: zone.cost_multiplier for zone in scenario.soil_zones
        }
        self._soil_zones = list(scenario.soil_zones)
        self._hazard_zones: list[Zone] = list(scenario.hazard_zones)
        self._pending_events = list(scenario.events)
        self._noise_stddev = scenario.sensor_noise_stddev
        self._fault_until: float | None = None
        self._fault_noise = 0.0

        self.world_x = self.base_x
        self.world_y = self.base_y
        self.time = 0.0
        self.distance_travelled = 0.0
        self.finished = False
        self.collisions = 0
        self.false_collects = 0
        self.hazard_hits = 0
        self.environment_log: list[dict[str, Any]] = []

        self._last_odom: tuple[float, float] | None = None
        self._inside_hazards: set[str] = set()
        self._last_collision_time = -float('inf')
        self._random = random.Random(scenario.seed)

    @property
    def collected_count(self) -> int:
        """Return the number of collected samples."""
        return sum(item.collected for item in self.samples)

    @property
    def score(self) -> float:
        """Return the local approximation of the final score."""
        return (
            SAMPLE_REWARD * self.collected_count
            + (FINISH_REWARD if self.finished else 0.0)
            - COLLISION_PENALTY * self.collisions
            - FALSE_COLLECT_PENALTY * self.false_collects
            - HAZARD_HIT_PENALTY * self.hazard_hits
        )

    def advance(self, time: float) -> None:
        """Move the simulation clock and apply due environment events."""
        self.time = time
        while self._pending_events and self._pending_events[0].at <= time:
            event = self._pending_events.pop(0)
            self._apply_event(event)
        if self._fault_until is not None and time >= self._fault_until:
            self._fault_until = None

    def _apply_event(self, event) -> None:
        if event.type == 'soil_change':
            self._soil_multiplier[event.zone] = event.cost_multiplier
            detail = {'zone': event.zone, 'cost_multiplier': event.cost_multiplier}
        elif event.type == 'hazard_appear':
            self._hazard_zones.append(event.zone)
            detail = {'zone': event.zone.id}
        else:
            self._fault_noise = event.noise_stddev
            self._fault_until = (
                self.time + event.duration if event.duration > 0.0 else float('inf')
            )
            detail = {'noise_stddev': event.noise_stddev, 'duration': event.duration}
        # The log is for the judge's own diagnostics: the agent is never told.
        self.environment_log.append({'t': self.time, 'type': event.type, **detail})

    def terrain_multiplier(self, x: float, y: float) -> float:
        """Return the battery cost multiplier of the floor at a world point."""
        value = 1.0
        for zone in self._soil_zones:
            if zone.contains(x, y):
                value = max(value, self._soil_multiplier[zone.id])
        return value

    def update_odometry(self, odom_x: float, odom_y: float) -> list[dict[str, Any]]:
        """Update position and charge from odometry anchored at the start pose.

        Returns the penalty events produced by this step.
        """
        events: list[dict[str, Any]] = []
        new_x = self.base_x + odom_x
        new_y = self.base_y + odom_y
        if self._last_odom is not None:
            step = hypot(odom_x - self._last_odom[0], odom_y - self._last_odom[1])
            middle = ((self.world_x + new_x) / 2.0, (self.world_y + new_y) / 2.0)
            self.distance_travelled += step
            self.battery = max(
                0.0,
                self.battery
                - step
                * self.scenario.battery_cost_per_meter
                * self.terrain_multiplier(*middle),
            )
        self._last_odom = (odom_x, odom_y)
        self.world_x = new_x
        self.world_y = new_y
        events.extend(self._check_hazards())
        return events

    def _check_hazards(self) -> list[dict[str, Any]]:
        events = []
        inside_now = {
            zone.id
            for zone in self._hazard_zones
            if zone.contains(self.world_x, self.world_y)
        }
        for zone in self._hazard_zones:
            if zone.id in inside_now and zone.id not in self._inside_hazards:
                self.hazard_hits += 1
                self.battery = max(0.0, self.battery - zone.penalty)
                events.append({'event': 'hazard_hit', 'zone': zone.id})
        self._inside_hazards = inside_now
        return events

    def report_scan(self, clearance: float) -> dict[str, Any] | None:
        """Register a collision when something touches the robot body.

        ``clearance`` is the smallest gap between the body and the lidar hits
        (see ``scan_clearance``); a gap below a centimetre or so is a contact.
        """
        if clearance > self.collision_clearance:
            return None
        if self.time - self._last_collision_time < self.collision_cooldown:
            return None
        self._last_collision_time = self.time
        self.collisions += 1
        return {'event': 'collision', 'clearance': round(clearance, 3)}

    def nearest_sample(self) -> tuple[SampleState | None, float]:
        """Return the nearest uncollected sample and its distance."""
        available = [item for item in self.samples if not item.collected]
        if not available:
            return None, float('inf')
        state = min(
            available,
            key=lambda item: hypot(
                item.sample.x - self.world_x,
                item.sample.y - self.world_y,
            ),
        )
        distance = hypot(state.sample.x - self.world_x, state.sample.y - self.world_y)
        return state, distance

    @property
    def sensor_noise_stddev(self) -> float:
        """Return the noise level currently applied by the sample sensor."""
        if self._fault_until is not None:
            return self._fault_noise
        return self._noise_stddev

    def sample_sensor(self) -> float:
        """Return a noisy, directionless proximity value in the [0, 1] range."""
        _, distance = self.nearest_sample()
        ideal = max(0.0, 1.0 - distance / self.sensor_range)
        noisy = ideal + self._random.gauss(0.0, self.sensor_noise_stddev)
        return min(1.0, max(0.0, noisy))

    def collect(self) -> tuple[bool, dict[str, Any]]:
        """Collect the nearest sample when it is within the allowed radius."""
        state, distance = self.nearest_sample()
        if state is None or distance > self.collection_radius:
            self.false_collects += 1
            return False, {'event': 'false_collect'}
        state.collected = True
        return True, {'event': 'sample_collected', 'collected': self.collected_count}

    def at_base(self) -> bool:
        """Return whether the robot is close enough to the start point."""
        return hypot(self.world_x - self.base_x, self.world_y - self.base_y) <= BASE_RADIUS

    def finish(self) -> bool:
        """Finish the run if the robot is on the base."""
        if not self.at_base():
            return False
        self.finished = True
        return True
