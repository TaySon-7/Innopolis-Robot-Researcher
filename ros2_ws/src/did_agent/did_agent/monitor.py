"""Telemetry segments and anomaly flags computed from the judge's public topics."""

from __future__ import annotations

from collections import deque
from math import hypot
from statistics import median
from typing import Any
from typing import Callable

PENALTY_EVENTS = ('collision', 'hazard_hit', 'false_collect')


class Monitor:
    """Watch pose, battery, sensor and events; know what is unusual.

    ``expected_cost(x, y)`` is the floor price the agent currently believes in;
    segments whose measured battery use per metre disagrees with it are flagged.
    """

    def __init__(
        self,
        expected_cost: Callable[[float, float], float] = lambda x, y: 1.0,
        cells_between: Callable[[float, float, float, float], list[list[int]]] | None = None,
        *,
        segment_length: float = 0.25,
        deviation_tolerance: float = 0.30,
        anomaly_hold: float = 15.0,
        burst_window: float = 20.0,
    ) -> None:
        self.expected_cost = expected_cost
        self.cells_between = cells_between
        self.segment_length = segment_length
        self.deviation_tolerance = deviation_tolerance
        self.anomaly_hold = anomaly_hold
        self.burst_window = burst_window

        self.battery: float | None = None
        self.sensor: float | None = None
        self.noise_baseline: float | None = None
        self.noise_estimate = 0.0
        self.t = 0.0

        self._origin: tuple[float, float] | None = None
        self._origin_battery = 0.0
        self._origin_time = 0.0
        self._last_pose: tuple[float, float] | None = None
        self._travelled = 0.0
        self._sensor_history: deque[float] = deque(maxlen=30)
        self._penalties: deque[float] = deque()
        self._deviation_streak = 0
        self._deviation_until = -1.0
        self._noise_until = -1.0
        self.events: deque[dict[str, Any]] = deque(maxlen=10)

    # --- inputs ------------------------------------------------------------------

    def on_battery(self, t: float, value: float) -> None:
        self.t = t
        self.battery = value

    def on_sensor(self, t: float, value: float) -> None:
        self.t = t
        self.sensor = value
        self._sensor_history.append(value)
        if len(self._sensor_history) >= 12:
            values = list(self._sensor_history)
            second = [
                values[i] - 2 * values[i - 1] + values[i - 2]
                for i in range(2, len(values))
            ]
            # Median, not RMS: a jump when a sample is collected must not look like noise.
            self.noise_estimate = median(abs(d) for d in second) * 0.605 / 1.0
            if self.noise_baseline is None or self.noise_estimate < self.noise_baseline:
                self.noise_baseline = max(self.noise_estimate, 0.005)
            if self.noise_estimate > max(3.0 * self.noise_baseline, 0.06):
                self._noise_until = t + self.anomaly_hold

    def on_event(self, t: float, event: dict[str, Any]) -> None:
        self.t = t
        self.events.append(event)
        if event.get('event') in PENALTY_EVENTS:
            self._penalties.append(t)

    def on_pose(self, t: float, x: float, y: float) -> dict[str, Any] | None:
        """Feed a pose; return a finished telemetry segment when one is complete."""
        self.t = t
        if self.battery is None:
            return None
        if self._origin is None:
            self._start_segment(t, x, y)
            self._last_pose = (x, y)
            return None
        self._travelled += hypot(x - self._last_pose[0], y - self._last_pose[1])
        self._last_pose = (x, y)
        if self._travelled < self.segment_length:
            return None
        segment = self._finish_segment(t, x, y)
        self._start_segment(t, x, y)
        return segment

    # --- segments --------------------------------------------------------------------

    def _start_segment(self, t: float, x: float, y: float) -> None:
        self._origin = (x, y)
        self._origin_battery = float(self.battery)
        self._origin_time = t
        self._travelled = 0.0

    def _finish_segment(self, t: float, x: float, y: float) -> dict[str, Any]:
        distance = self._travelled
        delta = self._origin_battery - float(self.battery)
        per_meter = delta / distance if distance > 1e-6 else 0.0
        middle = ((self._origin[0] + x) / 2.0, (self._origin[1] + y) / 2.0)
        expected = max(self.expected_cost(*middle), 1e-3)
        if abs(per_meter - expected) / expected > self.deviation_tolerance:
            self._deviation_streak += 1
            if self._deviation_streak >= 2:
                self._deviation_until = t + self.anomaly_hold
        else:
            self._deviation_streak = 0
        return {
            't': round(t, 2),
            'from': {'x': round(self._origin[0], 3), 'y': round(self._origin[1], 3)},
            'to': {'x': round(x, 3), 'y': round(y, 3)},
            'distance': round(distance, 3),
            'battery_delta': round(delta, 4),
            'per_meter': round(per_meter, 3),
            'expected_per_meter': round(expected, 3),
            'cells': (
                self.cells_between(*self._origin, x, y) if self.cells_between else []
            ),
        }

    # --- outputs ---------------------------------------------------------------------

    def anomaly(self, t: float | None = None) -> dict[str, bool]:
        now = self.t if t is None else t
        while self._penalties and now - self._penalties[0] > self.burst_window:
            self._penalties.popleft()
        return {
            'battery_deviation': now < self._deviation_until,
            'penalties_burst': len(self._penalties) >= 2,
            'sensor_noise_up': now < self._noise_until,
        }

    def recent_events(self) -> list[dict[str, Any]]:
        return list(self.events)
