"""Adaptation: notice that the world differs from the model, and update the model.

Glues the telemetry monitor, the floor-price learner and a small tracker that
turns what was learned into journal entries (hypotheses, confirmations, changes).
The same object runs in the ROS node and in the offline simulator.
"""

from __future__ import annotations

from collections import deque
from math import atan2
from math import cos
from math import hypot
from math import sin
from typing import Any
from typing import Callable

import numpy as np

from did_agent.costmap import CostMap
from did_agent.learner import CostLearner
from did_agent.monitor import Monitor

Journal = Callable[[str, str, str, str], None]  # kind, title, text, status

ZONE_THRESHOLD = 1.4        # a floor this dear counts as a "zone"
HAZARD_PRICE = 8.0
HAZARD_RADIUS = 0.5         # area blocked around a guessed hazard centre
HAZARD_AHEAD = 0.25         # the centre lies ahead of the point where we got hurt
PENALTY_SKIP = 2.0          # seconds of telemetry to ignore after a penalty hit
CONFIRM_METRES = 1.2


class Zone:
    """A connected area of expensive floor that the agent believes in."""

    def __init__(self, zone_id: int, x: float, y: float, price: float) -> None:
        self.id = zone_id
        self.x, self.y = x, y
        self.price = price
        self.reported_price = price
        self.confirmed = False


class Adaptation:
    """Feed it pose, battery, sensor and events; it keeps the cost map honest."""

    def __init__(
        self,
        costmap: CostMap,
        journal: Journal | None = None,
        *,
        learn: bool = True,
        segment_length: float = 0.12,
    ) -> None:
        self.costmap = costmap
        self.journal = journal or (lambda kind, title, text, status: None)
        self.learn = learn
        self.learner = CostLearner(costmap, prior=0.05, update_threshold=0.35)
        self.monitor = Monitor(self.expected_cost, self.cells_between, segment_length=segment_length)
        self.zones: list[Zone] = []
        self.hypotheses = 0
        self._pose: tuple[float, float] | None = None
        self._anchor: tuple[float, float] | None = None
        self._heading: float | None = None
        self._last_penalty = -float('inf')
        self._flags = {'sensor_noise_up': False, 'penalties_burst': False}
        self._last_zone_check = -float('inf')
        self._known_version = costmap.version

    # --- helpers for the monitor ------------------------------------------------------

    def expected_cost(self, x: float, y: float) -> float:
        row, col = self.costmap.world_to_cell(x, y)
        if self.costmap.grid.in_bounds(row, col):
            return float(self.costmap.terrain[row, col])
        return 1.0

    def cells_between(self, x0: float, y0: float, x1: float, y1: float) -> list[list[int]]:
        count = max(1, int(hypot(x1 - x0, y1 - y0) / 0.05))
        cells: list[list[int]] = []
        for i in range(count + 1):
            t = i / count
            cell = list(self.costmap.world_to_cell(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
            if cell not in cells:
                cells.append(cell)
        return cells

    # --- inputs --------------------------------------------------------------------------

    def on_battery(self, t: float, value: float) -> None:
        self.monitor.on_battery(t, value)

    def on_sensor(self, t: float, value: float) -> None:
        self.monitor.on_sensor(t, value)
        self._watch_flags(t)

    def on_pose(self, t: float, x: float, y: float) -> dict[str, Any] | None:
        """Return the finished telemetry segment, if there is one."""
        self._pose = (x, y)
        if self._anchor is None:
            self._anchor = (x, y)
        elif hypot(x - self._anchor[0], y - self._anchor[1]) >= 0.05:
            self._heading = atan2(y - self._anchor[1], x - self._anchor[0])
            self._anchor = (x, y)
        segment = self.monitor.on_pose(t, x, y)
        if segment is not None and self.learn and t - self._last_penalty > PENALTY_SKIP:
            if self.learner.observe(segment):
                self._review_zones(t)
        return segment

    def on_event(self, t: float, event: dict[str, Any]) -> None:
        self.monitor.on_event(t, event)
        kind = event.get('event')
        if kind in ('hazard_hit', 'collision', 'false_collect'):
            self._last_penalty = t  # the battery drop of a penalty is not floor price
        if kind == 'hazard_hit' and self._pose and self.learn:
            x, y = self._pose
            if self._heading is not None:  # we were hurt on the edge: the zone is ahead
                x, y = x + HAZARD_AHEAD * cos(self._heading), y + HAZARD_AHEAD * sin(self._heading)
            region = {'circle': {'x': x, 'y': y, 'r': HAZARD_RADIUS}}
            self.learner.clock = t
            self.learner.inject(region, HAZARD_PRICE)
            self.costmap.block_disk(x, y, HAZARD_RADIUS)
            self._hypothesis(
                'hypothesis', f'Опасная зона около ({x:.1f}; {y:.1f})',
                'Сработал штраф hazard_hit. Считаю участок опасным, блокирую его и обхожу.',
            )
        self._watch_flags(t)

    def on_external_update(self, region: dict[str, Any], price: float) -> None:
        """An analyst's estimate becomes evidence the learner keeps."""
        self.learner.clock = self.monitor.t
        self.learner.inject(region, price)

    def noise_level(self) -> float:
        return self.monitor.noise_estimate

    # --- journal ----------------------------------------------------------------------------

    def _hypothesis(self, kind: str, title: str, text: str, status: str = 'open') -> None:
        self.hypotheses += kind == 'hypothesis'
        self.journal(kind, title, text, status)

    def _watch_flags(self, t: float) -> None:
        anomaly = self.monitor.anomaly(t)
        noisy = anomaly['sensor_noise_up']
        if noisy and not self._flags['sensor_noise_up']:
            self._hypothesis(
                'hypothesis', 'Шум датчика образцов вырос',
                f'Оценка шума {self.monitor.noise_estimate:.2f} против обычных '
                f'{(self.monitor.noise_baseline or 0):.2f}. Возможен сбой: '
                'усредняю больше показаний и доверяю им меньше.',
            )
        elif not noisy and self._flags['sensor_noise_up']:
            self._hypothesis('result', 'Шум датчика вернулся в норму', '', 'confirmed')
        self._flags['sensor_noise_up'] = noisy
        burst = anomaly['penalties_burst']
        if burst and not self._flags['penalties_burst']:
            self._hypothesis(
                'decision', 'Серия штрафов',
                'Два штрафа за короткое время: пересматриваю маршрут и держусь дальше от стен.',
            )
        self._flags['penalties_burst'] = burst

    def _components(self) -> list[tuple[float, float, float, float]]:
        """Connected areas of expensive floor: (x, y, mean price, metres of evidence)."""
        terrain = self.costmap.terrain
        mask = terrain >= ZONE_THRESHOLD
        if not mask.any():
            return []
        rows, cols = np.where(mask)
        r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
        sub = mask[r0:r1 + 1, c0:c1 + 1]
        seen = np.zeros_like(sub)
        found = []
        for r, c in zip(*np.where(sub)):
            if seen[r, c]:
                continue
            queue, cells = deque([(r, c)]), []
            seen[r, c] = True
            while queue:
                a, b = queue.popleft()
                cells.append((a, b))
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        x, y = a + da, b + db
                        if 0 <= x < sub.shape[0] and 0 <= y < sub.shape[1] and sub[x, y] and not seen[x, y]:
                            seen[x, y] = True
                            queue.append((x, y))
            if len(cells) < 6:
                continue
            rr = np.array([a for a, _ in cells]) + r0
            cc = np.array([b for _, b in cells]) + c0
            x, y = self.costmap.cell_to_world(int(rr.mean()), int(cc.mean()))
            evidence = float(self.learner.weight[rr, cc].sum()) / max(1.0, len(cells) ** 0.5)
            found.append((x, y, float(terrain[rr, cc].mean()), evidence))
        return found

    def _review_zones(self, t: float) -> None:
        """Turn the learned prices into hypotheses, confirmations and change alerts."""
        if t - self._last_zone_check < 1.0:
            return
        self._last_zone_check = t
        for x, y, price, evidence in self._components():
            if price >= HAZARD_PRICE * 0.9:
                continue  # hazard areas are reported when they are hit
            zone = min(
                (z for z in self.zones if hypot(z.x - x, z.y - y) < 0.6),
                key=lambda z: hypot(z.x - x, z.y - y),
                default=None,
            )
            if zone is None:
                zone = Zone(len(self.zones) + 1, x, y, price)
                self.zones.append(zone)
                self._hypothesis(
                    'hypothesis', f'Дорогой участок около ({x:.1f}; {y:.1f})',
                    f'Расход на метр ≈ ×{price:.1f} от обычного. Проверю повторным проходом; '
                    'пока считаю дорогим и объезжаю, если есть дешёвый путь.',
                )
                continue
            zone.x, zone.y, zone.price = x, y, price
            change = abs(price - zone.reported_price) / zone.reported_price
            if not zone.confirmed:
                # Still collecting evidence: the estimate just gets better, that is no change.
                if evidence >= CONFIRM_METRES * 0.5:
                    zone.confirmed = True
                    zone.reported_price = price
                    self._hypothesis(
                        'result', f'Подтверждено: участок около ({x:.1f}; {y:.1f}) ×{price:.1f}',
                        'Повторные проходы дают тот же расход.', 'confirmed',
                    )
                continue
            if change > 0.35:
                self._hypothesis(
                    'hypothesis', f'Грунт около ({x:.1f}; {y:.1f}) мог измениться',
                    f'Расход вышел из прогноза: было ×{zone.reported_price:.1f}, '
                    f'теперь ×{price:.1f}. Пересматриваю карту стоимостей и маршрут.',
                )
                self._hypothesis('result', f'Оценка ×{zone.reported_price:.1f} устарела',
                                 f'Участок около ({x:.1f}; {y:.1f})', 'rejected')
                zone.reported_price = price
                zone.confirmed = False
