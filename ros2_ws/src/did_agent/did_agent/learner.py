"""Learn the price of each floor type from the battery use of driven segments.

Every telemetry segment says: "over these cells the battery went down by this
much per metre". Observations are collected per cell, smoothed in space with a
small kernel and blended with a prior of "ordinary floor" (price 1), so places
the robot never visited stay at 1 and a single noisy segment cannot invent a
zone. When fresh observations disagree strongly with what a cell already holds,
the old evidence is mostly forgotten: that is how a changed floor is noticed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from did_agent.costmap import CostMap
from did_agent.costmap import _disk_offsets

MAX_PRICE = 8.0
MIN_PRICE = 0.5


class CostLearner:
    """Turn measured battery use per metre into floor price multipliers."""

    def __init__(
        self,
        costmap: CostMap,
        *,
        radius: float = 0.25,
        prior: float = 0.3,
        unit_cost: float = 1.0,
        forget_ratio: float = 0.4,
        forget_keep: float = 0.3,
        update_threshold: float = 0.2,
    ) -> None:
        self.costmap = costmap
        self.unit_cost = unit_cost
        self.prior = prior
        self.forget_ratio = forget_ratio
        self.forget_keep = forget_keep
        self.update_threshold = update_threshold
        self.weight = np.zeros(costmap.grid.shape, dtype=np.float32)
        self.excess = np.zeros(costmap.grid.shape, dtype=np.float32)  # sum w * (price - 1)
        self.radius_cells = max(1, int(round(radius / costmap.resolution)))
        reach = float(self.radius_cells + 1)
        self._kernel = [
            (dr, dc, 1.0 - length / reach)
            for dr, dc, length in _disk_offsets(self.radius_cells)
        ]
        self.observations = 0
        self.clock: float | None = None  # time of the evidence being processed

    def _smooth(self, block: np.ndarray) -> np.ndarray:
        r = self.radius_cells
        padded = np.pad(block, r)
        out = np.zeros_like(block, dtype=np.float32)
        rows, cols = block.shape
        for dr, dc, k in self._kernel:
            out += k * padded[r + dr:r + dr + rows, r + dc:r + dc + cols]
        return out

    def _refresh(self, rows: list[int], cols: list[int]) -> bool:
        r = self.radius_cells
        shape = self.costmap.grid.shape
        r0, r1 = max(0, min(rows) - r), min(shape[0] - 1, max(rows) + r)
        c0, c1 = max(0, min(cols) - r), min(shape[1] - 1, max(cols) + r)
        weight = self._smooth(self.weight[r0:r1 + 1, c0:c1 + 1])
        if self.clock is not None:
            seen = self.costmap.last_seen[r0:r1 + 1, c0:c1 + 1]
            seen[weight > 0.02] = self.clock
        excess = self._smooth(self.excess[r0:r1 + 1, c0:c1 + 1])
        price = np.clip(1.0 + excess / (weight + self.prior), MIN_PRICE, MAX_PRICE)
        return self.costmap.set_terrain(r0, c0, price.astype(np.float32), self.update_threshold)

    def observe(self, segment: dict[str, Any]) -> bool:
        """Add a telemetry segment; return True if the cost map changed materially."""
        self.clock = float(segment.get('t', 0.0))
        cells = [(int(r), int(c)) for r, c in segment.get('cells', [])]
        distance = float(segment.get('distance', 0.0))
        price = float(segment.get('per_meter', 0.0)) / self.unit_cost
        shape = self.costmap.grid.shape
        cells = [(r, c) for r, c in cells if 0 <= r < shape[0] and 0 <= c < shape[1]]
        if not cells or distance <= 0.0 or not 0.2 < price < 12.0:
            return False
        share = distance / len(cells)
        for r, c in cells:
            w = float(self.weight[r, c])
            if w > 0.15:
                mean = 1.0 + float(self.excess[r, c]) / w
                if abs(price - mean) / max(mean, 0.1) > self.forget_ratio:
                    self.weight[r, c] *= self.forget_keep
                    self.excess[r, c] *= self.forget_keep
            self.weight[r, c] += share
            self.excess[r, c] += share * (price - 1.0)
        self.observations += 1
        return self._refresh([r for r, _ in cells], [c for _, c in cells])

    def inject(self, region: dict[str, Any], price: float, weight: float = 0.3) -> None:
        """Take an outside estimate (analyst, hazard hit) as strong evidence."""
        mask = self.costmap.cells_in_region(region)
        self.weight[mask] = weight
        self.excess[mask] = weight * (price - 1.0)
        rows, cols = np.where(mask)
        if len(rows):
            self._refresh(rows.tolist(), cols.tolist())

    def evidence_at(self, x: float, y: float) -> float:
        """Metres of travel behind the estimate at a world point (smoothed)."""
        row, col = self.costmap.world_to_cell(x, y)
        if not self.costmap.grid.in_bounds(row, col):
            return 0.0
        r = self.radius_cells
        block = self.weight[max(0, row - r):row + r + 1, max(0, col - r):col + r + 1]
        return float(block.sum())
