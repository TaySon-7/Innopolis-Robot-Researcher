"""Skills the plan executor and the autonomous agent are built from."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from did_agent.planner import plan_waypoints
from did_agent.robot import BASE
from did_agent.robot import Robot
from did_agent.search import SampleSearch


ENERGY_RISK_FLAGS = ('battery_deviation', 'penalties_burst')


def energy_risk(flags: dict[str, bool]) -> bool:
    """Whether observations make the return-energy forecast less trustworthy."""
    return any(flags.get(name, False) for name in ENERGY_RISK_FLAGS)


@dataclass
class SkillResult:
    """Outcome of a skill: success flag, reason for the planner, extra data."""

    ok: bool
    reason: str = ''
    data: dict[str, Any] = field(default_factory=dict)


class Skills:
    """goto / search_around / collect / return_to_base on top of a Robot."""

    def __init__(
        self,
        robot: Robot,
        *,
        reserve: float = 8.0,
        return_factor: float = 1.4,
        pessimism: float = 0.5,
    ) -> None:
        self.robot = robot
        self.reserve = reserve
        self.return_factor = return_factor
        self.pessimism = pessimism

    def goto(
        self,
        x: float,
        y: float,
        guarded: bool = False,
        stop_on_signal: bool = False,
    ) -> SkillResult:
        """Drive to a point, optionally yielding to battery or sample evidence."""
        guard = (lambda: not self.battery_allows_more()) if guarded else None
        result = self.robot.goto(x, y, guard=guard, stop_on_signal=stop_on_signal)
        data = {'distance_to_goal': result.distance_to_goal, 'replans': result.replans}
        if result.ok:
            return SkillResult(True, data=data)
        return SkillResult(False, result.reason or result.status, data)

    def return_cost_estimate(self) -> float:
        """Estimate the battery needed to drive from here to the base."""
        pose = self.robot.pose()
        route = plan_waypoints(self.robot.costmap, (pose.x, pose.y), BASE)
        if route is None:
            return float('inf')
        return self.robot.costmap.energy_cost(
            [(pose.x, pose.y), *route], pessimism=self.pessimism, now=self.robot.now()
        )

    def battery_allows_more(self) -> bool:
        """True while the battery covers the way home with a safety margin.

        While the world misbehaves (battery use off forecast, a burst of penalties)
        the forecast is less trustworthy, so the margin grows and we head home earlier.
        """
        factor, reserve = self.return_factor, self.reserve
        # Sensor noise changes how many readings search takes, not the energy
        # per metre.  Treating it as an energy anomaly sent hard missions home
        # early even when the return-cost model remained accurate.
        if energy_risk(self.robot.anomaly()):
            factor, reserve = factor * 1.15, reserve + 2.0
        return self.robot.battery() > self.return_cost_estimate() * factor + reserve

    def search_around(self, x: float, y: float, radius: float) -> SkillResult:
        search = SampleSearch(self.robot, budget_ok=self.battery_allows_more)
        found = search.run(x, y, radius)
        data = {
            'peak': round(found.peak, 3),
            'x': round(found.x, 3),
            'y': round(found.y, 3),
            'readings': len(found.trace),
            'trace': [
                {'x': round(tx, 3), 'y': round(ty, 3), 'signal': round(signal, 3)}
                for tx, ty, signal in found.trace
            ],
        }
        if found.gradient is not None:
            data['gradient'] = {
                'dx': round(found.gradient[0], 3),
                'dy': round(found.gradient[1], 3),
            }
        if found.found:
            return SkillResult(True, data=data)
        return SkillResult(False, found.reason or 'sample not found', data)

    def collect(self) -> SkillResult:
        ok, message = self.robot.collect()
        return SkillResult(ok, '' if ok else message,
                           {'collected': self.robot.collected()})

    def return_to_base(self) -> SkillResult:
        moved = self.goto(*BASE)
        if not moved.ok:
            return moved
        ok, message = self.robot.finish()
        return SkillResult(ok, '' if ok else message)
