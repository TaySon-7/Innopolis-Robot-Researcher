"""Run a plan of subgoals through the skills and report progress."""

from __future__ import annotations

from typing import Any
from typing import Callable

from did_agent.plan import Plan
from did_agent.plan import Subgoal
from did_agent.skills import SkillResult
from did_agent.skills import Skills


class PlanExecutor:
    """Execute subgoals in order; stop at the first failure and say why."""

    def __init__(
        self,
        skills: Skills,
        publish_status: Callable[[dict[str, Any]], None] = lambda status: None,
    ) -> None:
        self.skills = skills
        self.publish_status = publish_status

    def _execute(self, subgoal: Subgoal) -> SkillResult:
        if subgoal.type == 'goto':
            return self.skills.goto(subgoal.x, subgoal.y)
        if subgoal.type == 'search_around':
            return self.skills.search_around(subgoal.x, subgoal.y, subgoal.radius)
        if subgoal.type == 'collect':
            return self.skills.collect()
        return self.skills.return_to_base()

    def _status(self, plan: Plan, index: int, state: str, reason: str = '',
                data: dict[str, Any] | None = None) -> dict[str, Any]:
        subgoal = plan.subgoals[index] if index < len(plan.subgoals) else None
        status = {
            'plan_id': plan.plan_id,
            'index': index,
            'type': subgoal.type if subgoal else '',
            'subgoal': subgoal.describe() if subgoal else '',
            'state': state,
            'reason': reason,
            'data': data or {},
        }
        self.publish_status(status)
        return status

    def run(self, plan: Plan) -> dict[str, Any]:
        """Run the whole plan; return the last status that was published."""
        if not plan.subgoals:
            return self._status(plan, 0, 'idle', 'empty plan')
        last: dict[str, Any] = {}
        for index, subgoal in enumerate(plan.subgoals):
            self._status(plan, index, 'running')
            result = self._execute(subgoal)
            if self.skills.robot.preempted():
                return self._status(plan, index, 'preempted', 'replaced by a new plan')
            if not result.ok:
                return self._status(plan, index, 'failed', result.reason, result.data)
            last = self._status(plan, index, 'done', data=result.data)
        return last
