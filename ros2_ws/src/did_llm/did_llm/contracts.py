"""Contract mirrors: what the executor accepts and what it reports.

The planner sees two things: a plan it must publish to ``/agent/plan``, and a
state snapshot on ``/agent/state``. Both live in ``did_agent``, which this
package must not depend on — a planner that can only run next to the code it
plans for cannot be exercised against a stub, and cannot be changed when the
executor's internals move. So the parts that matter are re-declared here, and
kept deliberately thin.
"""

from did_llm.agent_link import (
    COMMAND_TOPIC,
    JOURNAL_TOPIC,
    PLAN_TOPIC,
    STATE_TOPIC,
    STATUS_TOPIC,
    AgentLink,
)
from did_llm.agent_plan import (
    ARENA_X_MAX,
    ARENA_X_MIN,
    ARENA_Y_MAX,
    ARENA_Y_MIN,
    BASE_X,
    BASE_Y,
    MAX_SUBGOALS,
    RADIUS_MAX,
    RADIUS_MIN,
    RETURN_MARGIN,
    RETURN_RESERVE,
    TARGET_SUBGOALS,
    Plan,
    PlanRejected,
    Subgoal,
    budget_floor,
    home_plan,
    must_return,
    parse_model_plan,
)

__all__ = [
    'AgentLink',
    'ARENA_X_MAX',
    'ARENA_X_MIN',
    'ARENA_Y_MAX',
    'ARENA_Y_MIN',
    'BASE_X',
    'BASE_Y',
    'COMMAND_TOPIC',
    'JOURNAL_TOPIC',
    'MAX_SUBGOALS',
    'PLAN_TOPIC',
    'Plan',
    'PlanRejected',
    'RADIUS_MAX',
    'RADIUS_MIN',
    'RETURN_MARGIN',
    'RETURN_RESERVE',
    'STATE_TOPIC',
    'STATUS_TOPIC',
    'Subgoal',
    'TARGET_SUBGOALS',
    'budget_floor',
    'home_plan',
    'must_return',
    'parse_model_plan',
]