"""Standalone math + model protocol demo; never connects to the robot.

Default: offline stub, project map, synthetic observations and status events.
An explicit --live option calls the supplied model endpoint for one decision.
Even in live mode, the resulting plan is printed, never published or executed.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os

from did_agent.costmap import CostMap
from did_agent.experimental_goals import (
    CandidateBackend, GoalValidationError, Observation, search_targets,
)
from did_agent.robot import BASE
from did_llm.experimental_selector import (
    BudgetStub, DecisionSession, SessionError, select_goal,
)
from did_llm.llm_client import LLMClient, LLMConfig


def run_demo(client=None, *, checks: bool = True) -> dict:
    costmap = CostMap()
    backend = CandidateBackend(costmap, BASE)
    session = DecisionSession(backend)
    observation = Observation('synthetic-demo', 0, BASE, battery=80, samples_total=3)
    targets = search_targets(costmap, limit=4, start=observation.pose)
    ticket, offer = session.prepare(observation, targets)
    selection = select_goal(offer, client or BudgetStub())
    decision = session.commit(ticket, selection, observation)
    result = {
        'mode': 'dry-run: synthetic state, no ROS publication or robot execution',
        'offer': offer, 'selection': selection.choice, 'source': selection.source,
        'validation_errors': selection.errors, 'plan': decision['plan'],
    }
    if not checks:
        return result

    # These are synthetic protocol events, not evidence of navigation success.
    try:
        session.prepare(observation, targets)
    except SessionError:
        result['active_plan_blocks_replanning'] = True
    statuses = []
    for index, step in enumerate(decision['plan']['subgoals']):
        terminated = session.on_status({'plan_id': decision['plan']['plan_id'],
                                        'index': index, 'state': 'done'})
        statuses.append({'step': step['type'], 'whole_plan_finished': terminated})
    result['synthetic_status_checks'] = statuses
    result['attempted_targets_after_completion'] = sorted(session.attempted_goal_ids)

    # Battery changed while the answer was in flight: discard the answer.
    ticket, second_offer = session.prepare(replace(observation, revision=1), targets)
    second_choice = select_goal(second_offer, BudgetStub())
    try:
        session.commit(ticket, second_choice, replace(observation, revision=1, battery=60))
    except GoalValidationError:
        result['changed_battery_rejects_old_answer'] = True

    low = replace(observation, revision=2, battery=8)
    ticket, low_offer = session.prepare(low, targets)
    low_choice = select_goal(low_offer, BudgetStub())
    result['low_battery_plan'] = session.commit(ticket, low_choice, low)['plan']
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true',
                        help='explicit opt-in: call the model, still never execute a plan')
    parser.add_argument('--base-url', default='', help='OpenAI-compatible endpoint for --live')
    parser.add_argument('--model', default='', help='model identifier for --live')
    args = parser.parse_args()
    client = None
    if args.live:
        key = os.environ.get('DID_LLM_API_KEY', '')
        if not key or not args.base_url or not args.model:
            parser.error('--live requires --base-url, --model and DID_LLM_API_KEY (key is never printed)')
        client = LLMClient(LLMConfig(
            base_url=args.base_url, api_key=key, model=args.model,
            timeout_sec=30, max_retries=0, min_interval_sec=0,
            max_calls_per_minute=2, max_calls_total=2, cache_enabled=False,
        ))
    print(json.dumps(run_demo(client, checks=not args.live), ensure_ascii=False,
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
