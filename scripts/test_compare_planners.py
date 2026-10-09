"""Offline mission protocol checks; never contacts ROS, Docker or a model."""

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from compare_planners import READ_METRICS, DockerPlanner, MissionComparison, clean_score, schedule
from test_compare_navigation import Clock, FakeAPI, snapshot


def frame(t=1.0, *, mode='autonomous', episode=2, finished=False, battery=60.0):
    result = snapshot(t, episode=episode)
    result['state'].update(control_mode=mode, t=t)
    result['score'].update(scenario='easy@7', battery=battery, finished=finished,
                           collected=2 if finished else 0, samples_total=3,
                           score=25 if finished else 0, false_collects=0, hazard_hits=0,
                           samples=[{'x': 100, 'y': 200}])
    result['journal'] = []
    return result


class Planner:
    def __init__(self, token, arm, max_calls):
        self.arm = arm
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def alive(self):
        return True

    def stop(self):
        self.stopped = True
        return {'selection_policy': self.arm, 'successful_exchanges': 1,
                'response_latency_s': [0.7], 'client_stats': {'calls_made': 2, 'failed': 1}}


class MissionTests(unittest.TestCase):
    def run_case(self, frames, arm='autonomous', **params):
        clock = Clock()
        api = FakeAPI([frame(mode='stopped'), *frames])
        self.planners = []

        def factory(*args):
            item = Planner(*args)
            self.planners.append(item)
            return item

        options = dict(sim_limit=10, wall_limit=3, poll=0.5, ready_timeout=2,
                       planner_factory=factory, clock=clock, sleep=clock.sleep, progress=lambda _: None)
        options.update(params)
        runner = MissionComparison(api, **options)
        runner.prepare = lambda *_: frame(mode='stopped')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.jsonl'
            result = runner.run_mission(arm, 'easy@7', 'safe-token', path)
            raw = [json.loads(line) for line in path.read_text().splitlines()]
        return result, raw, api

    def test_pilot_rotates_three_arms(self):
        self.assertEqual(schedule(['easy@7', 'medium@7', 'hard@7']), [
            ('easy@7', 'autonomous'), ('easy@7', 'budget'), ('easy@7', 'llm'),
            ('medium@7', 'budget'), ('medium@7', 'llm'), ('medium@7', 'autonomous'),
            ('hard@7', 'llm'), ('hard@7', 'autonomous'), ('hard@7', 'budget')])

    def test_finished_uses_judge_metrics_and_hides_sample_positions(self):
        result, raw, api = self.run_case([frame(2), frame(3, finished=True)])
        self.assertTrue(result['valid'])
        self.assertFalse(result['censored'])
        self.assertEqual(result['outcome'], 'finished')
        self.assertEqual(result['metrics']['score'], 25)
        self.assertTrue(result['metrics']['returned_to_base'])
        self.assertEqual(result['metrics']['simulation_elapsed_s'], 2)
        self.assertTrue(result['stop_confirmed'])
        self.assertNotIn('samples', clean_score(frame()))
        self.assertNotIn('"samples":', json.dumps(raw))
        self.assertIn(('command', {'cmd': 'auto'}), api.calls)
        self.assertEqual(self.planners, [])

    def test_budget_and_llm_use_owned_production_lifecycle(self):
        for arm in ('budget', 'llm'):
            with self.subTest(arm=arm):
                item = frame(2, mode='llm', finished=True)
                entry = {'source': 'llm_math_planner', 'decision_source': arm,
                         'plan_id': 'math-1', 'title': 'selected'}
                item['journal'] = [entry, deepcopy(entry)]
                result, raw, api = self.run_case([item], arm)
                self.assertTrue(self.planners[0].started)
                self.assertTrue(self.planners[0].stopped)
                self.assertEqual(result['decisions_published'], {arm: 1})
                self.assertEqual(sum(row['kind'] == 'journal' for row in raw), 1)
                self.assertIn(('command', {'cmd': 'llm'}), api.calls)
                self.assertEqual(result['planner_metrics']['client_stats']['calls_made'], 2)

    def test_failed_llm_fallback_is_counted_separately(self):
        item = frame(2, mode='fallback', finished=True)
        item['journal'] = [{'source': 'llm_math_planner', 'decision_source': 'fallback', 'plan_id': 'math-1'}]
        result, _, _ = self.run_case([item], 'llm')
        self.assertEqual(result['llm_decision_share'], 0)
        self.assertEqual(result['fallback_decision_share'], 1)

    def test_rejected_published_plan_is_not_an_accepted_execution(self):
        first = frame(2, mode='llm')
        first['journal'] = [{'source': 'llm_math_planner', 'decision_source': 'llm', 'plan_id': 'math-1'}]
        first['status'] = {'plan_id': 'math-1', 'state': 'failed', 'plan_complete': True,
                           'data': {'accepted': False}}
        last = frame(3, mode='llm', finished=True)
        last['journal'] = [*first['journal'],
                           {'source': 'llm_math_planner', 'decision_source': 'budget', 'plan_id': 'math-2'}]
        last['status'] = {'plan_id': 'math-2', 'state': 'done', 'plan_complete': True, 'data': {}}
        result, _, _ = self.run_case([first, last], 'llm')
        self.assertEqual(result['decisions_published'], {'llm': 1, 'budget': 1})
        self.assertEqual(result['decisions_accepted_observed'], {'budget': 1})
        self.assertEqual(result['decisions_rejected_observed'], {'llm': 1})
        self.assertEqual(result['terminal_plans_observed']['budget'], {'done': 1})
        self.assertEqual(result['terminal_plans_observed']['llm'], {})

    def test_simulation_timeout_is_a_censored_valid_run(self):
        result, _, _ = self.run_case([frame(10)])
        self.assertTrue(result['valid'])
        self.assertTrue(result['censored'])
        self.assertEqual(result['outcome'], 'simulation_timeout')
        self.assertFalse(result['metrics']['returned_to_base'])

    def test_frozen_clock_has_wall_timeout_without_no_progress_heuristic(self):
        result, _, _ = self.run_case([frame(1)])
        self.assertTrue(result['valid'])
        self.assertTrue(result['censored'])
        self.assertEqual(result['outcome'], 'wall_timeout')

    def test_stale_judge_telemetry_is_an_infrastructure_failure(self):
        result, _, _ = self.run_case([frame(1)], wall_limit=40)
        self.assertFalse(result['valid'])
        self.assertIn('telemetry', result['error'])

    def test_advancing_judge_does_not_hide_crashed_agent(self):
        frames = [frame(float(t)) for t in range(2, 75)]
        for item in frames:
            item['state']['t'] = 1.0
        result, raw, _ = self.run_case(frames, wall_limit=40, sim_limit=100)
        self.assertFalse(result['valid'])
        self.assertFalse(result['censored'])
        self.assertIn('agent state telemetry', result['error'])
        telemetry = [item for item in raw if item['kind'] == 'telemetry']
        self.assertGreater(telemetry[-1]['score']['t'], telemetry[0]['score']['t'])
        self.assertEqual(telemetry[-1]['state']['t'], 1.0)

    def test_missing_agent_timestamp_is_invalid(self):
        item = frame(2)
        item['state'].pop('t')
        result, _, _ = self.run_case([item])
        self.assertFalse(result['valid'])
        self.assertIn('agent state timestamp', result['error'])

    def test_autonomous_terminal_failure_is_counted_without_waiting_for_horizon(self):
        first, last = frame(2), frame(3)
        for item in (first, last):
            item['status'] = {'plan_id': 'auto', 'state': 'failed', 'plan_complete': True,
                              'reason': 'return failed'}
        result, _, _ = self.run_case([first, last])
        self.assertTrue(result['valid'])
        self.assertFalse(result['censored'])
        self.assertEqual(result['outcome'], 'autonomous_terminal')
        self.assertEqual(result['finish_reason'], 'return failed')

    def test_empty_battery_is_uncensored_failure(self):
        result, _, _ = self.run_case([frame(2, battery=0)])
        self.assertTrue(result['valid'])
        self.assertFalse(result['censored'])
        self.assertEqual(result['outcome'], 'battery_depleted')

    def test_scenario_change_is_invalid_and_stops_planner(self):
        result, _, _ = self.run_case([frame(2, episode=3, mode='llm')], 'llm')
        self.assertFalse(result['valid'])
        self.assertIn('episode', result['error'])
        self.assertTrue(self.planners[0].stopped)

    def test_controller_takeover_is_invalid(self):
        for finished in (False, True):
            with self.subTest(finished=finished):
                result, _, _ = self.run_case([frame(2, mode='manual', finished=finished)])
                self.assertFalse(result['valid'])
                self.assertIn('controller', result['error'])

    def test_slow_simulation_can_acknowledge_after_five_wall_seconds(self):
        frames = [frame(float(t), mode='stopped') for t in range(2, 24)]
        frames.append(frame(24, finished=True))
        result, _, _ = self.run_case(frames, wall_limit=40, sim_limit=100)
        self.assertTrue(result['valid'])
        self.assertEqual(result['outcome'], 'finished')
        self.assertGreater(result['wall_elapsed_s'], 5)

    def test_stop_after_acknowledgement_invalidates_immediately(self):
        result, _, _ = self.run_case([frame(2), frame(3, mode='stopped')])
        self.assertFalse(result['valid'])
        self.assertIn('controller', result['error'])
        self.assertLess(result['wall_elapsed_s'], 5)

    def test_backward_counters_invalidate_even_if_a_new_finish_arrives(self):
        result, _, _ = self.run_case([frame(3), frame(2, finished=True)])
        self.assertFalse(result['valid'])
        self.assertIn('backwards', result['error'])

    def test_planner_supervision_uses_argv_without_a_shell(self):
        command = DockerPlanner.command('some Python', 'token')
        self.assertEqual(command[:6], ['docker', 'compose', 'exec', '-T', 'sim', 'python3'])
        self.assertEqual(command[-2:], ['some Python', 'token'])

    def test_sanitized_exchange_diagnostics_match_budget_tie_break_and_ignore_malformed(self):
        candidates = [
            {'goal_id': 'home', 'kind': 'return', 'required_battery': 1, 'feasible': True},
            {'goal_id': 'b', 'kind': 'search', 'required_battery': 5, 'feasible': True, 'evidence': 'observed_signal'},
            {'goal_id': 'a', 'kind': 'search', 'required_battery': 5, 'feasible': True, 'evidence': 'exploration'},
            {'goal_id': 'blocked', 'kind': 'search', 'required_battery': 0, 'feasible': False},
        ]
        exchange = {'seconds': 0.4, 'user': json.dumps({'offer': {'snapshot_id': 'snap-1', 'candidates': candidates},
                                                      'private_test_field': 'do-not-copy'}),
                    'parsed': {'goal_id': 'b', 'reason': 'do-not-copy'}, 'system': 'do-not-copy'}
        home = deepcopy(exchange)
        home['user'] = json.dumps({'offer': {'snapshot_id': 'snap-2', 'candidates': [candidates[0]]}})
        home['parsed']['goal_id'] = 'home'
        malformed = {'seconds': 0.5, 'user': 'broken JSON', 'parsed': []}
        content = '\n'.join(json.dumps(value) for value in (exchange, home, malformed, [])) + '\ntruncated'

        def fake_open(path):
            if path.endswith('.exchanges.jsonl'):
                return io.StringIO(content)
            raise FileNotFoundError(path)

        output = io.StringIO()
        with patch('builtins.open', fake_open), patch('sys.argv', ['reader', 'test-token']), redirect_stdout(output):
            exec(compile(READ_METRICS, 'metrics-reader', 'exec'), {})
        data = json.loads(output.getvalue())
        self.assertEqual(data['successful_exchanges'], 3)
        self.assertEqual(data['decision_diagnostics'], [
            {'snapshot_id': 'snap-1', 'goal_id': 'b', 'feasible_search_count': 2,
             'budget_choice_id': 'a', 'llm_differs_from_budget': True, 'selected_evidence': 'observed_signal'},
            {'snapshot_id': 'snap-2', 'goal_id': 'home', 'feasible_search_count': 0,
             'budget_choice_id': 'home', 'llm_differs_from_budget': False, 'selected_evidence': None},
        ])
        self.assertNotIn('do-not-copy', output.getvalue())


if __name__ == '__main__':
    unittest.main()
