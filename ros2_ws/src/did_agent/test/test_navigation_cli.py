"""CLI navigation uses the agent plan contract and its matching final result."""

import json
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from did_agent.navigation_cli import goto_plan, main, terminal_result
from did_agent.plan import PlanError, parse_plan


class NavigationCliTests(unittest.TestCase):
    def test_repeated_gotos_have_independent_identity_and_no_backend_override(self):
        first_id, payload = goto_plan(-1.0, 2.0)
        second_id, _ = goto_plan(-1.0, 2.0)
        self.assertNotEqual(first_id, second_id)
        parsed = parse_plan(payload)
        self.assertEqual(parsed.plan_id, first_id)
        self.assertEqual(len(parsed.subgoals), 1)
        self.assertEqual(parsed.subgoals[0].type, 'goto')
        self.assertEqual((parsed.subgoals[0].x, parsed.subgoals[0].y), (-1.0, 2.0))
        self.assertEqual(json.loads(payload)['source'], 'manual')
        self.assertNotIn('navigation_backend', json.loads(payload))

    def test_invalid_coordinates_are_rejected_before_sending(self):
        for value in (float('nan'), float('inf'), -float('inf'), 10.01, True, '1'):
            for x, y in ((value, 0.0), (0.0, value)):
                with self.subTest(x=x, y=y), self.assertRaises(PlanError):
                    goto_plan(x, y)

    def test_unrelated_plans_and_nonterminal_statuses_cannot_finish_the_cli(self):
        messages = [
            'not json', 'null', '[]',
            json.dumps({'plan_id': 'other', 'state': 'done', 'index': 0}),
            json.dumps({'plan_id': 'mine', 'state': 'running', 'index': 0}),
            json.dumps({'plan_id': 'mine', 'state': 'idle', 'index': 0}),
            json.dumps({'plan_id': 'mine', 'state': 'done', 'index': 1}),
        ]
        for message in messages:
            with self.subTest(message=message):
                self.assertIsNone(terminal_result(message, 'mine'))

    def test_success_preserves_navigation_metrics_and_selected_backend(self):
        result = terminal_result(json.dumps({
            'plan_id': 'mine', 'state': 'done', 'index': 0, 'reason': '',
            'data': {'backend': 'nav2', 'distance_to_goal': 0.07, 'replans': 2},
        }), 'mine')
        self.assertEqual(result['status'], 'done')
        self.assertEqual(result['backend'], 'nav2')
        self.assertEqual(result['distance_to_goal'], 0.07)
        self.assertEqual(result['replans'], 2)

    def test_failure_and_preemption_never_become_success_from_nested_data(self):
        for state, reason in (('failed', 'no path'), ('preempted', '')):
            with self.subTest(state=state):
                result = terminal_result(json.dumps({
                    'plan_id': 'mine', 'state': state, 'index': 0, 'reason': reason,
                    'data': {'status': 'done', 'reason': 'old result'},
                }), 'mine')
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['reason'], reason or 'preempted')

    def _run_client(self, outcome):
        """Drive the real CLI loop with deterministic DDS and clock stand-ins."""
        clock, active = [0.0], [True]
        sent, callbacks = [], {}
        interrupted = [False]

        def publisher(message_type, topic, qos):
            return SimpleNamespace(
                get_subscription_count=lambda: 1,
                publish=lambda message: sent.append((topic, json.loads(message.data))),
            )

        node = SimpleNamespace(
            create_publisher=publisher,
            create_subscription=lambda message_type, topic, callback, qos: callbacks.update({topic: callback}),
            count_publishers=lambda topic: 1,
            destroy_node=lambda: None,
        )

        def spin_once(node, timeout_sec):
            clock[0] += timeout_sec
            plans = [payload for topic, payload in sent if topic == '/agent/plan']
            if plans and outcome == 'done':
                callbacks['/agent/status'](SimpleNamespace(data=json.dumps({
                    'plan_id': plans[0]['plan_id'], 'index': 0, 'state': 'done',
                    'data': {'backend': 'nav2', 'distance_to_goal': 0.04},
                })))
            if plans and outcome == 'interrupt' and not interrupted[0]:
                interrupted[0] = True
                raise KeyboardInterrupt

        ros = SimpleNamespace(
            init=lambda **kwargs: None, create_node=lambda name: node,
            ok=lambda: active[0], shutdown=lambda: active.__setitem__(0, False),
            spin_once=spin_once,
        )
        modules = {
            'rclpy': ros,
            'rclpy.signals': SimpleNamespace(SignalHandlerOptions=SimpleNamespace(NO=0)),
            'rclpy.utilities': SimpleNamespace(remove_ros_args=lambda args: args),
            'std_msgs': SimpleNamespace(),
            'std_msgs.msg': SimpleNamespace(String=lambda data: SimpleNamespace(data=data)),
        }
        output = StringIO()
        with patch.dict('sys.modules', modules), redirect_stdout(output), patch(
                'did_agent.navigation_cli.time.monotonic', side_effect=lambda: clock[0]):
            with self.assertRaises(SystemExit) as exit_result:
                main(['goto', '--x', '1', '--y', '0', '--timeout', '0.1'])
        return sent, json.loads(output.getvalue()), exit_result.exception.code

    def test_timeout_stops_agent_before_exiting_without_direct_motion_authority(self):
        sent, result, code = self._run_client('timeout')
        self.assertEqual(code, 1)
        self.assertEqual(result['reason'], 'timeout')
        self.assertEqual([topic for topic, payload in sent], ['/agent/plan', '/agent/command'])
        self.assertEqual(sent[-1][1], {'cmd': 'stop'})

    def test_keyboard_interrupt_delivers_stop_before_exiting(self):
        sent, result, code = self._run_client('interrupt')
        self.assertEqual(code, 1)
        self.assertIn('interrupted', result['reason'])
        self.assertEqual(sent[-1], ('/agent/command', {'cmd': 'stop'}))

    def test_success_returns_selected_backend_without_sending_stop(self):
        sent, result, code = self._run_client('done')
        self.assertEqual(code, 0)
        self.assertEqual(result['backend'], 'nav2')
        self.assertEqual([topic for topic, payload in sent], ['/agent/plan'])


if __name__ == '__main__':
    unittest.main()
