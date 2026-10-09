"""Offline tests for comparison timing, stale status and stop handling."""

from copy import deepcopy
import unittest

from compare_navigation import Comparison, RunError, measurements, metric_delta, route_from_json


def snapshot(t=0.0, *, plan_id='', status='idle', pose=(-2.0, -0.5), episode=1):
    return {
        'state': {'episode_id': episode, 'control_mode': 'stopped',
                  'navigation': {'backend': 'custom', 'ready': True}},
        'status': {'plan_id': plan_id, 'state': status, 'plan_complete': status == 'done'},
        'score': {'t': t, 'world_pose_valid': True, 'world_pose': dict(zip(('x', 'y'), pose)),
                  'battery': 60 - t / 10, 'distance_travelled': t / 10, 'collisions': 0,
                  'scenario': 'easy', 'finished': False, 'base_pose': {'x': -2.0, 'y': -0.5}},
    }


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, interval):
        self.now += interval


class FakeAPI:
    def __init__(self, frames):
        self.frames = deepcopy(frames)
        self.calls = []
        self.last = self.frames[0]

    def request(self, path, body=None):
        self.calls.append((path, body))
        if path == 'state':
            if self.frames:
                self.last = self.frames.pop(0)
            return deepcopy(self.last)
        if path == 'command':
            self.last['state']['control_mode'] = 'stopped'
            self.last['status']['state'] = 'preempted'
        return {'ok': True}


def comparison(api, **options):
    clock = Clock()
    params = dict(poll=0.2, ready_timeout=1, goal_timeout=2, wall_timeout=2,
                  tolerance=0.2, clock=clock, sleep=clock.sleep)
    params.update(options)
    return Comparison(api, **params)


class ComparisonTests(unittest.TestCase):
    def test_waypoints_reject_nonfinite_and_boolean_coordinates(self):
        for text in ('[]', '[[true, 0]]', '[[NaN, 0]]', '[[0]]', '{}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                route_from_json(text)
        self.assertEqual(route_from_json('[[1, -2]]'), [[1.0, -2.0]])

    def test_stale_plan_done_does_not_satisfy_new_waypoint(self):
        api = FakeAPI([snapshot(), snapshot(0.2, plan_id='old', status='done', pose=(0, 0)),
                       snapshot(0.4, plan_id='new', status='running'),
                       snapshot(0.6, plan_id='new', status='done', pose=(0.05, 0)),
                       snapshot(0.8, plan_id='new', status='done', pose=(0.05, 0))])
        result = comparison(api).goal('custom', 1, [0, 0], 'new')
        self.assertEqual(result['simulation_elapsed_s'], 0.8)
        self.assertEqual(result['goal_error_m'], 0.05)
        self.assertEqual(api.calls[1][1]['source'], 'manual')

    def test_done_requires_world_pose_within_tolerance(self):
        api = FakeAPI([snapshot(), snapshot(0.2, plan_id='new', status='done'),
                       snapshot(0.4, plan_id='new', status='done')])
        with self.assertRaisesRegex(RunError, 'goal error'):
            comparison(api).goal('custom', 1, [0, 0], 'new')

    def test_paused_simulation_has_wall_deadline(self):
        api = FakeAPI([snapshot(), snapshot(plan_id='new', status='running')])
        with self.assertRaisesRegex(RunError, 'wall-clock timeout'):
            comparison(api).goal('custom', 1, [0, 0], 'new')

    def test_advancing_simulation_has_independent_sim_deadline(self):
        api = FakeAPI([snapshot(), snapshot(2.1, plan_id='new', status='running')])
        with self.assertRaisesRegex(RunError, 'simulation timeout'):
            comparison(api).goal('custom', 1, [0, 0], 'new')

    def test_interrupted_episode_is_not_a_valid_measurement(self):
        api = FakeAPI([snapshot(), snapshot(0.2, episode=2)])
        with self.assertRaisesRegex(RunError, 'episode changed'):
            comparison(api).goal('custom', 1, [0, 0], 'new')

    def test_first_waypoint_failure_stops_route_and_commands_stop(self):
        api = FakeAPI([snapshot(), snapshot(0.2, plan_id='nav-comparison-id-custom-0', status='failed')])
        runner = comparison(api)
        runner.prepare = lambda *_: snapshot()
        result = runner.run('custom', 'easy', [[0, 0], [1, 1]], 'id')
        self.assertFalse(result['success'])
        self.assertTrue(result['stop_confirmed'])
        self.assertEqual(len(result['waypoints']), 1)
        self.assertEqual(sum(path == 'plan' for path, _ in api.calls), 1)
        self.assertIn(('command', {'cmd': 'stop'}), api.calls)

    def test_invalid_truth_pose_is_not_replaced_with_odom(self):
        frame = snapshot()
        frame['score']['world_pose_valid'] = False
        frame['pose'] = {'x': 0, 'y': 0}
        with self.assertRaisesRegex(RunError, 'world pose'):
            measurements(frame)

    def test_metric_reset_is_rejected(self):
        with self.assertRaisesRegex(RunError, 'backwards'):
            metric_delta(measurements(snapshot(2)), measurements(snapshot(1)))

    def test_prepare_fences_reset_and_waits_for_advancing_judge_data(self):
        before = snapshot()
        reset_without_judge = snapshot(episode=2)
        reset_without_judge['score'] = {}
        fresh = snapshot(episode=2)
        advancing = snapshot(0.2, episode=2)
        advancing['score']['distance_travelled'] = 0.0
        api = FakeAPI([before, before, reset_without_judge, fresh, fresh, fresh, fresh, advancing])
        ready = comparison(api).prepare('custom', 'easy')
        self.assertEqual(ready['state']['episode_id'], 2)
        self.assertEqual(ready['score']['t'], 0.2)
        mutations = [(path, body) for path, body in api.calls if path != 'state']
        self.assertEqual(mutations, [
            ('command', {'cmd': 'stop'}), ('navigation', {'backend': 'custom'}),
            ('scenario', {'scenario': 'easy'}), ('command', {'cmd': 'stop'}),
            ('command', {'cmd': 'stop'}),
        ])


if __name__ == '__main__':
    unittest.main()
