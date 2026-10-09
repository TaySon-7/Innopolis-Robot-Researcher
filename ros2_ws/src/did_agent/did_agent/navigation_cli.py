"""Send CLI navigation through the single agent that owns motion and maps."""

from __future__ import annotations

import argparse
import json
from math import isfinite
import sys
import time
from typing import Any
from uuid import uuid4

from did_agent.plan import PlanError, parse_plan


def goto_plan(x: float, y: float) -> tuple[str, str]:
    """Create a unique, validated manual plan without choosing a backend."""
    plan_id = f'goto-cli-{uuid4().hex}'
    payload = json.dumps({
        'plan_id': plan_id,
        'source': 'manual',
        'subgoals': [{'type': 'goto', 'x': x, 'y': y}],
    })
    parse_plan(payload)
    return plan_id, payload


def terminal_result(payload: str, plan_id: str) -> dict[str, Any] | None:
    """Ignore unrelated/progress messages and retain navigation result fields."""
    try:
        status = json.loads(payload)
    except (TypeError, ValueError):
        return None
    if not isinstance(status, dict) or status.get('plan_id') != plan_id:
        return None
    state = status.get('state')
    if state not in ('done', 'failed', 'preempted'):
        return None
    if state == 'done' and status.get('index') != 0:
        return None
    details = status.get('data')
    return {
        **(details if isinstance(details, dict) else {}),
        'status': 'done' if state == 'done' else 'failed',
        'reason': status.get('reason') or ('preempted' if state == 'preempted' else ''),
        'plan_id': plan_id,
    }


def main(args=None) -> None:
    """``ros2 run did_agent goto --x 1 --y 0`` using the active agent backend."""
    # Keep plan/result helpers usable in ordinary Python tests without ROS.
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from rclpy.utilities import remove_ros_args
    from std_msgs.msg import String

    parser = argparse.ArgumentParser(description='Drive through the active agent navigator.')
    parser.add_argument('--x', type=float, required=True)
    parser.add_argument('--y', type=float, required=True)
    parser.add_argument('--timeout', type=float, default=180.0,
                        help='Maximum CLI wait in wall-clock seconds (default: 180).')
    options = parser.parse_args(remove_ros_args(args if args is not None else sys.argv)[1:])
    if not isfinite(options.timeout) or options.timeout <= 0:
        parser.error('--timeout must be a positive finite number')
    try:
        plan_id, payload = goto_plan(options.x, options.y)
    except PlanError as error:
        parser.error(str(error))

    # Keep the ROS context alive when Ctrl-C raises KeyboardInterrupt so that
    # the stop command can be delivered before destroying our publishers.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node(f'goto_client_{uuid4().hex[:12]}')
    plans = node.create_publisher(String, '/agent/plan', 10)
    commands = node.create_publisher(String, '/agent/command', 10)
    result = None
    dispatched = False

    def on_status(message) -> None:
        nonlocal result
        final = terminal_result(message.data, plan_id)
        if final is not None:
            result = final

    node.create_subscription(String, '/agent/status', on_status, 50)

    def stop_agent() -> None:
        if not dispatched or not rclpy.ok():
            return
        commands.publish(String(data=json.dumps({'cmd': 'stop'})))
        # Reliable DDS delivery needs the publisher to stay alive briefly.
        stop_deadline = time.monotonic() + 0.75
        while rclpy.ok() and time.monotonic() < stop_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)

    try:
        deadline = time.monotonic() + options.timeout
        discovery_deadline = min(deadline, time.monotonic() + 15.0)
        while rclpy.ok() and time.monotonic() < discovery_deadline:
            if (plans.get_subscription_count() and commands.get_subscription_count()
                    and node.count_publishers('/agent/status')):
                break
            rclpy.spin_once(node, timeout_sec=0.05)
        if not (rclpy.ok() and plans.get_subscription_count()
                and commands.get_subscription_count() and node.count_publishers('/agent/status')):
            result = {'status': 'failed', 'reason': 'agent unavailable; start the demo before using goto',
                      'plan_id': plan_id}
        else:
            plans.publish(String(data=payload))
            dispatched = True
            while rclpy.ok() and result is None and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
            if result is None:
                stop_agent()
                result = {'status': 'failed', 'reason': 'timeout' if rclpy.ok() else 'ROS shutdown',
                          'plan_id': plan_id}
    except KeyboardInterrupt:
        stop_agent()
        result = {'status': 'failed', 'reason': 'interrupted; stop requested', 'plan_id': plan_id}
    except Exception:
        stop_agent()
        raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result and result['status'] == 'done' else 1)
