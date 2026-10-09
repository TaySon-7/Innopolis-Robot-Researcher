"""Switching navigation must preempt execution without resetting the episode."""

from types import SimpleNamespace

import pytest

from did_agent.agent_node import AgentNode
from did_agent.controller import Command
from did_agent.nav_node import Navigator


def test_switch_stops_before_confirming_and_preserves_scenario():
    events = []
    node = SimpleNamespace(
        navigation_backend='custom', _navigation_switch=None, _control_mode='stopped',
        _episode_id=7, _score={'scenario': 'hard@7'},
        _on_command=lambda message: events.append(('command', message.data)),
        stop=lambda: events.append(('zero',)),
        core=SimpleNamespace(cancel=lambda: events.append(('cancel',))),
        journal=lambda *_: None,
        _publish_state=lambda: events.append(('state', node.navigation_backend)),
    )
    AgentNode._on_navigation_backend(node, SimpleNamespace(data='nav2'))
    assert node.navigation_backend == 'custom'
    assert node._navigation_switch == 'nav2'
    assert events == [('command', '{"cmd":"stop"}')]
    AgentNode._apply_navigation_backend(node)
    assert events[-3:] == [('zero',), ('cancel',), ('state', 'nav2')]
    assert node._control_mode == 'stopped' and not node._preempt
    assert node._episode_id == 7 and node._score == {'scenario': 'hard@7'}


@pytest.mark.parametrize('backend', ['other', '', 'NAV2', '{"backend":"nav2"}'])
def test_invalid_backend_does_not_interrupt_or_change_navigation(backend):
    warnings = []
    node = SimpleNamespace(navigation_backend='custom', _navigation_switch=None,
                           get_logger=lambda: SimpleNamespace(warning=warnings.append))
    AgentNode._on_navigation_backend(node, SimpleNamespace(data=backend))
    assert warnings and node.navigation_backend == 'custom'
    assert node._navigation_switch is None


def test_reselecting_current_backend_does_not_interrupt_plan():
    node = SimpleNamespace(navigation_backend='nav2', _navigation_switch=None)
    AgentNode._on_navigation_backend(node, SimpleNamespace(data='nav2'))
    assert node._navigation_switch is None


def test_stop_closes_nav2_gate_before_publishing_zero():
    events = []
    node = SimpleNamespace(_nav2=SimpleNamespace(cancel=lambda: events.append('cancel')),
                           publish=lambda command: events.append(command))
    Navigator.stop(node)
    assert events == ['cancel', Command()]


def test_existing_goto_interface_dispatches_to_selected_nav2_and_keeps_guard():
    calls = []
    guard = lambda: False
    node = SimpleNamespace(navigation_backend='nav2', _nav2=SimpleNamespace(
        goto=lambda *args: (calls.append(args) or {'status': 'done', 'backend': 'nav2'})))
    assert Navigator.goto(node, 1, 2, timeout=45, guard=guard)['backend'] == 'nav2'
    assert calls == [(1, 2, 45, guard)]


@pytest.mark.parametrize('mode', ['manual', 'autonomous', 'llm', 'stopped'])
def test_apply_switch_preserves_an_explicit_command_received_after_request(mode):
    pending = object()
    node = SimpleNamespace(navigation_backend='custom', _navigation_switch='nav2',
                           _control_mode=mode, _pending=pending, _preempt=True,
                           core=SimpleNamespace(cancel=lambda: None), stop=lambda: None,
                           journal=lambda *_: None, _publish_state=lambda: None)
    AgentNode._apply_navigation_backend(node)
    assert node.navigation_backend == 'nav2' and node._control_mode == mode
    assert node._pending is pending and not node._preempt
