"""The live math planner reads observations through the agent contract only."""

import json
from types import SimpleNamespace

from did_llm.agent_link import AgentLink


class NodeStub:
    def __init__(self):
        self.subscriptions = []

    def get_logger(self):
        return SimpleNamespace(info=lambda *_: None)

    def create_publisher(self, *_args):
        return SimpleNamespace(publish=lambda *_: None)

    def create_subscription(self, _message_type, topic, _callback, _depth):
        self.subscriptions.append(topic)

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_000_000_000))


def test_math_link_reads_only_agent_observations_status_and_operator_commands():
    node = NodeStub()
    AgentLink(node, observed_only=True)
    assert set(node.subscriptions) == {'/agent/state', '/agent/status', '/agent/command'}


def test_finished_flag_and_episode_reset_come_from_agent_state():
    link = AgentLink(NodeStub(), observed_only=True)
    link._on_state(SimpleNamespace(data=json.dumps({'episode_id': 1, 'finished': True})))
    assert link.finished()
    link._on_state(SimpleNamespace(data=json.dumps({'episode_id': 2, 'finished': False})))
    assert not link.finished()
    assert link.episode_generation == 1
