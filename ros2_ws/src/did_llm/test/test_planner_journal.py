"""Readable journal entries and deterministic sensor reactions."""

from did_llm.agent_plan import Subgoal
from did_llm.planner_node import Planner


class _Log:
    def info(self, _message):
        pass

    def warn(self, _message):
        pass


class _Link:
    def __init__(self) -> None:
        self.log = _Log()
        self.state = {'samples_total': 3}
        self.events = [{'event': 'sample_collected', 'collected': 1}]
        self.collected_hint = 0
        self.entries = []
        self.published = []
        self.status = None
        self.clock = 10.0
        self.signal_value = 0.0
        self.noise_value = 0.0
        self.pose_value = (-0.5, -0.55)

    def take_event(self):
        return self.events.pop(0) if self.events else None

    def note_collected(self, value):
        self.collected_hint = value

    def journal(self, kind, title, text='', status='open', **extra):
        self.entries.append({
            'kind': kind,
            'title': title,
            'text': text,
            'status': status,
            **extra,
        })

    def signal(self):
        return self.signal_value

    def noise(self):
        return self.noise_value

    def pose(self):
        return self.pose_value

    def now(self):
        return self.clock

    def publish_plan(self, payload):
        self.published.append(payload)


def test_collection_event_announces_llm_replanning():
    link = _Link()
    planner = Planner(link, object())

    assert planner._handle_event() is True

    assert link.collected_hint == 1
    assert planner.force_replan is True
    assert link.entries == [{
        'kind': 'llm',
        'title': 'Получено новое состояние, готовится следующий план',
        'text': ('Собрано образцов: 1/3. Датчик теперь указывает на ближайший '
                 'оставшийся образец.'),
        'status': 'open',
        'source': 'llm_planner',
    }]


def test_collectable_signal_triggers_direct_collect_without_llm():
    link = _Link()
    link.signal_value = 0.82
    planner = Planner(link, object())

    assert planner._collect_when_close() is True

    assert [item['type'] for item in link.published[0]['subgoals']] == ['collect']
    assert link.published[0]['source'] == 'auto_collect'


def test_collectable_but_noisy_signal_does_not_trigger_collection():
    link = _Link()
    link.signal_value = 0.90
    link.noise_value = 0.15
    planner = Planner(link, object())

    assert planner._collect_when_close() is False
    assert link.published == []


def test_very_strong_signal_survives_hard_scenario_noise():
    link = _Link()
    link.signal_value = 1.0
    link.noise_value = 0.15
    planner = Planner(link, object())

    assert planner._collect_when_close() is True
    assert [item['type'] for item in link.published[0]['subgoals']] == ['collect']


def test_current_subgoal_uses_executor_zero_based_index():
    link = _Link()
    planner = Planner(link, object())
    planner.inflight = 'p1'
    goto = Subgoal(type='goto', x=0.5, y=0.5)
    search = Subgoal(type='search_around', x=0.5, y=0.5, radius=0.4)
    planner.sent_subgoals = [('p1', 0, goto), ('p1', 1, search)]
    link.status = {'plan_id': 'p1', 'index': 0, 'state': 'running'}

    assert planner._current_subgoal() is goto


def test_rising_signal_does_not_restart_an_active_search():
    link = _Link()
    link.signal_value = 0.35
    link.pose_value = (0.0, 0.0)
    planner = Planner(link, object())
    planner.inflight = 'p1'
    search = Subgoal(type='search_around', x=0.0, y=0.0, radius=0.9)
    planner.sent_subgoals = [('p1', 0, search)]
    link.status = {'plan_id': 'p1', 'index': 0, 'state': 'running'}

    assert planner._interrupt_for_sample() is False
    assert link.published == []
