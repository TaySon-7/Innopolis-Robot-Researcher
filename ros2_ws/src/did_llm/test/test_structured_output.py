"""Strict Structured Outputs contract for model-generated plans."""

import json

import pytest

from did_llm.agent_plan import PlanRejected
from did_llm.agent_plan import parse_model_plan
from did_llm.agent_plan import plan_response_format
from did_llm.llm_client import LLMClient
from did_llm.llm_client import LLMConfig


def test_response_format_is_strict_and_limited_to_six_subgoals():
    response_format = plan_response_format(6)
    declaration = response_format['json_schema']
    schema = declaration['schema']

    assert response_format['type'] == 'json_schema'
    assert declaration['strict'] is True
    assert schema['additionalProperties'] is False
    assert schema['properties']['subgoals']['maxItems'] == 6
    assert all(
        definition.get('additionalProperties') is False
        for definition in schema['$defs'].values()
    )


def test_model_contract_requires_coordinates_and_rejects_extra_fields():
    with pytest.raises(PlanRejected):
        parse_model_plan({
            'explanation': 'нет координаты x',
            'subgoals': [{'type': 'goto', 'y': 0.5}],
        }, 'p1', max_subgoals=6)

    with pytest.raises(PlanRejected):
        parse_model_plan({
            'explanation': 'лишнее поле',
            'subgoals': [{'type': 'collect', 'radius': 0.5}],
        }, 'p2', max_subgoals=6)


def test_client_sends_response_format_in_chat_completions(monkeypatch):
    captured = {}
    client = LLMClient(LLMConfig(
        base_url='https://example.test/v1',
        api_key='secret',
        model='test-model',
        reasoning_effort='none',
    ))
    monkeypatch.setattr(client, '_resolve', lambda _host: '127.0.0.1')

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                'choices': [{'message': {'content': json.dumps({
                    'explanation': 'test',
                    'subgoals': [{'type': 'collect'}],
                })}}],
            }).encode()

    def fake_urlopen(request, timeout):
        captured['payload'] = json.loads(request.data)
        captured['timeout'] = timeout
        return Response()

    monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
    response_format = plan_response_format(6)
    client._post('system', 'user', response_format=response_format)

    assert captured['payload']['response_format'] == response_format
    assert captured['payload']['reasoning_effort'] == 'none'
