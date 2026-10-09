#!/usr/bin/env python3
"""Render recorded Gazebo planner comparisons; no simulator, network or dependencies.

    python3 scripts/report_planner_comparison.py path/to/results.json --output-dir path/to/report

Only recorded metrics are reported. Missing runs stay visible, invalid runs are
excluded from paired differences, and valid timeouts retain their horizon score.
"""

from __future__ import annotations

import argparse
import csv
from html import escape
import io
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote


MISSING = '—'
ARMS = ('autonomous', 'budget', 'llm')
LABELS = {'autonomous': 'Автономный', 'budget': 'Математика', 'llm': 'Математика + LLM'}
OUTCOMES = {'finished': 'На базе', 'battery_depleted': 'Разряд',
            'simulation_timeout': 'Лимит sim', 'wall_timeout': 'Лимит wall',
            'autonomous_terminal': 'Алгоритм завершён', 'invalid': 'Невалиден',
            'interrupted': 'Прерван'}
CSV_FIELDS = ('scenario', 'arm', 'outcome', 'valid', 'censored', 'score', 'collected',
              'samples_total', 'finished', 'collisions', 'battery', 'distance_m',
              'simulation_elapsed_s', 'episode_simulation_s', 'wall_elapsed_s',
              'api_calls', 'successful_exchanges', 'accepted_llm_observed',
              'fallback_decision_share', 'llm_differs_from_budget', 'diagnostic_choices',
              'diagnostic_multiple_search_choices', 'response_latency_mean_s', 'response_latency_max_s')


def number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def numeric(value: Any) -> int | float | None:
    return value if number(value) else None


def fmt(value: Any, digits: int = 1) -> str:
    if isinstance(value, bool):
        return 'да' if value else 'нет'
    if not number(value):
        return MISSING
    return f'{value:.{digits}f}'.rstrip('0').rstrip('.') if digits else f'{value:.0f}'


def count(mapping: Any, key: str) -> int | None:
    if not isinstance(mapping, dict):
        return None
    value = mapping.get(key, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def latency_stats(values: Any) -> tuple[float | None, float | None]:
    valid = [value for value in values if number(value) and value > 0] if isinstance(values, list) else []
    if not valid:
        return None, None
    return math.fsum(value / len(valid) for value in valid), max(valid)


def diagnostics(run: dict) -> dict:
    """These compare valid API responses, not counterfactual mission outcomes."""
    source = (run.get('planner_metrics') or {}).get('decision_diagnostics')
    if not isinstance(source, list):
        return {'different': None, 'choices': None, 'multiple': None}
    choices = {}
    for item in source:
        if (isinstance(item, dict) and isinstance(item.get('snapshot_id'), str)
                and isinstance(item.get('goal_id'), str)
                and isinstance(item.get('llm_differs_from_budget'), bool)):
            choices[(item['snapshot_id'], item['goal_id'])] = item
    return {'different': sum(item['llm_differs_from_budget'] for item in choices.values()),
            'choices': len(choices),
            'multiple': sum(number(item.get('feasible_search_count'))
                            and item['feasible_search_count'] > 1 for item in choices.values())}


def rows_for(data: dict) -> list[dict]:
    runs = data.get('runs', [])
    if not isinstance(runs, list) or any(not isinstance(run, dict) for run in runs):
        raise ValueError('runs must be a list of objects')
    indexed = {}
    for run in runs:
        key = (run.get('scenario'), run.get('arm'))
        if not all(isinstance(value, str) for value in key) or key[1] not in ARMS:
            raise ValueError('each run requires scenario and a recognized arm')
        if key in indexed:
            raise ValueError(f'duplicate scenario/arm: {key}; do not silently choose a replicate')
        indexed[key] = run
    ordered = []
    for pair in data.get('schedule', []):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError('schedule entries must be [scenario, arm]')
        if not all(isinstance(value, str) for value in pair) or pair[1] not in ARMS:
            raise ValueError('schedule contains an invalid scenario/arm')
        if pair[0] not in ordered:
            ordered.append(pair[0])
    for scenario, _ in indexed:
        if scenario not in ordered:
            ordered.append(scenario)
    scheduled = {tuple(pair) for pair in data.get('schedule', [])} | set(indexed)
    result = []
    for scenario in ordered:
        for arm in ARMS:
            if (scenario, arm) not in scheduled:
                continue
            run = indexed.get((scenario, arm))
            metric = (run or {}).get('metrics') or {}
            planner = (run or {}).get('planner_metrics') or {}
            diag = diagnostics(run or {})
            mean_latency, max_latency = latency_stats(planner.get('response_latency_s'))
            row = dict.fromkeys(CSV_FIELDS)
            row.update(scenario=scenario, arm=arm, outcome=(run or {}).get('outcome', 'pending'),
                       valid=(run or {}).get('valid'), censored=(run or {}).get('censored'),
                       finished=metric.get('finished'), wall_elapsed_s=numeric((run or {}).get('wall_elapsed_s')),
                       api_calls=count(planner.get('client_stats'), 'calls_made'),
                       successful_exchanges=numeric(planner.get('successful_exchanges')),
                       accepted_llm_observed=count((run or {}).get('decisions_accepted_observed'), 'llm'),
                       fallback_decision_share=numeric((run or {}).get('fallback_decision_share')),
                       llm_differs_from_budget=diag['different'], diagnostic_choices=diag['choices'],
                       diagnostic_multiple_search_choices=diag['multiple'],
                       response_latency_mean_s=mean_latency, response_latency_max_s=max_latency,
                       error=(run or {}).get('error') or (run or {}).get('finish_reason'),
                       present=run is not None)
            for key in ('score', 'collected', 'samples_total', 'collisions', 'battery', 'distance_m',
                        'simulation_elapsed_s', 'episode_simulation_s'):
                row[key] = numeric(metric.get(key))
            result.append(row)
    return result


def paired_differences(rows: list[dict]) -> list[dict]:
    indexed = {(row['scenario'], row['arm']): row for row in rows}
    pairs = []
    for llm in rows:
        budget = indexed.get((llm['scenario'], 'budget'))
        if llm['arm'] != 'llm' or llm['valid'] is not True or not budget or budget['valid'] is not True:
            continue
        pair = {'scenario': llm['scenario'], 'censored': llm['censored'] is True or budget['censored'] is True,
                'llm_finished': llm['finished'], 'budget_finished': budget['finished']}
        for key in ('score', 'collected', 'battery', 'simulation_elapsed_s', 'wall_elapsed_s'):
            pair[key] = llm[key] - budget[key] if number(llm[key]) and number(budget[key]) else None
        pairs.append(pair)
    return pairs


def conclusion(rows: list[dict], pairs: list[dict]) -> str:
    pending = sum(not row['present'] for row in rows)
    prefix = f'Промежуточный отчёт: результатов ещё нет для {pending} прогонов. ' if pending else ''
    if not pairs:
        return prefix + 'Пока нет валидной пары LLM/математика; оценивать преимущество рано.'
    paired_scenarios = {pair['scenario'] for pair in pairs}
    llm_rows = [row for row in rows if row['arm'] == 'llm' and row['scenario'] in paired_scenarios]

    def evidence(row: dict) -> bool:
        return (number(row['accepted_llm_observed']) and row['accepted_llm_observed'] > 0
                and number(row['successful_exchanges']) and row['successful_exchanges'] > 0)

    mixed = any(not evidence(row) or (number(row['fallback_decision_share'])
                                     and row['fallback_decision_share'] > 0) for row in llm_rows)
    caveat = (' Сравнение описывает гибридную систему: есть fallback или пары без подтверждённого '
              'использования модели; весь эффект приписывать LLM нельзя.') if mixed else ''
    if not any(evidence(row) for row in llm_rows):
        return prefix + 'Настоящие ответы API и принятые LLM-планы пока не подтверждены вместе; полезность LLM не установлена.' + caveat
    scores = [pair['score'] for pair in pairs if number(pair['score'])]
    if not scores:
        return prefix + 'Для валидных пар нет полного счёта; численный эффект LLM не рассчитан.' + caveat
    worse_return = any(pair['budget_finished'] is True and pair['llm_finished'] is False for pair in pairs)
    if worse_return:
        text = 'В одной из пар LLM не завершил возврат, а математика завершила; пользу с сохранением надёжности возврата этот пилот не подтверждает.'
    elif all(value > 0 for value in scores):
        text = 'Режим с LLM получил больший счёт во всех измеренных парах. Это положительный результат пилота; нужны новые seed и baseline с приоритетом сигнала.'
    elif all(value == 0 for value in scores):
        text = 'Счёт в измеренных парах одинаков. Выгода LLM по основной метрике здесь не показана; сравните время и число API-запросов.'
    elif all(value <= 0 for value in scores):
        text = 'Преимущества LLM по счёту в измеренных парах нет. Журналы помогут разделить задержки API, выбор цели и ошибки общего поиска.'
    else:
        text = 'Результаты по счёту разнонаправленные; устойчивое преимущество LLM этот пилот не показывает.'
    if any(pair['censored'] for pair in pairs):
        text += ' Пары с лимитом сравнивают накопленный результат на горизонте, а не завершённые миссии.'
    return prefix + text + caveat


def render_csv(rows: list[dict]) -> str:
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow({key: MISSING if row.get(key) is None else row[key] for key in CSV_FIELDS})
    return output.getvalue()


def signed(value: Any) -> str:
    return ('+' if value > 0 else '') + fmt(value) if number(value) else MISSING


def table(headers: list[str], body: list[list[str]]) -> str:
    return '<table><thead><tr>' + ''.join(f'<th>{escape(v)}</th>' for v in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{escape(v)}</td>' for v in row) + '</tr>' for row in body) + '</tbody></table>'


def render_html(data: dict, rows: list[dict], source_href: str, csv_href: str) -> str:
    pairs = paired_differences(rows)
    body = []
    for row in rows:
        status = OUTCOMES.get(row['outcome'], 'Результат ещё не записан' if row['outcome'] == 'pending' else row['outcome'])
        if row['present'] and row['valid'] is not True and row['outcome'] not in ('invalid', 'interrupted'):
            status = 'Невалиден: ' + status
        share = fmt(row['fallback_decision_share'] * 100, 0) + '%' if number(row['fallback_decision_share']) else MISSING
        body.append([row['scenario'], LABELS[row['arm']], status, fmt(row['score']),
                     f"{fmt(row['collected'], 0)}/{fmt(row['samples_total'], 0)}", fmt(row['finished']),
                     fmt(row['collisions'], 0), fmt(row['battery']), fmt(row['simulation_elapsed_s']),
                     fmt(row['wall_elapsed_s']), fmt(row['api_calls'], 0), share])
    run_table = table(['Сценарий', 'Режим', 'Итог', 'Счёт', 'Образцы', 'Финиш', 'Столк.',
                       'Батарея', 'Sim, с', 'Wall, с', 'API', 'Fallback'], body)
    pair_table = table(['Пара: LLM − математика', 'Δ счёт', 'Δ образцы', 'Δ батарея', 'Δ sim, с', 'Δ wall, с', 'Лимит'],
                       [[pair['scenario'], *[signed(pair[key]) for key in
                         ('score', 'collected', 'battery', 'simulation_elapsed_s', 'wall_elapsed_s')],
                         'да' if pair['censored'] else 'нет'] for pair in pairs]) if pairs else '<p>Валидных пар пока нет.</p>'
    details = []
    for row in rows:
        if row['arm'] != 'llm' or not row['present']:
            continue
        n, different = row['diagnostic_choices'], row['llm_differs_from_budget']
        fraction = f'{different}/{n} ({100 * different / n:.0f}%)' if number(n) and n > 0 and number(different) else MISSING
        details.append(f"{row['scenario']}: ответы API {fmt(row['successful_exchanges'], 0)}; "
                       f"наблюдалось принятых LLM-планов {fmt(row['accepted_llm_observed'], 0)}; "
                       f"задержка ответа, средняя/макс. {fmt(row['response_latency_mean_s'])}/{fmt(row['response_latency_max_s'])} с; "
                       f"ответ отличается от budget {fraction}; "
                       f"ответов при ≥2 целях поиска {fmt(row['diagnostic_multiple_search_choices'], 0)}.")
    errors = [f"{row['scenario']}/{row['arm']}: {row['error']}" for row in rows
              if row['present'] and row['valid'] is not True and row.get('error')]
    limits = data.get('limits') or {}
    meta = (f"Gazebo · навигация {data.get('backend', MISSING)} · "
            f"записано {sum(row['present'] for row in rows)}/{len(rows)} прогонов · "
            f"лимиты: {fmt(limits.get('episode_simulation_s'), 0)} с эпизода / "
            f"{fmt(limits.get('mission_wall_s'), 0)} с wall · {data.get('created_at', MISSING)}")
    return '<!doctype html>\n<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' + '''
<title>Автономный робот и LLM — сравнение</title>
<style>body{font:14px/1.45 system-ui,sans-serif;color:#192936;max-width:1250px;margin:28px auto;padding:0 22px}h1{font-size:25px;margin:0 0 7px}h2{font-size:17px;margin:20px 0 8px}p{margin:8px 0}.meta,.note{color:#586873;font-size:12px}.lead{background:#eef4f7;border-left:4px solid #387390;padding:12px}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:7px 6px;text-align:right;border-bottom:1px solid #dce4e8}th{background:#edf2f5}td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){text-align:left}tr:nth-child(even){background:#fafbfc}.scroll{overflow-x:auto}a{color:#176285}@media print{body{margin:10mm;font-size:11px;padding:0}h1{font-size:20px}table{font-size:10px}td,th{padding:4px}h2{margin-top:12px}.lead{padding:8px}}</style>
<h1>Автономный робот и LLM: результат пилота</h1>''' + f'''
<p class="meta">{escape(meta)}</p><p class="lead">{escape(conclusion(rows, pairs))}</p>
<div class="scroll">{run_table}</div>
<p class="note">Sim и Wall — от команды запуска; лимит sim — абсолютное время эпизода. Финиш означает подтверждение судьи на базе. «{MISSING}» — нет данных. Fallback — доля опубликованных решений, заменённых резервным выбором; обязательный возврат budget в неё не входит.</p>
<h2>Вклад выбора LLM при одинаковых предложениях целей</h2>{pair_table}
<p class="note">Только валидные пары одного сценария; тайм-ауты включены и помечены. Положительная Δ батареи — больше остаток, отрицательная Δ времени — быстрее. Невалидные прогоны исключены из разностей.</p>
<p>{escape(' '.join(details) or 'Диагностика настоящих ответов API пока не записана.')}</p>
<p class="note">Счётчики принятых планов — нижние оценки из polling; ответы API не равны принятым планам. Различие с budget считается на одном и том же предложении целей, по ответам API; это не доказательство лучшего результата выбранной цели.</p>
<h2>Границы вывода</h2><p>Один seed на уровень — описательный пилот, без статистического доказательства полезности или 95% возврата. Математика выбирает самый дешёвый допустимый поиск; LLM дополнительно предложено предпочитать наблюдаемый сигнал. Следующая проверка — новые seed и простое правило «сначала сигнал, затем бюджет».</p>
<p class="note">Автономный режим использует сетку 1,0 м и поиск по сигналу; планировщики — пул 1,3 м, пять ближайших целей и иной порядок поисков. Поэтому LLM − математика лучше отделяет вклад модели, чем LLM − автономный. Ожидаемый счёт и вероятности находок не рассчитаны; денежная стоимость API не измерялась.</p>
''' + (f'<p class="note">Невалидные прогоны: {escape("; ".join(errors))}</p>' if errors else '') + f'''
<p class="note"><a href="{escape(source_href, quote=True)}">Исходный results.json</a> · <a href="{escape(csv_href, quote=True)}">Таблица CSV</a></p></html>\n'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.results.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or data.get('schema') != 'planner-comparison@1':
            raise ValueError('expected planner-comparison@1 results')
        rows = rows_for(data)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path, html_path = args.output_dir / 'planner-comparison.csv', args.output_dir / 'report.html'
        if args.results.resolve() in (csv_path.resolve(), html_path.resolve()):
            raise ValueError('report cannot overwrite its source')
        href = quote(os.path.relpath(args.results.resolve(), args.output_dir.resolve()), safe='/')
        csv_path.write_text(render_csv(rows), encoding='utf-8-sig')
        html_path.write_text(render_html(data, rows, href, csv_path.name), encoding='utf-8')
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(html_path.resolve())
    print(csv_path.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
