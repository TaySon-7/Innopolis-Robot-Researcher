"""Prompts for the planner and the hypothesis writer.

The planner prompt carries the state straight through as JSON. Earlier it was
compressed into a sector digest, which was a mistake: the fields that decide
whether the robot should keep going are exactly the ones a digest drops —
``return_cost_estimate``, the three anomaly flags and the reason from the
last subgoal. Those are the numbers a plan is judged by, so they go in
verbatim.
"""

from __future__ import annotations

from typing import Any

from did_llm.agent_plan import (
    ARENA,
    BASE_X,
    BASE_Y,
    NOISE_UNTRUSTWORTHY,
    PILLAR_KEEPOUT,
    RADIUS_MAX,
    RADIUS_MIN,
    SIGNAL_COLLECTABLE,
    SIGNAL_NEAR,
    TARGET_SUBGOALS,
    WALL_MARGIN,
)

PLANNER_PROMPT_VERSION = 'agent-plan@6'

# This is the LLM planning horizon, not the executor's wire-format safety
# ceiling (50).  The live Planner rebuilds the system prompt from its
# ``max_subgoals`` ROS parameter, whose configured value is currently six.
DEFAULT_PLANNER_MAX_SUBGOALS = 6

#: Starting battery in the judge's scenarios, from INTERFACES section 2.
BATTERY_FULL = 60.0

#: Planning angles, rotated every round.
#:
#: Two reasons, and the second is the one that matters. The endpoint caches by
#: exact prompt, so an unchanged state returns the previous answer in
#: milliseconds — the planner then republishes a plan the model never rethought,
#: and the run looks adaptive while nothing has been decided. Rotating the
#: directive makes every prompt unique. But a nonce alone would only break the
#: cache; these are genuine alternative priorities, so the model is also being
#: asked the same question from a different angle each time.
STRATEGIES: tuple[str, ...] = (
    'Ближайшая цель: если датчик что-то слышит — искать на месте; если нет — '
    'выбрать ближайшую к текущей позе свободную точку.',
    'Выгодная цель: предпочитай точки с обычным грунтом, даже если они дальше; '
    'дорогой пол может стоить дороже, чем найденный образец.',
    'Систематичный обход: иди по краю арены по часовой стрелке, точки выбирай '
    'равномерно, чтобы не пропускать углы.',
    'От базы: проверь сначала ближний к базе пояс, потом двигайся к центру — '
    'так при нехватке батареи ты останешься рядом с домом.',
    'По сигналу: если достоверный сигнал выше порога, запускай search_around '
    'из текущей pose — локальный навык сам измерит пространственный градиент.',
    'Экономия: план должен быть короче, иначе батареи не хватит; лучше один-два '
    'локальных поиска, а return_to_base оставь только для реальной нехватки батареи.',
)

def build_planner_system(
    max_subgoals: int = DEFAULT_PLANNER_MAX_SUBGOALS,
) -> str:
    """Build the system message from the planner's real plan-length limit."""
    if max_subgoals < 1:
        raise ValueError('max_subgoals must be positive')
    return f"""\
Ты — планировщик верхнего уровня мобильного робота. Ты выбираешь точки
и навыки, но НЕ управляешь скоростью, лидаром или колёсами. Низкоуровневый
исполнитель сам строит путь, обходит препятствия и вычисляет локальный градиент.

МИССИЯ
Собрать максимум доступных образцов и завершить эпизод на базе. Точное число
образцов и уже собранное число приходят в блоке «СЦЕНАРИЙ ТЕКУЩЕГО ЗАПУСКА».
return_to_base заканчивает эпизод. Он разрешён только, когда собраны все образцы или
когда безопасного бюджета на ещё один поиск уже нет. Обычный план не обязан заканчиваться
возвратом: после его выполнения ты получишь новое состояние и составишь следующий план.

СЦЕНАРИЙ
В каждом запросе тебе передаётся уровень easy, medium или hard, общее число образцов
и обзор возможных событий. Обзор описывает правила, но не раскрывает скрытые координаты или
время событий. Не выдумывай их из названия сценария или seed. В hard считай изменение
свершившимся только после наблюдаемого события, новой карты стоимости или флага anomaly.

ГЕОМЕТРИЯ
База: ({BASE_X}, {BASE_Y}). Координаты мировые, в метрах.
Точные границы и позиции столбов приходят в сообщении с состоянием, в разделе
«ГЕОМЕТРИЯ АРЕНЫ». Держись там подальше от столбов: план с точкой на столбе
не пройдёт проверку и будет отклонён.

ЖЁСТКИЕ ПРАВИЛА, НАРУШЕНИЕ ОТКЛОНЯЕТСЯ
1. После каждого search_around сразу ставь collect. Не начинай план с collect.
2. Если sensor.value >= {SIGNAL_NEAR:g} и sensor.noise_estimate < {NOISE_UNTRUSTWORTHY:g}, образец рядом:
начинай с search_around в текущей pose, затем collect. Не ставь перед ними goto в другую точку.
3. Если сигнала нет или он ненадёжен из-за шума, выбери новую свободную точку:
goto к этой точке, search_around с центром в ней, затем collect.
4. Не ставь return_to_base, если образцы ещё остались и «бюджет на новые поиски» больше 5.
5. Не ставь goto или search_around вне арены, на столбе или в известной дорогой зоне.

ПОДЦЕЛИ
- {{"type": "goto", "x": …, "y": …}} — доехать до точки.
- {{"type": "search_around", "x": …, "y": …, "radius": …}} — запустить
локальный градиентный поиск: исполнитель измерит сигнал в нескольких точках,
оценит направление его роста, уменьшит шаг и закончит у найденного максимума.
radius от {RADIUS_MIN} до {RADIUS_MAX} м.
- {{"type": "collect"}} — собрать найденный образец.
- {{"type": "return_to_base"}} — вернуться на базу и завершить эпизод.

СИГНАЛ И ГРАДИЕНТ
- sensor.value — одно скалярное измерение без направления. По одному значению
нельзя угадать, в какую сторону ехать.
- Градиент вычисляет исполнитель внутри search_around по серии измерений в
разных координатах. Не раскладывай этот локальный поиск на набор goto и не
пытайся сам угадать направление.
- Твоя задача — выбрать центр и радиус поиска по жёстким правилам выше. Не превращай
просто сильный сигнал в выдуманное направление.
- Исполнитель реагирует на датчик внутри навыка значительно чаще, чем приходит
следующий ответ модели, поэтому не нужен новый LLM-план для каждого измерения.
- При надёжном sensor.value >= {SIGNAL_COLLECTABLE:g} планировщик сам запускает
короткий search_around + collect на месте, не ожидая нового ответа модели.

КАК ЧИТАТЬ СОСТОЯНИЕ
- sensor.value — близость к ближайшему несобранному образцу, 0..1, но не направление.
- sensor.noise_estimate >= {NOISE_UNTRUSTWORTHY:g} — мгновенному sensor.value нельзя доверять.
- return_cost_estimate — сколько батареи стоит дорога домой по текущей карте
- anomaly — флаги: battery_deviation (расход выше ожидаемого), penalties_burst
(серия штрафов), sensor_noise_up (датчик шумит). Флаг требует перепланирования, но сам по себе
не разрешает ранний return_to_base.
- cost_map_updates — что агент уже знает о дорогом грунте. Не планируй путь
через известно дорогую зону.
- Если блок «ПОСЛЕДНИЙ РЕЗУЛЬТАТ ПОДЦЕЛИ» имеет state=failed, посмотри reason: «no path to goal» значит, что
через эту точку не пройти, «no sample within radius» — образца там нет,
«signal too weak» — сигнал недостаточный, «battery reserve reached» —
резерв на возврат достигнут, новый поиск начинать нельзя.

ПРАВИЛА
1. Обычно {min(TARGET_SUBGOALS, max_subgoals)}–{max_subgoals} подцелей,
максимум {max_subgoals}. Исключения: search_around + collect на текущей позиции — 2,
а return_to_base — 1. Не добавляй шаги для количества: короткий горизонт нужен для реакции на изменения.
2. При надёжном сигнале ищи из текущей pose; без него перемещайся к новой точке обхода.
3. Найден образец (collect) — потом ищи следующий.
4. Все образцы собраны или бюджета на новый поиск не осталось — return_to_base.
5. Не повторяй точку, на которой уже был и где ничего не нашлось.

ПРИМЕР ниже — для случая без сигнала, когда образцы остались и бюджета хватает.
ФОРМАТ ОТВЕТА — строго JSON по переданной JSON Schema, без текста вне JSON:
{{"explanation": "одно предложение: почему такой план", "subgoals": [
  {{"type": "goto", "x": -0.75, "y": 0.25}},
  {{"type": "search_around", "x": -0.75, "y": 0.25, "radius": 0.8}},
  {{"type": "collect"}},
  {{"type": "goto", "x": 0.75, "y": 0.25}},
  {{"type": "search_around", "x": 0.75, "y": 0.25, "radius": 0.8}},
  {{"type": "collect"}}
]}}
"""


# Compatibility export for tests and tools that inspect the default prompt.
# The live planner uses ``build_planner_system(self.cfg.max_subgoals)`` below.
PLANNER_SYSTEM = build_planner_system()

HYPOTHESIS_SYSTEM = """\
Ты — исследователь мобильного робота в арене. Формулируй ПРОВЕРЯЕМЫЕ гипотезы
о среде, а не описания.

ТРЕБОВАНИЯ
1. claim — одно утверждение о среде, которое можно опровергнуть измерением.
2. testable — машинно-проверяемое условие с числами, например
   "drain_A > 1.3 * drain_mean" или "sd_now > 2.5 * sd_baseline".
   Порог 1.3 выбран не произвольно: проверка гипотез идёт ровно этим правилом,
   и условие с другим порогом вердикта не получит.
3. measurement — что именно измеряем и в каких единицах.

ЗАПРЕЩЕНО
- Утверждения, которые нельзя опровергнуть ("робот работает хорошо").
- Ссылки на то, чего нет во входных данных.
- Выдуманные числа измерений.

ФОРМАТ ОТВЕТА — строго JSON:
{"hypotheses": [
  {"claim": "...", "testable": "...", "measurement": "..."}
]}
"""


def build_planner_prompt(mission: str, state: dict[str, Any],
                         status: dict[str, Any] | None,
                         feedback: str,
                         expensive: list[dict[str, float]] | None = None,
                         budget: dict[str, Any] | None = None,
                         round_number: int = 0) -> str:
    """Assemble the planner prompt from one state snapshot.

    ``round_number`` rotates the strategy directive, which both defeats the
    endpoint's prompt cache and gives the model a genuinely different angle.
    """
    parts = [
        f'МИССИЯ: {mission}',
        '',
        f'ПОДХОД К ЭТОМУ ПЛАНУ (раунд {round_number}): '
        f'{STRATEGIES[round_number % len(STRATEGIES)]}',
        '',
        _scenario_block(state),
        '',
        _geometry_block(),
        '',
        _ground_block(expensive),
        '',
        _budget_block(state, budget),
        '',
        'СОСТОЯНИЕ:',
        _json(state),
    ]
    if status:
        parts += ['', 'ПОСЛЕДНИЙ РЕЗУЛЬТАТ ПОДЦЕЛИ:', _json(status)]
    if feedback:
        # The executor's own wording goes in verbatim: it names the rule that
        # was broken, which is more useful than anything reworded here.
        parts += ['', 'ЧТО БЫЛО НЕ ТАК С ПРОШЛЫМ ПЛАНОМ:', feedback]
    parts += ['', 'ДАЙ СЛЕДУЮЩИЙ ПЛАН.']
    return '\n'.join(parts)


def _scenario_block(state: dict[str, Any]) -> str:
    """Describe scenario rules without leaking its hidden layout or schedule."""
    raw_name = str(state.get('scenario') or '').strip().lower()
    difficulty = raw_name.partition('@')[0]
    generated = '@' in raw_name

    total = state.get('samples_total')
    collected = state.get('collected')
    total_count = int(total) if isinstance(total, (int, float)) else None
    collected_count = int(collected) if isinstance(collected, (int, float)) else None

    lines = ['СЦЕНАРИЙ ТЕКУЩЕГО ЗАПУСКА:']
    if raw_name:
        lines.append(f'  режим: {raw_name}')
    else:
        lines.append('  режим: не передан; не угадывай его')

    if total_count is not None and collected_count is not None:
        remaining = max(0, total_count - collected_count)
        lines.append(
            f'  образцы: всего {total_count}, собрано {collected_count}, '
            f'осталось {remaining}'
        )
    elif total_count is not None:
        lines.append(f'  образцы: всего {total_count}')

    if difficulty == 'easy':
        lines += [
            '  профиль easy: 1 зона медленного/дорогого грунта',
            '  среда статична: событий во время прогона нет',
        ]
    elif difficulty == 'medium':
        lines += [
            '  профиль medium: 3 зоны медленного/дорогого грунта',
            '  среда статична: событий во время прогона нет',
        ]
    elif difficulty == 'hard':
        lines += [
            '  профиль hard: 4 исходные зоны медленного/дорогого грунта',
            '  во время прогона возможны: изменение стоимости грунта, '
            'появление новой опасной зоны и временный рост шума датчика',
            '  точные места и время этих событий тебе неизвестны',
        ]
    else:
        lines.append(
            '  профиль сложности неизвестен; опирайся только на наблюдения'
        )

    if generated:
        lines.append(
            '  seed делает раскладку воспроизводимой, но не раскрывает тебе '
            'координаты образцов, грунта или опасностей'
        )
    lines.append(
        '  скрытые координаты неизвестны; ищи по датчику и известной карте'
    )
    return '\n'.join(lines)


def _ground_block(expensive: list[dict[str, float]] | None) -> str:
    """Ground the agent has already measured, and what it costs.

    Without this the model cannot route around expensive floor: it has no way
    to know where the dear patches are, so it sweeps the arena at large and
    burns the battery crossing them.
    """
    if not expensive:
        return ('ДОРОГОЙ ГРУНТ: агент пока не измерил ни одной дорогой зоны. '
                'Точные места из профиля сценария неизвестны: не выдумывай их, '
                'а реагируй после измерения.')
    lines = ['ДОРОГОЙ ГРУНТ (измерен агентом, цена за метр пути):']
    for item in expensive[:12]:
        lines.append(f'  ({item["x"]:.2f}; {item["y"]:.2f}) цена ×{item["cost"]:.1f}'
                     f' в радиусе {item["reach"]:.2f} м')
    if len(expensive) > 12:
        lines.append(f'  …ещё {len(expensive) - 12}')
    lines.append('Через эти точки не ездить и не искать: план с такой точкой '
                 'будет отклонён.')
    return '\n'.join(lines)


def _budget_block(state: dict[str, Any],
                  budget: dict[str, Any] | None) -> str:
    """What is actually left, in terms the model can act on.

    ``return_cost_estimate`` is the agent's own figure. ``search_budget`` has
    already subtracted that trip and a safety reserve, so it must be described
    as money available for new work, not as a battery threshold.
    """
    battery = state.get('battery')
    parts = ['БЮДЖЕТ:']
    if isinstance(battery, (int, float)):
        parts.append(f'  батарея {battery:.1f} из {BATTERY_FULL}')
    if budget:
        search_budget = budget.get('search_budget')
        if isinstance(search_budget, (int, float)):
            parts.append(
                f'  бюджет на новые поиски: {search_budget:.1f}; '
                'это уже после резерва на возврат'
            )
        spent = budget.get('cost_to_come_back')
        if isinstance(spent, (int, float)) and spent > 1.0:
            parts.append(f'  дорога домой уже стоит ≈{spent:.1f}')
    parts.append('  Если батареи не хватает на круг — сокращай круг, '
                 'а не едь дальше.')
    return '\n'.join(parts)


def _geometry_block() -> str:
    """The arena as the agent actually measured it.

    Built from the live geometry when the dashboard is reachable, so the
    bounds and pillar positions in the prompt are the ones the plan will be
    checked against. Falling back to the documented constants keeps the prompt
    usable when the HTTP endpoint is not up.
    """
    arena = ARENA
    source = 'замерено в сцене Gazebo' if arena.from_scene else 'по описанию арены'
    pillars = '; '.join(f'({px:.2f}; {py:.2f})' for px, py, _ in arena.pillars)
    return (
        'ГЕОМЕТРИЯ АРЕНЫ '
        f'({source}):\n'
        f'  проходная часть: x от {arena.x_min + WALL_MARGIN:.2f} '
        f'до {arena.x_max - WALL_MARGIN:.2f}, '
        f'y от {arena.y_min + WALL_MARGIN:.2f} '
        f'до {arena.y_max - WALL_MARGIN:.2f}\n'
        f'  столбы: {pillars}\n'
        f'  держись не ближе {PILLAR_KEEPOUT:g} м от столба — '
        'точка на столбе не пройдёт проверку.'
    )


def build_hypothesis_prompt(state: dict[str, Any],
                            measurements: dict[str, float]) -> str:
    """Prompt for a hypothesis that the measurements can actually settle."""
    parts = [
        'НАБЛЮДЕНИЯ:',
        _json({'state': state, 'drain_per_metre': measurements}),
        '',
        'Сформулируй одну-две гипотезы, которые эти данные способны проверить.',
    ]
    return '\n'.join(parts)


def _json(value: Any) -> str:
    import json
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
