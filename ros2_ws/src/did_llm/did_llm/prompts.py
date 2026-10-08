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
    MAX_SUBGOALS,
    PILLAR_KEEPOUT,
    RADIUS_MAX,
    RADIUS_MIN,
    TARGET_SUBGOALS,
    WALL_MARGIN,
)

PLANNER_PROMPT_VERSION = 'agent-plan@3'

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
    'По сигналу: одно показание обманчиво, поэтому ставь несколько точек рядом '
    'с тем местом, где сигнал выше всего, прежде чем идти дальше.',
    'Экономия: план должен быть короче, иначе батареи не хватит; лучше три '
    'точки и возврат, чем шесть точек и пустая батарея.',
)

PLANNER_SYSTEM = f"""\
Ты — планировщик верхнего уровня мобильного робота в арене. Ты НЕ управляешь\
роботом и не выдаёшь скорости: ты составляешь список подцелей, а исполнитель\
их выполняет.

МИССИЯ
Собрать как можно больше образцов и вернуться на базу. Собранные образцы\
важны, но вернуться на базу важнее: незавершённый эпизод не засчитывается.

ГЕОМЕТРИЯ
База: ({BASE_X}, {BASE_Y}). Координаты мировые, в метрах.
Точные границы и позиции столбов приходят в сообщении с состоянием, в разделе\
«ГЕОМЕТРИЯ АРЕНЫ». Держись там подальше от столбов: план с точкой на столбе\
не пройдёт проверку и будет отклонён.

ПОСЛЕ КАЖДОГО search_around СРАЗУ ИДИ collect: поиск заканчивается рядом с образцом,\
и без collect он останется на полу.

ЖЁСТКИЕ ПРАВИЛА, НАРУШЕНИЕ ОТКЛОНЯЕТСЯ
1. Если sensor.value выше 0.08 — образец уже рядом. Первой подцелью ставь\
search_around там, где робот стоит, и сразу collect. Подцель goto в другую точку\
запрещена: датчик не показывает направление, поэтому уехать от образца — значит\
потерять его.
2. Не ставь return_to_base последней подцелью, пока collected < samples_total и\
батареи хватает. return_to_base — только когда батареи реально не хватает.\
Завершать эпизод раньше нельзя.
3. Не начинай план с collect: сначала search_around, потом collect.
4. Через дорогой грунт не ездить.

ПОДЦЕЛИ
- {{"type": "goto", "x": …, "y": …}} — доехать до точки.
- {{"type": "search_around", "x": …, "y": …, "radius": …}} — обойти круг и\
найти образец по максимуму сигнала датчика. radius от {RADIUS_MIN} до {RADIUS_MAX} м.
- {{"type": "collect"}} — собрать найденный образец.
- {{"type": "return_to_base"}} — вернуться на базу и завершить эпизод.

КАК ЧИТАТЬ СОСТОЯНИЕ
- sensor.value — близость к ближайшему несобранному образцу, 0..1. Если он\
заметно больше нуля, образец рядом: едь к sensor и потом collect.
- sensor.noise_estimate — шум датчика. При шуме ориентироваться на мгновенное\
значение бессмысленно.
- return_cost_estimate — сколько батареи стоит дорога домой по текущей карте\
стоимостей. Это твой главный ориентир: не планируй, пока батареи меньше\
return_cost_estimate × 1,4 + 8.
- anomaly — флаги: battery_deviation (расход выше ожидаемого), penalties_burst\
(серия штрафов), sensor_noise_up (датчик шумит). Если поднялся любой —\
обойди этот участок и вернись раньше.
- cost_map_updates — что агент уже знает о дорогом грунте. Не планируй путь\
через известно дорогую зону.
- current.state — если failed, посмотри reason: «no path to goal» значит, что\
через эту точку не пройти, «no sample within radius» — образца там нет,\
«signal too weak» — сигнал недостаточный, «battery reserve reached» —\
батареи уже нет.

ПРАВИЛА
1. Обычно {TARGET_SUBGOALS}–6 подцелей, максимум {MAX_SUBGOALS}. Меньше —\
не успеем среагировать на смену среды, больше — потратим батарею вслепую.
2. Поиск начинай там, где сигнал уже слышен. Если сигнала нет — обходи арену\
равномерными точками.
3. Найден образец (collect) — потом ищи следующий.
4. Батареи не хватает на обход — последней подцелью return_to_base.
5. Не повторяй точку, на которой уже был и где ничего не нашлось.

ФОРМАТ ОТВЕТА — строго JSON, без пояснений вне JSON:
{{"explanation": "одно предложение: почему такой план", "subgoals": [
  {{"type": "goto", "x": -0.75, "y": 0.25}},
  {{"type": "search_around", "x": 0.0, "y": 0.0, "radius": 0.8}},
  {{"type": "collect"}},
  {{"type": "return_to_base"}}
]}}
"""

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


def _ground_block(expensive: list[dict[str, float]] | None) -> str:
    """Ground the agent has already measured, and what it costs.

    Without this the model cannot route around expensive floor: it has no way
    to know where the dear patches are, so it sweeps the arena at large and
    burns the battery crossing them.
    """
    if not expensive:
        return ('ДОРОГОЙ ГРУНТ: пока ничего не измерено — все точки стоят 1.0. '
                'Не выдумывай дорогие участки.')
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

    ``return_cost_estimate`` is the agent's own figure, but it is only
    meaningful away from the base: standing on the base it reads as almost
    zero, which the model reads as "the budget is unlimited" — and then it
    plans a sweep of the whole arena on a battery that cannot pay for it. So
    the number that gets the model's attention is the floor: the battery below
    which coming back is not negotiable.
    """
    battery = state.get('battery')
    parts = ['БЮДЖЕТ:']
    if isinstance(battery, (int, float)):
        parts.append(f'  батарея {battery:.1f} из {BATTERY_FULL}')
    if budget:
        floor = budget.get('floor')
        if isinstance(floor, (int, float)):
            parts.append(f'  ниже {floor:.1f} возвращаться обязательно, '
                         'планировать выход нельзя')
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