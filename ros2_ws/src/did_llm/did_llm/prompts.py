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
    ARENA_X_MAX,
    ARENA_X_MIN,
    ARENA_Y_MAX,
    ARENA_Y_MIN,
    BASE_X,
    BASE_Y,
    MAX_SUBGOALS,
    RADIUS_MAX,
    RADIUS_MIN,
    TARGET_SUBGOALS,
)

PLANNER_PROMPT_VERSION = 'agent-plan@2'

PLANNER_SYSTEM = f"""\
Ты — планировщик верхнего уровня мобильного робота в арене. Ты НЕ управляешь\
роботом и не выдаёшь скорости: ты составляешь список подцелей, а исполнитель\
их выполняет.

МИССИЯ
Собрать как можно больше образцов и вернуться на базу. Собранные образцы\
важны, но вернуться на базу важнее: незавершённый эпизод не засчитывается.

ГЕОМЕТРИЯ
База: ({BASE_X}, {BASE_Y}).
Арена примерно x от {ARENA_X_MIN} до {ARENA_X_MAX}, y от {ARENA_Y_MIN} до {ARENA_Y_MAX}.
Внутри девять столбов сеткой 3x3 с шагом около 1,1 м (точки ±1,1 и 0).\
Координаты мировые, в метрах. Давай точки вне арены, столбы обходить.

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
                         feedback: str) -> str:
    """Assemble the planner prompt from one state snapshot."""
    parts = [
        f'МИССИЯ: {mission}',
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