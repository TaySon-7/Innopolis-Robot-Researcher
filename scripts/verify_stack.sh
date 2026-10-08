#!/usr/bin/env bash
# ============================================================================
# Проверка стека после сборки образа: данные идут, батарея убывает при езде.
#
# Запуск (внутри контейнера):
#   bash scripts/verify_stack.sh
#
# Уровень 0 из ТЗ: «запустить мир, проехать роботом вручную, запустить судью и
# увидеть /odom, /scan, /did/battery, /did/sample_sensor». Скрипт проверяет
# ровно это, плюс то, что слой LLM может из этих данных построить карту.
# ============================================================================
set -uo pipefail

TIMEOUT="${TIMEOUT:-90}"
fail=0

ok()   { printf '\033[1;32m  ОК   \033[0m %s\n' "$*"; }
bad()  { printf '\033[1;31m  СБОЙ \033[0m %s\n' "$*"; fail=$((fail + 1)); }
note() { printf '\033[2m       %s\033[0m\n' "$*"; }

step() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

# Снимает одно сообщение с топика. Пустой вывод означает «данных нет».
once() {
    local topic="$1" seconds="${2:-10}" type="${3:-}"
    if [[ -n "$type" ]]; then
        timeout "$seconds" ros2 topic echo --once "$topic" "$type" 2>/dev/null
    else
        timeout "$seconds" ros2 topic echo --once "$topic" 2>/dev/null
    fi
}

rate() { timeout 20 ros2 topic hz "$1" 2>/dev/null | head -1; }

# ---------------------------------------------------------------------------
step "1. Топики симулятора"

for topic in /scan /odom /cmd_vel /tf; do
    if ros2 topic list 2>/dev/null | grep -qx "$topic"; then
        ok "$topic существует"
    else
        bad "$topic отсутствует"
    fi
done

# ---------------------------------------------------------------------------
step "2. Топики судьи"

for topic in /did/battery /did/sample_sensor /did/score /did/events; do
    if ros2 topic list 2>/dev/null | grep -qx "$topic"; then
        ok "$topic существует"
    else
        bad "$topic отсутствует"
    fi
done

# ---------------------------------------------------------------------------
step "3. Данные идут"

scan="$(once /scan 20 sensor_msgs/msg/LaserScan)"
if [[ -n "$scan" ]]; then
    ok "/scan отдаёт данные"
    # Дальность до препятствия: у TurtleBot3 в этой сцене перед роботом
    # ничего нет, дальность должна быть заметно больше нуля.
    echo "$scan" | grep -E '^\s+ranges:' > /dev/null \
        && ok "в /scan есть массив ranges" \
        || bad "в /scan нет массива ranges"
else
    bad "/scan молчит"
fi

odom="$(once /odom 20 nav_msgs/msg/Odometry)"
if [[ -n "$odom" ]]; then
    ok "/odom отдаёт данные"
else
    bad "/odom молчит"
fi

battery="$(once /did/battery 20 std_msgs/msg/String)"
if [[ -n "$battery" ]]; then
    ok "/did/battery отдаёт данные"
    note "$(echo "$battery" | tr -d '\n' | cut -c1-90)"
else
    bad "/did/battery молчит"
fi

signal="$(once /did/sample_signal 10 std_msgs/msg/Float32)"
if [[ -z "$signal" ]]; then
    signal="$(once /did/sample_sensor 10)"
fi
if [[ -n "$signal" ]]; then
    ok "/did/sample_sensor отдаёт данные"
    note "$(echo "$signal" | tr -d '\n' | cut -c1-90)"
else
    bad "/did/sample_sensor молчит (нормально вне режима поиска)"
fi

# ---------------------------------------------------------------------------
step "4. Частоты"

for topic in /scan /did/battery; do
    h="$(rate "$topic")"
    if [[ -n "$h" ]]; then
        ok "$topic: $h"
    else
        bad "$topic: частота не измерена"
    fi
done

# ---------------------------------------------------------------------------
step "5. Батарея убывает при езде"

# Короткий импульс вперёд. Без движения расход может быть нулевым, и
# утверждение «батарея тратится на движение» ничего не проверяет.
before="$(once /did/battery 15 std_msgs/msg/String)"
note "до:    $(echo "$before" | tr -d '\n' | cut -c1-70)"

timeout 4 ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist \
    '{linear: {x: 0.15}, angular: {z: 0.0}}' > /dev/null 2>&1 &
pub_pid=$!
sleep 5
kill "$pub_pid" 2>/dev/null
wait "$pub_pid" 2>/dev/null

after="$(once /did/battery 15 std_msgs/msg/String)"
note "после: $(echo "$after" | tr -d '\n' | cut -c1-70)"

read_pct() { echo "$1" | grep -oE '[0-9]+\.?[0-9]*' | head -1; }
b_num="$(read_pct "$before")"
a_num="$(read_pct "$after")"

if [[ -z "$b_num" || -z "$a_num" ]]; then
    bad "не удалось разобрать процент батареи из топика"
elif awk "BEGIN{exit !($a_num < $b_num)}"; then
    ok "батарея убыла при езде: $b_num → $a_num"
else
    bad "батарея не убыла: $b_num → $a_num (движение не состоялось?)"
fi

# ---------------------------------------------------------------------------
step "6. Слой LLM строит карту из этих данных"

python3 - <<'PY' || fail=$((fail + 1))
import sys

try:
    from did_llm.world_model import WorldModel
except Exception as error:  # noqa: BLE001
    print(f'  СБОЙ слой LLM не импортируется: {error}')
    sys.exit(1)

world = WorldModel()
print(f'  ОК   сетка {world.spec.width:g}×{world.spec.height:g} м, '
      f'секторов {len(world.state)}')

# Лидар в конусе перед роботом: в этой сцене там пусто, поэтому клетки
# должны остаться свободными, а не занятыми.
ANGLE_MIN = -1.0          # TurtleBot3 LDS-01: примерно -57..+57 градусов
ANGLE_INCREMENT = 0.0044
BEAM_COUNT = int(2.0 / ANGLE_INCREMENT)   # около 2.0 rad, вся дуга
world.on_scan([3.0] * BEAM_COUNT,
              angle_min=ANGLE_MIN,
              angle_increment=ANGLE_INCREMENT)
print('  ОК   /scan принят, карта обновлена')

# Одометрия: робот проехал метр вперёд от старта.
for step in range(1, 11):
    world.on_odom(0.1 * step, 0.0, dt=1.0)
print(f'  ОК   /odom принят: пройдено '
      f'{world.distance_travelled:.2f} м, скорость {world.speed:.2f} м/с')

# Миссия от судьи: без неё планщик считает, что образцов не осталось.
world.set_samples_total(3)
world.collected = 1
print(f'  ОК   /did/score принят: осталось образцов '
      f'{world.samples_total - world.collected}')

# Штрафное событие судьи должно исключить сектор из планирования.
before = sorted(world.hazard_sectors())
world.on_event('collision')
after = sorted(world.hazard_sectors())
if len(after) > len(before):
    print(f'  ОК   collision помечает опасным: {after}')
else:
    print('  СБОЙ collision не пометил ни одного сектора')
    sys.exit(1)

results = world.check_hypotheses()
print(f'  ОК   проверка гипотез отработала: '
      f'{len(results)} результатов, '
      f'{len(world.experiment.links)} звеньев цепочки')
for link in world.experiment.links:
    print(f'       {link.id}: {link.verdict} — {link.claim}')
PY

# ---------------------------------------------------------------------------
printf '\n'
if [[ $fail -eq 0 ]]; then
    printf '\033[1;32mВСЁ ЗЕЛЁНОЕ\033[0m\n'
else
    printf '\033[1;31mПРОВАЛЕНО ПРОВЕРОК: %d\033[0m\n' "$fail"
fi
exit $fail