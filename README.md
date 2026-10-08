# Robot Researcher

Воспроизводимое окружение для задачи DID Hack «Автономный ИИ-исследователь на роботе-платформе».

## Уровень 0 и стенд

Текущий стенд использует:

- Ubuntu 24.04 в Docker;
- ROS 2 Jazzy;
- Gazebo Sim Harmonic;
- официальный TurtleBot3 Burger и мир `turtlebot3_world`;
- локального судью с интерфейсом `/did/*` и сценариями easy/medium/hard.

На macOS симулятор работает headless: физика, лидар, одометрия и управление полностью активны, но окно Gazebo не выводится. Это убирает зависимость от XQuartz и даёт одинаковый запуск на Apple Silicon и x86-64.

### Быстрый старт

Нужен запущенный Docker Desktop.

```bash
make build
make up
make check
```

`make check` отправляет команды скорости, ждёт реального смещения робота и проверяет `/odom`, `/scan`, `/did/battery` и `/did/sample_sensor`.

### Ручное управление

В одном терминале:

```bash
make teleop
```

Клавиши `w/x` меняют линейную скорость, `a/d` — угловую, `s` или пробел останавливают робота. В другом терминале можно наблюдать состояние:

```bash
make pose
make battery
make sensor
make scan
```

Полезные команды:

```bash
make topics   # список ROS-топиков
make logs     # журнал симулятора и судьи
make shell    # shell внутри ROS-контейнера
make down     # остановка стенда
```

## Дашборд

```bash
make demo         # стенд и дашборд, открывает http://localhost:8080
```

Карта лаборатории с роботом, следом и маршрутом, батарея и счёт, журнал гипотез и решений, управление мышью: клик — ехать, Shift+клик — искать образец, кнопка «Автономно» запускает агента без LLM. Переключатель «показать истину» рисует скрытые образцы и грунты сценария.

## Сценарии

Выбор сценария при запуске стенда:

```bash
SCENARIO=medium make restart      # easy | medium | hard
```

Сценарии лежат в [ros2_ws/src/did_judge/scenarios](ros2_ws/src/did_judge/scenarios): образцы, «грунты» с множителем расхода батареи, опасные зоны и тихие события (в `hard`). Формат описан в [docs/INTERFACES.md](docs/INTERFACES.md).

Новые сценарии генерируются: `SCENARIO=hard@7 make demo` (сложность `easy|medium|hard`, `@seed`) или `make scenario DIFF=hard SEED=7`. Генератор гарантирует достижимость образцов и бюджет батареи.

## Навигация (уровень 1)

Пакет `did_agent` строит сетку стоимостей по карте TurtleBot3 (стены и запас вокруг них запрещены), ищет путь A* с учётом цены грунта и едет по путевым точкам через `/cmd_vel` (`TwistStamped`). Нестандартные препятствия обнаруживаются лидаром, после чего маршрут перепланируется.

```bash
make goto X=0.55 Y=0.55
```

## Судья

Судья публикует:

- `/did/battery` (`std_msgs/msg/Float32`) — заряд от 60; расход = путь × множитель грунта под роботом;
- `/did/sample_sensor` (`std_msgs/msg/Float32`) — зашумлённая близость к ближайшему образцу в диапазоне `[0, 1]`, без направления;
- `/did/score` (`std_msgs/msg/String`) — состояние и счёт в JSON;
- `/did/events` (`std_msgs/msg/String`) — только штрафные события и сбор: `collision`, `false_collect`, `hazard_hit`, `sample_collected`.

Сервисы `/did/collect` и `/did/finish` имеют тип `std_srvs/srv/Trigger`. Сбор успешен на расстоянии не более 0,30 м, `finish` засчитывается на базе. Смена грунтов, новая опасная зона и сбой датчика в `hard` агенту не объявляются.

## Документы

С чего начать: [HANDOFF.md](HANDOFF.md) (состояние проекта, точки подключения для LLM-планировщика и аналитика, рабочий процесс).

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — как всё устроено и почему;
- [docs/REFERENCE.md](docs/REFERENCE.md) — что в каком файле, параметры, команды `make`, тесты;
- [docs/INTERFACES.md](docs/INTERFACES.md) — топики, JSON-форматы, формат сценария;
- [TESTING.md](TESTING.md) — как проверить всё локально;
- [PLAN.md](PLAN.md) — план работ роли Back/ROS и статус.
