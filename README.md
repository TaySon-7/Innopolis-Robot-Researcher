# Robot Researcher

Воспроизводимое окружение для задачи DID Hack «Автономный ИИ-исследователь на роботе-платформе».

## Уровень 0

Текущий стенд использует:

- Ubuntu 24.04 в Docker;
- ROS 2 Jazzy;
- Gazebo Sim Harmonic;
- официальный TurtleBot3 Burger и мир `turtlebot3_world`;
- локального судью с интерфейсом `/did/*`.

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

## Интерфейс судьи

Судья уровня 0 публикует:

- `/did/battery` (`std_msgs/msg/Float32`) — заряд от 60, уменьшается с пройденным расстоянием;
- `/did/sample_sensor` (`std_msgs/msg/Float32`) — зашумлённая близость к ближайшему образцу в диапазоне `[0, 1]`;
- `/did/score` (`std_msgs/msg/String`) — текущее состояние в JSON;
- `/did/events` (`std_msgs/msg/String`) — события в JSON.

Сервисы `/did/collect` и `/did/finish` имеют тип `std_srvs/srv/Trigger`. Для уровня `easy` заданы три образца; сбор успешен на расстоянии не более 0,30 м.

