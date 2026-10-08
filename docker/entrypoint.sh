#!/usr/bin/env bash
# Точка входа контейнера.
set -e

# setup.bash отсутствует, если пакеты примонтированы как есть, а не собраны
# colcon. В этом случае каждый скриптros2 запускается через python3.
if [[ -f /opt/did_ws/install/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    # shellcheck disable=SC1091
    source /opt/did_ws/install/setup.bash
fi

exec "$@"