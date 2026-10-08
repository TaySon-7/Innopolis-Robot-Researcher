#!/usr/bin/env bash
# ============================================================================
# DID Hack — сборка локального apt-репозитория для офлайн-сборки образа.
#
# Зачем: вендорные пакеты Gazebo и Nav2 весят ~850 МБ, а зеркало рвёт
# соединение. Docker при этом теряет ВЕСЬ слой целиком и начинает заново —
# два таких обрыва стоили больше получаса.
#
# Репозиторий собирается из кэша пакетов хоста плюс недостающие deb-пакеты.
# Папка vendor/ в git не попадает.
#
# Запуск:  bash scripts/make_vendor_repo.sh
# ============================================================================
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="${ROOT}/vendor/debs"

# Пакеты, которых на хосте может не быть в кэше, но они нужны в образе.
EXTRA=(
    mesa-utils
    python3-pip
    python3-pip-whl
    python3-setuptools-whl
    ros-jazzy-rmw-cyclonedds-cpp
    ros-jazzy-turtlebot3-gazebo
    ros-jazzy-turtlebot3-teleop
    ros-jazzy-turtlebot3-navigation2
    python3-colcon-common-extensions
)

log() { echo -e "\n\033[1;36m==> $*\033[0m"; }

mkdir -p "$VENDOR"

log "копирование кэша apt хоста"
shopt -s nullglob
copied=0
for deb in /var/cache/apt/archives/*.deb; do
    cp -n "$deb" "$VENDOR/" && copied=$((copied + 1))
done
echo "  скопировано: ${copied}"

log "докачивание недостающих пакетов"
cd "$VENDOR"
# shellcheck disable=SC2086
apt-get download ${EXTRA[*]} 2>&1 | grep -E "^(Пол|Err|Ошиб)" || true

log "индексация"
dpkg-scanpackages . /dev/null 2>/dev/null > Packages
gzip -kf Packages

count=$(grep -c '^Package:' Packages)
size=$(du -sh . | cut -f1)

echo
echo "  пакетов: ${count}"
echo "  размер : ${size}"

# Проверяем, что всё нужное действительно попало в индекс.
missing=0
for pkg in "${EXTRA[@]}"; do
    if ! grep -q "^Package: ${pkg}$" Packages; then
        echo -e "  \033[1;31mНЕТ в индексе: ${pkg}\033[0m"
        missing=$((missing + 1))
    fi
done

if [[ "$missing" -gt 0 ]]; then
    echo
    echo -e "\033[1;31mИндекс неполон, сборка образа не удастся.\033[0m"
    exit 1
fi

echo -e "\033[1;32mРепозиторий готов.\033[0m Теперь: make build"
