#!/usr/bin/env bash
# Запуск сборки образа в фоне с логом, переживающим разрыв сессии.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG="${LOG:-/tmp/opencode/build.log}"
exec docker build --progress=plain -t did:latest . > "$LOG" 2>&1