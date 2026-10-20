#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python virtual environment not found. Create .venv and install requirements.txt first." >&2
    exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required to prevent overlapping strategy runs." >&2
    exit 1
fi

mkdir -p "$ROOT/scanner_out"
exec flock -n "$ROOT/scanner_out/.all-strategies.lock" \
    "$PYTHON" "$ROOT/strategy_runner.py" "$@"
