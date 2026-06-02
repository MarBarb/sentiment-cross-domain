#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" scripts/prepare_real_data.py
"${PYTHON_BIN}" run.py experiment=final "$@"
