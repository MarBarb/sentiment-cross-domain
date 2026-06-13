#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
SOURCE_FILE="data/processed/source_full.csv"
TARGET_FILE="data/processed/social_full.csv"
START_TIME="$(date +%s)"

if [[ "${FORCE_PREPARE:-0}" == "1" || ! -s "${SOURCE_FILE}" || ! -s "${TARGET_FILE}" ]]; then
  echo "[run_final] preparing processed data"
  "${PYTHON_BIN}" scripts/prepare_real_data.py
else
  echo "[run_final] using existing processed data"
fi

echo "[run_final] running final experiment matrix"
"${PYTHON_BIN}" run.py experiment=final "$@"

END_TIME="$(date +%s)"
echo "[run_final] done in $((END_TIME - START_TIME))s"
