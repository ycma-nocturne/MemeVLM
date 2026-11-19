#!/usr/bin/env bash
# Run one study task inside an allocated GPU job.
set -euo pipefail
if [[ $# != 7 ]]; then
  echo 'Usage: reference_job.sh PYTHON PREPARED DATA_ROOT SHARED_RUNS MODEL TASK SPLIT' >&2
  exit 2
fi
reference_python="$1"
reference_prepared="$2"
reference_data="$3"
reference_runs="$4"
reference_model="$5"
reference_task="$6"
reference_split="$7"
reference_code="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${reference_code}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${reference_python}" -m meme_study.reference_cli run \
  --prepared "${reference_prepared}" --data-root "${reference_data}" \
  --runs "${reference_runs}" --model "${reference_model}" \
  --task "${reference_task}" --split "${reference_split}"
