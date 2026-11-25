#!/bin/bash -l
set -euo pipefail
umask 077

root=$(realpath -e "${1:?provide the scoped project directory}")
work=$(realpath -e "${2:?provide the follow-up directory}")
case "$work" in
    "$root"/*) ;;
    *) exit 2 ;;
esac
task=${SLURM_ARRAY_TASK_ID:?submit this script with a Slurm array}
case "$task" in
    0|1|2|3|4|5) ;;
    *) exit 2 ;;
esac
seed=$((42 + task % 3))
python="$root/.venv-cuda/bin/python"
arm=baseline
if (( task >= 3 )); then
    arm=$("$python" -B -c 'import json, sys; print(json.load(open(sys.argv[1]))["selected_by_seed"][sys.argv[2]])' \
        "$work/selection.json" "$seed")
fi
cd "$root"
export TMPDIR="$work/tmp/${SLURM_JOB_ID:?run within a Slurm allocation}/test-$arm-$seed"
export TMP="$TMPDIR" TEMP="$TMPDIR" XDG_CACHE_HOME="$work/cache" XDG_CONFIG_HOME="$work/config"
export HF_HOME="$XDG_CACHE_HOME/hf" HF_DATASETS_CACHE="$XDG_CACHE_HOME/hf/datasets"
export TORCH_HOME="$XDG_CACHE_HOME/torch" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"

exec "$python" -B "$work/predict.py" --root "$root" --work "$work" --split test --arm "$arm" --seed "$seed"
