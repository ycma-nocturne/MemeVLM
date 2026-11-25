#!/bin/bash -l
set -euo pipefail
umask 077

root=$(cd -- "${1:?provide the scoped project directory}" && pwd -P)
work=$(cd -- "${2:?provide the follow-up directory}" && pwd -P)
mode=${3:-fresh}
case "$work" in
    "$root"/*) ;;
    *) printf 'The follow-up directory must be inside the project directory.\n' >&2; exit 2 ;;
esac
case "$mode" in
    fresh|resume|auto) ;;
    *) printf 'Mode must be fresh, resume, or auto.\n' >&2; exit 2 ;;
esac
task=${SLURM_ARRAY_TASK_ID:?submit this script with a Slurm array}
case "$task" in
    0|1|2|3|4|5|6|7|8) ;;
    *) printf 'Slurm array index must be between 0 and 8.\n' >&2; exit 2 ;;
esac
arms=(baseline real real-low)
arm=${arms[task / 3]}
seed=$((42 + task % 3))

cd "$root"
export TMPDIR="$work/tmp/${SLURM_JOB_ID:?run within a Slurm allocation}/train-$arm-$seed"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$work/cache" XDG_CONFIG_HOME="$work/config"
export HF_HOME="$XDG_CACHE_HOME/hf" HF_DATASETS_CACHE="$XDG_CACHE_HOME/hf/datasets"
export TORCH_HOME="$XDG_CACHE_HOME/torch" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED="$seed"
export FORCE_TORCHRUN=0 USE_MCA=0 USE_V1=0 USE_RAY=0 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"

test -s "$work/predict.py"
test -s "$root/evaluation/freeze.json"
test -s "$root/final-expansion/protocol.json"
test -s "$work/prepared/native-dev/inputs.jsonl"
python="$root/.venv-cuda/bin/python"
arguments=(--root "$root" --work "$work" --arm "$arm" --seed "$seed")
run="$work/runs/$arm-$seed"
if [[ "$mode" = auto && ! -s "$run/run-receipt.json" && -d "$run" ]]; then
    if compgen -G "$run/trainer/checkpoint-*/trainer_state.json" > /dev/null; then
        mode=resume
    else
        mkdir -p "$work/interrupted"
        mv -- "$run" "$work/interrupted/$arm-$seed-$SLURM_JOB_ID"
    fi
fi
printf 'Running %s seed %s with Slurm job %s\n' "$arm" "$seed" "$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if test -s "$run/run-receipt.json"; then
    "$python" -B "$work/train.py" verify "${arguments[@]}"
else
    train_arguments=("${arguments[@]}")
    if test "$mode" = resume; then
        train_arguments+=(--resume)
    fi
    "$python" -B "$work/train.py" train "${train_arguments[@]}"
fi
exec "$python" -B "$work/predict.py" --split dev "${arguments[@]}"
