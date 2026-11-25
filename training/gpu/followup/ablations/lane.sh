#!/usr/bin/env bash
set -euo pipefail
umask 077

root=$(realpath -e "${1:?provide the experiment directory}")
work=$(realpath -e "${2:?provide the follow-up directory}")
phase=${3:?provide dev or test}
task=${4:?provide the lane index}
if [[ "$work" != "$root"/* || ! "$task" =~ ^([0-9]|1[01])$ ]]; then
    exit 2
fi
arms=(batch4 rank4 rank16 alpha8)
arm=${arms[task / 3]}
seed=$((42 + task % 3))
study="$work/ablations"
run="$study/runs/$arm-$seed"
python="$root/.venv-cuda/bin/python"

cd "$root"
export TMPDIR="$study/tmp/${SLURM_JOB_ID:?run within a Slurm allocation}/$phase-$arm-$seed"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$study/cache" XDG_CONFIG_HOME="$study/config"
export HF_HOME="$XDG_CACHE_HOME/hf" HF_DATASETS_CACHE="$XDG_CACHE_HOME/hf/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export TORCH_HOME="$XDG_CACHE_HOME/torch" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED="$seed"
export FORCE_TORCHRUN=0 USE_MCA=0 USE_V1=0 USE_RAY=0 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"

arguments=(--root "$root" --work "$work" --arm "$arm" --seed "$seed")
printf '%s %s seed %s\n' "$phase" "$arm" "$seed"
if [[ "$phase" = dev ]]; then
    if [[ -s "$run/run-receipt.json" ]]; then
        "$python" -B "$study/train_ablation.py" verify "${arguments[@]}"
    else
        flags=()
        if compgen -G "$run/trainer/checkpoint-*/trainer_state.json" >/dev/null; then
            flags=(--resume)
        elif [[ -d "$run" ]]; then
            mkdir -p "$study/interrupted"
            mv -- "$run" "$study/interrupted/$arm-$seed-$SLURM_JOB_ID"
        fi
        "$python" -B "$study/train_ablation.py" train "${arguments[@]}" "${flags[@]}"
    fi
elif [[ "$phase" != test ]]; then
    exit 2
fi
exec "$python" -B "$study/evaluate.py" "$phase" "${arguments[@]}"
