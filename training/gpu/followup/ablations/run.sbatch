#!/bin/bash -l
set -euo pipefail
umask 077

root=$(realpath -e "${1:?provide the experiment directory}")
work=$(realpath -e "${2:?provide the follow-up directory}")
lanes=${3:?provide the number of allocated GPUs}
if [[ "$work" != "$root"/* || ! "$lanes" =~ ^[1-8]$ ]]; then
    exit 2
fi
study="$work/ablations"
cd "$study"
job=${SLURM_JOB_ID:?run within a Slurm allocation}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 SLURM_EXPORT_ENV=ALL
unset SLURM_GPUS SLURM_GPUS_PER_NODE SLURM_GPUS_PER_SOCKET SLURM_GPUS_PER_TASK
python="$root/.venv-cuda/bin/python"
exec 9> .job.lock
flock -n 9

run_lanes() {
    local phase=$1 task pid result=0
    local children=()
    for task in {0..11}; do
        srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=40G --gres=gpu:1 --gpus-per-task=1 \
            --output="$study/logs/$phase-$job-$task.log" bash "$study/lane.sh" "$root" "$work" "$phase" "$task" &
        children+=("$!")
        if (( ${#children[@]} == lanes )); then
            for pid in "${children[@]}"; do wait "$pid" || result=1; done
            children=()
            (( result == 0 )) || return "$result"
        fi
    done
    for pid in "${children[@]}"; do wait "$pid" || result=1; done
    return "$result"
}

printf 'Starting ablation training and development evaluation, job %s\n' "$job"
run_lanes dev
"$python" -B "$study/evaluate.py" select --root "$root" --work "$work"
printf '%s\n' 'Starting fixed test inference for the four ablations'
run_lanes test
"$python" -B "$study/export.py" --root "$root" --work "$work" --job "$job"
