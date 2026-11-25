#!/bin/bash -l
set -euo pipefail
umask 077

root=$(realpath -e "${1:?provide the experiment directory}")
work=$(realpath -e "${2:?provide the follow-up directory}")
lanes=${3:-3}
case "$work" in
    "$root"/*) ;;
    *) exit 2 ;;
esac
if [[ ! "$lanes" =~ ^[1-6]$ ]]; then
    exit 2
fi
cd "$work"
job=${SLURM_JOB_ID:?run within a Slurm allocation}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 SLURM_EXPORT_ENV=ALL
unset SLURM_GPUS SLURM_GPUS_PER_NODE SLURM_GPUS_PER_SOCKET SLURM_GPUS_PER_TASK
python="$root/.venv-cuda/bin/python"
exec 9> .packed.lock
flock -n 9

run_lanes() {
    local phase=$1 script=$2
    shift 2
    local task result=0 pid
    local children=()
    for task in "$@"; do
        srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=40G --gres=gpu:1 --gpus-per-task=1 \
            --output="$work/logs/$phase-$job-$task.log" \
            env SLURM_ARRAY_TASK_ID="$task" bash "$work/$script" "$root" "$work" auto &
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

printf 'Starting matched data comparison in Slurm job %s\n' "$job"
run_lanes data train.sbatch 0 1 2 3 4 5
"$python" -B "$work/select.py" --work "$work" --stage data
printf '%s\n' 'Starting learning-rate comparison'
run_lanes lr train.sbatch 6 7 8
"$python" -B "$work/select.py" --work "$work" --stage lr
printf '%s\n' 'Starting fixed MemeQA test'
run_lanes test test.sbatch 0 1 2 3 4 5
"$python" -B "$work/score.py" --runs "$work/test-runs" --frozen "$root/final-expansion" \
    --selection "$work/selection.json" --output "$work/scores.json"
