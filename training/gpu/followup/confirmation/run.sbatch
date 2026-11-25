#!/bin/bash -l
set -euo pipefail
umask 077
root=$(realpath -e "${1:?provide the experiment directory}")
work=$(realpath -e "${2:?provide the follow-up directory}")
[[ "$work" = "$root"/* ]] || exit 2
study="$work/confirmation"
cd "$study"
id=${SLURM_JOB_ID:?run within a Slurm allocation}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 SLURM_EXPORT_ENV=ALL
unset SLURM_GPUS SLURM_GPUS_PER_NODE SLURM_GPUS_PER_SOCKET SLURM_GPUS_PER_TASK
python="$root/.venv-cuda/bin/python"
exec 9> .job.lock
flock -n 9

run_wave() {
    local phase=$1 arm=$2 seed pid failed=0
    local children=()
    for seed in 42 43 44; do
        srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=40G --gres=gpu:1 --gpus-per-task=1 \
            --output="$study/logs/$phase-$id-$arm-$seed.log" bash "$study/lane.sh" "$root" "$work" "$phase" "$arm" "$seed" &
        children+=("$!")
    done
    for pid in "${children[@]}"; do wait "$pid" || failed=1; done
    return "$failed"
}

printf 'Starting combined configuration training, job %s\n' "$id"
run_wave dev combined
"$python" -B "$study/confirm_eval.py" select --root "$root" --work "$work"
for arm in real batch4 combined; do
    printf 'Starting independent holdout evaluation for %s\n' "$arm"
    run_wave test "$arm"
done
"$python" -B "$study/export_confirmation.py" --root "$root" --work "$work" --job "$id"
