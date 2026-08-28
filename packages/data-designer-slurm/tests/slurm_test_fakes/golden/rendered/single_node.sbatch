#!/usr/bin/env bash
#SBATCH --job-name=data-designer
#SBATCH --account=research
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:55:00
#SBATCH --array=0
#SBATCH --gres=gpu:8
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

readonly DD_RUNTIME_ARCHIVE="/workspace/primary/runtime/eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.tar.gz"
readonly DD_RUNTIME_SHA256="eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
readonly DD_PLAN="/workspace/primary/runs/run-single/resolved-plan.json"
readonly DD_PLAN_SHA256="4a64741df1a3eaba63d1cd19f08ce5dfe9017509f839a78678d17324e46d98b4"
readonly DD_RUN_ROOT="/workspace/primary/runs/run-single"
readonly DD_ATTEMPT_ORDINAL="0001"

verify_sha256() {
    local actual_sha256
    actual_sha256="$(sha256sum < "$2")"
    [[ "${actual_sha256%% *}" == "$1" ]]
}

verify_sha256 "${DD_RUNTIME_SHA256}" "${DD_RUNTIME_ARCHIVE}"
verify_sha256 "${DD_PLAN_SHA256}" "${DD_PLAN}"
if [[ ! ${SLURM_ARRAY_TASK_ID:-} =~ ^[0-9]+$ ]]; then
    printf '%s\n' 'SLURM_ARRAY_TASK_ID must be a non-negative integer' >&2
    exit 64
fi
readonly DD_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID}"
printf -v DD_SHARD_ID 'shard-%05d' "${DD_ARRAY_TASK_ID}"
readonly DD_SHARD_ID
readonly DD_ATTEMPT_DIR="${DD_RUN_ROOT}/shards/${DD_SHARD_ID}/attempts/attempt-${DD_ATTEMPT_ORDINAL}"
install -d -m 0700 "${DD_ATTEMPT_DIR}"
DD_RUNTIME_DIR="$(mktemp -d "${DD_ATTEMPT_DIR}/runtime.${DD_RUNTIME_SHA256}.XXXXXX")"
readonly DD_RUNTIME_DIR
tar -xzf "${DD_RUNTIME_ARCHIVE}" -C "${DD_RUNTIME_DIR}"

source "${DD_RUNTIME_DIR}/entrypoint.sh"
dd_slurm_run_allocation "${DD_PLAN}" "${DD_ATTEMPT_DIR}"
