#!/usr/bin/env bash
#SBATCH --job-name=dd-two-model
#SBATCH --nodes=3
#SBATCH --time=03:55:00
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:8
set -Eeuo pipefail

readonly DD_RUNTIME_ARCHIVE="/workspace/primary/runtime/runtime.tar.gz"
readonly DD_RUNTIME_SHA256="eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
readonly DD_PLAN="/workspace/primary/runs/run-001/resolved-plan.json"
readonly DD_PLAN_SHA256="e101840b0ba888fd0f2d9feef70cb6bb2a668dc4dff4c2198a0f8ba2e8a1eb0a"
readonly DD_RUN_ROOT="/workspace/primary/runs/run-001"
readonly DD_ATTEMPT_ORDINAL="0001"

verify_sha256() {
    printf '%s  %s\n' "$1" "$2" | sha256sum --check --status -
}

verify_sha256 "${DD_RUNTIME_SHA256}" "${DD_RUNTIME_ARCHIVE}"
verify_sha256 "${DD_PLAN_SHA256}" "${DD_PLAN}"
readonly DD_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
printf -v DD_SHARD_ID 'shard-%05d' "${DD_ARRAY_TASK_ID}"
readonly DD_SHARD_ID
readonly DD_ATTEMPT_DIR="${DD_RUN_ROOT}/shards/${DD_SHARD_ID}/attempts/attempt-${DD_ATTEMPT_ORDINAL}"
install -d -m 0700 "${DD_ATTEMPT_DIR}"
readonly DD_RUNTIME_DIR="$(mktemp -d "${DD_ATTEMPT_DIR}/runtime.${DD_RUNTIME_SHA256}.XXXXXX")"
tar -xzf "${DD_RUNTIME_ARCHIVE}" -C "${DD_RUNTIME_DIR}"

source "${DD_RUNTIME_DIR}/entrypoint.sh"
dd_slurm_run_allocation "${DD_PLAN}" "${DD_ATTEMPT_DIR}"
