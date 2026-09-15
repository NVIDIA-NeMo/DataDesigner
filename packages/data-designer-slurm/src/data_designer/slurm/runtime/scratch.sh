# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

DD_SCRATCH_ROOT=${DD_SCRATCH_ROOT:-}
DD_SCRATCH_CONTAINER_ROOT=/run/data-designer-slurm
export DD_SCRATCH_CONTAINER_ROOT

dd_select_scratch_parent() {
    local candidate resolved
    command -v find >/dev/null || return 69
    for candidate in "${SLURM_TMPDIR:-}" "${TMPDIR:-}" /tmp; do
        [[ ${candidate} =~ ^/[A-Za-z0-9._/-]+$ && ${candidate} != / ]] || continue
        resolved=$(cd -P -- "${candidate}" 2>/dev/null && pwd -P) || continue
        [[ ${resolved} =~ ^/[A-Za-z0-9._/-]+$ && ${resolved} != / ]] || continue
        [[ -d ${resolved} && -w ${resolved} && -x ${resolved} ]] || continue
        DD_SCRATCH_PARENT=${resolved}
        return 0
    done
    printf '%s\n' 'no safe writable local scratch directory is available' >&2
    return 73
}

dd_initialize_local_scratch() {
    local expected=${1:-}
    local task_id=${SLURM_ARRAY_TASK_ID:-0}
    [[ ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ && ${task_id} =~ ^[0-9]+$ ]] || return 64
    dd_select_scratch_parent || return
    local root="${DD_SCRATCH_PARENT}/data-designer-slurm-${SLURM_JOB_ID}-${task_id}"
    [[ -z ${expected} || ${expected} == "${root}" ]] || {
        printf '%s\n' 'allocation nodes resolved inconsistent local scratch paths' >&2
        return 73
    }
    dd_ensure_private_scratch_directory "${root}" || return
    DD_SCRATCH_ROOT=${root}
    export DD_SCRATCH_ROOT
    local directory
    for directory in \
        home cache enroot enroot/cache enroot/config enroot/data enroot/tmp; do
        dd_ensure_private_scratch_directory "${root}/${directory}" || {
            dd_remove_local_scratch "${root}" >/dev/null 2>&1 || true
            return 73
        }
    done
    export HOME="${root}/home"
    export ENROOT_CACHE_PATH="${root}/enroot/cache"
    export ENROOT_CONFIG_PATH="${root}/enroot/config"
    export ENROOT_DATA_PATH="${root}/enroot/data"
    export ENROOT_TEMP_PATH="${root}/enroot/tmp"
}

dd_ensure_private_scratch_directory() {
    local path=$1 unsafe
    if [[ -e ${path} || -L ${path} ]]; then
        [[ -d ${path} && ! -L ${path} && -O ${path} ]] || return 73
    else
        (umask 077 && mkdir -- "${path}") || return 73
    fi
    chmod 0700 "${path}" || return 73
    [[ -d ${path} && ! -L ${path} && -O ${path} ]] || return 73
    if unsafe=$(find "${path}" -prune -type d -perm /0077 -print 2>/dev/null); then
        :
    elif unsafe=$(find "${path}" -prune -type d -perm +0077 -print 2>/dev/null); then
        :
    else
        return 69
    fi
    [[ -r ${path} && -w ${path} && -x ${path} && -z ${unsafe} ]] || return 73
}

dd_extract_runtime_bundle() {
    local archive=$1
    local expected_sha256=$2
    local actual_sha256 staging runtime
    actual_sha256=$(sha256sum < "${archive}") || return 65
    [[ ${actual_sha256%% *} == "${expected_sha256}" ]] || return 65
    staging="${DD_SCRATCH_ROOT}/.runtime-${expected_sha256}-$$"
    runtime="${DD_SCRATCH_ROOT}/runtime"
    [[ ! -e ${staging} && ! -L ${staging} && ! -e ${runtime} && ! -L ${runtime} ]] || return 73
    dd_ensure_private_scratch_directory "${staging}" || return
    (umask 077 && tar -xzf "${archive}" -C "${staging}") || return 65
    [[ -f ${staging}/entrypoint.sh && ! -L ${staging}/entrypoint.sh ]] || return 65
    [[ -f ${staging}/scratch.sh && ! -L ${staging}/scratch.sh ]] || return 65
    mv -- "${staging}" "${runtime}" || return 73
}

dd_stage_allocation_runtime() {
    [[ ${SLURM_JOB_NUM_NODES:-} =~ ^[1-9][0-9]*$ ]] || return 64
    local node_command='set -Eeuo pipefail
archive=$1
expected_sha256=$2
expected_root=$3
actual_sha256=$(sha256sum < "${archive}")
[[ ${actual_sha256%% *} == "${expected_sha256}" ]]
{
    printf "set -Eeuo pipefail\\n"
    tar -xOf "${archive}" scratch.sh
    printf "\ndd_initialize_local_scratch %q\ndd_extract_runtime_bundle %q %q\n" \
        "${expected_root}" "${archive}" "${expected_sha256}"
} | /bin/bash -s -- "${archive}" "${expected_sha256}" "${expected_root}"'
    srun \
        "--nodes=${SLURM_JOB_NUM_NODES}" \
        "--ntasks=${SLURM_JOB_NUM_NODES}" \
        --ntasks-per-node=1 \
        --cpus-per-task=1 \
        --gres=none \
        --exact \
        --overlap \
        --unbuffered \
        --kill-on-bad-exit=1 \
        --export=ALL \
        -- \
        /bin/bash -c "${node_command}" dd-scratch \
        "${DD_RUNTIME_ARCHIVE}" "${DD_RUNTIME_SHA256}" "${DD_SCRATCH_ROOT}"
}

dd_cleanup_allocation_scratch() {
    local status=0
    if [[ ${SLURM_JOB_NUM_NODES:-} =~ ^[1-9][0-9]*$ ]] && command -v srun >/dev/null; then
        local node_command='set -Eeuo pipefail
archive=$1
expected_sha256=$2
expected_root=$3
actual_sha256=$(sha256sum < "${archive}")
[[ ${actual_sha256%% *} == "${expected_sha256}" ]]
{
    printf "set -Eeuo pipefail\\n"
    tar -xOf "${archive}" scratch.sh
    printf "\ndd_remove_local_scratch %q\n" "${expected_root}"
} | /bin/bash -s -- "${archive}" "${expected_sha256}" "${expected_root}"'
        srun \
            "--nodes=${SLURM_JOB_NUM_NODES}" \
            "--ntasks=${SLURM_JOB_NUM_NODES}" \
            --ntasks-per-node=1 \
            --cpus-per-task=1 \
            --gres=none \
            --exact \
            --overlap \
            --unbuffered \
            --kill-on-bad-exit=1 \
            --time=00:01:00 \
            --wait=0 \
            --export=ALL \
            -- \
            /bin/bash -c "${node_command}" dd-scratch \
            "${DD_RUNTIME_ARCHIVE}" "${DD_RUNTIME_SHA256}" "${DD_SCRATCH_ROOT}" \
            >/dev/null 2>&1 || status=$?
    fi
    dd_remove_local_scratch "${DD_SCRATCH_ROOT}" || status=$?
    return "${status}"
}

dd_remove_local_scratch() {
    local path=${1:-}
    local task_id=${SLURM_ARRAY_TASK_ID:-0}
    local expected_name parent
    [[ ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ && ${task_id} =~ ^[0-9]+$ ]] || return 64
    expected_name="data-designer-slurm-${SLURM_JOB_ID}-${task_id}"
    [[ ${path} == /*/${expected_name} && ${path} != *//* && ${path} != */../* && ${path} != */./* ]] || return 73
    parent=$(cd -P -- "${path%/*}" 2>/dev/null && pwd -P) || return 73
    [[ ${path} == "${parent}/${expected_name}" ]] || return 73
    if [[ ! -e ${path} && ! -L ${path} ]]; then
        return 0
    fi
    [[ -d ${path} && ! -L ${path} && -O ${path} ]] || return 73
    rm -rf -- "${path}"
}
