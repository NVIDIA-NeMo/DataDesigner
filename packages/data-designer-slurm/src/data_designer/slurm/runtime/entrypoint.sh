# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

readonly DD_RUNTIME_DIR=$(cd -- "${BASH_SOURCE[0]%/*}" && pwd -P)
source "${DD_RUNTIME_DIR}/plan_reader.sh"
source "${DD_RUNTIME_DIR}/step_runner.sh"
source "${DD_RUNTIME_DIR}/cleanup.sh"

DD_RUNTIME_PREPARED=0
DD_RUNTIME_FINALIZED=0

dd_slurm_run_allocation() {
    if [[ $# -ne 2 ]]; then
        printf '%s\n' 'allocation runtime requires plan and attempt directory arguments' >&2
        return 64
    fi
    DD_PLAN_PATH=$1
    DD_ATTEMPT_PATH=$2
    DD_RUNTIME_MANIFEST=${DD_ATTEMPT_PATH}/runtime-manifest.json
    dd_read_control_plan "${DD_PLAN_PATH}"
    dd_read_plan_secret_names "${DD_PLAN_PATH}"
    dd_read_container_path "${DD_PLAN_PATH}" "${DD_PLAN_PATH}" false
    DD_PLAN_CONTAINER_PATH=${DD_CONTAINER_PATH}
    dd_read_container_path "${DD_PLAN_PATH}" "${DD_ATTEMPT_PATH}" true
    DD_ATTEMPT_CONTAINER_DIR=${DD_CONTAINER_PATH}
    dd_read_container_path "${DD_PLAN_PATH}" "${DD_RUNTIME_DIR}" true
    DD_RUNTIME_CONTAINER_ROOT=${DD_CONTAINER_PATH}
    dd_read_container_path "${DD_PLAN_PATH}" "${DD_RUNTIME_MANIFEST}" true
    DD_RUNTIME_MANIFEST_CONTAINER_PATH=${DD_CONTAINER_PATH}
    export DD_PLAN_PATH DD_ATTEMPT_PATH DD_RUNTIME_MANIFEST
    export DD_PLAN_CONTAINER_PATH DD_ATTEMPT_CONTAINER_DIR DD_RUNTIME_CONTAINER_ROOT

    dd_verify_host_context
    dd_start_runtime_timer
    trap dd_runtime_exit EXIT
    trap 'exit 130' INT TERM

    DD_RUNTIME_PREPARED=1
    dd_run_control_phase prepare \
        --runtime-root "${DD_RUNTIME_DIR}" \
        --manifest "${DD_RUNTIME_MANIFEST_CONTAINER_PATH}"
    dd_verify_runtime_manifest \
        "${DD_RUNTIME_MANIFEST}" \
        "${DD_PLAN_SHA256}" \
        "${DD_SHARD_ID}" \
        "attempt-${DD_ATTEMPT_ORDINAL}"
    dd_require_plan_secrets

    dd_read_step_ids "${DD_RUNTIME_MANIFEST}" client_preflight
    ((${#DD_STEP_IDS[@]} == 1))
    dd_run_step "${DD_RUNTIME_MANIFEST}" "${DD_STEP_IDS[0]}"

    dd_start_servers
    dd_wait_for_role_readiness server
    dd_start_endpoints
    dd_wait_for_role_readiness endpoint
    dd_require_running
    dd_run_control_phase ready

    dd_read_step_ids "${DD_RUNTIME_MANIFEST}" client
    ((${#DD_STEP_IDS[@]} == 1))
    dd_start_step "${DD_RUNTIME_MANIFEST}" "${DD_STEP_IDS[0]}"
    local client_pid=${DD_LAST_PID}
    DD_MANAGED_PIDS+=("${client_pid}")
    dd_wait_for_client "${client_pid}"
    dd_require_running

    dd_cleanup_steps
    dd_run_control_phase succeed
    DD_RUNTIME_FINALIZED=1
}

dd_verify_host_context() {
    local tool
    for tool in bash sha256sum tar jq srun scontrol getent curl; do
        command -v "${tool}" >/dev/null || {
            printf 'required allocation tool %q is unavailable\n' "${tool}" >&2
            return 69
        }
    done
    [[ -d ${DD_ATTEMPT_PATH} && ! -L ${DD_ATTEMPT_PATH} ]]
    [[ ${SLURM_ARRAY_TASK_ID:-} =~ ^[0-9]+$ ]]
    [[ ${SLURM_JOB_NUM_NODES:-} == 1 && ${SLURM_NODEID:-} == 0 ]]
    dd_verify_gpu_count
    dd_read_artifacts "${DD_PLAN_PATH}" "${SLURM_ARRAY_TASK_ID}"
    local index path digest actual
    for ((index = 0; index < ${#DD_ARTIFACT_FIELDS[@]}; index += 2)); do
        path=${DD_ARTIFACT_FIELDS[index]}
        digest=${DD_ARTIFACT_FIELDS[index + 1]}
        actual=$(sha256sum < "${path}")
        [[ ${actual%% *} == "${digest}" ]] || {
            printf 'allocation artifact digest mismatch: %s\n' "${path}" >&2
            return 65
        }
    done
}

dd_verify_gpu_count() {
    local visible=${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}
    local count=0 item
    local -A devices=()
    if [[ -z ${visible//[[:space:]]/} ]]; then
        count=0
    elif [[ ${visible} =~ ^gpu(:[^:]+)?:([0-9]+)$ ]]; then
        count=${BASH_REMATCH[2]}
    else
        IFS=, read -r -a device_values <<<"${visible}"
        for item in "${device_values[@]+"${device_values[@]}"}"; do
            item=${item//[[:space:]]/}
            [[ ${item} =~ ^[A-Za-z0-9_.:-]+$ ]] || return 65
            [[ -n ${item} && ! ${devices[${item}]+_} ]] || return 65
            devices[${item}]=1
            ((count += 1))
        done
    fi
    [[ ${DD_EXPECTED_GPUS} =~ ^[1-9][0-9]*$ && ${count} -ge ${DD_EXPECTED_GPUS} ]] || {
        printf '%s\n' 'allocation GPU visibility does not match the resolved plan' >&2
        return 65
    }
}

dd_start_servers() {
    dd_read_step_ids "${DD_RUNTIME_MANIFEST}" server
    local step_id deployment delay remaining
    local -A deployment_started=()
    for step_id in "${DD_STEP_IDS[@]+"${DD_STEP_IDS[@]}"}"; do
        dd_read_step "${DD_RUNTIME_MANIFEST}" "${step_id}"
        deployment=${step_id%%-replica-*}
        deployment_started[${deployment}]=${deployment_started[${deployment}]:-${SECONDS}}
        delay=${DD_STEP_DELAY}
        remaining=$((delay - (SECONDS - deployment_started[${deployment}])))
        ((remaining <= 0)) || dd_sleep "${remaining}"
        dd_start_step "${DD_RUNTIME_MANIFEST}" "${step_id}"
        dd_register_required_pid "${DD_LAST_PID}"
        dd_require_running
    done
}

dd_start_endpoints() {
    dd_read_step_ids "${DD_RUNTIME_MANIFEST}" endpoint
    local step_id
    for step_id in "${DD_STEP_IDS[@]+"${DD_STEP_IDS[@]}"}"; do
        dd_start_step "${DD_RUNTIME_MANIFEST}" "${step_id}"
        dd_register_required_pid "${DD_LAST_PID}"
        dd_require_running
    done
}

dd_wait_for_role_readiness() {
    local role=$1
    local step_id deadline
    dd_read_step_ids "${DD_RUNTIME_MANIFEST}" "${role}"
    for step_id in "${DD_STEP_IDS[@]+"${DD_STEP_IDS[@]}"}"; do
        dd_read_step "${DD_RUNTIME_MANIFEST}" "${step_id}"
        deadline=$((SECONDS + DD_STEP_PROBE_DEADLINE))
        until curl --fail --silent --max-time 1 \
            "http://${DD_STEP_PROBE_HOST}:${DD_STEP_PROBE_PORT}${DD_STEP_PROBE_PATH}" >/dev/null 2>&1; do
            dd_require_running
            ((SECONDS < deadline)) || {
                printf 'runtime step %q readiness timed out\n' "${step_id}" >&2
                return 70
            }
            dd_sleep 0.5
        done
    done
}

dd_runtime_exit() {
    local status=$?
    trap - EXIT INT TERM
    set +e
    dd_cleanup_steps
    if ((DD_RUNTIME_PREPARED == 1 && DD_RUNTIME_FINALIZED == 0)); then
        dd_run_control_phase fail >/dev/null
    fi
    dd_stop_runtime_timer
    exit "${status}"
}
