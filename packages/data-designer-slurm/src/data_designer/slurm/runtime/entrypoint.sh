# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

readonly DD_RUNTIME_DIR=$(cd -- "${BASH_SOURCE[0]%/*}" && pwd -P)
source "${DD_RUNTIME_DIR}/scratch.sh"
source "${DD_RUNTIME_DIR}/plan_reader.sh"
source "${DD_RUNTIME_DIR}/step_runner.sh"
source "${DD_RUNTIME_DIR}/cleanup.sh"

DD_RUNTIME_PREPARED=0
DD_RUNTIME_FINALIZED=0
DD_RUNTIME_PHASE=initialization
DD_RUNTIME_RETRY_ARGUMENTS=()
declare -A DD_STEP_STARTED_AT=()

dd_runtime_log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1"
}

dd_slurm_run_allocation() {
    if [[ $# -ne 2 && $# -ne 5 ]]; then
        printf '%s\n' 'allocation runtime requires plan and attempt with optional retry binding' >&2
        return 64
    fi
    DD_PLAN_PATH=$1
    DD_ATTEMPT_PATH=$2
    DD_RUNTIME_MANIFEST=${DD_ATTEMPT_PATH}/runtime-manifest.json
    DD_RUNTIME_RETRY_ARGUMENTS=()
    DD_STEP_STARTED_AT=()
    if [[ $# -eq 5 ]]; then
        DD_RUNTIME_RETRY_ARGUMENTS=(
            --retry-id "$3"
            --retry-plan-sha256 "$4"
            --effective-resume-mode "$5"
        )
    fi
    trap dd_runtime_exit EXIT
    trap 'exit 130' INT TERM
    DD_RUNTIME_PHASE=read_control_plan
    dd_read_control_plan "${DD_PLAN_PATH}"
    DD_RUNTIME_PHASE=read_plan_secrets
    dd_read_plan_secret_names "${DD_PLAN_PATH}"
    DD_RUNTIME_CONTAINER_ROOT=${DD_SCRATCH_CONTAINER_ROOT}/runtime
    export DD_PLAN_PATH DD_ATTEMPT_PATH DD_RUNTIME_MANIFEST
    export DD_RUNTIME_ROOT DD_RUNTIME_CONTAINER_ROOT

    DD_RUNTIME_PHASE=verify_host_context
    dd_verify_host_context
    DD_RUNTIME_PHASE=start_runtime_timer
    dd_start_runtime_timer

    DD_RUNTIME_PREPARED=1
    local host
    local -a host_arguments=()
    for host in "${DD_ALLOCATION_HOSTS[@]}"; do
        host_arguments+=(--node-host "${host}")
    done
    DD_RUNTIME_PHASE=prepare
    dd_run_bound_control_phase prepare \
        --runtime-root "${DD_RUNTIME_ROOT}" \
        --manifest "${DD_RUNTIME_MANIFEST}" \
        "${host_arguments[@]}"
    DD_RUNTIME_PHASE=verify_runtime_manifest
    dd_verify_runtime_manifest \
        "${DD_RUNTIME_MANIFEST}" \
        "${DD_PLAN_SHA256}" \
        "${DD_SHARD_ID}" \
        "attempt-${DD_ATTEMPT_ORDINAL}"
    DD_RUNTIME_PHASE=require_plan_secrets
    dd_require_plan_secrets

    DD_RUNTIME_PHASE=client_preflight
    dd_runtime_log 'Running client preflight...'
    dd_read_step_ids "${DD_RUNTIME_MANIFEST}" client_preflight
    ((${#DD_STEP_IDS[@]} == 1))
    dd_run_step "${DD_RUNTIME_MANIFEST}" "${DD_STEP_IDS[0]}"

    DD_RUNTIME_PHASE=server_preflight
    dd_runtime_log 'Running serving preflight...'
    dd_run_role_steps server_preflight
    DD_RUNTIME_PHASE=start_servers
    dd_runtime_log 'Starting vLLM servers...'
    dd_start_servers
    DD_RUNTIME_PHASE=server_readiness
    dd_runtime_log 'Waiting for vLLM servers to become ready...'
    dd_wait_for_role_readiness server
    dd_runtime_log 'vLLM servers are ready.'
    DD_RUNTIME_PHASE=start_endpoints
    dd_start_endpoints
    DD_RUNTIME_PHASE=endpoint_readiness
    dd_wait_for_role_readiness endpoint
    dd_require_running
    DD_RUNTIME_PHASE=ready
    dd_run_bound_control_phase ready

    DD_RUNTIME_PHASE=client
    dd_runtime_log 'Starting Data Designer client...'
    dd_read_step_ids "${DD_RUNTIME_MANIFEST}" client
    ((${#DD_STEP_IDS[@]} == 1))
    dd_start_step "${DD_RUNTIME_MANIFEST}" "${DD_STEP_IDS[0]}"
    local client_pid=${DD_LAST_PID}
    DD_MANAGED_PIDS+=("${client_pid}")
    local client_log_command progress_command
    printf -v client_log_command 'tail -F %q %q' "${DD_STEP_STDOUT}" "${DD_STEP_STDERR}"
    printf -v progress_command 'cat %q' "${DD_ATTEMPT_PATH}/client-progress.json"
    dd_runtime_log "Generation started. Follow client logs: ${client_log_command}"
    dd_runtime_log "Inspect generation progress: ${progress_command}"
    dd_wait_for_client "${client_pid}"
    dd_runtime_log 'Generation finished; finalizing run...'
    dd_require_running

    DD_RUNTIME_PHASE=cleanup
    dd_cleanup_steps
    DD_RUNTIME_PHASE=succeed
    dd_run_bound_control_phase succeed
    DD_RUNTIME_FINALIZED=1
    DD_RUNTIME_PHASE=completed
    dd_runtime_log 'Run completed.'
}

dd_run_bound_control_phase() {
    local operation=$1
    shift
    dd_run_control_phase \
        "${operation}" \
        "${DD_RUNTIME_RETRY_ARGUMENTS[@]+"${DD_RUNTIME_RETRY_ARGUMENTS[@]}"}" \
        "$@"
}

dd_verify_host_context() {
    local tool
    for tool in bash sha256sum tar jq srun scontrol getent curl find; do
        command -v "${tool}" >/dev/null || {
            printf 'required allocation tool %q is unavailable\n' "${tool}" >&2
            return 69
        }
    done
    [[ ${DD_SCRATCH_ROOT:-} == /* && ${DD_RUNTIME_DIR} == "${DD_SCRATCH_ROOT}/runtime" ]]
    [[ -d ${DD_SCRATCH_ROOT} && ! -L ${DD_SCRATCH_ROOT} && -O ${DD_SCRATCH_ROOT} ]]
    [[ -d ${DD_RUNTIME_DIR} && ! -L ${DD_RUNTIME_DIR} && -O ${DD_RUNTIME_DIR} ]]
    [[ -d ${DD_ATTEMPT_PATH} && ! -L ${DD_ATTEMPT_PATH} ]]
    [[ -x ${DD_CLIENT_PYTHON} ]]
    [[ ${SLURM_ARRAY_TASK_ID:-} =~ ^[0-9]+$ ]]
    [[ ${DD_EXPECTED_NODES} =~ ^[1-9][0-9]*$ ]]
    [[ ${DD_CLIENT_NODE_INDEX} =~ ^[0-9]+$ && ${DD_CLIENT_NODE_INDEX} -lt ${DD_EXPECTED_NODES} ]]
    [[ ${SLURM_JOB_NUM_NODES:-} == "${DD_EXPECTED_NODES}" ]]
    [[ ${SLURM_NODEID:-} == "${DD_CLIENT_NODE_INDEX}" ]]
    dd_resolve_allocation_hosts
    dd_verify_gpu_count
}

dd_resolve_allocation_hosts() {
    local node_list=${SLURM_JOB_NODELIST:-}
    [[ -n ${node_list} && ${node_list} != -* && ${#node_list} -le 4096 ]] || return 65
    [[ ${node_list} != *[$'\t\r\n ']* ]] || return 65
    local expanded
    expanded=$(scontrol show hostnames "${node_list}") || return 65
    DD_ALLOCATION_HOSTS=()
    local host
    while IFS= read -r host; do
        DD_ALLOCATION_HOSTS+=("${host}")
    done <<<"${expanded}"
    ((${#DD_ALLOCATION_HOSTS[@]} == DD_EXPECTED_NODES)) || return 65
    for host in "${DD_ALLOCATION_HOSTS[@]}"; do
        [[ ${host} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ && ${#host} -le 253 ]] || return 65
    done
    DD_CLIENT_HOST=${DD_ALLOCATION_HOSTS[DD_CLIENT_NODE_INDEX]}
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
    local step_id deployment delay remaining tail_command
    local -A deployment_started=()
    for step_id in "${DD_STEP_IDS[@]+"${DD_STEP_IDS[@]}"}"; do
        dd_read_step "${DD_RUNTIME_MANIFEST}" "${step_id}"
        deployment=${step_id%%-replica-*}
        deployment_started[${deployment}]=${deployment_started[${deployment}]:-${SECONDS}}
        delay=${DD_STEP_DELAY}
        remaining=$((delay - (SECONDS - deployment_started[${deployment}])))
        ((remaining <= 0)) || dd_sleep "${remaining}"
        DD_STEP_STARTED_AT[${step_id}]=${SECONDS}
        dd_start_step "${DD_RUNTIME_MANIFEST}" "${step_id}"
        dd_register_required_pid "${DD_LAST_PID}"
        dd_require_running
        printf -v tail_command 'tail -F %q %q' "${DD_STEP_STDOUT}" "${DD_STEP_STDERR}"
        dd_runtime_log "Serving step ${step_id} started. Follow vLLM logs: ${tail_command}"
    done
}

dd_run_role_steps() {
    local role=$1
    local step_id
    dd_read_step_ids "${DD_RUNTIME_MANIFEST}" "${role}"
    for step_id in "${DD_STEP_IDS[@]+"${DD_STEP_IDS[@]}"}"; do
        dd_run_step "${DD_RUNTIME_MANIFEST}" "${step_id}"
    done
}

dd_start_endpoints() {
    dd_read_step_ids "${DD_RUNTIME_MANIFEST}" endpoint
    local step_id
    for step_id in "${DD_STEP_IDS[@]+"${DD_STEP_IDS[@]}"}"; do
        DD_STEP_STARTED_AT[${step_id}]=${SECONDS}
        dd_start_step "${DD_RUNTIME_MANIFEST}" "${step_id}"
        dd_register_required_pid "${DD_LAST_PID}"
        dd_require_running
    done
}

dd_wait_for_role_readiness() {
    local role=$1
    local step_id index started_at probe pending
    local -a probe_steps=() probe_hosts=() probe_ports=() probe_paths=() probe_deadlines=() probe_ready=()
    dd_read_step_ids "${DD_RUNTIME_MANIFEST}" "${role}"
    for step_id in "${DD_STEP_IDS[@]+"${DD_STEP_IDS[@]}"}"; do
        dd_read_step "${DD_RUNTIME_MANIFEST}" "${step_id}"
        ((${#DD_STEP_PROBE_FIELDS[@]} % 4 == 0 && ${#DD_STEP_PROBE_FIELDS[@]} > 0)) || return 65
        started_at=${DD_STEP_STARTED_AT[${step_id}]:-${SECONDS}}
        for ((index = 0; index < ${#DD_STEP_PROBE_FIELDS[@]}; index += 4)); do
            probe_steps+=("${step_id}")
            probe_hosts+=("${DD_STEP_PROBE_FIELDS[index]}")
            probe_ports+=("${DD_STEP_PROBE_FIELDS[index + 1]}")
            probe_paths+=("${DD_STEP_PROBE_FIELDS[index + 2]}")
            probe_deadlines+=("$((started_at + DD_STEP_PROBE_FIELDS[index + 3]))")
            probe_ready+=(0)
        done
    done
    pending=${#probe_steps[@]}
    while ((pending > 0)); do
        dd_require_running
        for ((probe = 0; probe < ${#probe_steps[@]}; probe++)); do
            [[ ${probe_ready[probe]} == 0 ]] || continue
            ((SECONDS < probe_deadlines[probe])) || {
                printf 'runtime step %q readiness timed out\n' "${probe_steps[probe]}" >&2
                return 70
            }
            if curl --fail --silent --max-time 1 \
                "http://${probe_hosts[probe]}:${probe_ports[probe]}${probe_paths[probe]}" >/dev/null 2>&1; then
                ((SECONDS < probe_deadlines[probe])) || {
                    printf 'runtime step %q readiness timed out\n' "${probe_steps[probe]}" >&2
                    return 70
                }
                probe_ready[probe]=1
                pending=$((pending - 1))
            fi
        done
        ((pending == 0)) || dd_sleep 0.5
    done
}

dd_runtime_exit() {
    local status=$?
    trap - EXIT INT TERM
    set +e
    if ((status != 0)); then
        printf 'allocation runtime failed: phase=%s status=%d\n' "${DD_RUNTIME_PHASE}" "${status}" >&2
    fi
    dd_cleanup_steps
    if ((DD_RUNTIME_PREPARED == 1 && DD_RUNTIME_FINALIZED == 0)); then
        dd_run_bound_control_phase fail >/dev/null
    fi
    dd_cleanup_allocation_scratch >/dev/null 2>&1 || true
    dd_stop_runtime_timer
    exit "${status}"
}
