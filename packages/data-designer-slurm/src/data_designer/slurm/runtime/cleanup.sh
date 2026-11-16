# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

DD_MANAGED_PIDS=()
DD_REQUIRED_PIDS=()
DD_ENDPOINT_PIDS=()
declare -A DD_ENDPOINT_STEP_FILES=()
DD_HEALTH_PID=
DD_CLEANUP_COMPLETE=0

dd_start_runtime_timer() {
    coproc DD_RUNTIME_TIMER { while IFS= read -r _; do :; done; }
}

dd_sleep() {
    if [[ ${DD_RUNTIME_TIMER[0]:-} =~ ^[0-9]+$ ]]; then
        read -r -t "$1" -u "${DD_RUNTIME_TIMER[0]}" _ || true
    else
        sleep "$1"
    fi
}

dd_register_required_pid() {
    DD_MANAGED_PIDS+=("$1")
    DD_REQUIRED_PIDS+=("$1")
}

dd_require_running() {
    local pid
    for pid in "${DD_REQUIRED_PIDS[@]+"${DD_REQUIRED_PIDS[@]}"}"; do
        kill -0 "${pid}" 2>/dev/null || return 1
    done
}

dd_wait_for_client() {
    local client_pid=$1
    local status=0
    while kill -0 "${client_pid}" 2>/dev/null; do
        wait -n || status=$?
        if ! kill -0 "${client_pid}" 2>/dev/null; then
            return "${status}"
        fi
        dd_require_running || return 70
    done
    wait "${client_pid}"
}

dd_cleanup_steps() {
    ((DD_CLEANUP_COMPLETE == 0)) || return 0
    if [[ -n ${DD_HEALTH_PID} ]]; then
        dd_stop_steps 1 "${DD_HEALTH_PID}"
    fi
    dd_drain_endpoints
    dd_stop_steps 1 "${DD_MANAGED_PIDS[@]+"${DD_MANAGED_PIDS[@]}"}"
    DD_CLEANUP_COMPLETE=1
}

dd_drain_endpoints() {
    local pid identity_file step_identity
    for pid in "${DD_ENDPOINT_PIDS[@]+"${DD_ENDPOINT_PIDS[@]}"}"; do
        kill -0 "${pid}" 2>/dev/null || continue
        identity_file=${DD_ENDPOINT_STEP_FILES[${pid}]:-}
        step_identity=
        if [[ -f ${identity_file} && ! -L ${identity_file} ]]; then
            IFS= read -r step_identity < "${identity_file}" || true
        fi
        if [[ ${step_identity} =~ ^([1-9][0-9]*)\.([0-9]+)$ && ${BASH_REMATCH[1]} == "${SLURM_JOB_ID:-}" ]]; then
            # TERM to srun becomes KILL; signal only this endpoint's task step.
            scancel --quiet --signal=TERM "${step_identity}" || true
        fi
    done
    # Keep launchers and backends alive through the proxy's 10-second drain.
    dd_wait_steps 12 "${DD_ENDPOINT_PIDS[@]+"${DD_ENDPOINT_PIDS[@]}"}"
    dd_stop_steps 1 "${DD_ENDPOINT_PIDS[@]+"${DD_ENDPOINT_PIDS[@]}"}"
}

dd_stop_steps() {
    local grace_seconds=$1
    shift
    (($# > 0)) || return 0
    local pid
    for pid in "$@"; do
        kill -0 "${pid}" 2>/dev/null && kill -TERM "${pid}" 2>/dev/null || true
    done
    dd_wait_steps "${grace_seconds}" "$@"
    for pid in "$@"; do
        kill -0 "${pid}" 2>/dev/null && kill -KILL "${pid}" 2>/dev/null || true
    done
    for pid in "$@"; do
        wait "${pid}" 2>/dev/null || true
    done
}

dd_wait_steps() {
    local grace_seconds=$1
    shift
    (($# > 0)) || return 0
    local pid pending deadline=$((SECONDS + grace_seconds))
    while ((SECONDS < deadline)); do
        pending=0
        for pid in "$@"; do
            if kill -0 "${pid}" 2>/dev/null; then
                pending=1
                break
            fi
        done
        ((pending != 0)) || break
        dd_sleep 0.1
    done
}

dd_stop_runtime_timer() {
    if [[ ${DD_RUNTIME_TIMER_PID:-} =~ ^[0-9]+$ ]]; then
        kill -TERM "${DD_RUNTIME_TIMER_PID}" 2>/dev/null || true
        wait "${DD_RUNTIME_TIMER_PID}" 2>/dev/null || true
    fi
}
