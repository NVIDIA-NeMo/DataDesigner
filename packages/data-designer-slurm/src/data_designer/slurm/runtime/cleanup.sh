# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

DD_MANAGED_PIDS=()
DD_REQUIRED_PIDS=()
DD_CLEANUP_COMPLETE=0

dd_start_runtime_timer() {
    coproc DD_RUNTIME_TIMER { while IFS= read -r _; do :; done; }
}

dd_sleep() {
    read -r -t "$1" -u "${DD_RUNTIME_TIMER[0]}" _ || true
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
    local index pid position
    local -a indices=()
    ((${#DD_MANAGED_PIDS[@]} == 0)) || indices=("${!DD_MANAGED_PIDS[@]}")
    for ((position = ${#indices[@]} - 1; position >= 0; position--)); do
        index=${indices[position]}
        pid=${DD_MANAGED_PIDS[index]}
        kill -0 "${pid}" 2>/dev/null && kill -TERM "${pid}" 2>/dev/null || true
    done
    dd_sleep 0.2
    for ((position = ${#indices[@]} - 1; position >= 0; position--)); do
        index=${indices[position]}
        pid=${DD_MANAGED_PIDS[index]}
        kill -0 "${pid}" 2>/dev/null && kill -KILL "${pid}" 2>/dev/null || true
    done
    for ((position = ${#indices[@]} - 1; position >= 0; position--)); do
        index=${indices[position]}
        wait "${DD_MANAGED_PIDS[index]}" 2>/dev/null || true
    done
    DD_CLEANUP_COMPLETE=1
}

dd_stop_runtime_timer() {
    if [[ ${DD_RUNTIME_TIMER_PID:-} =~ ^[0-9]+$ ]]; then
        kill -TERM "${DD_RUNTIME_TIMER_PID}" 2>/dev/null || true
        wait "${DD_RUNTIME_TIMER_PID}" 2>/dev/null || true
    fi
}
