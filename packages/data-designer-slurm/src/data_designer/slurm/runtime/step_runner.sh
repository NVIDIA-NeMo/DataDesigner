# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

dd_validate_environment_name() {
    [[ $1 =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]
}

dd_require_plan_secrets() {
    local name
    for name in "${DD_REQUIRED_SECRET_NAMES[@]+"${DD_REQUIRED_SECRET_NAMES[@]}"}"; do
        dd_validate_environment_name "${name}" || return 64
        [[ ${!name+x} ]] || {
            printf 'required secret environment %q is unavailable\n' "${name}" >&2
            return 78
        }
    done
}

dd_scope_control_environment() {
    local name
    for name in "${DD_ALL_SECRET_NAMES[@]+"${DD_ALL_SECRET_NAMES[@]}"}"; do
        unset "${name}"
    done
    unset CUDA_VISIBLE_DEVICES
    export LC_ALL=C
    export PYTHONPATH=${DD_RUNTIME_CONTAINER_ROOT}
}

dd_materialize_step_environment() {
    local index name source value prefix current
    local -a secret_values=()
    for ((index = 0; index < ${#DD_STEP_SECRET_ENV[@]}; index += 2)); do
        name=${DD_STEP_SECRET_ENV[index]}
        source=${DD_STEP_SECRET_ENV[index + 1]}
        dd_validate_environment_name "${name}" && dd_validate_environment_name "${source}" || return 64
        [[ ${!source+x} ]] || return 78
        secret_values+=("${!source}")
    done
    for name in "${DD_ALL_SECRET_NAMES[@]+"${DD_ALL_SECRET_NAMES[@]}"}"; do
        unset "${name}"
    done
    unset CUDA_VISIBLE_DEVICES
    for ((index = 0; index < ${#DD_STEP_LITERAL_ENV[@]}; index += 2)); do
        name=${DD_STEP_LITERAL_ENV[index]}
        value=${DD_STEP_LITERAL_ENV[index + 1]}
        dd_validate_environment_name "${name}" || return 64
        printf -v "${name}" '%s' "${value}"
        export "${name}"
    done
    for ((index = 0; index < ${#DD_STEP_SECRET_ENV[@]}; index += 2)); do
        name=${DD_STEP_SECRET_ENV[index]}
        printf -v "${name}" '%s' "${secret_values[index / 2]}"
        export "${name}"
    done
    for ((index = 0; index < ${#DD_STEP_ENV_PREFIXES[@]}; index += 2)); do
        name=${DD_STEP_ENV_PREFIXES[index]}
        prefix=${DD_STEP_ENV_PREFIXES[index + 1]}
        dd_validate_environment_name "${name}" || return 64
        current=${!name}
        printf -v "${name}" '%s' "${prefix}:${current}"
        export "${name}"
    done
    if [[ -n ${DD_STEP_VISIBLE_GPUS} ]]; then
        export CUDA_VISIBLE_DEVICES=${DD_STEP_VISIBLE_GPUS}
    fi
}

dd_build_srun_command() {
    local index gpu_mask=0 container_names= node_list=
    ((${#DD_STEP_NODE_HOSTS[@]})) || return 64
    printf -v node_list '%s,' "${DD_STEP_NODE_HOSTS[@]}"
    node_list=${node_list%,}
    DD_STEP_VISIBLE_GPUS=
    DD_SRUN_COMMAND=(
        srun
        "--nodes=${#DD_STEP_NODE_HOSTS[@]}"
        "--ntasks=${#DD_STEP_NODE_HOSTS[@]}"
        --ntasks-per-node=1
        "--nodelist=${node_list}"
        --exact
        --overlap
        --unbuffered
        --export=ALL
        "--cpus-per-task=${DD_STEP_CPUS}"
        "--container-image=${DD_STEP_IMAGE}"
    )
    if [[ ${DD_STEP_KILL_ON_BAD_EXIT} == true ]]; then
        DD_SRUN_COMMAND+=(--kill-on-bad-exit=1)
    fi
    if ((${#DD_STEP_GPU_INDICES[@]})); then
        if [[ ${DD_GPU_REQUEST_MODE} == gres ]]; then
            for index in "${DD_STEP_GPU_INDICES[@]+"${DD_STEP_GPU_INDICES[@]}"}"; do
                ((gpu_mask |= 1 << index))
            done
            printf -v gpu_mask '0x%x' "${gpu_mask}"
            DD_SRUN_COMMAND+=(
                "--gpus-per-task=${#DD_STEP_GPU_INDICES[@]}"
                "--gpu-bind=mask_gpu:${gpu_mask}"
            )
        else
            local visible_gpus
            printf -v visible_gpus '%s,' "${DD_STEP_GPU_INDICES[@]}"
            DD_STEP_VISIBLE_GPUS=${visible_gpus%,}
            DD_STEP_CONTAINER_ENV+=(CUDA_VISIBLE_DEVICES)
        fi
    else
        DD_SRUN_COMMAND+=(--gres=none)
    fi
    [[ -z ${DD_CONTAINER_MOUNTS} ]] || DD_SRUN_COMMAND+=("--container-mounts=${DD_CONTAINER_MOUNTS}")
    if ((${#DD_STEP_CONTAINER_ENV[@]})); then
        printf -v container_names '%s,' "${DD_STEP_CONTAINER_ENV[@]}"
        DD_SRUN_COMMAND+=("--container-env=${container_names%,}")
    fi
}

dd_start_step() {
    local manifest=$1
    local step_id=$2
    dd_read_step "${manifest}" "${step_id}"
    dd_build_srun_command
    (
        dd_materialize_step_environment
        exec "${DD_SRUN_COMMAND[@]}" -- "${DD_STEP_COMMAND[@]}"
    ) >"${DD_STEP_STDOUT}" 2>"${DD_STEP_STDERR}" &
    DD_LAST_PID=$!
}

dd_run_step() {
    local manifest=$1
    local step_id=$2
    dd_start_step "${manifest}" "${step_id}"
    local pid=${DD_LAST_PID}
    local index=${#DD_MANAGED_PIDS[@]}
    local status=0
    DD_MANAGED_PIDS+=("${pid}")
    wait "${pid}" || status=$?
    unset "DD_MANAGED_PIDS[${index}]"
    DD_MANAGED_PIDS=("${DD_MANAGED_PIDS[@]+"${DD_MANAGED_PIDS[@]}"}")
    return "${status}"
}

dd_run_control_phase() {
    local operation=$1
    shift
    local -a command=(
        srun
        --nodes=1
        --ntasks=1
        "--nodelist=${DD_CLIENT_HOST}"
        --exact
        --overlap
        --unbuffered
        --export=ALL
        "--cpus-per-task=${DD_CLIENT_CPUS}"
        "--container-image=${DD_CLIENT_IMAGE}"
        --gres=none
    )
    [[ -z ${DD_CONTAINER_MOUNTS} ]] || command+=("--container-mounts=${DD_CONTAINER_MOUNTS}")
    command+=(
        --container-env=PYTHONPATH,SLURM_JOB_GPUS
        --
        python3
        -m
        data_designer.slurm.runtime.entrypoint
        "${operation}"
        --plan
        "${DD_PLAN_CONTAINER_PATH}"
        --attempt-dir
        "${DD_ATTEMPT_CONTAINER_DIR}"
        "$@"
    )
    (
        dd_scope_control_environment
        exec "${command[@]}"
    )
}
