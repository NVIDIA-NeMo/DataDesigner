# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

dd_read_null_values() {
    local array_name=$1
    local index=0 value
    unset "${array_name}"
    declare -g -a "${array_name}"
    printf -v "${array_name}[0]" '%s' ""
    unset "${array_name}[0]"
    while IFS= read -r -d '' value; do
        printf -v "${array_name}[${index}]" '%s' "${value}"
        ((index += 1))
    done
}

dd_read_control_plan() {
    local plan=$1
    jq -e '
        .schema_version == 1
        and ([.deployments[].node_indices[]] | length > 0)
        and ([.deployments[].node_indices[]] | all(type == "number" and . >= 0))
        and ([.container_mounts[] | (.source + .target)] | all(test("[,:]") | not))
    ' "${plan}" >/dev/null
    DD_CLIENT_IMAGE=$(jq -er '.client.image.path' "${plan}")
    DD_CLIENT_CPUS=$(jq -er '.client.authored.cpus | tostring' "${plan}")
    DD_EXPECTED_GPUS=$(jq -er '.resolved_gpus_per_node | tostring' "${plan}")
    DD_EXPECTED_NODES=$(jq -er '[.deployments[].node_indices[]] | max + 1 | tostring' "${plan}")
    DD_CLIENT_NODE_INDEX=$(jq -er '.client.host_node_index | tostring' "${plan}")
    DD_GPU_REQUEST_MODE=$(jq -er '.selected_profile.profile.gpu_request_mode' "${plan}")
    DD_CONTAINER_MOUNTS=$(jq -jr '
        [.container_mounts[] | .source + ":" + .target + (if .read_only then ":ro" else "" end)]
        | join(",")
    ' "${plan}")
}

dd_read_container_path() {
    local plan=$1
    local host_path=$2
    local require_writable=$3
    DD_CONTAINER_PATH=$(jq -er \
        --arg path "${host_path}" \
        --argjson require_writable "${require_writable}" '
        [
            .container_mounts[]
            | . as $mount
            | select(($path == $mount.source) or ($path | startswith($mount.source + "/")))
            | select(($require_writable | not) or ($mount.read_only | not))
        ]
        | sort_by(.source | length)
        | last
        | select(. != null)
        | . as $mount
        | if $path == $mount.source then
              $mount.target
          else
              $mount.target + ($path | ltrimstr($mount.source))
          end
    ' "${plan}")
}

dd_read_artifacts() {
    local plan=$1
    local task_id=$2
    dd_read_null_values DD_ARTIFACT_FIELDS < <(
        jq -j --argjson task_id "${task_id}" '
            [
                .runtime_bundle,
                .client.dependency_lock,
                {path: .client.image.path, sha256: .client.image.sha256},
                (.deployments[] | {path: .image.path, sha256: .image.sha256}),
                .builder.source,
                (.shards[] | select(.array_task_index == $task_id) | .input_partition)
            ]
            | map(select(. != null))
            | unique_by([.path, .sha256])
            | .[]
            | .path, "\u0000", .sha256, "\u0000"
        ' "${plan}"
    )
}

dd_read_plan_secret_names() {
    local plan=$1
    dd_read_null_values DD_REQUIRED_SECRET_NAMES < <(
        jq -j '[.. | objects | select(.type? == "secret") | .environment] | unique | .[] | ., "\u0000"' "${plan}"
    )
    dd_read_null_values DD_ALL_SECRET_NAMES < <(
        jq -j '
            [
                (.. | objects | select(.type? == "secret") | .environment),
                (
                    .deployments[].authored.server.environment
                    | to_entries[]
                    | select(.value.type == "secret")
                    | .key
                )
            ]
            | unique
            | .[]
            | ., "\u0000"
        ' "${plan}"
    )
}

dd_verify_runtime_manifest() {
    local manifest=$1
    local plan_sha256=$2
    local shard_id=$3
    local attempt_id=$4
    jq -e \
        --arg plan_sha256 "${plan_sha256}" \
        --arg shard_id "${shard_id}" \
        --arg attempt_id "${attempt_id}" '
        .schema_version == 1
        and .plan_sha256 == $plan_sha256
        and .shard_id == $shard_id
        and .attempt_id == $attempt_id
        and ([.steps[].step_id] | length == (unique | length))
    ' "${manifest}" >/dev/null
}

dd_read_step_ids() {
    local manifest=$1
    local role=$2
    dd_read_null_values DD_STEP_IDS < <(
        jq -j --arg role "${role}" '.steps[] | select(.role == $role) | .step_id, "\u0000"' "${manifest}"
    )
}

dd_read_step() {
    local manifest=$1
    local step_id=$2
    jq -e --arg step_id "${step_id}" '[.steps[] | select(.step_id == $step_id)] | length == 1' \
        "${manifest}" >/dev/null
    dd_read_null_values DD_STEP_FIELDS < <(
        jq -j --arg step_id "${step_id}" '
            .steps[]
            | select(.step_id == $step_id)
            | .image_path, "\u0000",
              (.cpus | tostring), "\u0000",
              .stdout_path, "\u0000",
              .stderr_path, "\u0000",
              (.launch_delay_seconds | tostring), "\u0000",
              (.kill_on_bad_exit | tostring), "\u0000"
        ' "${manifest}"
    )
    DD_STEP_IMAGE=${DD_STEP_FIELDS[0]}
    DD_STEP_CPUS=${DD_STEP_FIELDS[1]}
    DD_STEP_STDOUT=${DD_STEP_FIELDS[2]}
    DD_STEP_STDERR=${DD_STEP_FIELDS[3]}
    DD_STEP_DELAY=${DD_STEP_FIELDS[4]}
    DD_STEP_KILL_ON_BAD_EXIT=${DD_STEP_FIELDS[5]}
    dd_read_null_values DD_STEP_COMMAND < <(
        jq -j --arg step_id "${step_id}" '.steps[] | select(.step_id == $step_id) | .command[] | ., "\u0000"' \
            "${manifest}"
    )
    dd_read_null_values DD_STEP_GPU_INDICES < <(
        jq -j --arg step_id "${step_id}" \
            '.steps[] | select(.step_id == $step_id) | .gpu_indices[] | tostring, "\u0000"' "${manifest}"
    )
    dd_read_null_values DD_STEP_NODE_HOSTS < <(
        jq -j --arg step_id "${step_id}" \
            '.steps[] | select(.step_id == $step_id) | .node_hosts[] | ., "\u0000"' "${manifest}"
    )
    dd_read_null_values DD_STEP_PROBE_FIELDS < <(
        jq -j --arg step_id "${step_id}" '
            .steps[]
            | select(.step_id == $step_id)
            | .readiness[]
            | .host, "\u0000",
              (.port | tostring), "\u0000",
              .path, "\u0000",
              (.deadline_seconds | tostring), "\u0000"
        ' "${manifest}"
    )
    dd_read_null_values DD_STEP_CONTAINER_ENV < <(
        jq -j --arg step_id "${step_id}" \
            '.steps[] | select(.step_id == $step_id) | .container_environment[] | ., "\u0000"' "${manifest}"
    )
    dd_read_null_values DD_STEP_LITERAL_ENV < <(
        jq -j --arg step_id "${step_id}" '
            .steps[]
            | select(.step_id == $step_id)
            | .literal_environment
            | to_entries[]
            | .key, "\u0000", .value, "\u0000"
        ' "${manifest}"
    )
    dd_read_null_values DD_STEP_SECRET_ENV < <(
        jq -j --arg step_id "${step_id}" '
            .steps[]
            | select(.step_id == $step_id)
            | .secret_environment
            | to_entries[]
            | .key, "\u0000", .value, "\u0000"
        ' "${manifest}"
    )
    dd_read_null_values DD_STEP_ENV_PREFIXES < <(
        jq -j --arg step_id "${step_id}" '
            .steps[]
            | select(.step_id == $step_id)
            | .environment_prefixes
            | to_entries[]
            | .key, "\u0000", .value, "\u0000"
        ' "${manifest}"
    )
}
