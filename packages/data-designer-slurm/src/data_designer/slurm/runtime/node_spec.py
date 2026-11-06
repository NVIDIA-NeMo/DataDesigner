# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict transport records for one coordinated deployment's node workers."""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import asdict, dataclass

from data_designer.slurm.contracts import validate_absolute_path
from data_designer.slurm.runtime.network import validate_host_name, validate_network_port

_MAXIMUM_SPEC_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class NodeReadinessSpec:
    """Bounded loopback readiness for a node-local process."""

    port: int
    path: str
    deadline_seconds: int

    def __post_init__(self) -> None:
        validate_network_port(self.port)
        if (
            type(self.path) is not str
            or not self.path.startswith("/")
            or any(ord(value) < 32 or ord(value) == 127 for value in self.path)
        ):
            raise ValueError("node readiness path is invalid")
        if type(self.deadline_seconds) is not int or self.deadline_seconds <= 0:
            raise ValueError("node readiness deadline is invalid")


@dataclass(frozen=True, slots=True)
class NodeProcessSpec:
    """One resolved lane process owned by a node worker."""

    process_id: str
    command: tuple[str, ...]
    gpu_indices: tuple[int, ...]
    launch_delay_seconds: int
    readiness: NodeReadinessSpec | None = None

    def __post_init__(self) -> None:
        if (
            type(self.process_id) is not str
            or not self.process_id
            or any(ord(value) < 32 or ord(value) == 127 for value in self.process_id)
        ):
            raise ValueError("node process identity is invalid")
        if (
            type(self.command) is not tuple
            or not self.command
            or any(type(value) is not str or not value or "\0" in value for value in self.command)
        ):
            raise ValueError("node process command is invalid")
        if (
            type(self.gpu_indices) is not tuple
            or not self.gpu_indices
            or self.gpu_indices != tuple(sorted(set(self.gpu_indices)))
            or any(type(value) is not int or value < 0 for value in self.gpu_indices)
        ):
            raise ValueError("node process GPU indices are invalid")
        if type(self.launch_delay_seconds) is not int or self.launch_delay_seconds < 0:
            raise ValueError("node process launch delay is invalid")
        if self.readiness is not None and not isinstance(self.readiness, NodeReadinessSpec):
            raise ValueError("node process readiness is invalid")


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """Resolved work and ports for one node in a coordinated step."""

    node_index: int
    host: str
    ports: tuple[int, ...]
    processes: tuple[NodeProcessSpec, ...]

    def __post_init__(self) -> None:
        if type(self.node_index) is not int or self.node_index < 0:
            raise ValueError("node index is invalid")
        validate_host_name(self.host)
        if type(self.ports) is not tuple or type(self.processes) is not tuple:
            raise ValueError("node ports and processes must be tuples")
        for port in self.ports:
            validate_network_port(port)
        if self.ports != tuple(sorted(set(self.ports))):
            raise ValueError("node ports must be sorted and unique")
        if not self.processes or len({process.process_id for process in self.processes}) != len(self.processes):
            raise ValueError("node process identities must be present and unique")
        if any(
            process.readiness is not None and process.readiness.port not in self.ports for process in self.processes
        ):
            raise ValueError("node readiness port is not assigned to the node")


@dataclass(frozen=True, slots=True)
class NodeWorkerSpec:
    """Strict, bounded input shared by every task in one deployment step."""

    schema_version: int
    resolved_gpus_per_node: int
    required_model_path: str | None
    nodes: tuple[NodeSpec, ...]
    required_model_paths: tuple[str, ...] = ()
    allow_gpu_sharing: bool = False

    def __post_init__(self) -> None:
        if type(self.allow_gpu_sharing) is not bool:
            raise ValueError("GPU sharing policy must be a boolean")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("node worker schema version is unsupported")
        if type(self.resolved_gpus_per_node) is not int or self.resolved_gpus_per_node <= 0:
            raise ValueError("resolved GPU count is invalid")
        if self.required_model_path is not None and type(self.required_model_path) is not str:
            raise ValueError("required model path is invalid")
        if self.required_model_path is not None:
            validate_absolute_path(self.required_model_path)
        if type(self.required_model_paths) is not tuple:
            raise ValueError("required model paths must be a tuple")
        for path in self.required_model_paths:
            if type(path) is not str:
                raise ValueError("required model path is invalid")
            validate_absolute_path(path)
        if (
            type(self.nodes) is not tuple
            or not self.nodes
            or tuple(node.node_index for node in self.nodes) != tuple(sorted({node.node_index for node in self.nodes}))
        ):
            raise ValueError("node worker nodes must use sorted unique indices")
        process_ids = tuple(process.process_id for node in self.nodes for process in node.processes)
        if len({node.host for node in self.nodes}) != len(self.nodes) or len(set(process_ids)) != len(process_ids):
            raise ValueError("node worker hosts and process identities must be unique")
        for node in self.nodes:
            if any(index >= self.resolved_gpus_per_node for process in node.processes for index in process.gpu_indices):
                raise ValueError("node process GPU index is outside the allocation")
            assigned_gpus = tuple(index for process in node.processes for index in process.gpu_indices)
            if len(set(assigned_gpus)) != len(assigned_gpus):
                if not self.allow_gpu_sharing:
                    raise ValueError("node processes must not share GPUs")
                if any(process.readiness is None for process in node.processes):
                    raise ValueError("GPU sharing requires process readiness probes")


def encode_node_worker_spec(spec: NodeWorkerSpec) -> str:
    """Serialize one validated node-worker specification for an argv boundary."""
    payload = {
        "schema_version": spec.schema_version,
        "resolved_gpus_per_node": spec.resolved_gpus_per_node,
        "required_model_path": spec.required_model_path,
        "nodes": [
            {
                "node_index": node.node_index,
                "host": node.host,
                "ports": list(node.ports),
                "processes": [
                    {
                        "process_id": process.process_id,
                        "command": list(process.command),
                        "gpu_indices": list(process.gpu_indices),
                        "launch_delay_seconds": process.launch_delay_seconds,
                        **({"readiness": asdict(process.readiness)} if process.readiness else {}),
                    }
                    for process in node.processes
                ],
            }
            for node in spec.nodes
        ],
    }
    if spec.required_model_paths:
        payload["required_model_paths"] = list(spec.required_model_paths)
    if spec.allow_gpu_sharing:
        payload["allow_gpu_sharing"] = True
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if len(serialized) > _MAXIMUM_SPEC_BYTES:
        raise ValueError("node worker specification exceeds its size limit")
    return base64.urlsafe_b64encode(serialized).decode()


def decode_node_worker_spec(encoded: str) -> NodeWorkerSpec:
    """Decode and validate an untrusted node-worker specification."""
    if type(encoded) is not str or not encoded or len(encoded) > 2 * _MAXIMUM_SPEC_BYTES:
        raise ValueError("node worker specification is invalid")
    try:
        serialized = base64.b64decode(encoded, altchars=b"-_", validate=True)
        if len(serialized) > _MAXIMUM_SPEC_BYTES:
            raise ValueError("node worker specification exceeds its size limit")
        payload = json.loads(serialized)
    except (binascii.Error, json.JSONDecodeError, UnicodeEncodeError) as error:
        raise ValueError("node worker specification is invalid") from error
    return _parse_worker_spec(payload)


def _parse_worker_spec(payload: object) -> NodeWorkerSpec:
    keys = {"schema_version", "resolved_gpus_per_node", "required_model_path", "nodes"}
    if isinstance(payload, dict):
        keys.update(set(payload) & {"required_model_paths", "allow_gpu_sharing"})
    root = _require_mapping(payload, keys)
    nodes = tuple(_parse_node(value) for value in _require_list(root["nodes"]))
    return NodeWorkerSpec(
        schema_version=_require_integer(root["schema_version"]),
        resolved_gpus_per_node=_require_integer(root["resolved_gpus_per_node"]),
        required_model_path=_require_optional_string(root["required_model_path"]),
        nodes=nodes,
        required_model_paths=tuple(
            _require_string(path) for path in _require_list(root.get("required_model_paths", []))
        ),
        allow_gpu_sharing=_require_boolean(root.get("allow_gpu_sharing", False)),
    )


def _parse_node(payload: object) -> NodeSpec:
    value = _require_mapping(payload, {"node_index", "host", "ports", "processes"})
    return NodeSpec(
        node_index=_require_integer(value["node_index"]),
        host=_require_string(value["host"]),
        ports=tuple(_require_integer(port) for port in _require_list(value["ports"])),
        processes=tuple(_parse_process(process) for process in _require_list(value["processes"])),
    )


def _parse_process(payload: object) -> NodeProcessSpec:
    keys = {"process_id", "command", "gpu_indices", "launch_delay_seconds"}
    if isinstance(payload, dict) and "readiness" in payload:
        keys.add("readiness")
    value = _require_mapping(payload, keys)
    return NodeProcessSpec(
        process_id=_require_string(value["process_id"]),
        command=tuple(_require_string(argument) for argument in _require_list(value["command"])),
        gpu_indices=tuple(_require_integer(index) for index in _require_list(value["gpu_indices"])),
        launch_delay_seconds=_require_integer(value["launch_delay_seconds"]),
        readiness=_parse_readiness(value["readiness"]) if "readiness" in value else None,
    )


def _parse_readiness(payload: object) -> NodeReadinessSpec:
    value = _require_mapping(payload, {"port", "path", "deadline_seconds"})
    return NodeReadinessSpec(
        port=_require_integer(value["port"]),
        path=_require_string(value["path"]),
        deadline_seconds=_require_integer(value["deadline_seconds"]),
    )


def _require_mapping(payload: object, keys: set[str]) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != keys or not all(type(key) is str for key in payload):
        raise ValueError("node worker object shape is invalid")
    return payload


def _require_list(payload: object) -> list[object]:
    if not isinstance(payload, list):
        raise ValueError("node worker list value is invalid")
    return payload


def _require_string(payload: object) -> str:
    if type(payload) is not str:
        raise ValueError("node worker string value is invalid")
    return payload


def _require_optional_string(payload: object) -> str | None:
    if payload is None:
        return None
    return _require_string(payload)


def _require_integer(payload: object) -> int:
    if type(payload) is not int:
        raise ValueError("node worker integer value is invalid")
    return payload


def _require_boolean(payload: object) -> bool:
    if type(payload) is not bool:
        raise ValueError("node worker boolean value is invalid")
    return payload


__all__ = [
    "NodeProcessSpec",
    "NodeReadinessSpec",
    "NodeSpec",
    "NodeWorkerSpec",
    "decode_node_worker_spec",
    "encode_node_worker_spec",
]
