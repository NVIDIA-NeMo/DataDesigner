# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict transport records for one coordinated deployment's node workers."""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass

from data_designer.slurm.runtime.network import validate_host_name, validate_network_port

_MAXIMUM_SPEC_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class NodeProcessSpec:
    """One resolved lane process owned by a node worker."""

    process_id: str
    command: tuple[str, ...]
    gpu_indices: tuple[int, ...]
    launch_delay_seconds: int

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


@dataclass(frozen=True, slots=True)
class NodeWorkerSpec:
    """Strict, bounded input shared by every task in one deployment step."""

    schema_version: int
    resolved_gpus_per_node: int
    nodes: tuple[NodeSpec, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("node worker schema version is unsupported")
        if type(self.resolved_gpus_per_node) is not int or self.resolved_gpus_per_node <= 0:
            raise ValueError("resolved GPU count is invalid")
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
                raise ValueError("node processes must not share GPUs")


def encode_node_worker_spec(spec: NodeWorkerSpec) -> str:
    """Serialize one validated node-worker specification for an argv boundary."""
    payload = {
        "schema_version": spec.schema_version,
        "resolved_gpus_per_node": spec.resolved_gpus_per_node,
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
                    }
                    for process in node.processes
                ],
            }
            for node in spec.nodes
        ],
    }
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
    root = _require_mapping(payload, {"schema_version", "resolved_gpus_per_node", "nodes"})
    nodes = tuple(_parse_node(value) for value in _require_list(root["nodes"]))
    return NodeWorkerSpec(
        schema_version=_require_integer(root["schema_version"]),
        resolved_gpus_per_node=_require_integer(root["resolved_gpus_per_node"]),
        nodes=nodes,
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
    value = _require_mapping(payload, {"process_id", "command", "gpu_indices", "launch_delay_seconds"})
    return NodeProcessSpec(
        process_id=_require_string(value["process_id"]),
        command=tuple(_require_string(argument) for argument in _require_list(value["command"])),
        gpu_indices=tuple(_require_integer(index) for index in _require_list(value["gpu_indices"])),
        launch_delay_seconds=_require_integer(value["launch_delay_seconds"]),
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


def _require_integer(payload: object) -> int:
    if type(payload) is not int:
        raise ValueError("node worker integer value is invalid")
    return payload


__all__ = [
    "NodeProcessSpec",
    "NodeSpec",
    "NodeWorkerSpec",
    "decode_node_worker_spec",
    "encode_node_worker_spec",
]
