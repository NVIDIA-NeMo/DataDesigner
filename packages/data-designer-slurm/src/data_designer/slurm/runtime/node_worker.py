# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the resolved vLLM lanes assigned to one physical allocation node."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence

from data_designer.slurm.runtime.node_spec import (
    NodeProcessSpec as NodeProcessSpec,
)
from data_designer.slurm.runtime.node_spec import (
    NodeSpec as NodeSpec,
)
from data_designer.slurm.runtime.node_spec import (
    NodeWorkerSpec as NodeWorkerSpec,
)
from data_designer.slurm.runtime.node_spec import (
    decode_node_worker_spec as decode_node_worker_spec,
)
from data_designer.slurm.runtime.node_spec import (
    encode_node_worker_spec as encode_node_worker_spec,
)

_POLL_INTERVAL_SECONDS = 0.1
_TERMINATION_GRACE_SECONDS = 10.0


class NodeProcessSupervisor:
    """Own every lane child on one node and stop them as a unit."""

    def __init__(self, *, environment: Mapping[str, str]) -> None:
        self._environment = dict(environment)
        self._children: list[subprocess.Popen[bytes]] = []
        self._cleanup_started = False
        self._cleanup_complete = False
        self._stop_requested = False

    @property
    def cleanup_complete(self) -> bool:
        """Return whether every registered lane child has exited."""
        return self._cleanup_complete

    def run(self, node: NodeSpec, visible_gpus: tuple[str, ...]) -> int:
        """Launch local lanes and fail the node task when any lane exits."""
        started_at = time.monotonic()
        try:
            for process in node.processes:
                self._wait_for_launch(process.launch_delay_seconds, started_at)
                if self._stop_requested:
                    return 1
                self._children.append(self._start_process(process, visible_gpus))
            return self._wait_for_first_exit()
        finally:
            self.cleanup()

    def request_stop(self) -> None:
        """Ask the poll loop to enter cleanup at its next boundary."""
        self._stop_requested = True

    def cleanup(self) -> None:
        """Idempotently terminate every registered lane child."""
        if self._cleanup_complete:
            return
        self._cleanup_started = True
        live = tuple(child for child in reversed(self._children) if child.poll() is None)
        _signal_children(live, signal.SIGTERM)
        deadline = time.monotonic() + _TERMINATION_GRACE_SECONDS
        for child in live:
            _wait_until_deadline(child, deadline)
        _signal_children(tuple(child for child in live if child.poll() is None), signal.SIGKILL)
        for child in live:
            try:
                child.wait(timeout=_TERMINATION_GRACE_SECONDS)
            except (OSError, subprocess.SubprocessError):
                pass
        self._cleanup_complete = all(child.poll() is not None for child in self._children)
        if not self._cleanup_complete:
            raise RuntimeError("node process cleanup is incomplete")

    def _start_process(self, process: NodeProcessSpec, visible_gpus: tuple[str, ...]) -> subprocess.Popen[bytes]:
        if self._cleanup_started:
            raise RuntimeError("cannot launch a node process after cleanup")
        environment = dict(self._environment)
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpus[index] for index in process.gpu_indices)
        return subprocess.Popen(
            process.command,
            stdin=subprocess.DEVNULL,
            env=environment,
            shell=False,
            start_new_session=True,
            close_fds=True,
        )

    def _wait_for_launch(self, delay_seconds: int, started_at: float) -> None:
        while not self._stop_requested:
            remaining = delay_seconds - (time.monotonic() - started_at)
            if remaining <= 0:
                return
            time.sleep(min(remaining, _POLL_INTERVAL_SECONDS))

    def _wait_for_first_exit(self) -> int:
        while True:
            if self._stop_requested:
                return 1
            for child in self._children:
                returncode = child.poll()
                if returncode is not None:
                    return returncode if returncode != 0 else 1
            time.sleep(_POLL_INTERVAL_SECONDS)


def main(arguments: Sequence[str] | None = None) -> int:
    """Preflight or serve the node-local slice of a coordinated deployment."""
    parser = argparse.ArgumentParser(prog="data-designer-slurm-node-worker")
    parser.add_argument("operation", choices=("preflight", "serve"))
    parser.add_argument("--spec", required=True)
    parsed = parser.parse_args(arguments)
    try:
        spec = decode_node_worker_spec(parsed.spec)
        node = _select_node(spec, os.environ)
        visible_gpus = _parse_visible_gpus(os.environ.get("CUDA_VISIBLE_DEVICES"))
        _verify_node(spec, node, visible_gpus, os.environ)
        if parsed.operation == "preflight":
            _verify_ports(node.ports)
            return 0
        return _run_node(node, visible_gpus)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        print("node worker failed at a validated runtime boundary", file=sys.stderr)
        return 70


def _run_node(node: NodeSpec, visible_gpus: tuple[str, ...]) -> int:
    supervisor = NodeProcessSupervisor(environment=os.environ)
    interrupted: list[int] = []

    def handle_termination(signum: int, frame: object) -> None:
        del frame
        if not interrupted:
            interrupted.append(signum)
        supervisor.request_stop()

    previous = {selected: signal.signal(selected, handle_termination) for selected in (signal.SIGINT, signal.SIGTERM)}
    try:
        return _run_until_exit_or_signal(supervisor, node, visible_gpus, interrupted)
    finally:
        try:
            supervisor.cleanup()
        finally:
            for selected, handler in previous.items():
                signal.signal(selected, handler)


def _run_until_exit_or_signal(
    supervisor: NodeProcessSupervisor,
    node: NodeSpec,
    visible_gpus: tuple[str, ...],
    interrupted: list[int],
) -> int:
    if interrupted:
        return 128 + interrupted[0]
    result = supervisor.run(node, visible_gpus)
    if interrupted:
        return 128 + interrupted[0]
    return result


def _select_node(spec: NodeWorkerSpec, environment: Mapping[str, str]) -> NodeSpec:
    process_id = _parse_non_negative_integer(environment.get("SLURM_PROCID"), "SLURM_PROCID")
    if process_id >= len(spec.nodes):
        raise ValueError("Slurm process identity is outside the deployment")
    return spec.nodes[process_id]


def _verify_node(
    spec: NodeWorkerSpec,
    node: NodeSpec,
    visible_gpus: tuple[str, ...],
    environment: Mapping[str, str],
) -> None:
    if len(visible_gpus) != spec.resolved_gpus_per_node:
        raise ValueError("node GPU visibility does not match the resolved plan")
    task_count = _parse_non_negative_integer(environment.get("SLURM_NTASKS"), "SLURM_NTASKS")
    if task_count != len(spec.nodes):
        raise ValueError("Slurm task count does not match the deployment")
    scheduler_host = environment.get("SLURMD_NODENAME")
    if scheduler_host is not None and scheduler_host != node.host:
        raise ValueError("Slurm node identity does not match the deployment")


def _verify_ports(ports: tuple[int, ...]) -> None:
    reservations: list[socket.socket] = []
    try:
        for port in ports:
            reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            reservations.append(reservation)
            reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            reservation.bind(("0.0.0.0", port))
    finally:
        for reservation in reservations:
            reservation.close()


def _parse_visible_gpus(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return ()
    values = tuple(item.strip() for item in value.split(","))
    if not all(values) or len(values) != len(set(values)) or any("\0" in item for item in values):
        raise ValueError("node GPU visibility is invalid")
    return values


def _parse_non_negative_integer(value: str | None, name: str) -> int:
    if value is None or not value.isascii() or not value.isdigit():
        raise ValueError(f"{name} is unavailable or invalid")
    return int(value)


def _signal_process_group(process: subprocess.Popen[bytes], selected: signal.Signals) -> None:
    try:
        os.killpg(process.pid, selected)
    except ProcessLookupError:
        pass


def _signal_children(children: tuple[subprocess.Popen[bytes], ...], selected: signal.Signals) -> None:
    for child in children:
        try:
            _signal_process_group(child, selected)
        except OSError:
            continue


def _wait_until_deadline(process: subprocess.Popen[bytes], deadline: float) -> None:
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_SECONDS)


if __name__ == "__main__":  # pragma: no cover - exercised as a managed step
    raise SystemExit(main())


__all__ = [
    "NodeProcessSpec",
    "NodeProcessSupervisor",
    "NodeSpec",
    "NodeWorkerSpec",
    "decode_node_worker_spec",
    "encode_node_worker_spec",
    "main",
]
