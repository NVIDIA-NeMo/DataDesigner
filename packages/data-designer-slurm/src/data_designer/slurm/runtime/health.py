# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Continuous aggregate health checks for a running allocation."""

from __future__ import annotations

import http.client
import json
import time
from collections.abc import Mapping
from datetime import UTC, datetime

from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.state import (
    AttemptReadiness,
    DeploymentReadiness,
    ProbeEvidence,
    ProbeOutcome,
    ReadinessState,
    SlurmStateWriter,
)

_HEALTH_TIMEOUT_SECONDS = 6.0
_MAXIMUM_HEALTH_BYTES = 4096


def monitor_runtime_health(
    writer: SlurmStateWriter,
    readiness: AttemptReadiness,
    endpoint_ports: Mapping[str, int],
    *,
    poll_interval_seconds: float = 1.0,
) -> None:
    """Publish health evidence until readiness stops or a required endpoint fails.

    The allocation controller owns process termination and attempt finalization.
    This monitor fails on service loss even if all serving processes remain alive.
    """
    if poll_interval_seconds <= 0:
        raise ValueError("health poll interval must be positive")
    if set(endpoint_ports) != {deployment.deployment_id for deployment in readiness.deployments}:
        raise ValueError("health endpoints must match readiness deployments")
    while readiness.state is ReadinessState.READY:
        readiness = writer.write_readiness(probe_runtime_health(readiness, endpoint_ports))
        if readiness.state is ReadinessState.FAILED:
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.STEP_FAILED, "required inference service lost health")
        time.sleep(poll_interval_seconds)
        readiness = writer.load_readiness(readiness.shard_id, readiness.attempt_id)


def probe_runtime_health(readiness: AttemptReadiness, endpoint_ports: Mapping[str, int]) -> AttemptReadiness:
    """Check every logical endpoint and preserve per-deployment backend counts."""
    deployments: list[DeploymentReadiness] = []
    for deployment in readiness.deployments:
        count = _ready_backend_count(endpoint_ports[deployment.deployment_id], deployment.expected_backends)
        healthy = count == deployment.expected_backends
        observed_at = datetime.now(UTC)
        deployments.append(
            deployment.model_copy(
                update={
                    "state": ReadinessState.READY if healthy else ReadinessState.FAILED,
                    "ready_backends": count,
                    "last_probe": ProbeEvidence(
                        observed_at=observed_at,
                        outcome=ProbeOutcome.SUCCESS if healthy else ProbeOutcome.FAILURE,
                        reason_code="runtime_healthy" if healthy else "runtime_health_lost",
                        redacted_message=f"{count} of {deployment.expected_backends} backends ready",
                    ),
                }
            )
        )
    return readiness.model_copy(
        update={
            "revision": readiness.revision + 1,
            "updated_at": datetime.now(UTC),
            "state": (
                ReadinessState.READY
                if all(deployment.state is ReadinessState.READY for deployment in deployments)
                else ReadinessState.FAILED
            ),
            "deployments": tuple(deployments),
        }
    )


def _ready_backend_count(port: int, expected: int) -> int:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=_HEALTH_TIMEOUT_SECONDS)
    try:
        connection.request("GET", "/health", headers={"Connection": "close"})
        response = connection.getresponse()
        body = response.read(_MAXIMUM_HEALTH_BYTES + 1)
        if response.status not in {200, 503} or len(body) > _MAXIMUM_HEALTH_BYTES:
            return 0
        payload = json.loads(body)
        if not isinstance(payload, dict) or type(payload.get("backends")) is not int or payload["backends"] != expected:
            return 0
        count = payload.get("backends_ready")
        if type(count) is not int or not 0 <= count <= expected:
            return 0
        return count if response.status == 200 or count < expected else 0
    except (OSError, http.client.HTTPException, ValueError):
        return 0
    finally:
        connection.close()
