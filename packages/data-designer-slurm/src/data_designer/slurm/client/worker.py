# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib
import signal
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from data_designer.slurm.client.environment import (
    ClientEnvironmentBuilder,
    PreparedClientEnvironment,
    activate_environment,
)
from data_designer.slurm.client.errors import ClientWorkerError
from data_designer.slurm.client.filesystem import replace_private_text
from data_designer.slurm.client.records import (
    ClientEnvironmentManifest,
    ClientEnvironmentOutcome,
    ClientErrorCode,
    ClientPluginEntryPoint,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one allocation-local client preflight or generation operation."""
    arguments = _parse_arguments(argv)
    previous_sigterm_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    prepared: PreparedClientEnvironment | None = None
    plugins: tuple[ClientPluginEntryPoint, ...] = ()
    try:
        endpoints = _parse_endpoints(arguments.endpoint)
        prepared = ClientEnvironmentBuilder().prepare(
            arguments.plan,
            shard_id=arguments.shard_id,
            attempt_id=arguments.attempt_id,
            attempt_dir=arguments.attempt_dir,
        )
        activate_environment(prepared)
        plugins_module = importlib.import_module("data_designer.slurm.client.plugins")
        discover_plugins = getattr(plugins_module, "discover_plugins")
        plugins = discover_plugins(prepared.installed_distributions)
        execution_module = importlib.import_module("data_designer.slurm.client.execution")
        ClientWorker = getattr(execution_module, "ClientWorker")
        worker = ClientWorker()
        if arguments.operation == "preflight":
            worker.preflight(arguments.plan, prepared=prepared, endpoints=endpoints, plugins=plugins)
        else:
            worker.run(arguments.plan, prepared=prepared, endpoints=endpoints, plugins=plugins)
        return 0
    except ClientWorkerError as error:
        if prepared is not None and error.code is ClientErrorCode.PLUGIN_LOAD_FAILED:
            _write_plugin_failure(prepared, plugins, error)
        print(f"client worker failed: {error.code.value}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, SystemExit):
        print(f"client worker failed: {ClientErrorCode.INTERRUPTED.value}", file=sys.stderr)
        return 2
    except Exception:
        print(f"client worker failed: {ClientErrorCode.INVALID_INPUT.value}", file=sys.stderr)
        return 2
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


def _handle_sigterm(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m data_designer.slurm.client.worker")
    parser.add_argument("operation", choices=("preflight", "run"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--endpoint", action="append", default=[])
    return parser.parse_args(argv)


def _parse_endpoints(values: list[str]) -> dict[str, str]:
    endpoints: dict[str, str] = {}
    for value in values:
        alias, separator, endpoint = value.partition("=")
        if not separator or not alias or not endpoint or alias in endpoints:
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "runtime endpoint argument is invalid")
        endpoints[alias] = endpoint
    return endpoints


def _write_plugin_failure(
    prepared: PreparedClientEnvironment,
    plugins: tuple[ClientPluginEntryPoint, ...],
    error: ClientWorkerError,
) -> None:
    manifest = ClientEnvironmentManifest.from_prepared(
        prepared,
        created_at=datetime.now(timezone.utc),
        outcome=ClientEnvironmentOutcome.FAILED,
        plugins=plugins,
        error_code=error.code,
        redacted_message=error.redacted_message,
    )
    replace_private_text(prepared.attempt_dir / "client-environment.json", manifest.serialize_json())


if __name__ == "__main__":
    raise SystemExit(main())
