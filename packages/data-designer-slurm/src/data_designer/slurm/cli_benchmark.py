# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin benchmark commands for the Slurm CLI."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import typer
from pydantic import BaseModel

from data_designer.slurm.config import load_benchmark_config
from data_designer.slurm.services import create_slurm_benchmark_service
from data_designer.slurm.services.errors import SlurmServiceOperation

_Invoke = Callable[[SlurmServiceOperation, Callable[[], BaseModel]], BaseModel]
_Emit = Callable[[BaseModel], None]


def _run_benchmark(
    benchmark_file: Path,
    *,
    profile_file: Path | None,
    cluster: str | None,
    force: bool,
) -> BaseModel:
    config = load_benchmark_config(benchmark_file)
    service = create_slurm_benchmark_service(profile_file=profile_file, cluster=cluster)
    return service.run(config, source_root=benchmark_file.resolve().parent, force=force)


def _analyze_benchmark(
    benchmark: str,
    *,
    profile_file: Path | None,
    cluster: str | None,
    refresh: bool,
    fail_if_incomplete: bool,
) -> BaseModel:
    benchmark_id = Path(benchmark.rstrip("/")).name
    service = create_slurm_benchmark_service(profile_file=profile_file, cluster=cluster)
    return service.analyze(
        benchmark_id,
        refresh_state=refresh,
        fail_if_incomplete=fail_if_incomplete,
    )


def create_benchmark_app(invoke: _Invoke, emit: _Emit) -> typer.Typer:
    """Create benchmark commands using the root CLI error and output policy."""
    benchmark_app = typer.Typer(help="Run and analyze Slurm benchmarks", no_args_is_help=True)

    @benchmark_app.command("run")
    def run_command(
        benchmark_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
        profile_file: Path | None = typer.Option(None, "--profile-file", dir_okay=False),
        cluster: str | None = typer.Option(None, "--cluster"),
        force: bool = typer.Option(False, "--force"),
    ) -> None:
        """Expand and submit ordinary child runs."""
        emit(
            invoke(
                SlurmServiceOperation.RUN_BENCHMARK,
                partial(
                    _run_benchmark,
                    benchmark_file,
                    profile_file=profile_file,
                    cluster=cluster,
                    force=force,
                ),
            )
        )

    @benchmark_app.command("analyze")
    def analyze_command(
        benchmark: str = typer.Argument(..., help="Managed benchmark ID or benchmark directory"),
        profile_file: Path | None = typer.Option(None, "--profile-file", dir_okay=False),
        cluster: str | None = typer.Option(None, "--cluster"),
        refresh: bool = typer.Option(False, "--refresh-state", "--refresh"),
        fail_if_incomplete: bool = typer.Option(False, "--fail-if-incomplete"),
    ) -> None:
        """Write one point-in-time report from ordinary child state."""
        emit(
            invoke(
                SlurmServiceOperation.ANALYZE_BENCHMARK,
                partial(
                    _analyze_benchmark,
                    benchmark,
                    profile_file=profile_file,
                    cluster=cluster,
                    refresh=refresh,
                    fail_if_incomplete=fail_if_incomplete,
                ),
            )
        )

    return benchmark_app


__all__ = ["create_benchmark_app"]
