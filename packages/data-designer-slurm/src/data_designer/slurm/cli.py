# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click
import typer

app = typer.Typer(
    name="slurm",
    help="Run Data Designer workloads on Slurm",
    no_args_is_help=True,
)


@app.callback()
def slurm_callback() -> None:
    pass


def create_cli() -> click.Command:
    """Create the Slurm CLI group."""
    return typer.main.get_command(app)
