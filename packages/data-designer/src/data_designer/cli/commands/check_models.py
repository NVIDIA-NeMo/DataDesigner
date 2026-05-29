# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from data_designer.cli.controllers.generation_controller import GenerationController


def check_models_command(
    config_source: str = typer.Argument(
        help=(
            "Path or URL to a config file (.yaml/.yml/.json), or a local Python module (.py)"
            " that defines a load_config_builder() function."
        ),
    ),
) -> None:
    """Check that every model and MCP tool referenced by the configuration is reachable.

    Runs the same readiness probes performed at the start of ``preview`` and
    ``create``: a tiny generation against each referenced model alias, plus a
    connectivity probe to each referenced MCP tool. Models with
    ``skip_health_check=True`` are skipped.

    Complements ``validate``: ``validate`` checks the configuration is
    well-formed (internal readiness); ``check-models`` checks the providers
    it depends on are responsive (external readiness).

    Examples:
        # Check models referenced by a YAML config
        data-designer check-models my_config.yaml

        # Check models referenced by a remote config URL
        data-designer check-models https://example.com/my_config.yaml

        # Check models referenced by a Python module
        data-designer check-models my_config.py
    """
    controller = GenerationController()
    controller.run_check_models(config_source=config_source)
