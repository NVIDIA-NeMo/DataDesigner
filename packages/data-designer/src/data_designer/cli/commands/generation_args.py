# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import click


@dataclass(frozen=True)
class GenerationConfigTarget:
    """Resolved config target for create, preview, and validate commands."""

    config_source: str | None
    recipe: str | None
    workflow_args: tuple[str, ...]


def resolve_generation_config_target(
    raw_args: list[str] | None,
    recipe: str | None,
) -> GenerationConfigTarget:
    """Split variadic CLI args into a config source or recipe plus workflow args."""
    args = tuple(raw_args or ())
    if recipe is not None:
        return GenerationConfigTarget(config_source=None, recipe=recipe, workflow_args=args)

    if not args:
        raise click.UsageError("Missing argument 'CONFIG_SOURCE'. Provide a config source or use --recipe.")

    config_source, *workflow_args = args
    return GenerationConfigTarget(config_source=config_source, recipe=None, workflow_args=tuple(workflow_args))
