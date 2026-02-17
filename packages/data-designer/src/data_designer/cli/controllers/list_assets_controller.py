# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import typer

from data_designer.cli.repositories.persona_repository import PersonaRepository
from data_designer.cli.services.download_service import DownloadService


class ListAssetsController:
    """Controller for listing managed dataset assets."""

    def __init__(self, config_dir: Path) -> None:
        self.persona_repository = PersonaRepository()
        self.service = DownloadService(config_dir, self.persona_repository)

    def list_assets(self, output_format: str) -> None:
        """List installed and available Nemotron-Persona datasets.

        Args:
            output_format: "text" or "json".
        """
        all_locales = self.persona_repository.list_all()
        installed: list[str] = []
        not_installed: list[str] = []

        for locale in all_locales:
            if self.service.is_locale_downloaded(locale.code):
                installed.append(locale.code)
            else:
                not_installed.append(locale.code)

        if output_format == "json":
            typer.echo(json.dumps({"installed": installed, "not_installed": not_installed}))
            return

        typer.echo("Nemotron-Persona Datasets")
        typer.echo("-" * 25)

        if installed:
            typer.echo(f"Usable locales in PersonSamplerParams: {', '.join(installed)}")
        else:
            typer.echo("No persona datasets installed.")

        if not_installed:
            typer.echo(f"Not installed: {', '.join(not_installed)}")
            typer.echo("The user can run `data-designer download personas --locale <LOCALE>` to install.")
