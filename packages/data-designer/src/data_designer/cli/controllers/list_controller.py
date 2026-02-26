# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import typer

from data_designer.cli.repositories.model_repository import ModelRepository
from data_designer.cli.repositories.persona_repository import PersonaRepository
from data_designer.cli.repositories.provider_repository import ProviderRepository
from data_designer.cli.services.download_service import DownloadService
from data_designer.cli.services.introspection.discovery import (
    discover_column_configs,
    discover_processor_configs,
    discover_sampler_types,
    discover_validator_types,
)
from data_designer.config.default_model_settings import get_providers_with_missing_api_keys

_IMPORT_HINT = "# import data_designer.config as dd"


class ListController:
    """Controller for listing valid configuration values."""

    def __init__(self, config_dir: Path) -> None:
        self._config_dir = config_dir
        self._model_repository = ModelRepository(config_dir)
        self._provider_repository = ProviderRepository(config_dir)
        self._persona_repository = PersonaRepository()
        self._download_service = DownloadService(config_dir, self._persona_repository)

    def list_model_aliases(self) -> None:
        """List configured model aliases.

        Only shows aliases whose backing provider has a valid API key.
        """
        provider_registry = self._provider_repository.load()

        if not provider_registry or not provider_registry.providers:
            typer.echo("No model providers configured. Run `data-designer config models` to configure your models.")
            return

        missing_key_providers = get_providers_with_missing_api_keys(provider_registry.providers)
        valid_provider_names = {p.name for p in provider_registry.providers} - {p.name for p in missing_key_providers}

        if not valid_provider_names:
            typer.echo(
                "No model providers are configured with valid API keys. "
                "Run `data-designer config models` to configure your models."
            )
            return

        default_provider = provider_registry.default or provider_registry.providers[0].name

        model_registry = self._model_repository.load()
        configs = model_registry.model_configs if model_registry else []

        if not configs:
            typer.echo("No model aliases configured.")
            typer.echo("Run `data-designer config models` to add models.")
            return

        filtered = [mc for mc in configs if (mc.provider or default_provider) in valid_provider_names]

        if not filtered:
            typer.echo(
                "All configured model aliases use providers without valid API keys. "
                "Run `data-designer config models` to configure your models."
            )
            return

        c1, c2, c3 = "model_alias", "model", "provider"
        w1 = max(len(c1), max(len(mc.alias) for mc in filtered))
        w2 = max(len(c2), max(len(mc.model) for mc in filtered))
        w3 = max(len(c3), max(len(mc.provider or "default") for mc in filtered))
        typer.echo(f"{c1:<{w1}}  {c2:<{w2}}  {c3}")
        typer.echo(f"{'-' * w1}  {'-' * w2}  {'-' * w3}")
        for mc in filtered:
            typer.echo(f"{mc.alias:<{w1}}  {mc.model:<{w2}}  {mc.provider or 'default'}")

        if len(filtered) < len(configs):
            typer.echo(f"\n({len(configs) - len(filtered)} model alias(es) hidden — providers missing API keys)")

    def list_persona_datasets(self) -> None:
        """List persona datasets available for PersonSamplerParams."""
        managed_locales = self._persona_repository.list_all()
        if not managed_locales:
            typer.echo("No persona datasets found.")
            return

        entries: list[dict[str, str | bool]] = []
        for locale in managed_locales:
            installed = self._download_service.is_locale_downloaded(locale.code)
            entries.append({"locale": locale.code, "installed": installed})

        typer.echo(_IMPORT_HINT)
        typer.echo("")
        col1 = "locale"
        col2 = "status"
        max_width = max(len(col1), max(len(str(entry["locale"])) for entry in entries))
        typer.echo(f"{col1:<{max_width}}  {col2}")
        typer.echo(f"{'-' * max_width}  {'-' * len('not installed')}")
        for entry in entries:
            status = "installed" if entry["installed"] else "not installed"
            typer.echo(f"{str(entry['locale']):<{max_width}}  {status}")
        typer.echo("")
        typer.echo("Use the PersonSamplerParams locale parameter to select a dataset.")
        typer.echo("Run `data-designer download personas --locale <locale>` to install a dataset.")

    def _print_type_table(
        self,
        items: dict[str, type],
        col1: str,
        col2: str,
        inspect_command: str,
    ) -> None:
        """Print a two-column table of discovered types with an inspect tip."""
        if not items:
            typer.echo("No items found.")
            return

        sorted_types = sorted(items.keys())
        max_width = max(len(col1), max(len(t) for t in sorted_types))

        typer.echo(_IMPORT_HINT)
        typer.echo("")
        typer.echo(f"{col1:<{max_width}}  {col2}")
        typer.echo(f"{'-' * max_width}  {'-' * max(len(items[t].__name__) for t in sorted_types)}")
        for t in sorted_types:
            typer.echo(f"{t:<{max_width}}  {items[t].__name__}")
        typer.echo("")
        typer.echo(f"Run `data-designer inspect {inspect_command}` to see that type's full schema.")

    def list_column_types(self) -> None:
        """List available column configuration types."""
        self._print_type_table(discover_column_configs(), "column_type", "config_class", "column <column_type>")

    def list_sampler_types(self) -> None:
        """List available sampler types."""
        self._print_type_table(discover_sampler_types(), "sampler_type", "params_class", "sampler <sampler_type>")

    def list_validator_types(self) -> None:
        """List available validator types."""
        self._print_type_table(
            discover_validator_types(), "validator_type", "params_class", "validator <validator_type>"
        )

    def list_processor_types(self) -> None:
        """List available processor types."""
        self._print_type_table(
            discover_processor_configs(), "processor_type", "config_class", "processor <processor_type>"
        )
