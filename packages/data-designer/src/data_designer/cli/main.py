# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
import sys
from typing import TextIO

import typer

from data_designer.cli.agent_command_defs import AGENT_COMMANDS
from data_designer.cli.lazy_group import create_lazy_typer_group
from data_designer.cli.runtime import ensure_cli_default_model_settings
from data_designer.config.utils.constants import DATA_DESIGNER_PACKAGE_NAME

_CMD = "data_designer.cli.commands"


def should_show_update_notice(stream: TextIO | None = None) -> bool:
    stream = sys.stdout if stream is None else stream
    return stream.isatty()


def _version_callback(value: bool) -> None:
    if not value:
        return
    try:
        installed_version = importlib.metadata.version(DATA_DESIGNER_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        typer.echo(f"Unable to resolve installed {DATA_DESIGNER_PACKAGE_NAME} package version.", err=True)
        raise typer.Exit(1) from None

    typer.echo(installed_version)
    if not should_show_update_notice():
        raise typer.Exit()

    try:
        # The update CTA is opportunistic; version output should stay usable if lookup fails.
        from data_designer.cli.ui import print_update_notice
        from data_designer.cli.version_notice import get_update_notice

        notice = get_update_notice(installed_version)
        if notice is not None:
            print_update_notice(notice.latest_version, notice.upgrade_command)
    except (ImportError, OSError, RuntimeError, ValueError):
        pass
    raise typer.Exit()


def _is_version_request(args: list[str]) -> bool:
    return "--version" in args


# Initialize Typer app with custom configuration
app = typer.Typer(
    name="data-designer",
    help="Data Designer CLI - Configure model providers and models for synthetic data generation",
    cls=create_lazy_typer_group(
        {
            "preview": {
                "module": f"{_CMD}.preview",
                "attr": "preview_command",
                "help": "Generate a preview dataset for fast iteration",
                "rich_help_panel": "Generation",
            },
            "create": {
                "module": f"{_CMD}.create",
                "attr": "create_command",
                "help": "Create a full dataset and save results to disk",
                "rich_help_panel": "Generation",
            },
            "validate": {
                "module": f"{_CMD}.validate",
                "attr": "validate_command",
                "help": "Validate a Data Designer configuration",
                "rich_help_panel": "Generation",
            },
        }
    ),
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)


# Create config subcommand group
config_app = typer.Typer(
    name="config",
    help="Manage configuration files",
    cls=create_lazy_typer_group(
        {
            "providers": {
                "module": f"{_CMD}.providers",
                "attr": "providers_command",
                "help": "Configure model providers interactively",
            },
            "models": {
                "module": f"{_CMD}.models",
                "attr": "models_command",
                "help": "Configure models interactively",
            },
            "mcp": {
                "module": f"{_CMD}.mcp",
                "attr": "mcp_command",
                "help": "Configure MCP providers interactively",
            },
            "tools": {
                "module": f"{_CMD}.tools",
                "attr": "tools_command",
                "help": "Configure tool configs interactively",
            },
            "list": {
                "module": f"{_CMD}.list",
                "attr": "list_command",
                "help": "List current configurations",
            },
            "reset": {
                "module": f"{_CMD}.reset",
                "attr": "reset_command",
                "help": "Reset configuration files",
            },
        }
    ),
    no_args_is_help=True,
)

# Create download command group
download_app = typer.Typer(
    name="download",
    help="Download assets for Data Designer",
    cls=create_lazy_typer_group(
        {
            "personas": {
                "module": f"{_CMD}.download",
                "attr": "personas_command",
                "help": "Download Nemotron-Persona datasets",
            },
        }
    ),
    no_args_is_help=True,
)

# Create plugin command group
plugin_app = typer.Typer(
    name="plugin",
    help="Discover, install, and uninstall Data Designer plugin packages from catalogs",
    cls=create_lazy_typer_group(
        {
            "list": {
                "module": f"{_CMD}.plugin",
                "attr": "list_command",
                "help": "List plugin packages from a catalog",
            },
            "search": {
                "module": f"{_CMD}.plugin",
                "attr": "search_command",
                "help": "Search plugin packages from a catalog",
            },
            "info": {
                "module": f"{_CMD}.plugin",
                "attr": "info_command",
                "help": "Show plugin package metadata and install plan",
            },
            "install": {
                "module": f"{_CMD}.plugin",
                "attr": "install_command",
                "help": "Install a plugin package and verify declared runtime entry points",
            },
            "uninstall": {
                "module": f"{_CMD}.plugin",
                "attr": "uninstall_command",
                "help": "Uninstall a plugin package and verify declared runtime entry points are removed",
            },
            "installed": {
                "module": f"{_CMD}.plugin",
                "attr": "installed_command",
                "help": "List installed runtime plugins and their packages",
            },
        }
    ),
    no_args_is_help=True,
)


@plugin_app.callback()
def plugin_callback(
    catalog: str | None = typer.Option(
        None,
        "--catalog",
        help="Plugin catalog alias to use for commands that read package metadata.",
    ),
) -> None:
    _ = catalog


plugin_catalog_app = typer.Typer(
    name="catalog",
    help="Manage plugin catalog aliases",
    cls=create_lazy_typer_group(
        {
            "list": {
                "module": f"{_CMD}.plugin",
                "attr": "catalog_list_command",
                "help": "List configured plugin catalogs",
            },
            "add": {
                "module": f"{_CMD}.plugin",
                "attr": "catalog_add_command",
                "help": "Add a plugin catalog alias",
            },
            "remove": {
                "module": f"{_CMD}.plugin",
                "attr": "catalog_remove_command",
                "help": "Remove a plugin catalog alias",
            },
        }
    ),
    no_args_is_help=True,
)

_AGENT_CMD = f"{_CMD}.agent"


def _build_agent_lazy_group(prefix: str) -> dict[str, dict[str, str]]:
    return {
        cmd.name.removeprefix(f"{prefix}."): {"module": _AGENT_CMD, "attr": cmd.attr, "help": cmd.help}
        for cmd in AGENT_COMMANDS
        if (prefix == "" and "." not in cmd.name) or cmd.name.startswith(f"{prefix}.")
    }


agent_app = typer.Typer(
    name="agent",
    help="Agent-only interface for dynamic Data Designer introspection",
    cls=create_lazy_typer_group(_build_agent_lazy_group("")),
    no_args_is_help=True,
)

agent_state_app = typer.Typer(
    name="state",
    help="Return current local state relevant to agents",
    cls=create_lazy_typer_group(_build_agent_lazy_group("state")),
    no_args_is_help=True,
)

agent_app.add_typer(agent_state_app, name="state")
plugin_app.add_typer(plugin_catalog_app, name="catalog")

# Add setup command groups
app.add_typer(config_app, name="config", rich_help_panel="Setup")
app.add_typer(download_app, name="download", rich_help_panel="Setup")
app.add_typer(plugin_app, name="plugin", rich_help_panel="Setup")
app.add_typer(agent_app, name="agent", rich_help_panel="Agent")


@app.callback()
def app_callback(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed data-designer package version and exit.",
    ),
) -> None:
    _ = version


def main() -> None:
    """Main entry point for the CLI."""
    if not _is_version_request(sys.argv[1:]):
        ensure_cli_default_model_settings()
    app()


if __name__ == "__main__":
    main()
