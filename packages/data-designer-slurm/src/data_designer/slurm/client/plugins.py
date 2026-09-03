# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib.metadata import entry_points

from packaging.utils import canonicalize_name

from data_designer.plugins import Plugin
from data_designer.slurm.client.errors import ClientWorkerError
from data_designer.slurm.client.records import ClientErrorCode, ClientPluginEntryPoint
from data_designer.slurm.contracts import InstalledDistribution


def discover_plugins(
    installed_distributions: tuple[InstalledDistribution, ...],
) -> tuple[ClientPluginEntryPoint, ...]:
    """Load and validate every installed Data Designer plugin entry point."""
    installed = {distribution.name: distribution.version for distribution in installed_distributions}
    discovered: list[ClientPluginEntryPoint] = []
    plugin_names: set[str] = set()
    try:
        for entry_point in entry_points(group="data_designer.plugins"):
            if entry_point.dist is None:
                raise ValueError("plugin entry point has no owning distribution")
            distribution = canonicalize_name(entry_point.dist.name)
            version = entry_point.dist.version
            if installed.get(distribution) != version:
                raise ValueError("plugin entry point is outside the verified environment")
            plugin = entry_point.load()
            if not isinstance(plugin, Plugin):
                raise TypeError("plugin entry point did not load a Plugin")
            plugin.config_cls
            plugin.impl_cls
            if plugin.name in plugin_names:
                raise ValueError("plugin names must be unique")
            plugin_names.add(plugin.name)
            discovered.append(
                ClientPluginEntryPoint(
                    entry_point=entry_point.name,
                    value=entry_point.value,
                    distribution=distribution,
                    distribution_version=version,
                    plugin_name=plugin.name,
                    plugin_type=plugin.plugin_type.value,
                )
            )
    except ClientWorkerError:
        raise
    except Exception as error:
        raise ClientWorkerError(ClientErrorCode.PLUGIN_LOAD_FAILED, "Data Designer plugin validation failed") from error
    return tuple(sorted(discovered, key=lambda plugin: (plugin.distribution, plugin.entry_point)))
