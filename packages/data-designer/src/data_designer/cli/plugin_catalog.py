# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from data_designer.plugins.plugin import PluginType

DEFAULT_PLUGIN_TAP_ALIAS = "nvidia"
DEFAULT_PLUGIN_TAP_URL = "https://raw.githubusercontent.com/NVIDIA-NeMo/DataDesignerPlugins/main/catalog/plugins.json"
PLUGIN_TAPS_FILE_NAME = "plugin_taps.yaml"
PLUGIN_TAP_CACHE_DIR_NAME = "plugin-tap-cache"
PLUGIN_TAP_DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60
MAX_PLUGIN_CATALOG_SIZE_BYTES = 1 * 1024 * 1024
SUPPORTED_PLUGIN_CATALOG_SCHEMA_VERSIONS = {1, 2}
PLUGIN_TAP_ALIAS_PATTERN = r"^[A-Za-z0-9_.-]+$"


class PluginCatalogError(ValueError):
    """Raised when a plugin catalog cannot be loaded or validated."""


class PluginCompatibilityTarget(BaseModel):
    """Version requirement for one environment target."""

    model_config = ConfigDict(extra="allow")

    specifier: str | None = None
    marker: str | None = None


class PluginCompatibility(BaseModel):
    """Compatibility requirements declared by a catalog entry."""

    model_config = ConfigDict(extra="allow")

    python: PluginCompatibilityTarget | None = None
    data_designer: PluginCompatibilityTarget | None = None


class PluginPackageInfo(BaseModel):
    """Python package metadata for a catalog entry."""

    model_config = ConfigDict(extra="allow")

    name: str
    version: str | None = None
    path: str | None = None


class PluginEntryPointInfo(BaseModel):
    """Runtime entry point exposed by an installable plugin package."""

    model_config = ConfigDict(extra="allow")

    group: str = "data_designer.plugins"
    name: str
    value: str


class PluginSourceInfo(BaseModel):
    """Install source metadata for a catalog entry."""

    model_config = ConfigDict(extra="allow")

    type: str
    package: str | None = None
    url: str | None = None
    ref: str | None = None
    path: str | None = None
    subdirectory: str | None = None
    editable: bool = False


class PluginDocsInfo(BaseModel):
    """Documentation metadata for a catalog entry."""

    model_config = ConfigDict(extra="allow")

    url: str | None = None


class PluginCatalogEntry(BaseModel):
    """One discoverable Data Designer plugin entry from a tap catalog."""

    model_config = ConfigDict(extra="allow")

    name: str
    plugin_type: PluginType
    description: str = ""
    package: PluginPackageInfo
    entry_point: PluginEntryPointInfo
    compatibility: PluginCompatibility | None = None
    source: PluginSourceInfo | None = None
    docs: PluginDocsInfo | None = None
    tags: list[str] = Field(default_factory=list)
    maintainers: list[str] = Field(default_factory=list)
    release_notes_url: str | None = None


class PluginCatalog(BaseModel):
    """Versioned plugin tap catalog."""

    model_config = ConfigDict(extra="allow")

    schema_version: int
    plugins: list[PluginCatalogEntry] = Field(default_factory=list)


class PluginTapConfig(BaseModel):
    """Persisted tap configuration."""

    alias: str = Field(pattern=PLUGIN_TAP_ALIAS_PATTERN)
    url: str
    trusted: bool = False
    cache_ttl_seconds: int = Field(default=PLUGIN_TAP_DEFAULT_CACHE_TTL_SECONDS, ge=0)


class PluginTapRegistry(BaseModel):
    """Persisted collection of user-configured plugin taps."""

    taps: list[PluginTapConfig] = Field(default_factory=list)


@dataclass(frozen=True)
class CompatibilityResult:
    """Compatibility result for one catalog entry in the local environment."""

    is_compatible: bool
    reasons: list[str]


@dataclass(frozen=True)
class InstallPlan:
    """Resolved package-manager command for installing one plugin entry."""

    plugin_name: str
    package_name: str
    source_description: str
    command: list[str]
    manager: str
    tap_alias: str
    trusted_tap: bool


@dataclass(frozen=True)
class InstalledPluginInfo:
    """Runtime plugin discovered from installed entry points."""

    name: str
    plugin_type: PluginType
    config_qualified_name: str
    impl_qualified_name: str
