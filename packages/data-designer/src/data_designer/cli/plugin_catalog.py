# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from packaging.markers import InvalidMarker, Marker
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import InvalidName, canonicalize_name
from pydantic import BaseModel, ConfigDict, Field

from data_designer.plugins.plugin import PluginType

DEFAULT_PLUGIN_CATALOG_ALIAS = "nvidia"
DEFAULT_PLUGIN_CATALOG_URL = "https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json"
DEFAULT_PLUGIN_CATALOG_URL_ENV_VAR = "DATA_DESIGNER_DEFAULT_PLUGIN_CATALOG_URL"
PLUGIN_CATALOGS_FILE_NAME = "plugin_catalogs.yaml"
PLUGIN_CATALOG_CACHE_DIR_NAME = "plugin-catalog-cache"
PLUGIN_CATALOG_DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60
MAX_PLUGIN_CATALOG_SIZE_BYTES = 1 * 1024 * 1024
PLUGIN_CATALOG_SCHEMA_VERSION = 2
PLUGIN_CATALOG_ALIAS_PATTERN = r"^[A-Za-z0-9_.-]+$"
DATA_DESIGNER_DISTRIBUTION_NAME = "data-designer"
DATA_DESIGNER_PLUGIN_PACKAGE_PREFIX = "data-designer-"
PLUGIN_ENTRY_POINT_GROUP = "data_designer.plugins"
PYPI_SIMPLE_INDEX_URL = "https://pypi.org/simple/"
CATALOG_DOCUMENT_KEYS = {"packages", "schema_version"}
CATALOG_PACKAGE_KEYS = {
    "compatibility",
    "description",
    "docs",
    "install",
    "name",
    "plugins",
}
CATALOG_PLUGIN_KEYS = {"entry_point", "name", "plugin_type"}
CATALOG_ENTRY_POINT_KEYS = {"group", "name", "value"}
CATALOG_COMPATIBILITY_KEYS = {"data_designer", "python"}
CATALOG_PYTHON_COMPATIBILITY_KEYS = {"specifier"}
CATALOG_DATA_DESIGNER_COMPATIBILITY_KEYS = {"marker", "requirement", "specifier"}
CATALOG_DOCS_KEYS = {"url"}
CATALOG_INSTALL_REQUIRED_KEYS = {"requirement"}
CATALOG_INSTALL_OPTIONAL_KEYS = {"index_url"}
SUPPORTED_PLUGIN_TYPE_VALUES = {plugin_type.value for plugin_type in PluginType}


class PluginCatalogError(ValueError):
    """Raised when a plugin catalog cannot be loaded or validated."""


class PluginCompatibilityTarget(BaseModel):
    """Version requirement for one environment target."""

    model_config = ConfigDict(extra="forbid")

    requirement: str | None = None
    specifier: str = Field(min_length=1)
    marker: str | None = None


class PluginCompatibility(BaseModel):
    """Compatibility requirements declared by a catalog package."""

    model_config = ConfigDict(extra="forbid")

    python: PluginCompatibilityTarget
    data_designer: PluginCompatibilityTarget


class PluginPackageInfo(BaseModel):
    """Python distribution metadata for a catalog entry."""

    model_config = ConfigDict(extra="forbid")

    name: str


class PluginEntryPointInfo(BaseModel):
    """Runtime entry point exposed by an installable plugin package."""

    model_config = ConfigDict(extra="forbid")

    group: str = PLUGIN_ENTRY_POINT_GROUP
    name: str
    value: str


class PluginInstallInfo(BaseModel):
    """Resolver-native install metadata for a catalog package."""

    model_config = ConfigDict(extra="forbid")

    requirement: str
    index_url: str | None = None


class PluginDocsInfo(BaseModel):
    """Documentation metadata for a catalog package."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1)


class PluginCatalogEntry(BaseModel):
    """One discoverable runtime plugin entry from a catalog package."""

    model_config = ConfigDict(extra="forbid")

    name: str
    plugin_type: PluginType
    description: str = Field(min_length=1)
    package: PluginPackageInfo
    install: PluginInstallInfo
    entry_point: PluginEntryPointInfo
    compatibility: PluginCompatibility
    docs: PluginDocsInfo


class PluginCatalogRuntimePlugin(BaseModel):
    """Runtime plugin metadata nested under one catalog package."""

    model_config = ConfigDict(extra="forbid")

    name: str
    plugin_type: PluginType
    entry_point: PluginEntryPointInfo


class PluginCatalogPackage(BaseModel):
    """One installable package from a package-first plugin catalog."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = Field(min_length=1)
    install: PluginInstallInfo
    compatibility: PluginCompatibility
    docs: PluginDocsInfo
    plugins: list[PluginCatalogRuntimePlugin] = Field(min_length=1)

    def entries(self) -> list[PluginCatalogEntry]:
        """Flatten nested runtime plugins while preserving package-level metadata."""
        package = PluginPackageInfo(name=self.name)
        return [
            PluginCatalogEntry(
                name=plugin.name,
                plugin_type=plugin.plugin_type,
                description=self.description,
                package=package,
                install=self.install,
                entry_point=plugin.entry_point,
                compatibility=self.compatibility,
                docs=self.docs,
            )
            for plugin in self.plugins
        ]


class PluginCatalog(BaseModel):
    """Versioned plugin catalog."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    packages: list[PluginCatalogPackage] = Field(default_factory=list)

    @property
    def entries(self) -> list[PluginCatalogEntry]:
        """Return the runtime plugin entries described by every package."""
        return [entry for package in self.packages for entry in package.entries()]

    @property
    def plugins(self) -> list[PluginCatalogEntry]:
        """Convenience alias for flattened runtime plugin entries."""
        return self.entries


class PluginCatalogConfig(BaseModel):
    """Persisted catalog configuration."""

    alias: str = Field(pattern=PLUGIN_CATALOG_ALIAS_PATTERN)
    url: str
    cache_ttl_seconds: int = Field(default=PLUGIN_CATALOG_DEFAULT_CACHE_TTL_SECONDS, ge=0)


class PluginCatalogRegistry(BaseModel):
    """Persisted collection of user-configured plugin catalogs."""

    catalogs: list[PluginCatalogConfig] = Field(default_factory=list)


@dataclass(frozen=True)
class CompatibilityResult:
    """Compatibility result for one catalog entry in the local environment."""

    is_compatible: bool
    reasons: list[str]


@dataclass(frozen=True)
class InstallCommandTemporaryFile:
    """Temporary file needed only while executing one install command."""

    placeholder: str
    filename: str
    content: str


@dataclass(frozen=True)
class InstallPlan:
    """Resolved package-manager command for installing one plugin package."""

    package_name: str
    source_description: str
    command: list[str]
    manager: str
    catalog_alias: str
    source_warning: str | None = None
    data_designer_protection: str | None = None
    command_stdin: str | None = None
    temporary_file: InstallCommandTemporaryFile | None = None
    install_mode: str = "environment"
    project_root: str | None = None


@dataclass(frozen=True)
class UninstallPlan:
    """Resolved package-manager command for uninstalling one plugin package."""

    package_name: str
    command: list[str]
    manager: str
    catalog_alias: str
    commands: list[list[str]] | None = None
    uninstall_mode: str = "environment"
    project_root: str | None = None


@dataclass(frozen=True)
class InstalledPluginInfo:
    """Installed runtime plugin entry point discovered without importing plugin code."""

    name: str
    entry_point_value: str


def get_default_plugin_catalog_url() -> str:
    """Return the built-in plugin catalog URL, honoring a local override for QA/staging."""
    return os.getenv(DEFAULT_PLUGIN_CATALOG_URL_ENV_VAR, DEFAULT_PLUGIN_CATALOG_URL)


def validate_plugin_catalog_payload(payload: object, *, source: str) -> None:
    """Validate a decoded plugin catalog against the schema v2 contract."""
    try:
        _validate_plugin_catalog_payload(payload)
    except PluginCatalogError as e:
        raise PluginCatalogError(f"Invalid plugin catalog at {source!r}: {e}") from e


def _validate_plugin_catalog_payload(payload: object) -> None:
    catalog = _required_catalog_object("catalog document", payload, CATALOG_DOCUMENT_KEYS)
    schema_version = catalog["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != PLUGIN_CATALOG_SCHEMA_VERSION
    ):
        raise PluginCatalogError(
            f"unsupported catalog schema_version {schema_version!r}; expected {PLUGIN_CATALOG_SCHEMA_VERSION}"
        )

    packages = catalog["packages"]
    if not isinstance(packages, list):
        raise PluginCatalogError("catalog document has invalid packages; expected a list")

    package_names: dict[str, str] = {}
    runtime_names: dict[str, tuple[str, str]] = {}
    for index, raw_package in enumerate(packages):
        validated_plugins = _validate_catalog_package(raw_package, index)
        package_name = validated_plugins[0][0]
        canonical_package_name = canonicalize_name(package_name)
        previous_package_name = package_names.get(canonical_package_name)
        if previous_package_name is not None:
            raise PluginCatalogError(
                f"duplicate package name {package_name!r}; canonical name {canonical_package_name!r} "
                f"already used by {previous_package_name!r}"
            )
        package_names[canonical_package_name] = package_name

        for package_name, plugin_name, entry_point_name in validated_plugins:
            previous = runtime_names.get(plugin_name)
            if previous is not None:
                previous_package, previous_entry_point_name = previous
                raise PluginCatalogError(
                    f"duplicate runtime plugin name {plugin_name!r} from "
                    f"{previous_package!r} entry point {previous_entry_point_name!r} and "
                    f"{package_name!r} entry point {entry_point_name!r}"
                )
            runtime_names[plugin_name] = (package_name, entry_point_name)


def _validate_catalog_package(raw_package: object, index: int) -> list[tuple[str, str, str]]:
    context = f"catalog packages[{index}]"
    package = _required_catalog_object(context, raw_package, CATALOG_PACKAGE_KEYS)
    compatibility = _required_catalog_object(
        f"{context}.compatibility",
        package["compatibility"],
        CATALOG_COMPATIBILITY_KEYS,
    )
    python_compatibility = _required_catalog_object(
        f"{context}.compatibility.python",
        compatibility["python"],
        CATALOG_PYTHON_COMPATIBILITY_KEYS,
    )
    data_designer_compatibility = _required_catalog_object(
        f"{context}.compatibility.data_designer",
        compatibility["data_designer"],
        CATALOG_DATA_DESIGNER_COMPATIBILITY_KEYS,
    )
    install = _required_catalog_object(f"{context}.install", package["install"])
    docs = _required_catalog_object(f"{context}.docs", package["docs"], CATALOG_DOCS_KEYS)

    package_name = _catalog_package_name(f"{context}.name", package["name"])
    _required_catalog_string(f"{context}.description", package["description"])
    _catalog_version_specifier(
        package_name,
        f"{context}.compatibility.python.specifier",
        python_compatibility["specifier"],
    )
    _catalog_data_designer_compatibility(
        package_name,
        f"{context}.compatibility.data_designer",
        data_designer_compatibility,
    )
    _validate_install_metadata(package_name, f"{context}.install", install)
    _catalog_http_url(f"{context}.docs.url", docs["url"])

    plugins = package["plugins"]
    if not isinstance(plugins, list) or not plugins:
        raise PluginCatalogError(f"{context}.plugins is invalid; expected a non-empty list")

    return [
        _validate_catalog_plugin(
            raw_plugin,
            package_name=package_name,
            context=f"{context}.plugins[{plugin_index}]",
        )
        for plugin_index, raw_plugin in enumerate(plugins)
    ]


def _validate_catalog_plugin(raw_plugin: object, *, package_name: str, context: str) -> tuple[str, str, str]:
    plugin = _required_catalog_object(context, raw_plugin, CATALOG_PLUGIN_KEYS)
    entry_point = _required_catalog_object(
        f"{context}.entry_point",
        plugin["entry_point"],
        CATALOG_ENTRY_POINT_KEYS,
    )

    plugin_type = _required_catalog_string(f"{context}.plugin_type", plugin["plugin_type"])
    if plugin_type not in SUPPORTED_PLUGIN_TYPE_VALUES:
        raise PluginCatalogError(
            f"{context}.plugin_type {plugin_type!r} is invalid; expected one of "
            f"{_format_catalog_choices(SUPPORTED_PLUGIN_TYPE_VALUES)}"
        )

    plugin_name = _required_catalog_string(f"{context}.name", plugin["name"])
    entry_point_group = _required_catalog_string(f"{context}.entry_point.group", entry_point["group"])
    if entry_point_group != PLUGIN_ENTRY_POINT_GROUP:
        raise PluginCatalogError(
            f"{context}.entry_point.group {entry_point_group!r} is invalid; expected {PLUGIN_ENTRY_POINT_GROUP!r}"
        )
    entry_point_name = _required_catalog_string(f"{context}.entry_point.name", entry_point["name"])
    _required_catalog_string(f"{context}.entry_point.value", entry_point["value"])
    return package_name, plugin_name, entry_point_name


def _validate_install_metadata(package_name: str, context: str, install: dict[str, object]) -> None:
    keys = set(install)
    missing_keys = CATALOG_INSTALL_REQUIRED_KEYS - keys
    extra_keys = keys - CATALOG_INSTALL_REQUIRED_KEYS - CATALOG_INSTALL_OPTIONAL_KEYS
    if missing_keys or extra_keys:
        expected_required = _format_catalog_keys(CATALOG_INSTALL_REQUIRED_KEYS)
        expected_optional = _format_catalog_keys(CATALOG_INSTALL_OPTIONAL_KEYS)
        raise PluginCatalogError(
            f"package {package_name!r} has invalid install fields; "
            f"expected {{{expected_required}; optional {{{expected_optional}}}}}, "
            f"got {{{_format_catalog_keys(keys)}}}"
        )

    requirement_text = _required_catalog_string(f"{context}.requirement", install["requirement"])
    try:
        requirement = Requirement(requirement_text)
    except InvalidRequirement as e:
        raise PluginCatalogError(
            f"package {package_name!r} has invalid {context}.requirement {requirement_text!r}: {e}"
        ) from e
    if canonicalize_name(requirement.name) != canonicalize_name(package_name):
        raise PluginCatalogError(
            f"package {package_name!r} has invalid {context}.requirement {requirement_text!r}; "
            f"expected a requirement for {package_name!r}"
        )

    if "index_url" in install:
        _catalog_http_url(f"package {package_name!r} install.index_url", install["index_url"])


def _required_catalog_object(
    context: str,
    value: object,
    expected_keys: set[str] | None = None,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PluginCatalogError(f"{context} is invalid; expected an object")
    if expected_keys is not None:
        _validate_catalog_object_keys(context, value, expected_keys)
    return value


def _validate_catalog_object_keys(context: str, value: dict[str, object], expected_keys: set[str]) -> None:
    keys = set(value)
    if keys != expected_keys:
        # Catalog v2 is strict by design: additive wire-schema changes should bump
        # schema_version so older CLIs do not silently ignore new fields.
        raise PluginCatalogError(
            f"{context} has invalid fields; expected {{{_format_catalog_keys(expected_keys)}}}, "
            f"got {{{_format_catalog_keys(keys)}}}"
        )


def _required_catalog_string(context: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise PluginCatalogError(f"{context} is invalid; expected a non-empty string")
    return value


def _required_catalog_nullable_string(context: str, value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise PluginCatalogError(f"{context} is invalid; expected a string or null")


def _catalog_package_name(context: str, value: object) -> str:
    package_name = _required_catalog_string(context, value)
    try:
        canonicalize_name(package_name, validate=True)
    except InvalidName as e:
        raise PluginCatalogError(f"{context} {package_name!r} is invalid; expected a valid package name") from e
    return package_name


def _catalog_version_specifier(package_name: str, context: str, value: object) -> str:
    raw_specifier = _required_catalog_string(context, value)
    try:
        specifier = SpecifierSet(raw_specifier)
    except InvalidSpecifier as e:
        raise PluginCatalogError(f"package {package_name!r} has invalid {context} {raw_specifier!r}: {e}") from e
    if not str(specifier):
        raise PluginCatalogError(f"package {package_name!r} has invalid {context}; expected at least one specifier")
    return str(specifier)


def _catalog_data_designer_compatibility(
    package_name: str,
    context: str,
    compatibility: dict[str, object],
) -> None:
    requirement_text = _required_catalog_string(f"{context}.requirement", compatibility["requirement"])
    try:
        requirement = Requirement(requirement_text)
    except InvalidRequirement as e:
        raise PluginCatalogError(
            f"package {package_name!r} has invalid {context}.requirement {requirement_text!r}: {e}"
        ) from e
    if canonicalize_name(requirement.name) != DATA_DESIGNER_DISTRIBUTION_NAME:
        raise PluginCatalogError(
            f"package {package_name!r} has invalid {context}.requirement {requirement_text!r}; "
            f"expected a {DATA_DESIGNER_DISTRIBUTION_NAME!r} requirement"
        )
    if not requirement.specifier:
        raise PluginCatalogError(f"package {package_name!r} has invalid {context}.requirement; expected a specifier")

    specifier = _catalog_version_specifier(package_name, f"{context}.specifier", compatibility["specifier"])
    if specifier != str(requirement.specifier):
        raise PluginCatalogError(
            f"package {package_name!r} has invalid {context}.specifier {specifier!r}; "
            f"expected {str(requirement.specifier)!r} from requirement"
        )

    marker = _catalog_marker(package_name, f"{context}.marker", compatibility["marker"])
    expected_marker = str(requirement.marker) if requirement.marker is not None else None
    if marker != expected_marker:
        raise PluginCatalogError(
            f"package {package_name!r} has invalid {context}.marker {marker!r}; expected {expected_marker!r}"
        )


def _catalog_marker(package_name: str, context: str, value: object) -> str | None:
    raw_marker = _required_catalog_nullable_string(context, value)
    if raw_marker is None:
        return None
    try:
        return str(Marker(raw_marker))
    except InvalidMarker as e:
        raise PluginCatalogError(f"package {package_name!r} has invalid {context} {raw_marker!r}: {e}") from e


def _catalog_http_url(context: str, value: object) -> str:
    url = _required_catalog_string(context, value)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PluginCatalogError(f"{context} {url!r} is invalid; expected an absolute HTTP(S) URL")
    return url


def _format_catalog_keys(keys: set[str]) -> str:
    return ", ".join(sorted(keys))


def _format_catalog_choices(choices: set[str]) -> str:
    return ", ".join(repr(choice) for choice in sorted(choices))
