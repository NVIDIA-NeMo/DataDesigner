# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from data_designer.config import DataDesignerConfigBuilder, ModelConfig
from data_designer.slurm.config import (
    DEFAULT_PROFILE_FILE_NAME,
    PROFILE_FILE_ENVIRONMENT,
    ConfigBuilderError,
    ConfigLoadError,
    DataDesignerSlurmConfig,
    DataDesignerSlurmConfigBuilder,
    ProfileSelectionSource,
    SlurmProfileCatalog,
    load_profile_catalog,
    load_run_config,
    resolve_profile,
)


def _config_builder() -> DataDesignerSlurmConfigBuilder:
    data_designer = DataDesignerConfigBuilder(
        model_configs=[ModelConfig(alias="generator", model="example/generator", provider="openai")]
    )
    return (
        DataDesignerSlurmConfigBuilder.from_config_builder(data_designer, name="generated-run")
        .with_invocation(
            num_records=8,
            dataset_name="generated",
            model_concurrency={"generator": 8},
        )
        .with_client(image={"name": "dd-client"})
        .with_deployment(
            {
                "model_alias": "generator",
                "model": "example/generator",
                "server": {"type": "vllm", "image": {"name": "vllm"}},
                "topology": {"tensor_parallel": 8},
            }
        )
    )


def test_builder_builds_without_file_discovery_or_serialization(tmp_path: Path) -> None:
    builder = _config_builder()

    config = builder.build()

    assert config.name == "generated-run"
    assert config.builder.inline is not None
    assert config.deployments[0].model_alias == "generator"
    assert not tuple(tmp_path.iterdir())


def test_builder_requires_complete_authored_intent() -> None:
    builder = DataDesignerSlurmConfigBuilder.from_builder_source("builder.json")

    with pytest.raises(ConfigBuilderError, match="invocation, client, deployment"):
        builder.build()


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_builder_write_config_round_trips_supported_formats(tmp_path: Path, suffix: str) -> None:
    builder = _config_builder()
    path = tmp_path / f"run{suffix}"

    builder.write_config(path)

    assert load_run_config(path) == builder.build()


def test_builder_rejects_unsupported_output_format(tmp_path: Path) -> None:
    with pytest.raises(ConfigBuilderError, match="must end"):
        _config_builder().write_config(tmp_path / "run.txt")


@pytest.mark.parametrize(
    ("suffix", "contents", "message"),
    [
        (".json", '{"schema_version": 1, "schema_version": 1}', "duplicate"),
        (".yaml", "schema_version: 1\nschema_version: 1\n", "duplicate"),
        (".yaml", "defaults: &defaults\n  schema_version: 1\nrun: *defaults\n", "anchors"),
        (".yaml", "schema_version: 1\nname: ${RUN_NAME}\n", "interpolation"),
    ],
)
def test_strict_loader_rejects_ambiguous_yaml_and_json(
    tmp_path: Path,
    suffix: str,
    contents: str,
    message: str,
) -> None:
    path = tmp_path / f"run{suffix}"
    path.write_text(contents)

    with pytest.raises(ConfigLoadError, match=message):
        load_run_config(path)


def test_strict_loader_rejects_non_object_and_unknown_extension(tmp_path: Path) -> None:
    json_path = tmp_path / "run.json"
    json_path.write_text("[]")

    with pytest.raises(ConfigLoadError, match="root must be an object"):
        load_run_config(json_path)
    with pytest.raises(ConfigLoadError, match="must end"):
        load_run_config(tmp_path / "run.toml")


def test_profile_source_and_selection_precedence(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
) -> None:
    explicit_path = tmp_path / "explicit.json"
    environment_path = tmp_path / "environment.yaml"
    default_path = tmp_path / DEFAULT_PROFILE_FILE_NAME
    explicit_path.write_text(profile_catalog.serialize_json())
    environment_path.write_text(yaml.safe_dump(profile_catalog.model_dump(mode="json"), sort_keys=True))
    default_path.write_text(profile_catalog.serialize_json())

    explicit = resolve_profile(
        profile_file=explicit_path,
        cluster="lab",
        hostname_resolver=lambda: ("primary-login-1",),
        environ={PROFILE_FILE_ENVIRONMENT: str(environment_path)},
    )
    environment = resolve_profile(
        hostname_resolver=lambda: ("LAB-LOGIN-1",),
        environ={PROFILE_FILE_ENVIRONMENT: str(environment_path)},
    )
    default = resolve_profile(
        hostname_resolver=lambda: ("unmatched",),
        environ={},
        home_directory=tmp_path,
    )

    assert (explicit.cluster_name, explicit.selection_source) == ("lab", ProfileSelectionSource.EXPLICIT)
    assert explicit.catalog_path == explicit_path.as_posix()
    assert (environment.cluster_name, environment.selection_source) == (
        "lab",
        ProfileSelectionSource.HOSTNAME,
    )
    assert (default.cluster_name, default.selection_source) == ("primary", ProfileSelectionSource.DEFAULT)
    assert load_profile_catalog(default_path) == profile_catalog


def test_injected_profile_bypasses_catalog_lookup(profile_catalog: SlurmProfileCatalog) -> None:
    selected = resolve_profile(
        profile=profile_catalog.clusters["primary"],
        hostname_resolver=lambda: pytest.fail("hostname lookup must not run"),
        environ={PROFILE_FILE_ENVIRONMENT: "/missing/profile.json"},
    )

    assert selected.selection_source is ProfileSelectionSource.INJECTED
    assert selected.catalog_path is None


def test_profile_resolution_rejects_conflicting_or_empty_sources(
    profile_catalog: SlurmProfileCatalog,
) -> None:
    with pytest.raises(ConfigLoadError, match="mutually exclusive"):
        resolve_profile(catalog=profile_catalog, profile_file="profile.json")
    with pytest.raises(ConfigLoadError, match="must not be empty"):
        resolve_profile(environ={PROFILE_FILE_ENVIRONMENT: ""})
    with pytest.raises(ConfigLoadError, match="cluster selection"):
        resolve_profile(profile=profile_catalog.clusters["primary"], cluster="primary")


def test_json_builder_output_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    builder = _config_builder()

    builder.write_config(path)

    assert path.read_text() == builder.build().serialize_json()
    assert json.loads(path.read_text())["schema_version"] == 1
    assert DataDesignerSlurmConfig.model_validate_json(path.read_text()) == builder.build()
