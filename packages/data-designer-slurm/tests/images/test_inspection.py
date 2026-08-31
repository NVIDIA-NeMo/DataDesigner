# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from slurm_test_fakes import FakeInspectionEnvironment

from data_designer.slurm.config import ImageInspectionRecord, InstalledDistribution
from data_designer.slurm.images.errors import ImageInspectionError
from data_designer.slurm.images.inspection import (
    ClientImageInspector,
    ServingImageInspector,
    SystemInspectionEnvironment,
)
from data_designer.slurm.images.resources import inspect_image as resource_inspector


def _get_client_distributions() -> tuple[InstalledDistribution, ...]:
    return (
        InstalledDistribution(name="data-designer", version="0.9.2"),
        InstalledDistribution(name="data-designer-config", version="0.9.2"),
        InstalledDistribution(name="data-designer-engine", version="0.9.2"),
        InstalledDistribution(name="data-designer-slurm", version="0.9.2"),
        InstalledDistribution(name="pip", version="26.1"),
    )


def test_client_inspector_produces_digest_bound_golden_facts(
    client_image_inspection: ImageInspectionRecord,
) -> None:
    environment = FakeInspectionEnvironment(
        distributions=_get_client_distributions(),
        distribution_versions={"pip": "26.1"},
        executables={"pip": "/usr/bin/pip"},
    )

    inspected = ClientImageInspector(environment).inspect(client_image_inspection.sqsh_sha256)

    assert inspected == client_image_inspection


def test_serving_inspector_produces_digest_bound_golden_facts(
    serving_image_inspection: ImageInspectionRecord,
) -> None:
    environment = FakeInspectionEnvironment(
        distribution_versions={"vllm": "0.21.0"},
        executables={"vllm": "/usr/local/bin/vllm"},
    )

    inspected = ServingImageInspector(environment).inspect(serving_image_inspection.sqsh_sha256)

    assert inspected == serving_image_inspection


@pytest.mark.parametrize(
    "environment",
    (
        FakeInspectionEnvironment(
            distributions=_get_client_distributions(),
        ),
        FakeInspectionEnvironment(
            distributions=tuple(
                distribution for distribution in _get_client_distributions() if distribution.name != "pip"
            ),
            executables={"pip": "/usr/bin/pip"},
        ),
    ),
    ids=("missing-installer", "missing-installer-distribution"),
)
def test_client_inspector_rejects_missing_installer_capabilities(
    environment: FakeInspectionEnvironment,
) -> None:
    with pytest.raises(ImageInspectionError, match="required"):
        ClientImageInspector(environment).inspect("a" * 64)


def test_client_inspector_rejects_generic_python_image_without_data_designer() -> None:
    environment = FakeInspectionEnvironment(
        distributions=(InstalledDistribution(name="pip", version="26.1"),),
        distribution_versions={"pip": "26.1"},
        executables={"pip": "/usr/bin/pip"},
    )

    with pytest.raises(ImageInspectionError, match="data-designer"):
        ClientImageInspector(environment).inspect("a" * 64)


@pytest.mark.parametrize(
    "missing_name",
    ("data-designer", "data-designer-config", "data-designer-engine", "data-designer-slurm"),
)
def test_client_inspector_rejects_incomplete_data_designer_package_set(missing_name: str) -> None:
    environment = FakeInspectionEnvironment(
        distributions=tuple(
            distribution for distribution in _get_client_distributions() if distribution.name != missing_name
        ),
        executables={"pip": "/usr/bin/pip"},
    )

    with pytest.raises(ImageInspectionError, match=missing_name):
        ClientImageInspector(environment).inspect("a" * 64)


@pytest.mark.parametrize(
    "environment",
    (
        FakeInspectionEnvironment(distribution_versions={"vllm": "0.21.0"}),
        FakeInspectionEnvironment(executables={"vllm": "/usr/local/bin/vllm"}),
    ),
    ids=("missing-executable", "missing-runtime-distribution"),
)
def test_serving_inspector_rejects_missing_runtime_capabilities(
    environment: FakeInspectionEnvironment,
) -> None:
    with pytest.raises(ImageInspectionError, match="required"):
        ServingImageInspector(environment).inspect("a" * 64)


def test_client_inspector_normalizes_invalid_facts_to_canonical_error() -> None:
    environment = FakeInspectionEnvironment(
        distributions=_get_client_distributions()
        + (
            InstalledDistribution(name="plugin", version="1"),
            InstalledDistribution(name="plugin", version="2"),
        ),
        distribution_versions={"pip": "26.1"},
        executables={"pip": "/usr/bin/pip"},
    )

    with pytest.raises(ImageInspectionError, match="invalid facts"):
        ClientImageInspector(environment).inspect("a" * 64)


def test_client_inspector_sorts_distribution_inventory() -> None:
    environment = FakeInspectionEnvironment(
        distributions=tuple(reversed(_get_client_distributions())),
        distribution_versions={"pip": "26.1"},
        executables={"pip": "/usr/bin/pip"},
    )

    inspection = ClientImageInspector(environment).inspect("a" * 64).inspection

    assert tuple(distribution.name for distribution in inspection.distributions) == tuple(
        distribution.name for distribution in _get_client_distributions()
    )


def test_client_inspector_rejects_installer_version_outside_distribution_inventory() -> None:
    environment = FakeInspectionEnvironment(
        distributions=_get_client_distributions(),
        distribution_versions={"pip": "unexpected"},
        executables={"pip": "/usr/bin/pip"},
    )

    with pytest.raises(ImageInspectionError, match="does not match"):
        ClientImageInspector(environment).inspect("a" * 64)


def test_system_inspection_environment_normalizes_distribution_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distributions = (
        SimpleNamespace(metadata={"Name": "My_Package"}, version="1.2.3"),
        SimpleNamespace(metadata={"Name": "another.package"}, version="2"),
        SimpleNamespace(metadata={"Name": "my-package"}, version="1.2.3"),
    )
    monkeypatch.setattr(importlib.metadata, "distributions", Mock(return_value=distributions))

    inspected = SystemInspectionEnvironment().list_distributions()

    assert tuple((distribution.name, distribution.version) for distribution in inspected) == (
        ("another-package", "2"),
        ("my-package", "1.2.3"),
    )


@pytest.mark.parametrize(
    "distributions",
    (
        (
            SimpleNamespace(metadata={"Name": "plugin"}, version="1"),
            SimpleNamespace(metadata={"Name": "plugin"}, version="2"),
        ),
        (SimpleNamespace(metadata={}, version="1"),),
    ),
    ids=("conflicting-versions", "missing-name"),
)
def test_system_inspection_environment_rejects_invalid_distribution_inventory(
    monkeypatch: pytest.MonkeyPatch,
    distributions: tuple[SimpleNamespace, ...],
) -> None:
    monkeypatch.setattr(importlib.metadata, "distributions", Mock(return_value=distributions))

    with pytest.raises(ImageInspectionError):
        SystemInspectionEnvironment().list_distributions()


def test_system_inspection_environment_normalizes_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        Mock(side_effect=importlib.metadata.PackageNotFoundError("missing")),
    )

    with pytest.raises(ImageInspectionError, match="not installed"):
        SystemInspectionEnvironment().get_distribution_console_script("missing")


def test_package_and_standalone_serving_inspectors_share_distribution_owned_console_script_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable_path = tmp_path / "active-environment" / "bin" / "vllm"
    executable_path.parent.mkdir(parents=True)
    executable_path.write_text("#!/bin/sh\n")
    executable_path.chmod(0o700)
    shadow_path = tmp_path / "shadow" / "bin" / "vllm"
    shadow_path.parent.mkdir(parents=True)
    shadow_path.write_text("#!/bin/sh\n")
    shadow_path.chmod(0o700)
    distribution = SimpleNamespace(
        metadata={"Name": "vllm"},
        version="0.21.0",
        entry_points=(SimpleNamespace(group="console_scripts", name="vllm"),),
        files=(Path("../../../bin/vllm"),),
        locate_file=lambda _installed_file: executable_path,
    )
    monkeypatch.setenv("PATH", shadow_path.parent.as_posix())
    monkeypatch.setattr(importlib.metadata, "distribution", Mock(return_value=distribution))

    package_record = ServingImageInspector().inspect("d" * 64)
    standalone_record = ImageInspectionRecord.model_validate(resource_inspector.inspect_image("serving", "d" * 64))

    assert package_record == standalone_record
    assert package_record.inspection.executable_path == executable_path.as_posix()  # type: ignore[union-attr]


def test_system_inspection_environment_rejects_unowned_console_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distribution = SimpleNamespace(
        version="26.1",
        entry_points=(SimpleNamespace(group="console_scripts", name="pip"),),
        files=(Path("pip"),),
        locate_file=lambda installed_file: installed_file,
    )
    monkeypatch.setattr(importlib.metadata, "distribution", Mock(return_value=distribution))

    with pytest.raises(ImageInspectionError, match="does not own"):
        SystemInspectionEnvironment().get_distribution_console_script("pip")


def test_system_inspection_environment_reports_current_python_facts() -> None:
    environment = SystemInspectionEnvironment()

    assert environment.get_python_implementation()
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", environment.get_python_version())
    assert re.fullmatch(r"[A-Za-z0-9._-]+", environment.get_python_abi())
