# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from slurm_test_fakes import FakeInspectionEnvironment

from data_designer.slurm.config import ImageInspectionRecord, InstalledDistribution
from data_designer.slurm.images import ClientImageInspector, ImageInspectionError, ServingImageInspector


def test_client_inspector_produces_digest_bound_golden_facts(
    client_image_inspection: ImageInspectionRecord,
) -> None:
    environment = FakeInspectionEnvironment(
        distributions=(
            InstalledDistribution(name="data-designer", version="0.9.2"),
            InstalledDistribution(name="pip", version="26.1"),
        ),
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
        FakeInspectionEnvironment(distribution_versions={"pip": "26.1"}),
        FakeInspectionEnvironment(executables={"pip": "/usr/bin/pip"}),
    ),
    ids=("missing-installer", "missing-installer-distribution"),
)
def test_client_inspector_rejects_missing_installer_capabilities(
    environment: FakeInspectionEnvironment,
) -> None:
    with pytest.raises(ImageInspectionError, match="required"):
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
        distributions=(
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
        distributions=(
            InstalledDistribution(name="pip", version="26.1"),
            InstalledDistribution(name="data-designer", version="0.9.2"),
        ),
        distribution_versions={"pip": "26.1"},
        executables={"pip": "/usr/bin/pip"},
    )

    inspection = ClientImageInspector(environment).inspect("a" * 64).inspection

    assert tuple(distribution.name for distribution in inspection.distributions) == ("data-designer", "pip")
