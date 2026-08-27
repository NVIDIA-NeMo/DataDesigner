# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.slurm.config import ClientImageInspection, ImageInspectionRecord, ImageKind, InstalledDistribution
from data_designer.slurm.images.records import ImageRegistryDocument, RegisteredImage

_SQSH_SHA256 = "a" * 64


def test_registered_image_exposes_inspected_kind() -> None:
    image = _registered_image("alpha")

    assert image.kind is ImageKind.CLIENT
    assert image.immutable_facts == (_SQSH_SHA256, None, image.inspection)


def test_registry_document_accepts_aliases_with_identical_facts() -> None:
    alpha = _registered_image("alpha")
    beta = alpha.model_copy(update={"name": "beta"})

    document = ImageRegistryDocument(schema_version=1, images=(alpha, beta))

    assert document.images == (alpha, beta)


def _registered_image(name: str) -> RegisteredImage:
    inspection = ImageInspectionRecord(
        schema_version=1,
        inspector_version="inspector-1",
        sqsh_sha256=_SQSH_SHA256,
        inspection=ClientImageInspection(
            kind="client",
            python_implementation="cpython",
            python_version="3.13.3",
            python_abi="cp313",
            distributions=(InstalledDistribution(name="data-designer", version="0.9.2"),),
            installer_path="/usr/bin/pip",
            installer_version="26.1",
        ),
    )
    return RegisteredImage(
        schema_version=1,
        name=name,
        path="/workspace/images/shared.sqsh",
        sqsh_sha256=_SQSH_SHA256,
        inspection=inspection,
    )
