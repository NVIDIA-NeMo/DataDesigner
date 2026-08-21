# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from data_designer.slurm.contracts import (
    AuthoredConfig,
    ContractRecord,
    ContractValue,
    Identifier,
    Sha256Digest,
    validate_absolute_path,
    validate_plain_text,
)

DistributionName = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    ),
]


class ImageKind(str, Enum):
    CLIENT = "client"
    SERVING = "serving"


class ImageRef(AuthoredConfig):
    """Authored reference to one registered image alias or SQSH path."""

    name: Identifier | None = None
    path: str | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        validate_absolute_path(value)
        if not value.endswith(".sqsh"):
            raise ValueError("image path must end in .sqsh")
        return value

    @model_validator(mode="after")
    def validate_reference(self) -> ImageRef:
        if (self.name is None) == (self.path is None):
            raise ValueError("image reference requires exactly one of name or path")
        return self


class ImageBuildRequest(AuthoredConfig):
    """Typed input for one image import or existing-SQSH registration."""

    name: Identifier
    kind: Literal["client", "serving"]
    source: str

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        validate_plain_text(value, field_name="source")
        if value.endswith(".sqsh"):
            return validate_absolute_path(value)
        if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", value):
            raise ValueError("OCI image source must be digest-qualified")
        return value


class InstalledDistribution(ContractValue):
    name: DistributionName
    version: Annotated[str, StringConstraints(min_length=1, max_length=128)]


class ClientImageInspection(ContractValue):
    kind: Literal[ImageKind.CLIENT]
    python_implementation: Identifier
    python_version: Annotated[str, StringConstraints(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")]
    python_abi: Identifier
    distributions: tuple[InstalledDistribution, ...]
    installer_path: str
    installer_version: Annotated[str, StringConstraints(min_length=1, max_length=128)]

    _installer_path_is_absolute = field_validator("installer_path")(validate_absolute_path)

    @model_validator(mode="after")
    def validate_distributions(self) -> ClientImageInspection:
        names = tuple(distribution.name for distribution in self.distributions)
        if len(names) != len(set(names)):
            raise ValueError("installed distribution names must be unique")
        return self


class ServingImageInspection(ContractValue):
    kind: Literal[ImageKind.SERVING]
    server_type: Literal["vllm"]
    runtime_version: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    executable_path: str

    _executable_path_is_absolute = field_validator("executable_path")(validate_absolute_path)


ImageInspection = Annotated[ClientImageInspection | ServingImageInspection, Field(discriminator="kind")]


class ImageInspectionRecord(ContractRecord):
    """Digest-bound factual inspection output produced inside an SQSH."""

    inspector_version: Identifier
    sqsh_sha256: Sha256Digest
    inspection: ImageInspection
