# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public image-resolution facade for Data Designer Slurm."""

from __future__ import annotations

from typing import Protocol

from data_designer.slurm.config import ImageKind, ImageRef
from data_designer.slurm.planning import ResolvedImage
from data_designer.slurm.services.errors import (
    SlurmServiceOperation,
    _invoke_service_backend,
    _make_invalid_request_error,
)


class SlurmImageResolver(Protocol):
    """Resolve verified images through a supported service dependency."""

    def resolve(self, reference: ImageRef, *, expected_kind: ImageKind) -> ResolvedImage:
        """Return immutable facts for one verified image reference.

        Any non-``INTERNAL`` service error must contain a caller-safe message.
        """


class SlurmImageService:
    """Expose stable image results through a package-owned boundary.

    The service borrows its injected dependency and does not manage its lifecycle.

    Args:
        resolver: Image-resolution dependency implementing
            ``SlurmImageResolver``.
    """

    def __init__(self, resolver: SlurmImageResolver) -> None:
        self._resolver = resolver

    def resolve(self, reference: ImageRef, *, expected_kind: ImageKind) -> ResolvedImage:
        """Resolve one registered image for planning.

        The result must match both the authored reference and expected image kind.

        Raises:
            SlurmServiceError: If the request is invalid or resolution fails.
        """
        operation = SlurmServiceOperation.RESOLVE_IMAGE
        if not isinstance(reference, ImageRef):
            raise _make_invalid_request_error(operation, "reference must be an ImageRef")
        if not isinstance(expected_kind, ImageKind):
            raise _make_invalid_request_error(operation, "expected_kind must be an ImageKind")

        def resolve_image() -> ResolvedImage:
            image = self._resolver.resolve(reference, expected_kind=expected_kind)
            if not isinstance(image, ResolvedImage):
                raise TypeError("image resolver returned an invalid result")
            if image.authored_ref != reference:
                raise ValueError("resolved image does not match the requested reference")
            if image.kind is not expected_kind:
                raise ValueError("resolved image does not match the expected kind")
            return image

        return _invoke_service_backend(operation, resolve_image)
