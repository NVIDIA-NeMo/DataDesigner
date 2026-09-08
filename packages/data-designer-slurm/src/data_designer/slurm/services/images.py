# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public image-resolution facade for Data Designer Slurm."""

from __future__ import annotations

from typing import Protocol

from pydantic import TypeAdapter, ValidationError

from data_designer.slurm.config import ImageKind, ImageRef
from data_designer.slurm.config.images import ImageBuildRequest
from data_designer.slurm.contracts import Identifier
from data_designer.slurm.images.records import RegisteredImage
from data_designer.slurm.planning import ResolvedImage
from data_designer.slurm.services.errors import (
    SlurmServiceError,
    SlurmServiceErrorCode,
    SlurmServiceOperation,
    _invoke_service_backend,
    _make_invalid_request_error,
)

_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)


class SlurmImageResolver(Protocol):
    """Resolve verified images through a supported service dependency."""

    def resolve(self, reference: ImageRef, *, expected_kind: ImageKind) -> ResolvedImage:
        """Return immutable facts for one verified image reference.

        Any non-``INTERNAL`` service error must contain a caller-safe message.
        """


class SlurmImageManager(Protocol):
    """Manage verified images through a supported service dependency."""

    def add(self, request: ImageBuildRequest, *, replace: bool) -> RegisteredImage:
        """Build or inspect and register one image."""

    def list(self) -> tuple[RegisteredImage, ...]:
        """Return all registered images in alias order."""

    def get(self, name: Identifier) -> RegisteredImage:
        """Return one registered image."""

    def remove(self, name: Identifier) -> RegisteredImage:
        """Unregister one alias without deleting its SQSH."""


class SlurmImageService:
    """Expose stable image results through a package-owned boundary.

    The service borrows its injected dependency and does not manage its lifecycle.

    Args:
        resolver: Image-resolution dependency implementing ``SlurmImageResolver``.
        manager: Optional image-operation dependency. Package-owned factories provide it.
    """

    def __init__(self, resolver: SlurmImageResolver, manager: SlurmImageManager | None = None) -> None:
        self._resolver = resolver
        self._manager = manager

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

    def add(self, request: ImageBuildRequest, *, replace: bool = False) -> RegisteredImage:
        """Build or inspect and atomically register one image."""
        operation = SlurmServiceOperation.ADD_IMAGE
        if not isinstance(request, ImageBuildRequest):
            raise _make_invalid_request_error(operation, "request must be an ImageBuildRequest")
        if type(replace) is not bool:
            raise _make_invalid_request_error(operation, "replace must be a boolean")
        manager = self._require_manager(operation)

        def add_image() -> RegisteredImage:
            image = manager.add(request, replace=replace)
            if not isinstance(image, RegisteredImage) or image.name != request.name or image.kind.value != request.kind:
                raise TypeError("image manager returned an invalid registration result")
            return image

        return _invoke_service_backend(operation, add_image)

    def list(self) -> tuple[RegisteredImage, ...]:
        """Return all registered images in deterministic alias order."""
        operation = SlurmServiceOperation.LIST_IMAGES
        manager = self._require_manager(operation)

        def list_images() -> tuple[RegisteredImage, ...]:
            images = manager.list()
            if not isinstance(images, tuple) or any(not isinstance(image, RegisteredImage) for image in images):
                raise TypeError("image manager returned an invalid image list")
            names = tuple(image.name for image in images)
            if names != tuple(sorted(names)) or len(names) != len(set(names)):
                raise ValueError("image manager returned an invalid image order")
            return images

        return _invoke_service_backend(operation, list_images)

    def get(self, name: Identifier) -> RegisteredImage:
        """Return one registered image by alias."""
        operation = SlurmServiceOperation.GET_IMAGE
        normalized_name = _validate_name(name, operation)
        manager = self._require_manager(operation)

        def get_image() -> RegisteredImage:
            image = manager.get(normalized_name)
            if not isinstance(image, RegisteredImage) or image.name != normalized_name:
                raise TypeError("image manager returned an invalid image result")
            return image

        return _invoke_service_backend(operation, get_image)

    def remove(self, name: Identifier) -> RegisteredImage:
        """Unregister one alias without deleting its SQSH artifact."""
        operation = SlurmServiceOperation.REMOVE_IMAGE
        normalized_name = _validate_name(name, operation)
        manager = self._require_manager(operation)

        def remove_image() -> RegisteredImage:
            image = manager.remove(normalized_name)
            if not isinstance(image, RegisteredImage) or image.name != normalized_name:
                raise TypeError("image manager returned an invalid removal result")
            return image

        return _invoke_service_backend(operation, remove_image)

    def _require_manager(self, operation: SlurmServiceOperation) -> SlurmImageManager:
        if self._manager is None:
            raise SlurmServiceError(
                SlurmServiceErrorCode.UNAVAILABLE,
                operation,
                "image operations require package-owned service construction",
            )
        return self._manager


def _validate_name(name: object, operation: SlurmServiceOperation) -> Identifier:
    try:
        return _IDENTIFIER_ADAPTER.validate_python(name, strict=True)
    except ValidationError:
        raise _make_invalid_request_error(operation, "name must be a valid identifier") from None
