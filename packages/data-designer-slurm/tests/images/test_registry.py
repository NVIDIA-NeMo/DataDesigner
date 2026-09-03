# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from slurm_test_fakes import FakeInspectionEnvironment

from data_designer.slurm.config import (
    ImageBuildRequest,
    ImageInspectionRecord,
    ImageKind,
    ImageRef,
    InstalledDistribution,
)
from data_designer.slurm.images.errors import (
    ImageConflictError,
    ImageNotFoundError,
    ImageVerificationError,
)
from data_designer.slurm.images.inspection import ClientImageInspector, ServingImageInspector
from data_designer.slurm.images.records import ImageRegistryDocument
from data_designer.slurm.images.service import (
    VerifiedImageRegistry,
    compute_sqsh_file_sha256,
)


def _create_registry(workspace: Path) -> VerifiedImageRegistry:
    return VerifiedImageRegistry(workspace)


def _get_registry_path(workspace: Path) -> Path:
    return workspace / "images" / "registry.yaml"


def test_register_and_resolve_existing_sqsh_by_alias_and_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image_path = _write_sqsh(tmp_path / "artifacts" / "client.sqsh", b"client-image")
    inspection = _inspect_client(image_path)
    service = _create_registry(workspace)

    registered = service.register_existing(
        ImageBuildRequest(name="client", kind="client", source=image_path.as_posix()),
        inspection,
    )

    by_alias = service.resolve_for_planning(ImageRef(name="client"), expected_kind=ImageKind.CLIENT)
    by_path = service.resolve_for_planning(ImageRef(path=image_path.as_posix()), expected_kind=ImageKind.CLIENT)
    persisted = ImageRegistryDocument.model_validate_json(
        json.dumps(yaml.safe_load(_get_registry_path(workspace).read_text()))
    )
    assert by_alias.path == by_path.path
    assert by_alias.sha256 == by_path.sha256
    assert by_alias.inspection == by_path.inspection
    assert by_alias.path == image_path.as_posix()
    assert registered == persisted.images[0]
    assert _get_registry_path(workspace) == workspace / "images" / "registry.yaml"


def test_registry_lists_aliases_deterministically(tmp_path: Path) -> None:
    service = _create_registry(tmp_path / "workspace")
    for name in ("zeta", "alpha"):
        image_path = _write_sqsh(tmp_path / f"{name}.sqsh", name.encode())
        service.register_existing(
            ImageBuildRequest(name=name, kind="serving", source=image_path.as_posix()),
            _inspect_serving(image_path),
        )

    assert tuple(image.name for image in service.list_images()) == ("alpha", "zeta")


def test_unregister_removes_only_alias(tmp_path: Path) -> None:
    image_path = _write_sqsh(tmp_path / "client.sqsh", b"client-image")
    service = _create_registry(tmp_path / "workspace")
    service.register_existing(
        ImageBuildRequest(name="client", kind="client", source=image_path.as_posix()),
        _inspect_client(image_path),
    )

    removed = service.unregister("client")

    assert removed.path == image_path.as_posix()
    assert image_path.read_bytes() == b"client-image"
    with pytest.raises(ImageNotFoundError, match="not registered"):
        service.resolve_for_planning(ImageRef(name="client"), expected_kind=ImageKind.CLIENT)


def test_registration_requires_explicit_replace_for_alias_updates(tmp_path: Path) -> None:
    first_path = _write_sqsh(tmp_path / "first.sqsh", b"first")
    second_path = _write_sqsh(tmp_path / "second.sqsh", b"second")
    service = _create_registry(tmp_path / "workspace")
    request = ImageBuildRequest(name="serving", kind="serving", source=first_path.as_posix())
    service.register_existing(request, _inspect_serving(first_path))

    replacement = ImageBuildRequest(name="serving", kind="serving", source=second_path.as_posix())
    with pytest.raises(ImageConflictError, match="already registered"):
        service.register_existing(replacement, _inspect_serving(second_path))

    service.register_existing(replacement, _inspect_serving(second_path), replace=True)

    assert (
        service.resolve_for_planning(ImageRef(name="serving"), expected_kind=ImageKind.SERVING).path
        == second_path.as_posix()
    )
    assert first_path.exists()


def test_failed_replacement_keeps_previous_alias(tmp_path: Path) -> None:
    first_path = _write_sqsh(tmp_path / "first.sqsh", b"first")
    second_path = _write_sqsh(tmp_path / "second.sqsh", b"second")
    service = _create_registry(tmp_path / "workspace")
    service.register_existing(
        ImageBuildRequest(name="serving", kind="serving", source=first_path.as_posix()),
        _inspect_serving(first_path),
    )
    second_inspection = _inspect_serving(second_path)
    original_token_hex = secrets.token_hex

    def mutate_replacement_before_registry_write(nbytes: int | None = None) -> str:
        second_path.write_bytes(b"modified")
        return original_token_hex(nbytes)

    with (
        patch(
            "data_designer.slurm.filesystem.secrets.token_hex",
            side_effect=mutate_replacement_before_registry_write,
        ),
        pytest.raises(ImageVerificationError, match="no longer matches"),
    ):
        service.register_existing(
            ImageBuildRequest(name="serving", kind="serving", source=second_path.as_posix()),
            second_inspection,
            replace=True,
        )

    reloaded = _create_registry(tmp_path / "workspace")
    assert (
        reloaded.resolve_for_planning(ImageRef(name="serving"), expected_kind=ImageKind.SERVING).path
        == first_path.as_posix()
    )


def test_registration_does_not_publish_alias_when_sqsh_changes_before_atomic_publish(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image_path = _write_sqsh(tmp_path / "client.sqsh", b"client")
    inspection = _inspect_client(image_path)
    service = _create_registry(workspace)
    original_token_hex = secrets.token_hex

    def mutate_image_before_registry_write(nbytes: int | None = None) -> str:
        image_path.write_bytes(b"modified")
        return original_token_hex(nbytes)

    with (
        patch(
            "data_designer.slurm.filesystem.secrets.token_hex",
            side_effect=mutate_image_before_registry_write,
        ),
        pytest.raises(ImageVerificationError, match="no longer matches"),
    ):
        service.register_existing(
            ImageBuildRequest(name="client", kind="client", source=image_path.as_posix()),
            inspection,
        )

    assert _create_registry(workspace).list_images() == ()


def test_hash_rejects_path_replaced_while_reading(tmp_path: Path) -> None:
    image_path = _write_sqsh(tmp_path / "client.sqsh", b"client")
    replacement = _write_sqsh(tmp_path / "replacement.sqsh", b"replacement")
    original_lstat = Path.lstat
    lstat_calls = 0

    def replace_before_final_path_check(path: Path) -> os.stat_result:
        nonlocal lstat_calls
        lstat_calls += 1
        if lstat_calls == 2:
            replacement.replace(image_path)
        return original_lstat(path)

    with (
        patch.object(Path, "lstat", replace_before_final_path_check),
        pytest.raises(ImageVerificationError, match="changed while it was being verified"),
    ):
        compute_sqsh_file_sha256(image_path)


def test_hash_uses_exact_sqsh_bytes(tmp_path: Path) -> None:
    content = b"exact SQSH bytes\x00\xff"
    image_path = _write_sqsh(tmp_path / "client.sqsh", content)

    assert compute_sqsh_file_sha256(image_path) == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("mutation", (b"modified", b""), ids=("modified", "truncated"))
def test_resolution_rejects_modified_registered_sqsh(tmp_path: Path, mutation: bytes) -> None:
    image_path = _write_sqsh(tmp_path / "client.sqsh", b"original")
    service = _create_registry(tmp_path / "workspace")
    service.register_existing(
        ImageBuildRequest(name="client", kind="client", source=image_path.as_posix()),
        _inspect_client(image_path),
    )
    image_path.write_bytes(mutation)

    with pytest.raises(ImageVerificationError, match="no longer matches"):
        service.resolve_for_planning(ImageRef(name="client"), expected_kind=ImageKind.CLIENT)


def test_resolution_rejects_kind_mismatch(tmp_path: Path) -> None:
    image_path = _write_sqsh(tmp_path / "client.sqsh", b"client")
    service = _create_registry(tmp_path / "workspace")
    service.register_existing(
        ImageBuildRequest(name="client", kind="client", source=image_path.as_posix()),
        _inspect_client(image_path),
    )

    with pytest.raises(ImageVerificationError, match="does not match"):
        service.resolve_for_planning(ImageRef(name="client"), expected_kind=ImageKind.SERVING)


def test_registration_rejects_missing_symlink_and_wrong_inspection(tmp_path: Path) -> None:
    target = _write_sqsh(tmp_path / "target.sqsh", b"target")
    symlink = tmp_path / "link.sqsh"
    symlink.symlink_to(target)
    service = _create_registry(tmp_path / "workspace")

    with pytest.raises(ImageVerificationError, match="does not exist"):
        service.register_existing(
            ImageBuildRequest(name="missing", kind="client", source=(tmp_path / "missing.sqsh").as_posix()),
            _inspect_client(target),
        )
    with pytest.raises(ImageVerificationError, match="regular non-symlink"):
        service.register_existing(
            ImageBuildRequest(name="symlink", kind="client", source=symlink.as_posix()),
            _inspect_client(target),
        )
    with pytest.raises(ImageVerificationError, match="digest"):
        service.register_existing(
            ImageBuildRequest(name="mismatch", kind="client", source=target.as_posix()),
            ClientImageInspector(_get_client_environment()).inspect("f" * 64),
        )
    assert service.list_images() == ()


def test_registration_rejects_kind_mismatched_inspection(tmp_path: Path) -> None:
    image_path = _write_sqsh(tmp_path / "image.sqsh", b"image")
    service = _create_registry(tmp_path / "workspace")

    with pytest.raises(ImageVerificationError, match="kind"):
        service.register_existing(
            ImageBuildRequest(name="client", kind="client", source=image_path.as_posix()),
            _inspect_serving(image_path),
        )
    assert service.list_images() == ()


def test_registration_rejects_unsupported_inspector_version(tmp_path: Path) -> None:
    image_path = _write_sqsh(tmp_path / "client.sqsh", b"client")
    inspection = _inspect_client(image_path).model_copy(update={"inspector_version": "inspector-2"})
    service = _create_registry(tmp_path / "workspace")

    with pytest.raises(ImageVerificationError, match="unsupported image inspector version"):
        service.register_existing(
            ImageBuildRequest(name="client", kind="client", source=image_path.as_posix()),
            inspection,
        )

    assert service.list_images() == ()


def test_existing_registration_rejects_oci_source(tmp_path: Path) -> None:
    service = _create_registry(tmp_path / "workspace")
    request = ImageBuildRequest(
        name="serving",
        kind="serving",
        source=f"registry.example.test/serving@sha256:{'a' * 64}",
    )

    with pytest.raises(ImageVerificationError, match="CPU Slurm image lifecycle"):
        service.register_existing(request, ServingImageInspector(_get_serving_environment()).inspect("a" * 64))


def test_direct_path_must_be_registered(tmp_path: Path) -> None:
    image_path = _write_sqsh(tmp_path / "unregistered.sqsh", b"unregistered")
    service = _create_registry(tmp_path / "workspace")

    with pytest.raises(ImageNotFoundError, match="not registered"):
        service.resolve_for_planning(ImageRef(path=image_path.as_posix()), expected_kind=ImageKind.CLIENT)


def _write_sqsh(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _get_client_environment() -> FakeInspectionEnvironment:
    return FakeInspectionEnvironment(
        distributions=(
            InstalledDistribution(name="data-designer", version="0.9.2"),
            InstalledDistribution(name="data-designer-config", version="0.9.2"),
            InstalledDistribution(name="data-designer-engine", version="0.9.2"),
            InstalledDistribution(name="data-designer-slurm", version="0.9.2"),
            InstalledDistribution(name="pip", version="26.1"),
        ),
        distribution_versions={"pip": "26.1"},
        executables={"pip": "/usr/bin/pip"},
    )


def _inspect_client(path: Path) -> ImageInspectionRecord:
    return ClientImageInspector(_get_client_environment()).inspect(compute_sqsh_file_sha256(path))


def _inspect_serving(path: Path) -> ImageInspectionRecord:
    return ServingImageInspector(_get_serving_environment()).inspect(compute_sqsh_file_sha256(path))


def _get_serving_environment() -> FakeInspectionEnvironment:
    return FakeInspectionEnvironment(
        distribution_versions={"vllm": "0.21.0"},
        executables={"vllm": "/usr/local/bin/vllm"},
    )
