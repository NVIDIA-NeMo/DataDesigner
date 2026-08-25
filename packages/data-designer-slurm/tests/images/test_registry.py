# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from slurm_test_fakes import FakeInspectionEnvironment

from data_designer.slurm.config import (
    ImageBuildRequest,
    ImageInspectionRecord,
    ImageKind,
    ImageRef,
    InstalledDistribution,
)
from data_designer.slurm.images import (
    ClientImageInspector,
    ImageConflictError,
    ImageNotFoundError,
    ImageRegistryError,
    ImageRegistrySnapshot,
    ImageVerificationError,
    ServingImageInspector,
    SlurmImageService,
    compute_file_sha256,
)


def test_register_and_resolve_existing_sqsh_by_alias_and_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image_path = _write_sqsh(tmp_path / "artifacts" / "client.sqsh", b"client-image")
    inspection = _inspect_client(image_path)
    service = SlurmImageService(workspace)

    registered = service.register_existing(
        ImageBuildRequest(name="client", kind="client", source=image_path.as_posix()),
        inspection,
    )

    by_alias = service.resolve(ImageRef(name="client"), expected_kind=ImageKind.CLIENT)
    by_path = service.resolve(ImageRef(path=image_path.as_posix()), expected_kind=ImageKind.CLIENT)
    persisted = ImageRegistrySnapshot.model_validate_json(service.registry_path.read_text())
    assert by_alias.path == by_path.path
    assert by_alias.sha256 == by_path.sha256
    assert by_alias.inspection == by_path.inspection
    assert by_alias.path == image_path.as_posix()
    assert registered == persisted.images[0]
    assert service.registry_path == workspace / "images" / "registry.json"


def test_registry_lists_aliases_deterministically(tmp_path: Path) -> None:
    service = SlurmImageService(tmp_path / "workspace")
    for name in ("zeta", "alpha"):
        image_path = _write_sqsh(tmp_path / f"{name}.sqsh", name.encode())
        service.register_existing(
            ImageBuildRequest(name=name, kind="serving", source=image_path.as_posix()),
            _inspect_serving(image_path),
        )

    assert tuple(image.name for image in service.list_images()) == ("alpha", "zeta")


def test_unregister_removes_only_alias(tmp_path: Path) -> None:
    image_path = _write_sqsh(tmp_path / "client.sqsh", b"client-image")
    service = SlurmImageService(tmp_path / "workspace")
    service.register_existing(
        ImageBuildRequest(name="client", kind="client", source=image_path.as_posix()),
        _inspect_client(image_path),
    )

    removed = service.unregister("client")

    assert removed.path == image_path.as_posix()
    assert image_path.read_bytes() == b"client-image"
    with pytest.raises(ImageNotFoundError, match="not registered"):
        service.resolve(ImageRef(name="client"), expected_kind=ImageKind.CLIENT)


def test_registration_requires_explicit_replace_for_alias_updates(tmp_path: Path) -> None:
    first_path = _write_sqsh(tmp_path / "first.sqsh", b"first")
    second_path = _write_sqsh(tmp_path / "second.sqsh", b"second")
    service = SlurmImageService(tmp_path / "workspace")
    request = ImageBuildRequest(name="serving", kind="serving", source=first_path.as_posix())
    service.register_existing(request, _inspect_serving(first_path))

    replacement = ImageBuildRequest(name="serving", kind="serving", source=second_path.as_posix())
    with pytest.raises(ImageConflictError, match="already registered"):
        service.register_existing(replacement, _inspect_serving(second_path))

    service.register_existing(replacement, _inspect_serving(second_path), replace=True)

    assert service.resolve(ImageRef(name="serving"), expected_kind=ImageKind.SERVING).path == second_path.as_posix()
    assert first_path.exists()


@pytest.mark.parametrize("mutation", (b"modified", b""), ids=("modified", "truncated"))
def test_resolution_rejects_modified_registered_sqsh(tmp_path: Path, mutation: bytes) -> None:
    image_path = _write_sqsh(tmp_path / "client.sqsh", b"original")
    service = SlurmImageService(tmp_path / "workspace")
    service.register_existing(
        ImageBuildRequest(name="client", kind="client", source=image_path.as_posix()),
        _inspect_client(image_path),
    )
    image_path.write_bytes(mutation)

    with pytest.raises(ImageVerificationError, match="no longer matches"):
        service.resolve(ImageRef(name="client"), expected_kind=ImageKind.CLIENT)


def test_resolution_rejects_kind_mismatch(tmp_path: Path) -> None:
    image_path = _write_sqsh(tmp_path / "client.sqsh", b"client")
    service = SlurmImageService(tmp_path / "workspace")
    service.register_existing(
        ImageBuildRequest(name="client", kind="client", source=image_path.as_posix()),
        _inspect_client(image_path),
    )

    with pytest.raises(ImageVerificationError, match="does not match"):
        service.resolve(ImageRef(name="client"), expected_kind=ImageKind.SERVING)


def test_registration_rejects_missing_symlink_and_wrong_inspection(tmp_path: Path) -> None:
    target = _write_sqsh(tmp_path / "target.sqsh", b"target")
    symlink = tmp_path / "link.sqsh"
    symlink.symlink_to(target)
    service = SlurmImageService(tmp_path / "workspace")

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
            ClientImageInspector(_client_environment()).inspect("f" * 64),
        )
    assert service.list_images() == ()


def test_registration_rejects_kind_mismatched_inspection(tmp_path: Path) -> None:
    image_path = _write_sqsh(tmp_path / "image.sqsh", b"image")
    service = SlurmImageService(tmp_path / "workspace")

    with pytest.raises(ImageVerificationError, match="kind"):
        service.register_existing(
            ImageBuildRequest(name="client", kind="client", source=image_path.as_posix()),
            _inspect_serving(image_path),
        )
    assert service.list_images() == ()


def test_existing_registration_rejects_oci_source(tmp_path: Path) -> None:
    service = SlurmImageService(tmp_path / "workspace")
    request = ImageBuildRequest(
        name="serving",
        kind="serving",
        source=f"registry.example.test/serving@sha256:{'a' * 64}",
    )

    with pytest.raises(ImageVerificationError, match="CPU Slurm image lifecycle"):
        service.register_existing(request, ServingImageInspector(_serving_environment()).inspect("a" * 64))


def test_direct_path_must_be_registered(tmp_path: Path) -> None:
    image_path = _write_sqsh(tmp_path / "unregistered.sqsh", b"unregistered")
    service = SlurmImageService(tmp_path / "workspace")

    with pytest.raises(ImageNotFoundError, match="not registered"):
        service.resolve(ImageRef(path=image_path.as_posix()), expected_kind=ImageKind.CLIENT)


def test_registry_normalizes_corrupt_persisted_state(tmp_path: Path) -> None:
    service = SlurmImageService(tmp_path / "workspace")
    service.registry_path.parent.mkdir(parents=True)
    service.registry_path.write_text("not-json\n")

    with pytest.raises(ImageRegistryError, match="cannot load"):
        service.list_images()


def test_concurrent_alias_writers_preserve_both_registrations(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    barrier = Barrier(2)

    def register(name: str) -> str:
        image_path = _write_sqsh(tmp_path / f"{name}.sqsh", name.encode())
        barrier.wait()
        image = SlurmImageService(workspace).register_existing(
            ImageBuildRequest(name=name, kind="serving", source=image_path.as_posix()),
            _inspect_serving(image_path),
        )
        return image.name

    with ThreadPoolExecutor(max_workers=2) as executor:
        names = tuple(executor.map(register, ("first", "second")))

    assert names == ("first", "second")
    assert tuple(image.name for image in SlurmImageService(workspace).list_images()) == ("first", "second")


def test_concurrent_writers_cannot_publish_conflicting_aliases(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    barrier = Barrier(2)

    def register(content: bytes) -> str:
        image_path = _write_sqsh(tmp_path / f"{content.decode()}.sqsh", content)
        barrier.wait()
        image = SlurmImageService(workspace).register_existing(
            ImageBuildRequest(name="shared", kind="serving", source=image_path.as_posix()),
            _inspect_serving(image_path),
        )
        return image.path

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(register, content) for content in (b"first", b"second"))

    failures = tuple(future.exception() for future in futures if future.exception() is not None)
    outcomes = tuple(future.result() for future in futures if future.exception() is None)
    assert len(outcomes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], ImageConflictError)
    assert SlurmImageService(workspace).list_images()[0].path == outcomes[0]


def _write_sqsh(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _client_environment() -> FakeInspectionEnvironment:
    return FakeInspectionEnvironment(
        distributions=(InstalledDistribution(name="data-designer", version="0.9.2"),),
        distribution_versions={"pip": "26.1"},
        executables={"pip": "/usr/bin/pip"},
    )


def _inspect_client(path: Path) -> ImageInspectionRecord:
    return ClientImageInspector(_client_environment()).inspect(compute_file_sha256(path))


def _inspect_serving(path: Path) -> ImageInspectionRecord:
    return ServingImageInspector(_serving_environment()).inspect(compute_file_sha256(path))


def _serving_environment() -> FakeInspectionEnvironment:
    return FakeInspectionEnvironment(
        distribution_versions={"vllm": "0.21.0"},
        executables={"vllm": "/usr/local/bin/vllm"},
    )
