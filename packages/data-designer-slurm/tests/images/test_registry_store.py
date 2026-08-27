# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event

import pytest
import yaml
from slurm_test_fakes import FakeInspectionEnvironment

from data_designer.slurm.config import ImageBuildRequest, ImageInspectionRecord, InstalledDistribution
from data_designer.slurm.images.errors import ImageConflictError, ImageRegistryError
from data_designer.slurm.images.inspection import ClientImageInspector, ServingImageInspector
from data_designer.slurm.images.records import RegisteredImage
from data_designer.slurm.images.registry import ImageRegistryStore
from data_designer.slurm.images.service import VerifiedImageRegistry, compute_sqsh_file_sha256


@pytest.mark.parametrize(
    "content",
    (
        b"not-yaml: [\n",
        b"\xff",
        b"schema_version: 1\nimages: &recursive\n  - *recursive\n",
        b"images: []\n",
        b"schema_version: 2\nimages: []\n",
        b"schema_version: 1\nschema_version: 1\nimages: []\n",
    ),
    ids=(
        "invalid-yaml",
        "invalid-utf8",
        "recursive-alias",
        "missing-version",
        "unsupported-version",
        "duplicate-key",
    ),
)
def test_registry_normalizes_corrupt_persisted_state(tmp_path: Path, content: bytes) -> None:
    workspace = tmp_path / "workspace"
    _write_registry_content(workspace, content)

    with pytest.raises(ImageRegistryError, match="cannot load"):
        ImageRegistryStore(workspace).list_images()


def test_registry_rejects_persisted_non_sqsh_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image = _registered_client(_write_sqsh(tmp_path / "client.sqsh", b"client"))
    payload = image.model_dump(mode="json")
    payload["path"] = (tmp_path / "client.bin").as_posix()
    _write_registry_images(workspace, (payload,))

    with pytest.raises(ImageRegistryError, match="cannot load"):
        ImageRegistryStore(workspace).list_images()


def test_registry_rejects_persisted_inspection_digest_mismatch(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image = _registered_client(_write_sqsh(tmp_path / "client.sqsh", b"client"))
    payload = image.model_dump(mode="json")
    payload["sqsh_sha256"] = "f" * 64
    _write_registry_images(workspace, (payload,))

    with pytest.raises(ImageRegistryError, match="cannot load"):
        ImageRegistryStore(workspace).list_images()


def test_registry_rejects_persisted_unsorted_aliases(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    zeta = _registered_client(_write_sqsh(tmp_path / "zeta.sqsh", b"zeta"), name="zeta")
    alpha = _registered_client(_write_sqsh(tmp_path / "alpha.sqsh", b"alpha"), name="alpha")
    _write_registry_images(workspace, (zeta.model_dump(mode="json"), alpha.model_dump(mode="json")))

    with pytest.raises(ImageRegistryError, match="cannot load"):
        ImageRegistryStore(workspace).list_images()


def test_registry_rejects_persisted_duplicate_aliases(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image = _registered_client(_write_sqsh(tmp_path / "client.sqsh", b"client"))
    payload = image.model_dump(mode="json")
    _write_registry_images(workspace, (payload, payload))

    with pytest.raises(ImageRegistryError, match="cannot load"):
        ImageRegistryStore(workspace).list_images()


def test_registry_rejects_persisted_conflicting_facts_for_one_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image_path = _write_sqsh(tmp_path / "shared.sqsh", b"shared")
    client = _registered_client(image_path, name="client")
    serving_inspection = _inspect_serving(image_path)
    serving = RegisteredImage(
        schema_version=1,
        name="serving",
        path=image_path.as_posix(),
        sqsh_sha256=serving_inspection.sqsh_sha256,
        inspection=serving_inspection,
    )
    _write_registry_images(
        workspace,
        (client.model_dump(mode="json"), serving.model_dump(mode="json")),
    )

    with pytest.raises(ImageRegistryError, match="cannot load"):
        ImageRegistryStore(workspace).list_images()


def test_registry_allows_aliases_with_identical_facts_for_one_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image = _registered_client(_write_sqsh(tmp_path / "shared.sqsh", b"shared"), name="alpha")
    alias = image.model_copy(update={"name": "beta"})
    store = ImageRegistryStore(workspace)

    store.register(image, verify_before_publish=lambda _image: None)
    store.register(alias, verify_before_publish=lambda _image: None)

    assert store.list_images() == (image, alias)


def test_registry_rejects_new_alias_with_conflicting_facts_for_one_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image_path = _write_sqsh(tmp_path / "shared.sqsh", b"shared")
    client = _registered_client(image_path, name="client")
    serving_inspection = _inspect_serving(image_path)
    serving = RegisteredImage(
        schema_version=1,
        name="serving",
        path=image_path.as_posix(),
        sqsh_sha256=serving_inspection.sqsh_sha256,
        inspection=serving_inspection,
    )
    store = ImageRegistryStore(workspace)
    store.register(client, verify_before_publish=lambda _image: None)

    with pytest.raises(ImageConflictError, match="different immutable facts"):
        store.register(serving, verify_before_publish=lambda _image: None)


def test_registry_rejects_symlinked_state_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    registry_path = _registry_path(workspace)
    registry_path.parent.mkdir(parents=True)
    target = tmp_path / "outside.yaml"
    target.write_text("schema_version: 1\nimages: []\n")
    registry_path.symlink_to(target)

    with pytest.raises(ImageRegistryError, match="cannot load"):
        ImageRegistryStore(workspace).list_images()


def test_registry_rejects_reads_through_symlinked_image_root(tmp_path: Path) -> None:
    image_path = _write_sqsh(tmp_path / "client.sqsh", b"client")
    outside_workspace = tmp_path / "outside-workspace"
    outside_registry = VerifiedImageRegistry(outside_workspace)
    outside_registry.register_existing(
        ImageBuildRequest(name="client", kind="client", source=image_path.as_posix()),
        _inspect_client(image_path),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "images").symlink_to(outside_workspace / "images", target_is_directory=True)

    with pytest.raises(ImageRegistryError, match="cannot load"):
        ImageRegistryStore(workspace).list_images()


def test_registry_rejects_nonregular_state_file_without_blocking(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    registry_path = _registry_path(workspace)
    registry_path.parent.mkdir(parents=True)
    os.mkfifo(registry_path)

    with pytest.raises(ImageRegistryError, match="cannot load"):
        ImageRegistryStore(workspace).list_images()


def test_registry_rejects_symlinked_lock_file_without_changing_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    lock_directory = workspace / "images" / ".locks"
    lock_directory.mkdir(parents=True)
    target = tmp_path / "outside.lock"
    target.write_text("outside")
    target.chmod(0o640)
    (lock_directory / "alias-client.lock").symlink_to(target)
    image = _registered_client(_write_sqsh(tmp_path / "client.sqsh", b"client"))

    with pytest.raises(ImageRegistryError, match="cannot lock"):
        ImageRegistryStore(workspace).register(image, verify_before_publish=lambda _image: None)

    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_registry_rejects_nonregular_lock_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    lock_directory = workspace / "images" / ".locks"
    lock_directory.mkdir(parents=True)
    os.mkfifo(lock_directory / "alias-client.lock")
    image = _registered_client(_write_sqsh(tmp_path / "client.sqsh", b"client"))

    with pytest.raises(ImageRegistryError, match="cannot lock"):
        ImageRegistryStore(workspace).register(image, verify_before_publish=lambda _image: None)


@pytest.mark.parametrize("directory", ("images", "locks"))
def test_registry_rejects_symlinked_storage_directories(tmp_path: Path, directory: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image_root = workspace / "images"
    outside = tmp_path / "outside"
    outside.mkdir()
    if directory == "images":
        image_root.symlink_to(outside, target_is_directory=True)
    else:
        image_root.mkdir()
        (image_root / ".locks").symlink_to(outside, target_is_directory=True)
    image = _registered_client(_write_sqsh(tmp_path / "client.sqsh", b"client"))

    with pytest.raises(ImageRegistryError, match="cannot initialize"):
        ImageRegistryStore(workspace).register(image, verify_before_publish=lambda _image: None)

    assert tuple(outside.iterdir()) == ()


def test_registry_uses_restrictive_atomic_workspace_storage(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image_root = workspace / "images"
    lock_directory = image_root / ".locks"
    lock_directory.mkdir(parents=True)
    image_root.chmod(0o775)
    lock_directory.chmod(0o775)
    image = _registered_client(_write_sqsh(tmp_path / "client.sqsh", b"client"))

    ImageRegistryStore(workspace).register(image, verify_before_publish=lambda _image: None)

    registry_path = _registry_path(workspace)
    assert stat.S_IMODE(registry_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(image_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(lock_directory.stat().st_mode) == 0o700
    assert not tuple(registry_path.parent.glob(".registry.*.tmp"))


def test_registration_remains_unpublished_until_final_verification(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image = _registered_client(_write_sqsh(tmp_path / "client.sqsh", b"client"))
    store = ImageRegistryStore(workspace)
    verification_started = Event()
    finish_verification = Event()

    def verify_before_publish(_image: RegisteredImage) -> None:
        verification_started.set()
        assert finish_verification.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            store.register,
            image,
            verify_before_publish=verify_before_publish,
        )
        assert verification_started.wait(timeout=5)
        assert ImageRegistryStore(workspace).list_images() == ()
        finish_verification.set()
        assert future.result().name == "client"

    assert tuple(registered.name for registered in ImageRegistryStore(workspace).list_images()) == ("client",)


def test_concurrent_alias_writers_preserve_both_registrations(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    barrier = Barrier(2)

    def register(name: str) -> str:
        image = _registered_client(_write_sqsh(tmp_path / f"{name}.sqsh", name.encode()), name=name)
        barrier.wait()
        return ImageRegistryStore(workspace).register(image, verify_before_publish=lambda _image: None).name

    with ThreadPoolExecutor(max_workers=2) as executor:
        names = tuple(executor.map(register, ("first", "second")))

    assert names == ("first", "second")
    assert tuple(image.name for image in ImageRegistryStore(workspace).list_images()) == ("first", "second")


def test_concurrent_writers_cannot_publish_conflicting_aliases(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    barrier = Barrier(2)

    def register(content: bytes) -> str:
        image = _registered_client(_write_sqsh(tmp_path / f"{content.decode()}.sqsh", content), name="shared")
        barrier.wait()
        return ImageRegistryStore(workspace).register(image, verify_before_publish=lambda _image: None).path

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(register, content) for content in (b"first", b"second"))

    failures = tuple(future.exception() for future in futures if future.exception() is not None)
    outcomes = tuple(future.result() for future in futures if future.exception() is None)
    assert len(outcomes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], ImageConflictError)
    assert ImageRegistryStore(workspace).list_images()[0].path == outcomes[0]


def _write_registry_content(workspace: Path, content: bytes) -> None:
    registry_path = _registry_path(workspace)
    registry_path.parent.mkdir(parents=True)
    registry_path.write_bytes(content)


def _write_registry_images(workspace: Path, images: tuple[dict[str, object], ...]) -> None:
    content = yaml.safe_dump({"schema_version": 1, "images": list(images)}, sort_keys=True).encode()
    _write_registry_content(workspace, content)


def _registry_path(workspace: Path) -> Path:
    return workspace / "images" / "registry.yaml"


def _write_sqsh(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _registered_client(path: Path, *, name: str = "client") -> RegisteredImage:
    inspection = _inspect_client(path)
    return RegisteredImage(
        schema_version=1,
        name=name,
        path=path.as_posix(),
        sqsh_sha256=inspection.sqsh_sha256,
        inspection=inspection,
    )


def _inspect_client(path: Path) -> ImageInspectionRecord:
    environment = FakeInspectionEnvironment(
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
    return ClientImageInspector(environment).inspect(compute_sqsh_file_sha256(path))


def _inspect_serving(path: Path) -> ImageInspectionRecord:
    environment = FakeInspectionEnvironment(
        distribution_versions={"vllm": "0.21.0"},
        executables={"vllm": "/usr/local/bin/vllm"},
    )
    return ServingImageInspector(environment).inspect(compute_sqsh_file_sha256(path))
