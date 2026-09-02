# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from unittest.mock import patch

import pytest
import yaml
from slurm_test_fakes import FakeInspectionEnvironment

import data_designer.slurm.images.service as image_service
from data_designer.slurm.config import ImageBuildRequest, ImageInspectionRecord, InstalledDistribution
from data_designer.slurm.images.errors import ImageConflictError, ImageRegistryError, ImageVerificationError
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
    image = _get_registered_client_image(_write_sqsh(tmp_path / "client.sqsh", b"client"))
    payload = image.model_dump(mode="json")
    payload["path"] = (tmp_path / "client.bin").as_posix()
    _write_registry_images(workspace, (payload,))

    with pytest.raises(ImageRegistryError, match="cannot load"):
        ImageRegistryStore(workspace).list_images()


def test_registry_rejects_persisted_inspection_digest_mismatch(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image = _get_registered_client_image(_write_sqsh(tmp_path / "client.sqsh", b"client"))
    payload = image.model_dump(mode="json")
    payload["sqsh_sha256"] = "f" * 64
    _write_registry_images(workspace, (payload,))

    with pytest.raises(ImageRegistryError, match="cannot load"):
        ImageRegistryStore(workspace).list_images()


def test_registry_rejects_persisted_unsorted_aliases(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    zeta = _get_registered_client_image(_write_sqsh(tmp_path / "zeta.sqsh", b"zeta"), name="zeta")
    alpha = _get_registered_client_image(_write_sqsh(tmp_path / "alpha.sqsh", b"alpha"), name="alpha")
    _write_registry_images(workspace, (zeta.model_dump(mode="json"), alpha.model_dump(mode="json")))

    with pytest.raises(ImageRegistryError, match="cannot load"):
        ImageRegistryStore(workspace).list_images()


def test_registry_rejects_persisted_duplicate_aliases(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image = _get_registered_client_image(_write_sqsh(tmp_path / "client.sqsh", b"client"))
    payload = image.model_dump(mode="json")
    _write_registry_images(workspace, (payload, payload))

    with pytest.raises(ImageRegistryError, match="cannot load"):
        ImageRegistryStore(workspace).list_images()


def test_registry_rejects_persisted_conflicting_facts_for_one_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image_path = _write_sqsh(tmp_path / "shared.sqsh", b"shared")
    client = _get_registered_client_image(image_path, name="client")
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
    image = _get_registered_client_image(_write_sqsh(tmp_path / "shared.sqsh", b"shared"), name="alpha")
    alias = image.model_copy(update={"name": "beta"})
    store = ImageRegistryStore(workspace)

    store.register(image, verify_before_publish=lambda _image: None)
    store.register(alias, verify_before_publish=lambda _image: None)

    assert store.list_images() == (image, alias)


def test_registry_rejects_new_alias_with_conflicting_facts_for_one_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image_path = _write_sqsh(tmp_path / "shared.sqsh", b"shared")
    client = _get_registered_client_image(image_path, name="client")
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
    registry_path = _get_registry_path(workspace)
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
    registry_path = _get_registry_path(workspace)
    registry_path.parent.mkdir(parents=True)
    os.mkfifo(registry_path)

    with pytest.raises(ImageRegistryError, match="cannot load"):
        ImageRegistryStore(workspace).list_images()


def test_registry_rejects_fifo_substituted_between_check_and_open(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image = _get_registered_client_image(_write_sqsh(tmp_path / "client.sqsh", b"client"))
    store = ImageRegistryStore(workspace)
    store.register(image, verify_before_publish=lambda _image: None)
    registry_path = store.registry_path
    original_stat = os.stat
    replaced = False

    def replace_registry_after_stat(path: str, **kwargs: object) -> os.stat_result:
        nonlocal replaced
        status = original_stat(path, **kwargs)
        if path == "registry.yaml" and not replaced:
            replaced = True
            registry_path.unlink()
            os.mkfifo(registry_path)
        return status

    with (
        patch("data_designer.slurm.images.filesystem.os.stat", side_effect=replace_registry_after_stat),
        pytest.raises(ImageRegistryError, match="cannot load"),
    ):
        store.list_images()


def test_registry_rejects_state_mutated_during_read(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image = _get_registered_client_image(_write_sqsh(tmp_path / "client.sqsh", b"client"))
    store = ImageRegistryStore(workspace)
    store.register(image, verify_before_publish=lambda _image: None)
    registry_path = store.registry_path
    registry_inode = registry_path.stat().st_ino
    original_fstat = os.fstat
    registry_fstat_count = 0

    def mutate_before_final_registry_fstat(descriptor: int) -> os.stat_result:
        nonlocal registry_fstat_count
        status = original_fstat(descriptor)
        if status.st_ino == registry_inode:
            registry_fstat_count += 1
            if registry_fstat_count == 2:
                registry_path.write_text("mutated")
                status = original_fstat(descriptor)
        return status

    with (
        patch("data_designer.slurm.images.filesystem.os.fstat", side_effect=mutate_before_final_registry_fstat),
        pytest.raises(ImageRegistryError, match="cannot load"),
    ):
        store.list_images()


def test_sqsh_reader_rejects_fifo_substituted_between_check_and_open(tmp_path: Path) -> None:
    image_path = _write_sqsh(tmp_path / "client.sqsh", b"client")
    original_lstat = Path.lstat
    replaced = False

    def replace_sqsh_after_lstat(path: Path) -> os.stat_result:
        nonlocal replaced
        status = original_lstat(path)
        if path == image_path and not replaced:
            replaced = True
            image_path.unlink()
            os.mkfifo(image_path)
        return status

    with (
        patch.object(Path, "lstat", replace_sqsh_after_lstat),
        pytest.raises(ImageVerificationError, match="changed while it was being opened"),
    ):
        compute_sqsh_file_sha256(image_path)


def test_sqsh_reader_rejects_path_replaced_during_hash(tmp_path: Path) -> None:
    image_path = _write_sqsh(tmp_path / "client.sqsh", b"client")
    replacement = _write_sqsh(tmp_path / "replacement.sqsh", b"replacement")
    original_hash_descriptor = image_service._hash_descriptor

    def replace_path_after_hash(descriptor: int) -> str:
        digest = original_hash_descriptor(descriptor)
        replacement.replace(image_path)
        return digest

    with (
        patch("data_designer.slurm.images.service._hash_descriptor", side_effect=replace_path_after_hash),
        pytest.raises(ImageVerificationError, match="changed while it was being verified"),
    ):
        compute_sqsh_file_sha256(image_path)


def test_registry_rejects_symlinked_lock_file_without_changing_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    lock_directory = workspace / "images" / ".locks"
    lock_directory.mkdir(parents=True)
    target = tmp_path / "outside.lock"
    target.write_text("outside")
    target.chmod(0o640)
    (lock_directory / "alias-client.lock").symlink_to(target)
    image = _get_registered_client_image(_write_sqsh(tmp_path / "client.sqsh", b"client"))

    with pytest.raises(ImageRegistryError, match="cannot lock"):
        ImageRegistryStore(workspace).register(image, verify_before_publish=lambda _image: None)

    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_registry_rejects_nonregular_lock_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    lock_directory = workspace / "images" / ".locks"
    lock_directory.mkdir(parents=True)
    os.mkfifo(lock_directory / "alias-client.lock")
    image = _get_registered_client_image(_write_sqsh(tmp_path / "client.sqsh", b"client"))

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
    image = _get_registered_client_image(_write_sqsh(tmp_path / "client.sqsh", b"client"))

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
    image = _get_registered_client_image(_write_sqsh(tmp_path / "client.sqsh", b"client"))

    ImageRegistryStore(workspace).register(image, verify_before_publish=lambda _image: None)

    registry_path = _get_registry_path(workspace)
    assert stat.S_IMODE(registry_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(image_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(lock_directory.stat().st_mode) == 0o700
    assert not tuple(registry_path.parent.glob(".registry.*.tmp"))
    assert not (registry_path.parent / ".registry.rollback.yaml").exists()
    assert not (registry_path.parent / ".registry.committed.yaml").exists()


def test_registration_blocks_readers_until_final_verification(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image = _get_registered_client_image(_write_sqsh(tmp_path / "client.sqsh", b"client"))
    store = ImageRegistryStore(workspace)
    verification_started = Event()
    finish_verification = Event()

    def verify_before_publish(_image: RegisteredImage) -> None:
        verification_started.set()
        assert finish_verification.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=2) as executor:
        writer = executor.submit(
            store.register,
            image,
            verify_before_publish=verify_before_publish,
        )
        assert verification_started.wait(timeout=5)
        reader = executor.submit(ImageRegistryStore(workspace).list_images)
        assert not reader.done()
        finish_verification.set()
        assert writer.result().name == "client"
        assert tuple(registered.name for registered in reader.result()) == ("client",)

    assert tuple(registered.name for registered in ImageRegistryStore(workspace).list_images()) == ("client",)


def test_existing_sqsh_hashing_does_not_block_registry_readers(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    first_path = _write_sqsh(tmp_path / "first.sqsh", b"first")
    second_path = _write_sqsh(tmp_path / "second.sqsh", b"second")
    registry = VerifiedImageRegistry(workspace)
    first = registry.register_existing(
        ImageBuildRequest(name="first", kind="client", source=first_path.as_posix()),
        _inspect_client(first_path),
    )
    second_inspection = _inspect_client(second_path)
    hash_started = Event()
    finish_hash = Event()
    original_hash_descriptor = image_service._hash_descriptor

    def block_second_hash(descriptor: int) -> str:
        descriptor_status = os.fstat(descriptor)
        second_status = second_path.stat()
        if (descriptor_status.st_dev, descriptor_status.st_ino) == (second_status.st_dev, second_status.st_ino):
            hash_started.set()
            assert finish_hash.wait(timeout=5)
        return original_hash_descriptor(descriptor)

    with (
        patch("data_designer.slurm.images.service._hash_descriptor", side_effect=block_second_hash),
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        writer = executor.submit(
            registry.register_existing,
            ImageBuildRequest(name="second", kind="client", source=second_path.as_posix()),
            second_inspection,
        )
        assert hash_started.wait(timeout=5)
        reader = executor.submit(VerifiedImageRegistry(workspace).list_images)
        assert reader.result(timeout=5) == (first,)
        finish_hash.set()
        assert writer.result(timeout=5).name == "second"


def test_failed_registration_runs_rollback_before_releasing_target_lock(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    first = _get_registered_client_image(_write_sqsh(tmp_path / "shared.sqsh", b"shared"), name="first")
    second = first.model_copy(update={"name": "second"})
    store = ImageRegistryStore(workspace)
    rollback_started = Event()
    finish_rollback = Event()

    def fail_after_publish(_image: RegisteredImage) -> None:
        raise ImageVerificationError("injected post-publication failure")

    def rollback_after_failure(_image: RegisteredImage) -> None:
        rollback_started.set()
        assert finish_rollback.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=2) as executor:
        failed_writer = executor.submit(
            store.register,
            first,
            verify_before_publish=lambda _image: None,
            verify_after_publish=fail_after_publish,
            rollback_after_failure=rollback_after_failure,
        )
        assert rollback_started.wait(timeout=5)
        succeeding_writer = executor.submit(
            store.register,
            second,
            verify_before_publish=lambda _image: None,
        )
        assert not succeeding_writer.done()
        finish_rollback.set()
        with pytest.raises(ImageVerificationError, match="injected post-publication failure"):
            failed_writer.result()
        assert succeeding_writer.result() == second

    assert ImageRegistryStore(workspace).list_images() == (second,)


def test_fresh_reader_recovers_previous_registry_after_interrupted_publication(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    original = _get_registered_client_image(_write_sqsh(tmp_path / "original.sqsh", b"original"))
    replacement = _get_registered_client_image(_write_sqsh(tmp_path / "replacement.sqsh", b"replacement"))
    store = ImageRegistryStore(workspace)
    store.register(original, verify_before_publish=lambda _image: None)
    registry_path = store.registry_path
    previous_content = registry_path.read_bytes()
    (registry_path.parent / ".registry.rollback.yaml").write_bytes(previous_content)
    registry_path.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "images": [replacement.model_dump(mode="json")]},
            sort_keys=True,
        )
    )

    assert _list_image_names_in_fresh_process(workspace) == (original.name,)
    assert not (registry_path.parent / ".registry.rollback.yaml").exists()
    assert ImageRegistryStore(workspace).list_images() == (original,)


def test_fresh_reader_keeps_committed_registry_after_interrupted_marker_cleanup(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    original = _get_registered_client_image(_write_sqsh(tmp_path / "original.sqsh", b"original"))
    replacement = _get_registered_client_image(_write_sqsh(tmp_path / "replacement.sqsh", b"replacement"))
    store = ImageRegistryStore(workspace)
    store.register(original, verify_before_publish=lambda _image: None)
    registry_path = store.registry_path
    (registry_path.parent / ".registry.committed.yaml").write_bytes(registry_path.read_bytes())
    registry_path.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "images": [replacement.model_dump(mode="json")]},
            sort_keys=True,
        )
    )

    assert _list_image_names_in_fresh_process(workspace) == (replacement.name,)
    assert not (registry_path.parent / ".registry.committed.yaml").exists()
    assert ImageRegistryStore(workspace).list_images() == (replacement,)


def test_committed_marker_sync_failure_reports_failure_and_keeps_recovery_state(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image = _get_registered_client_image(_write_sqsh(tmp_path / "client.sqsh", b"client"))
    store = ImageRegistryStore(workspace)
    image_root_sync_count = 0
    original_fsync = os.fsync

    def fail_committed_marker_sync(descriptor: int) -> None:
        nonlocal image_root_sync_count
        descriptor_status = os.fstat(descriptor)
        image_root_status = store.image_root.stat()
        if (descriptor_status.st_dev, descriptor_status.st_ino) == (
            image_root_status.st_dev,
            image_root_status.st_ino,
        ):
            image_root_sync_count += 1
            if image_root_sync_count == 3:
                raise OSError("injected committed marker sync failure")
        original_fsync(descriptor)

    with (
        patch("data_designer.slurm.images.registry.os.fsync", side_effect=fail_committed_marker_sync),
        pytest.raises(ImageRegistryError, match="commit state requires recovery"),
    ):
        store.register(image, verify_before_publish=lambda _image: None)

    committed_marker = store.image_root / ".registry.committed.yaml"
    assert committed_marker.is_file()
    assert _list_image_names_in_fresh_process(workspace) == (image.name,)
    assert not committed_marker.exists()
    assert store.list_images() == (image,)


def test_registry_transaction_stays_bound_when_image_root_path_is_replaced(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image_root = workspace / "images"
    detached_image_root = workspace / "detached-images"
    replacement_image_root = workspace / "replacement-images"
    replacement_image_root.mkdir(parents=True)
    image = _get_registered_client_image(_write_sqsh(tmp_path / "client.sqsh", b"client"))

    def replace_image_root(_image: RegisteredImage) -> None:
        image_root.rename(detached_image_root)
        replacement_image_root.rename(image_root)

    ImageRegistryStore(workspace).register(image, verify_before_publish=replace_image_root)

    assert not (image_root / "registry.yaml").exists()
    relocated_registry = yaml.safe_load((detached_image_root / "registry.yaml").read_text())
    assert relocated_registry["images"][0]["name"] == "client"


def test_concurrent_alias_writers_preserve_both_registrations(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    barrier = Barrier(2)

    def register(name: str) -> str:
        image = _get_registered_client_image(_write_sqsh(tmp_path / f"{name}.sqsh", name.encode()), name=name)
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
        image = _get_registered_client_image(_write_sqsh(tmp_path / f"{content.decode()}.sqsh", content), name="shared")
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
    registry_path = _get_registry_path(workspace)
    registry_path.parent.mkdir(parents=True)
    registry_path.write_bytes(content)


def _list_image_names_in_fresh_process(workspace: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "import json,sys; "
                "from data_designer.slurm.images.registry import ImageRegistryStore; "
                "print(json.dumps([image.name for image in ImageRegistryStore(sys.argv[1]).list_images()]))"
            ),
            workspace.as_posix(),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return tuple(json.loads(completed.stdout))


def _write_registry_images(workspace: Path, images: tuple[dict[str, object], ...]) -> None:
    content = yaml.safe_dump({"schema_version": 1, "images": list(images)}, sort_keys=True).encode()
    _write_registry_content(workspace, content)


def _get_registry_path(workspace: Path) -> Path:
    return workspace / "images" / "registry.yaml"


def _write_sqsh(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _get_registered_client_image(path: Path, *, name: str = "client") -> RegisteredImage:
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
