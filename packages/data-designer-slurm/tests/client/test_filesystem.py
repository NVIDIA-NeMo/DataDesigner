# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

import data_designer.slurm.client.filesystem as filesystem_module
from data_designer.slurm.client.errors import ClientWorkerError
from data_designer.slurm.client.filesystem import ensure_private_directory, publish_private_text, replace_private_text
from data_designer.slurm.client.records import ClientErrorCode


def test_ensure_private_directory_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ClientWorkerError) as error:
        ensure_private_directory(link)

    assert error.value.code is ClientErrorCode.INVALID_INPUT


def test_ensure_private_directory_does_not_follow_intermediate_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ClientWorkerError) as error:
        ensure_private_directory(link / "nested")

    assert error.value.code is ClientErrorCode.INVALID_INPUT
    assert not (target / "nested").exists()


@pytest.mark.parametrize("operation_name", ("read_regular_bytes", "compute_file_sha256"))
def test_regular_file_operation_rejects_swap_to_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation_name: str,
) -> None:
    path = tmp_path / "input.json"
    path.write_text("original")
    replacement = tmp_path / "replacement.json"
    replacement.write_text("replacement")
    original_open = filesystem_module.os.open
    swapped = False

    def swapping_open(name: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        nonlocal swapped
        if name == path.name and dir_fd is not None and not swapped:
            swapped = True
            path.unlink()
            path.symlink_to(replacement)
        return original_open(name, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(filesystem_module.os, "open", swapping_open)

    with pytest.raises(ClientWorkerError) as error:
        getattr(filesystem_module, operation_name)(path, missing_code=ClientErrorCode.INVALID_INPUT)

    assert error.value.code is ClientErrorCode.INVALID_INPUT


def test_publish_private_text_is_convergent_and_immutable(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    publish_private_text(path, "same")
    publish_private_text(path, "same")

    with pytest.raises(ClientWorkerError) as error:
        publish_private_text(path, "different")

    assert error.value.code is ClientErrorCode.OUTPUT_INVALID
    assert path.read_text() == "same"


@pytest.mark.parametrize("interruption", (KeyboardInterrupt, SystemExit))
def test_replace_private_text_cleans_temporary_file_on_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
) -> None:
    def interrupt_write(_descriptor: int, _payload: object) -> int:
        raise interruption

    monkeypatch.setattr(filesystem_module.os, "write", interrupt_write)

    with pytest.raises(interruption):
        replace_private_text(tmp_path / "record.json", "value")

    assert not tuple(tmp_path.glob(".client.*.tmp"))
