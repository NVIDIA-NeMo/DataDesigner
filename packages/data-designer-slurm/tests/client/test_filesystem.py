# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from data_designer.slurm.client.errors import ClientWorkerError
from data_designer.slurm.client.filesystem import ensure_private_directory, publish_private_text
from data_designer.slurm.client.records import ClientErrorCode


def test_ensure_private_directory_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ClientWorkerError) as error:
        ensure_private_directory(link)

    assert error.value.code is ClientErrorCode.INVALID_INPUT


def test_publish_private_text_is_convergent_and_immutable(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    publish_private_text(path, "same")
    publish_private_text(path, "same")

    with pytest.raises(ClientWorkerError) as error:
        publish_private_text(path, "different")

    assert error.value.code is ClientErrorCode.OUTPUT_INVALID
    assert path.read_text() == "same"
