# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import data_designer
import data_designer.slurm


def test_slurm_uses_shared_namespace() -> None:
    assert data_designer.__file__ is None
    assert data_designer.slurm.__name__ == "data_designer.slurm"
