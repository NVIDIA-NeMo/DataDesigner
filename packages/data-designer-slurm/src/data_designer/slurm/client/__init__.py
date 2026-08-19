# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic client records shared with Slurm state consumers."""

from __future__ import annotations

from data_designer.slurm.client.records import ClientOutcome, ClientResult

__all__ = ["ClientOutcome", "ClientResult"]
