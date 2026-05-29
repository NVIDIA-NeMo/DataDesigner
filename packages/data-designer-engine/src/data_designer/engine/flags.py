# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-wide feature flags read from environment variables.

This module exists so the engine, the public interface, and the readiness
module can share a single source of truth for runtime mode flags without
forming an import cycle. Tests patch values here to flip behavior for a
single test scope.
"""

from __future__ import annotations

import os

# Async engine is the default execution path. Set ``DATA_DESIGNER_ASYNC_ENGINE=0``
# to opt back into the legacy sync engine for one transitional release; the sync
# path is scheduled for removal afterwards.
DATA_DESIGNER_ASYNC_ENGINE: bool = os.environ.get("DATA_DESIGNER_ASYNC_ENGINE", "1") == "1"
