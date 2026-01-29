# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom hatch metadata hook to sync README from root.

This hook runs during metadata resolution (before build hooks) to ensure
the README.md from the repository root is copied before hatchling validates
that the readme file exists.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from hatchling.metadata.plugin.interface import MetadataHookInterface


class ReadmeSyncHook(MetadataHookInterface):
    """Metadata hook that copies README.md from repository root before building."""

    PLUGIN_NAME = "readme-sync"

    def update(self, metadata: dict[str, Any]) -> None:
        """Copy README.md from repository root to package directory."""
        root_readme = Path(self.root) / ".." / ".." / "README.md"
        package_readme = Path(self.root) / "README.md"

        if root_readme.exists():
            shutil.copy2(root_readme, package_readme)
