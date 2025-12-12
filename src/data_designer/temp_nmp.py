# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This module will actually exist in NMP as a plugin, but providing it here temporarily as a sketch.

We need to make a few updates to plugins:
1. Dependency management (the "engine" problem); separate PR with an approach already up for this
2. Add support for PluginType.SEED_DATASET
"""

from typing import Literal

import duckdb

from data_designer.config.seed_source import SeedSource
from data_designer.engine.resources.seed_reader import SeedReader


class NMPFileSeedConfig(SeedSource):
    seed_type: Literal["nmp"] = "nmp"  # or "fileset" since that's the fsspec client protocol?

    # Check with MG: what do we expect to be more common in scenarios like this?
    # 1. Just "fileset", with optional workspace prefix, e.g. "myworkspace/myfileset", "myfileset" (implicit default workspace), etc.
    # 2. Separate "fileset" and "workspace" fields
    fileset: str
    path: str


class NMPFileSeedReader(SeedReader[NMPFileSeedConfig]):
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        # NMP helper function
        sdk = get_platform_sdk()  # noqa:F821
        # New fsspec client for Files service
        fs = FilesetFileSystem(sdk)  # noqa:F821

        conn = duckdb.connect()
        conn.register_filesystem(fs)
        return conn

    def get_dataset_uri(self) -> str:
        workspace, fileset_name = self._get_workspace_and_fileset_name()
        return f"fileset://{workspace}/{fileset_name}/{self.config.path}"

    def _get_workspace_and_fileset_name(self) -> tuple[str, str]:
        match self.config.fileset.split("/"):
            case [fileset_name]:
                return ("default", fileset_name)
            case [workspace, fileset_name]:
                return (workspace, fileset_name)
            case _:
                raise ValueError("Malformed fileset")
