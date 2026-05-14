# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.resources.seed_reader import SeedReader
from data_designer_e2e_tests.plugins.seed_reader.config import DemoSeedSource

if TYPE_CHECKING:
    import duckdb


class DemoSeedReader(SeedReader[DemoSeedSource]):
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        return lazy.duckdb.connect()

    def get_dataset_uri(self) -> str:
        return f"{self.source.directory}/{self.source.filename}"
