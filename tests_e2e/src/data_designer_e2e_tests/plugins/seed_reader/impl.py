# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import duckdb

from data_designer.engine.resources.seed_reader import SeedReader
from data_designer_e2e_tests.plugins.seed_reader.config import DemoSeedSource


class DemoSeedReader(SeedReader[DemoSeedSource]):
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect()

    def get_dataset_uri(self) -> str:
        return f"{self.source.directory}/{self.source.filename}"
