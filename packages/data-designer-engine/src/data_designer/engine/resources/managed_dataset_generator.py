# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from data_designer.engine.resources.person_reader import PersonReader

if TYPE_CHECKING:
    import pandas as pd


class ManagedDatasetGenerator:
    def __init__(self, reader: PersonReader, locale: str) -> None:
        self._conn = reader.create_duckdb_connection()
        self._uri = reader.get_dataset_uri(locale)

    def generate_samples(
        self,
        size: int = 1,
        evidence: dict[str, Any | list[Any]] = {},
    ) -> pd.DataFrame:
        parameters = []
        query = f"select * from '{self._uri}'"
        if evidence:
            where_conditions = []
            for column, values in evidence.items():
                if values:
                    values = values if isinstance(values, list) else [values]
                    formatted_values = ["?"] * len(values)
                    condition = f"{column} IN ({', '.join(formatted_values)})"
                    where_conditions.append(condition)
                    parameters.extend(values)
            if where_conditions:
                query += " where " + " and ".join(where_conditions)
        query += f" order by random() limit {size}"
        cursor = self._conn.cursor()
        try:
            return cursor.execute(query, parameters).df()
        finally:
            cursor.close()
