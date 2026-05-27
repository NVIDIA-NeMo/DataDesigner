# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from data_designer.engine.resources.person_reader import PersonReader

if TYPE_CHECKING:
    import pandas as pd


class ManagedDatasetGenerator:
    def __init__(self, reader: PersonReader, locale: str) -> None:
        self._person_reader = reader
        self._locale = locale

    def generate_samples(
        self,
        size: int = 1,
        evidence: dict[str, Any | list[Any]] | None = None,
    ) -> pd.DataFrame:
        parameters = []
        uri = self._person_reader.get_dataset_uri(self._locale)
        query = f"select * from '{uri}'"
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

        return self._person_reader.execute(query, parameters)
