# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pandas as pd

from data_designer.engine.resources.managed_dataset_repository import ManagedDatasetRepository


class ManagedDatasetGenerator:
    def __init__(self, managed_datasets: ManagedDatasetRepository, dataset_name: str):
        self.managed_datasets = managed_datasets
        self.dataset_name = dataset_name

    def generate_samples(
        self,
        size: int = 1,
        evidence: dict[str, Any | list[Any]] = {},
    ) -> pd.DataFrame:
        parameters = []
        query = f"select * from {self.dataset_name}"
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
        return self.managed_datasets.query(query, parameters)
