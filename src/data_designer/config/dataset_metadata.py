# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel


class DatasetMetadata(BaseModel):
    """Metadata about a generated dataset.

    This object is created by the engine and passed to results objects for use
    in visualization and other client-side utilities. It is designed to be
    serializable so it can be sent over the wire in a client-server architecture.

    Attributes:
        seed_column_names: Names of columns from the seed dataset. Empty list if no seed dataset.
    """

    seed_column_names: list[str] = []
