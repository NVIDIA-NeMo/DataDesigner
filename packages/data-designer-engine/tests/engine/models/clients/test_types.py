# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.engine.models.clients.types import Usage
from data_designer.engine.models.usage import TokenCountSource


def test_usage_reasoning_token_count_source_is_required() -> None:
    with pytest.raises(ValueError, match="reasoning_tokens requires reasoning_token_count_source"):
        Usage(reasoning_tokens=1)

    with pytest.raises(ValueError, match="reasoning_token_count_source requires reasoning_tokens"):
        Usage(reasoning_token_count_source=TokenCountSource.ESTIMATED)
