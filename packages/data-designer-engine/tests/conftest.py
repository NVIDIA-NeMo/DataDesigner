# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Import fixtures from config and engine testing modules
pytest_plugins = [
    "data_designer.config.testing.fixtures",
    "data_designer.engine.testing.fixtures",
]
