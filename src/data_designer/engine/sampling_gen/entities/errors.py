# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.errors import DataDesignerError


class MissingPersonFieldsError(DataDesignerError):
    """Exception for all errors related to missing person fields."""
