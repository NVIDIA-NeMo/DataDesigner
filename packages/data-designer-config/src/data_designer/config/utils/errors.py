# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.errors import DataDesignerError


class UserJinjaTemplateSyntaxError(DataDesignerError): ...


class InvalidEnumValueError(DataDesignerError): ...


class InvalidTypeUnionError(DataDesignerError): ...


class InvalidDiscriminatorFieldError(DataDesignerError): ...


class DatasetSampleDisplayError(DataDesignerError): ...
