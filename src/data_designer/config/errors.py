# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.errors import DataDesignerError


class BuilderConfigurationError(DataDesignerError): ...


class BuilderSerializationError(DataDesignerError): ...


class InvalidColumnTypeError(DataDesignerError): ...


class InvalidConfigError(DataDesignerError): ...


class InvalidFilePathError(DataDesignerError): ...


class InvalidFileFormatError(DataDesignerError): ...
