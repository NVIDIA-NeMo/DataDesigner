# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.engine.errors import DataDesignerError


class ArtifactStorageError(DataDesignerError): ...


class DatasetGenerationError(DataDesignerError): ...


class DatasetProcessingError(DataDesignerError): ...
