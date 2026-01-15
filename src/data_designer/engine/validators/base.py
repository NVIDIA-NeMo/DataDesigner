# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from pydantic import BaseModel, ConfigDict
from typing_extensions import Self


class ValidationOutput(BaseModel):
    is_valid: bool | None
    model_config = ConfigDict(extra="allow")


class ValidationResult(BaseModel):
    data: list[ValidationOutput]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> ValidationOutput:
        return self.data[index]

    def __iter__(self) -> Iterator[ValidationOutput]:
        return iter(self.data)

    @classmethod
    def empty(cls, size: int) -> Self:
        return cls(data=[ValidationOutput(is_valid=None) for _ in range(size)])


class BaseValidator(ABC):
    @abstractmethod
    def run_validation(self, data: list[dict]) -> ValidationResult:
        pass
