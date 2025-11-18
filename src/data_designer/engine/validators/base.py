# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Iterator, Optional

from pydantic import BaseModel, ConfigDict
from typing_extensions import Self


class ValidationOutput(BaseModel):
    is_valid: Optional[bool]
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
