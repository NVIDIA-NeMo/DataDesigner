# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from data_designer.config.seed_source import AgentRolloutFormat
from data_designer.engine.resources.agent_rollout.types import NormalizedAgentRolloutRecord


@dataclass(frozen=True)
class AgentRolloutParseContext:
    """Base context — format-specific subclasses add their own fields."""


class AgentRolloutFormatHandler(ABC):
    format: ClassVar[AgentRolloutFormat]

    def build_parse_context(self, *, root_path: Path, recursive: bool) -> AgentRolloutParseContext | None:
        """Build format-specific context once per attachment. Default: None."""
        return None

    @abstractmethod
    def is_handled_file(self, relative_path: str) -> bool: ...

    @abstractmethod
    def parse_file(
        self,
        *,
        root_path: Path,
        relative_path: str,
        parse_context: AgentRolloutParseContext | None = None,
    ) -> list[NormalizedAgentRolloutRecord]: ...
