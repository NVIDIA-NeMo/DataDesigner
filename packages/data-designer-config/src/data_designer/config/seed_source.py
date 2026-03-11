# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Self

from data_designer.config.errors import InvalidFilePathError
from data_designer.config.utils.io_helpers import (
    VALID_DATASET_FILE_EXTENSIONS,
    validate_dataset_file_path,
    validate_path_contains_files_of_type,
)

if TYPE_CHECKING:
    import pandas as pd


class SeedSource(BaseModel, ABC):
    """Base class for seed dataset configurations.

    All subclasses must define a `seed_type` field with a Literal value.
    This serves as a discriminated union discriminator.
    """

    seed_type: str


class LocalFileSeedSource(SeedSource):
    seed_type: Literal["local"] = "local"

    path: str

    @field_validator("path", mode="after")
    def validate_path(cls, v: str) -> str:
        valid_wild_card_versions = {f"*{ext}" for ext in VALID_DATASET_FILE_EXTENSIONS}
        if any(v.endswith(wildcard) for wildcard in valid_wild_card_versions):
            parts = v.split("*.")
            file_path = parts[0]
            file_extension = parts[-1]
            validate_path_contains_files_of_type(file_path, file_extension)
        else:
            validate_dataset_file_path(v)
        return v

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, path: str) -> Self:
        df.to_parquet(path, index=False)
        return cls(path=path)


class HuggingFaceSeedSource(SeedSource):
    seed_type: Literal["hf"] = "hf"

    path: str = Field(
        ...,
        description=(
            "Path to the seed data in HuggingFace. Wildcards are allowed. Examples include "
            "'datasets/my-username/my-dataset/data/000_00000.parquet', 'datasets/my-username/my-dataset/data/*.parquet', "
            "and 'datasets/my-username/my-dataset/**/*.parquet'"
        ),
    )
    token: str | None = None
    endpoint: str = "https://huggingface.co"


class DirectorySeedTransform(BaseModel, ABC):
    """Base class for full-batch directory seed transforms."""

    transform_type: str


class ClaudeCodeTraceNormalizer(DirectorySeedTransform):
    transform_type: Literal["claude_code_trace"] = "claude_code_trace"


class CodexTraceNormalizer(DirectorySeedTransform):
    transform_type: Literal["codex_trace"] = "codex_trace"


class ChatCompletionJsonlNormalizer(DirectorySeedTransform):
    transform_type: Literal["chat_completion_jsonl"] = "chat_completion_jsonl"


DirectorySeedTransformT = Annotated[
    ClaudeCodeTraceNormalizer | CodexTraceNormalizer | ChatCompletionJsonlNormalizer,
    Field(discriminator="transform_type"),
]


class DirectorySeedSource(SeedSource):
    seed_type: Literal["directory"] = "directory"

    path: str = Field(..., description="Directory containing seed artifacts to normalize into a seed dataset.")
    glob: str = Field("**/*", description="Glob pattern used to discover files under the provided directory.")
    transform: DirectorySeedTransformT | None = Field(
        default=None,
        description="Optional full-batch transform applied to the matched files before seeding.",
    )

    @field_validator("path", mode="after")
    def validate_path(cls, value: str) -> str:
        path = Path(value)
        if not path.is_dir():
            raise InvalidFilePathError(f"🛑 Path {path} is not a directory.")
        return str(path)

    @field_validator("glob", mode="after")
    def validate_glob(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("🛑 DirectorySeedSource.glob must be a non-empty string.")
        return value


def _get_claude_code_default_path() -> str:
    return str(Path("~/.claude/projects").expanduser())


def _get_codex_default_path() -> str:
    return str(Path("~/.codex/sessions").expanduser())


class ClaudeCodeTraceSeedSource(DirectorySeedSource):
    path: str = Field(
        default_factory=_get_claude_code_default_path,
        description="Directory containing Claude Code session traces. Defaults to ~/.claude/projects.",
    )
    glob: str = Field("**/*.jsonl", description="Glob pattern used to discover Claude Code trace files.")
    transform: DirectorySeedTransformT | None = Field(
        default_factory=ClaudeCodeTraceNormalizer,
        description="Full-batch Claude Code trace normalizer.",
    )


class CodexTraceSeedSource(DirectorySeedSource):
    path: str = Field(
        default_factory=_get_codex_default_path,
        description="Directory containing Codex rollout traces. Defaults to ~/.codex/sessions.",
    )
    glob: str = Field("**/*.jsonl", description="Glob pattern used to discover Codex trace files.")
    transform: DirectorySeedTransformT | None = Field(
        default_factory=CodexTraceNormalizer,
        description="Full-batch Codex trace normalizer.",
    )


class ChatCompletionJsonlSeedSource(DirectorySeedSource):
    glob: str = Field("**/*.jsonl", description="Glob pattern used to discover chat-completion JSONL files.")
    transform: DirectorySeedTransformT | None = Field(
        default_factory=ChatCompletionJsonlNormalizer,
        description="Full-batch chat-completion JSONL normalizer.",
    )
