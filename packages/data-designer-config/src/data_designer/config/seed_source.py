# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import codecs
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
from typing_extensions import Self

from data_designer.config.errors import InvalidFilePathError
from data_designer.config.utils.io_helpers import (
    VALID_DATASET_FILE_EXTENSIONS,
    validate_dataset_file_path,
    validate_path_contains_files_of_type,
)
from data_designer.config.utils.type_helpers import StrEnum

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
    _runtime_path: str | None = PrivateAttr(default=None)

    path: str = Field(
        ...,
        description=(
            "Path to a local seed dataset file or wildcard pattern. Relative paths are resolved from the "
            "current working directory when the config is loaded, not from the config file location."
        ),
    )

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

    def model_post_init(self, __context: Any) -> None:
        self._runtime_path = _resolve_local_file_runtime_path(self.path)

    @property
    def runtime_path(self) -> str:
        if self._runtime_path is None:
            self._runtime_path = _resolve_local_file_runtime_path(self.path)
        return self._runtime_path

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


class FileSystemSeedSource(SeedSource, ABC):
    _runtime_path: str | None = PrivateAttr(default=None)

    path: str = Field(
        ...,
        description=(
            "Directory containing seed artifacts. Relative paths are resolved from the current working "
            "directory when the config is loaded, not from the config file location."
        ),
    )
    file_pattern: str = Field(
        "*",
        description=(
            "Case-sensitive filename pattern used to match files under the provided directory. "
            "Patterns match basenames only, not relative paths."
        ),
    )
    recursive: bool = Field(
        True,
        description="Whether to search nested subdirectories under the provided directory for matching files.",
    )

    @field_validator("path", mode="after")
    def validate_path(cls, value: str | None) -> str | None:
        # Signature is str | None because AgentRolloutSeedSource overrides path to str | None
        # and inherited validators fire for all subclasses.
        return _validate_filesystem_seed_source_path(value)

    def model_post_init(self, __context: Any) -> None:
        # None guard is exercised by AgentRolloutSeedSource (path: str | None) via inheritance.
        self._runtime_path = None if self.path is None else _resolve_filesystem_runtime_path(self.path)

    @property
    def runtime_path(self) -> str:
        if self._runtime_path is None:
            self._runtime_path = _resolve_filesystem_runtime_path(self.path)
        return self._runtime_path

    @field_validator("file_pattern", mode="after")
    def validate_file_pattern(cls, value: str | None) -> str | None:
        return _validate_filesystem_seed_source_file_pattern(value)


class DirectorySeedSource(FileSystemSeedSource):
    seed_type: Literal["directory"] = "directory"


class FileContentsSeedSource(FileSystemSeedSource):
    seed_type: Literal["file_contents"] = "file_contents"

    encoding: str = Field(
        "utf-8",
        description="Text encoding used when reading matching files into the `content` column.",
    )

    @field_validator("encoding", mode="after")
    def validate_encoding(cls, value: str) -> str:
        try:
            codecs.lookup(value)
        except LookupError as error:
            raise ValueError(f"🛑 Unknown encoding: {value!r}. Use a valid Python codec name.") from error
        return value


def _resolve_filesystem_runtime_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _resolve_local_file_runtime_path(path: str) -> str:
    if "*" not in path:
        return _resolve_filesystem_runtime_path(path)

    path_prefix, glob_suffix = path.split("*", 1)
    resolved_prefix = Path(path_prefix or ".").expanduser().resolve()
    return str(resolved_prefix / f"*{glob_suffix}")


def get_claude_code_default_path() -> str:
    return str(Path("~/.claude/projects").expanduser())


def get_codex_default_path() -> str:
    return str(Path("~/.codex/sessions").expanduser())


def _validate_filesystem_seed_source_path(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise InvalidFilePathError(f"🛑 Path {path} is not a directory.")
    return value


def _validate_filesystem_seed_source_file_pattern(value: str | None) -> str | None:
    if value is None:
        return None
    if not value.strip():
        raise ValueError("🛑 FileSystemSeedSource.file_pattern must be a non-empty string.")
    if "/" in value or "\\" in value:
        raise ValueError("🛑 FileSystemSeedSource.file_pattern must match file names, not relative paths.")
    return value


class AgentRolloutFormat(StrEnum):
    ATIF = "atif"
    CLAUDE_CODE = "claude_code"
    CODEX = "codex"


def get_agent_rollout_format_defaults(fmt: AgentRolloutFormat) -> tuple[str | None, str]:
    if fmt == AgentRolloutFormat.ATIF:
        return (None, "*.json")
    if fmt == AgentRolloutFormat.CLAUDE_CODE:
        return (get_claude_code_default_path(), "*.jsonl")
    if fmt == AgentRolloutFormat.CODEX:
        return (get_codex_default_path(), "*.jsonl")
    raise ValueError(f"🛑 Unknown agent rollout format: {fmt!r}")


class AgentRolloutSeedSource(FileSystemSeedSource):
    seed_type: Literal["agent_rollout"] = "agent_rollout"

    format: AgentRolloutFormat = Field(
        ...,
        description="Built-in agent rollout format to use for parsing trace files.",
    )

    path: str | None = Field(
        None,
        description=(
            "Directory containing agent rollout artifacts. This field is required for ATIF trajectories. "
            "When omitted, built-in defaults are used for formats that define one. "
            "Relative paths are resolved from the current working directory when the config is loaded, "
            "not from the config file location."
        ),
    )

    file_pattern: str | None = Field(
        None,
        description=(
            "Case-sensitive filename pattern used to match agent rollout files. When omitted, "
            "ATIF defaults to '*.json' while Claude Code and Codex default to '*.jsonl'."
        ),
    )

    @model_validator(mode="after")
    def validate_runtime_path_source(self) -> Self:
        default_path, _ = get_agent_rollout_format_defaults(self.format)
        if self.path is None and default_path is None:
            raise ValueError(f"🛑 AgentRolloutSeedSource.path is required for format {self.format.value!r}.")
        return self

    @property
    def runtime_path(self) -> str:
        if self._runtime_path is not None:
            return self._runtime_path
        default_path, _ = get_agent_rollout_format_defaults(self.format)
        resolved_path = self.path if self.path is not None else default_path
        if resolved_path is None:
            raise ValueError(f"🛑 AgentRolloutSeedSource.path is required for format {self.format.value!r}.")
        self._runtime_path = _resolve_filesystem_runtime_path(resolved_path)
        return self._runtime_path

    @property
    def resolved_file_pattern(self) -> str:
        if self.file_pattern is not None:
            return self.file_pattern
        _, default_file_pattern = get_agent_rollout_format_defaults(self.format)
        return default_file_pattern
