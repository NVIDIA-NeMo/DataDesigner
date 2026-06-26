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

    Attributes:
        seed_type: Discriminator field that identifies the specific seed source type.
            Subclasses must override this field with a ``Literal`` value.
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
    """Base class for seed sources backed by a directory of files.

    Use this base when a seed reader needs to enumerate files under a directory
    on disk and turn each (or groups of them) into seed rows. Concrete plugin
    configs declare a ``Literal`` ``seed_type`` and pair with a
    ``FileSystemSeedReader`` implementation.

    Attributes:
        path: Directory containing seed artifacts. Relative local paths are
            resolved by the active filesystem provider when the seed is
            validated or read, not when the config object is constructed.
        file_pattern: Case-sensitive filename pattern used to match files under
            the provided directory. Patterns match basenames only, not relative
            paths. Defaults to ``'*'``.
        recursive: Whether to search nested subdirectories under the provided
            directory for matching files. Defaults to ``True``.
    """

    path: str = Field(
        ...,
        description=(
            "Directory containing seed artifacts. Relative local paths are resolved by the active filesystem "
            "provider when the seed is validated or read, not when the config object is constructed."
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
        if value is None:
            return None
        if not value.strip():
            raise InvalidFilePathError("🛑 FileSystemSeedSource.path must be a non-empty string.")
        return value

    @field_validator("file_pattern", mode="after")
    def validate_file_pattern(cls, value: str | None) -> str | None:
        return _validate_filesystem_seed_source_file_pattern(value)

    @property
    def runtime_path(self) -> str:
        # Path resolution and existence checks are the filesystem provider's job at read
        # time, not the config object's. Keeping the raw value here preserves relative
        # paths and avoids assuming a local filesystem.
        return self.path


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


def get_hermes_agent_default_path() -> str:
    return str(Path("~/.hermes/sessions").expanduser())


def get_pi_coding_agent_default_path() -> str:
    return str(Path("~/.pi/agent/sessions").expanduser())


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
    HERMES_AGENT = "hermes_agent"
    PI_CODING_AGENT = "pi_coding_agent"


def get_agent_rollout_format_defaults(fmt: AgentRolloutFormat) -> tuple[str | None, str]:
    if fmt == AgentRolloutFormat.ATIF:
        return (None, "*.json")
    if fmt == AgentRolloutFormat.CLAUDE_CODE:
        return (get_claude_code_default_path(), "*.jsonl")
    if fmt == AgentRolloutFormat.CODEX:
        return (get_codex_default_path(), "*.jsonl")
    if fmt == AgentRolloutFormat.HERMES_AGENT:
        return (get_hermes_agent_default_path(), "*.json*")
    if fmt == AgentRolloutFormat.PI_CODING_AGENT:
        return (get_pi_coding_agent_default_path(), "*.jsonl")
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
            "Claude Code defaults to ~/.claude/projects, Codex defaults to ~/.codex/sessions, "
            "Hermes Agent defaults to ~/.hermes/sessions, "
            "and Pi Coding Agent defaults to ~/.pi/agent/sessions. "
            "Relative local paths are resolved by the active filesystem provider when the seed is "
            "validated or read, not when the config object is constructed."
        ),
    )

    file_pattern: str | None = Field(
        None,
        description=(
            "Case-sensitive filename pattern used to match agent rollout files. When omitted, "
            "ATIF defaults to '*.json', Claude Code, Codex, and Pi Coding Agent default to '*.jsonl', "
            "and Hermes Agent defaults to '*.json*'."
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
        # Path resolution and existence checks happen in the filesystem provider at read
        # time. When no explicit path is given, fall back to the format's default root.
        if self.path is not None:
            return self.path
        default_path, _ = get_agent_rollout_format_defaults(self.format)
        if default_path is None:
            raise ValueError(f"🛑 AgentRolloutSeedSource.path is required for format {self.format.value!r}.")
        return default_path

    @property
    def resolved_file_pattern(self) -> str:
        if self.file_pattern is not None:
            return self.file_pattern
        _, default_file_pattern = get_agent_rollout_format_defaults(self.format)
        return default_file_pattern
