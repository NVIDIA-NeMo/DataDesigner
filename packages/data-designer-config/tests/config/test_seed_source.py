# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

import data_designer.config as dd
import data_designer.lazy_heavy_imports as lazy
from data_designer.config.errors import InvalidFilePathError
from data_designer.config.seed_source import (
    AgentRolloutFormat,
    AgentRolloutSeedSource,
    DirectorySeedSource,
    FileContentsSeedSource,
    LocalFileSeedSource,
)
from data_designer.config.seed_source_dataframe import DataFrameSeedSource


def create_partitions_in_path(temp_dir: Path, extension: str, num_files: int = 2) -> Path:
    df = lazy.pd.DataFrame({"col": [1, 2, 3]})

    for i in range(num_files):
        file_path = temp_dir / f"partition_{i}.{extension}"
        if extension == "parquet":
            df.to_parquet(file_path)
        elif extension == "csv":
            df.to_csv(file_path, index=False)
        elif extension in {"json", "jsonl"}:
            df.to_json(file_path, orient="records", lines=True)
    return temp_dir


def test_local_seed_dataset_reference_validation(tmp_path: Path) -> None:
    with pytest.raises(InvalidFilePathError, match="🛑 Path test/dataset.parquet is not a file."):
        LocalFileSeedSource(path="test/dataset.parquet")

    create_partitions_in_path(tmp_path, "parquet")
    create_partitions_in_path(tmp_path, "csv")
    create_partitions_in_path(tmp_path, "json")
    create_partitions_in_path(tmp_path, "jsonl")

    for extension in ["parquet", "csv", "json", "jsonl"]:
        config = LocalFileSeedSource(path=f"{tmp_path}/*.{extension}")
        assert config.path == f"{tmp_path}/*.{extension}"


def test_local_seed_dataset_reference_validation_error(tmp_path: Path) -> None:
    create_partitions_in_path(tmp_path, "parquet")
    with pytest.raises(InvalidFilePathError, match="does not contain files of type 'csv'"):
        LocalFileSeedSource(path=f"{tmp_path}/*.csv")


def test_local_source_from_dataframe(tmp_path: Path) -> None:
    df = lazy.pd.DataFrame({"col": [1, 2, 3]})
    filepath = f"{tmp_path}/data.parquet"

    source = LocalFileSeedSource.from_dataframe(df, filepath)

    assert source.path == filepath
    lazy.pd.testing.assert_frame_equal(df, lazy.pd.read_parquet(filepath))


def test_local_seed_source_caches_runtime_path_across_cwd_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial_root = tmp_path / "initial"
    later_root = tmp_path / "later"
    initial_seed_dir = initial_root / "seed-dir"
    initial_seed_dir.mkdir(parents=True)
    create_partitions_in_path(initial_seed_dir, "parquet", num_files=1)
    later_root.mkdir()

    monkeypatch.chdir(initial_root)
    source = LocalFileSeedSource(path="seed-dir/*.parquet")
    expected_runtime_path = str(initial_seed_dir.resolve() / "*.parquet")

    monkeypatch.chdir(later_root)

    assert source.path == "seed-dir/*.parquet"
    assert source.runtime_path == expected_runtime_path
    assert source.model_dump(mode="json")["path"] == "seed-dir/*.parquet"


def test_dataframe_seed_source_serialization() -> None:
    """Test that DataFrameSeedSource excludes the DataFrame field during serialization."""
    df = lazy.pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    source = DataFrameSeedSource(df=df)

    serialized = source.model_dump(mode="json")
    assert "df" not in serialized
    assert serialized == {"seed_type": "df"}


def test_directory_seed_source_requires_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("alpha", encoding="utf-8")

    with pytest.raises(InvalidFilePathError, match="is not a directory"):
        DirectorySeedSource(path=str(file_path))


def test_directory_seed_source_preserves_relative_path_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_dir = tmp_path / "seed-dir"
    seed_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    source = DirectorySeedSource(path="seed-dir")

    assert source.path == "seed-dir"
    assert source.model_dump(mode="json")["path"] == "seed-dir"
    assert source.file_pattern == "*"
    assert source.recursive is True


def test_file_contents_seed_source_defaults() -> None:
    source = FileContentsSeedSource(path=".", file_pattern="*.md", recursive=False)

    assert source.seed_type == "file_contents"
    assert source.file_pattern == "*.md"
    assert source.recursive is False
    assert source.encoding == "utf-8"


def test_file_contents_seed_source_preserves_relative_path_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_dir = tmp_path / "seed-dir"
    seed_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    source = FileContentsSeedSource(path="seed-dir", file_pattern="*.txt")

    assert source.path == "seed-dir"
    assert source.model_dump(mode="json")["path"] == "seed-dir"


@pytest.mark.parametrize(
    ("source_type", "source_kwargs"),
    [
        pytest.param(DirectorySeedSource, {}, id="directory"),
        pytest.param(FileContentsSeedSource, {"file_pattern": "*.txt"}, id="file-contents"),
    ],
)
def test_filesystem_seed_sources_cache_runtime_path_across_cwd_changes(
    source_type: type[DirectorySeedSource] | type[FileContentsSeedSource],
    source_kwargs: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_root = tmp_path / "initial"
    later_root = tmp_path / "later"
    initial_seed_dir = initial_root / "seed-dir"
    initial_seed_dir.mkdir(parents=True)
    later_root.mkdir()

    monkeypatch.chdir(initial_root)
    source = source_type(path="seed-dir", **source_kwargs)
    expected_runtime_path = str(initial_seed_dir.resolve())

    monkeypatch.chdir(later_root)

    assert source.path == "seed-dir"
    assert source.runtime_path == expected_runtime_path
    assert source.model_dump(mode="json")["path"] == "seed-dir"


def test_seed_source_path_descriptions_document_cwd_resolution() -> None:
    local_path_description = LocalFileSeedSource.model_json_schema()["properties"]["path"]["description"]
    directory_path_description = DirectorySeedSource.model_json_schema()["properties"]["path"]["description"]
    file_contents_path_description = FileContentsSeedSource.model_json_schema()["properties"]["path"]["description"]

    assert "current working directory" in local_path_description
    assert "config file location" in local_path_description
    assert "current working directory" in directory_path_description
    assert "config file location" in directory_path_description
    assert "current working directory" in file_contents_path_description
    assert "config file location" in file_contents_path_description


def test_file_contents_seed_source_parses_from_dict(tmp_path: Path) -> None:
    source = FileContentsSeedSource.model_validate(
        {
            "path": str(tmp_path),
            "file_pattern": "*.txt",
            "recursive": False,
            "encoding": "latin-1",
        }
    )

    assert source.file_pattern == "*.txt"
    assert source.recursive is False
    assert source.encoding == "latin-1"


def test_file_contents_seed_source_rejects_unknown_encoding(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown encoding"):
        FileContentsSeedSource(path=str(tmp_path), file_pattern="*.txt", encoding="utf-999")


@pytest.mark.parametrize(
    ("source_type", "file_pattern", "error_message"),
    [
        pytest.param(DirectorySeedSource, "", "non-empty string", id="directory-empty"),
        pytest.param(DirectorySeedSource, "subdir/*.txt", "match file names, not relative paths", id="directory-posix"),
        pytest.param(FileContentsSeedSource, "", "non-empty string", id="contents-empty"),
        pytest.param(
            FileContentsSeedSource, r"subdir\\*.txt", "match file names, not relative paths", id="contents-windows"
        ),
    ],
)
def test_filesystem_seed_sources_reject_path_like_file_patterns(
    source_type: type[DirectorySeedSource] | type[FileContentsSeedSource],
    file_pattern: str,
    error_message: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=error_message):
        source_type(path=str(tmp_path), file_pattern=file_pattern)


@pytest.mark.parametrize(
    ("rollout_format", "file_pattern", "error_message"),
    [
        pytest.param(
            AgentRolloutFormat.ATIF,
            "nested/trace.json",
            "match file names, not relative paths",
            id="atif-posix",
        ),
        pytest.param(
            AgentRolloutFormat.CLAUDE_CODE,
            "",
            "non-empty string",
            id="claude-empty",
        ),
        pytest.param(
            AgentRolloutFormat.CODEX,
            "nested/*.jsonl",
            "match file names, not relative paths",
            id="codex-posix",
        ),
        pytest.param(
            AgentRolloutFormat.HERMES_AGENT,
            r"nested\\session_*.json",
            "match file names, not relative paths",
            id="hermes-windows",
        ),
    ],
)
def test_agent_rollout_seed_source_rejects_invalid_file_patterns(
    rollout_format: AgentRolloutFormat,
    file_pattern: str,
    error_message: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=error_message):
        AgentRolloutSeedSource(path=str(tmp_path), file_pattern=file_pattern, format=rollout_format)


def test_agent_rollout_seed_source_requires_explicit_atif_path() -> None:
    with pytest.raises(ValueError, match="path is required for format 'atif'"):
        AgentRolloutSeedSource(format=AgentRolloutFormat.ATIF)


def test_agent_rollout_seed_source_uses_default_atif_file_pattern(tmp_path: Path) -> None:
    trace_dir = tmp_path / "atif"
    trace_dir.mkdir()

    source = AgentRolloutSeedSource(path=str(trace_dir), format=AgentRolloutFormat.ATIF)

    assert source.seed_type == "agent_rollout"
    assert source.resolved_file_pattern == "*.json"
    assert source.recursive is True
    assert source.format == AgentRolloutFormat.ATIF


def test_agent_rollout_seed_source_uses_default_claude_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    claude_dir = tmp_path / ".claude" / "projects"
    claude_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))

    source = AgentRolloutSeedSource(format=AgentRolloutFormat.CLAUDE_CODE)

    assert source.seed_type == "agent_rollout"
    assert source.path is None
    assert source.file_pattern is None
    assert source.resolved_file_pattern == "*.jsonl"
    assert source.recursive is True
    assert source.format == AgentRolloutFormat.CLAUDE_CODE


def test_agent_rollout_seed_source_uses_default_codex_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    codex_dir = tmp_path / ".codex" / "sessions"
    codex_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))

    source = AgentRolloutSeedSource(format=AgentRolloutFormat.CODEX)

    assert source.seed_type == "agent_rollout"
    assert source.path is None
    assert source.file_pattern is None
    assert source.resolved_file_pattern == "*.jsonl"
    assert source.recursive is True
    assert source.format == AgentRolloutFormat.CODEX


def test_agent_rollout_seed_source_uses_default_hermes_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    hermes_dir = tmp_path / ".hermes" / "sessions"
    hermes_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))

    source = AgentRolloutSeedSource(format=AgentRolloutFormat.HERMES_AGENT)

    assert source.seed_type == "agent_rollout"
    assert source.path is None
    assert source.file_pattern is None
    assert source.resolved_file_pattern == "*.json*"
    assert source.recursive is True
    assert source.format == AgentRolloutFormat.HERMES_AGENT


def test_agent_rollout_seed_source_round_trips_none_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    claude_dir = tmp_path / ".claude" / "projects"
    claude_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))

    source = AgentRolloutSeedSource(format=AgentRolloutFormat.CLAUDE_CODE)
    serialized = source.model_dump(mode="json")

    assert serialized["path"] is None
    assert serialized["file_pattern"] is None

    restored = AgentRolloutSeedSource.model_validate(serialized)
    assert restored.path is None
    assert restored.file_pattern is None
    assert restored.format == AgentRolloutFormat.CLAUDE_CODE


def test_agent_rollout_seed_source_parses_format_from_dict(tmp_path: Path) -> None:
    source = AgentRolloutSeedSource.model_validate(
        {
            "path": str(tmp_path),
            "format": "codex",
        }
    )

    assert source.format == AgentRolloutFormat.CODEX
    assert source.resolved_file_pattern == "*.jsonl"


def test_seed_sources_are_exported_from_config_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    claude_dir = tmp_path / ".claude" / "projects"
    codex_dir = tmp_path / ".codex" / "sessions"
    hermes_dir = tmp_path / ".hermes" / "sessions"
    claude_dir.mkdir(parents=True)
    codex_dir.mkdir(parents=True)
    hermes_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))

    directory_source = dd.DirectorySeedSource(path=str(tmp_path))
    file_contents_source = dd.FileContentsSeedSource(path=str(tmp_path), file_pattern="*.txt")
    rollout_source = dd.AgentRolloutSeedSource(
        path=str(tmp_path),
        format=dd.AgentRolloutFormat.CLAUDE_CODE,
    )

    assert directory_source.seed_type == "directory"
    assert file_contents_source.seed_type == "file_contents"
    assert rollout_source.seed_type == "agent_rollout"
    assert rollout_source.format == dd.AgentRolloutFormat.CLAUDE_CODE
