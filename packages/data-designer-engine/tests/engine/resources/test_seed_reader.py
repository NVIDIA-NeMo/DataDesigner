# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.seed import IndexRange
from data_designer.config.seed_source import (
    AgentRolloutFormat,
    AgentRolloutSeedSource,
    DirectorySeedSource,
    FileContentsSeedSource,
    LocalFileSeedSource,
)
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.resources.seed_reader import (
    AgentRolloutSeedReader,
    DataFrameSeedReader,
    DirectorySeedReader,
    FileContentsSeedReader,
    FileSystemSeedReader,
    LocalFileSeedReader,
    SeedReaderError,
    SeedReaderFileSystemContext,
    SeedReaderRegistry,
)
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.engine.testing.seed_readers import LineFanoutDirectorySeedReader


class TrackingFileContentsSeedReader(FileContentsSeedReader):
    def __init__(self) -> None:
        self.hydrated_relative_paths: list[str] = []

    def hydrate_row(
        self,
        *,
        manifest_row: dict[str, str],
        context: SeedReaderFileSystemContext,
    ) -> dict[str, str]:
        self.hydrated_relative_paths.append(manifest_row["relative_path"])
        return super().hydrate_row(manifest_row=manifest_row, context=context)


class PluginStyleDirectorySeedReader(FileSystemSeedReader[DirectorySeedSource]):
    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> lazy.pd.DataFrame | list[dict[str, str]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [
            {
                "relative_path": relative_path,
                "file_name": Path(relative_path).name,
            }
            for relative_path in matched_paths
        ]


class CountingDataFrameSeedReader(DataFrameSeedReader):
    def __init__(self) -> None:
        self.create_duckdb_connection_calls = 0

    def create_duckdb_connection(self) -> lazy.duckdb.DuckDBPyConnection:
        self.create_duckdb_connection_calls += 1
        return super().create_duckdb_connection()


class OnAttachDirectorySeedReader(FileSystemSeedReader[DirectorySeedSource]):
    def __init__(self, label_prefix: str) -> None:
        self.label_prefix = label_prefix
        self.attach_call_count = 0

    def on_attach(self) -> None:
        self.attach_call_count += 1

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> lazy.pd.DataFrame | list[dict[str, str]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [
            {
                "relative_path": relative_path,
                "label": f"{self.label_prefix}:{Path(relative_path).name}",
            }
            for relative_path in matched_paths
        ]


class UndeclaredHydrationColumnSeedReader(FileSystemSeedReader[DirectorySeedSource]):
    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> lazy.pd.DataFrame | list[dict[str, str]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [{"relative_path": relative_path} for relative_path in matched_paths]

    def hydrate_row(
        self,
        *,
        manifest_row: dict[str, str],
        context: SeedReaderFileSystemContext,
    ) -> dict[str, str]:
        hydrated_row = dict(manifest_row)
        hydrated_row["content"] = str(context.root_path / manifest_row["relative_path"])
        return hydrated_row


class MissingHydrationColumnSeedReader(FileSystemSeedReader[DirectorySeedSource]):
    output_columns = ["relative_path", "content"]

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> lazy.pd.DataFrame | list[dict[str, str]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [{"relative_path": relative_path} for relative_path in matched_paths]

    def hydrate_row(
        self,
        *,
        manifest_row: dict[str, str],
        context: SeedReaderFileSystemContext,
    ) -> dict[str, str]:
        if manifest_row["relative_path"] == "beta.txt":
            return {
                "relative_path": manifest_row["relative_path"],
                "content": str(context.root_path / manifest_row["relative_path"]),
            }
        return {
            "relative_path": manifest_row["relative_path"],
        }


class ConfigurableHydrationDirectorySeedReader(FileSystemSeedReader[DirectorySeedSource]):
    def __init__(self, *, hydrated_return: Any, output_columns: list[str] | None = None) -> None:
        self._hydrated_return = hydrated_return
        self.output_columns = output_columns

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> lazy.pd.DataFrame | list[dict[str, str]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [{"relative_path": relative_path} for relative_path in matched_paths]

    def hydrate_row(
        self,
        *,
        manifest_row: dict[str, Any],
        context: SeedReaderFileSystemContext,
    ) -> Any:
        del manifest_row, context
        return self._hydrated_return


class ContextCountingDirectorySeedReader(FileSystemSeedReader[DirectorySeedSource]):
    def __init__(self) -> None:
        self.filesystem_context_calls = 0

    def create_filesystem_context(self, root_path: Path | str) -> SeedReaderFileSystemContext:
        self.filesystem_context_calls += 1
        return super().create_filesystem_context(root_path)

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> lazy.pd.DataFrame | list[dict[str, str]]:
        matched_paths = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [{"relative_path": relative_path} for relative_path in matched_paths]


class TrackingAgentRolloutSeedReader(AgentRolloutSeedReader):
    def __init__(self) -> None:
        super().__init__()
        self.hydrated_relative_paths: list[str] = []

    def hydrate_row(
        self,
        *,
        manifest_row: dict[str, Any],
        context: SeedReaderFileSystemContext,
    ) -> list[dict[str, Any]]:
        self.hydrated_relative_paths.append(str(manifest_row["relative_path"]))
        return super().hydrate_row(manifest_row=manifest_row, context=context)


WriteJsonl = Callable[[Path, list[dict[str, Any]]], None]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_claude_trace_directory(root_path: Path, write_jsonl: WriteJsonl) -> None:
    session_dir = root_path / "project-a"
    write_jsonl(
        session_dir / "session-1.jsonl",
        [
            {"type": "user", "sessionId": "session-1", "message": {"content": "Hello"}},
            {
                "type": "assistant",
                "sessionId": "session-1",
                "message": {"content": [{"type": "text", "text": "Hi there"}]},
            },
        ],
    )
    write_jsonl(
        session_dir / "session-2.jsonl",
        [
            {"type": "user", "sessionId": "session-2", "message": {"content": "Bye"}},
            {
                "type": "assistant",
                "sessionId": "session-2",
                "message": {"content": [{"type": "text", "text": "Goodbye"}]},
            },
        ],
    )


def _write_atif_trace_directory(root_path: Path) -> None:
    session_dir = root_path / "project-a"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "session-1.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.6",
                "session_id": "atif-session-1",
                "agent": {"name": "harbor-agent", "model_name": "gpt-5"},
                "steps": [
                    {
                        "step_id": 1,
                        "timestamp": "2026-04-06T12:00:00Z",
                        "source": "user",
                        "message": "Inspect the repo",
                    },
                    {
                        "step_id": 2,
                        "timestamp": "2026-04-06T12:00:02Z",
                        "source": "agent",
                        "message": "Repo inspected",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (session_dir / "session-2.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.6",
                "session_id": "atif-session-2",
                "agent": {"name": "harbor-agent", "model_name": "gpt-5"},
                "steps": [
                    {
                        "step_id": 1,
                        "timestamp": "2026-04-06T13:00:00Z",
                        "source": "user",
                        "message": "Run tests",
                    },
                    {
                        "step_id": 2,
                        "timestamp": "2026-04-06T13:00:03Z",
                        "source": "agent",
                        "message": "Tests checked",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_hermes_trace_directory(root_path: Path, write_jsonl: WriteJsonl) -> None:
    _write_json(
        root_path / "session_20260407_092759_baeaac.json",
        {
            "session_id": "20260407_092759_baeaac",
            "model": "aws/anthropic/bedrock-claude-opus-4-6",
            "base_url": "https://inference-api.nvidia.com/v1",
            "platform": "cli",
            "session_start": "2026-04-07T09:39:07.028463",
            "last_updated": "2026-04-07T09:51:07.905570",
            "system_prompt": "You are Hermes.",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "description": "Run shell commands.",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                }
            ],
            "messages": [
                {"role": "user", "content": "Set up a uv project."},
                {
                    "role": "assistant",
                    "content": "I'll initialize the project.",
                    "finish_reason": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "tooluse_init",
                            "call_id": "tooluse_init",
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "arguments": '{"command":"uv init"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tooluse_init",
                    "content": '{"output":"Initialized project","exit_code":0,"error":null}',
                },
                {
                    "role": "assistant",
                    "content": "Done.",
                    "finish_reason": "stop",
                    "tool_calls": [],
                },
            ],
        },
    )
    _write_json(
        root_path / "sessions.json",
        {
            "slack:thread-1": {
                "session_key": "slack:thread-1",
                "session_id": "gateway-session-1",
                "created_at": "2026-04-07T08:00:00",
                "updated_at": "2026-04-07T08:05:00",
                "display_name": "ops-thread",
                "platform": "slack",
                "chat_type": "thread",
            }
        },
    )
    write_jsonl(
        root_path / "gateway-session-1.jsonl",
        [
            {"role": "user", "content": "Check the deployment status."},
            {
                "role": "assistant",
                "content": "I'll inspect the logs.",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {
                        "id": "tooluse_logs",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command":"kubectl logs deploy/app"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tooluse_logs",
                "content": '{"output":"healthy","exit_code":0,"error":null}',
            },
        ],
    )


@pytest.fixture
def write_alpha_beta_text_files(tmp_path: Path) -> Callable[[str, str], Path]:
    def _write_alpha_beta_text_files(alpha_contents: str, beta_contents: str) -> Path:
        (tmp_path / "alpha.txt").write_text(alpha_contents, encoding="utf-8")
        (tmp_path / "beta.txt").write_text(beta_contents, encoding="utf-8")
        return tmp_path

    return _write_alpha_beta_text_files


def test_one_reader_per_seed_type():
    local_1 = LocalFileSeedReader()
    local_2 = LocalFileSeedReader()

    with pytest.raises(SeedReaderError):
        SeedReaderRegistry([local_1, local_2])

    registry = SeedReaderRegistry([local_1])

    with pytest.raises(SeedReaderError):
        registry.add_reader(local_2)


def test_get_reader_basic():
    local_reader = LocalFileSeedReader()
    df_reader = DataFrameSeedReader()
    registry = SeedReaderRegistry([local_reader, df_reader])

    df = lazy.pd.DataFrame(data={"a": [1, 2, 3]})
    local_seed_config = DataFrameSeedSource(df=df)

    reader = registry.get_reader(local_seed_config, PlaintextResolver())

    assert reader == df_reader


def test_get_reader_missing():
    local_reader = LocalFileSeedReader()
    registry = SeedReaderRegistry([local_reader])

    df = lazy.pd.DataFrame(data={"a": [1, 2, 3]})
    local_seed_config = DataFrameSeedSource(df=df)

    with pytest.raises(SeedReaderError):
        registry.get_reader(local_seed_config, PlaintextResolver())


def test_filesystem_seed_readers_expose_seed_type() -> None:
    assert DirectorySeedReader().get_seed_type() == "directory"
    assert FileContentsSeedReader().get_seed_type() == "file_contents"
    assert AgentRolloutSeedReader().get_seed_type() == "agent_rollout"


def test_seed_reader_requires_attach_before_use() -> None:
    reader = DataFrameSeedReader()

    with pytest.raises(SeedReaderError, match="must be attached to a source"):
        reader.get_seed_dataset_size()


def test_plugin_style_filesystem_seed_reader_needs_only_manifest_builder(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "beta.txt").write_text("beta", encoding="utf-8")

    reader = PluginStyleDirectorySeedReader()
    reader.attach(
        DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt"),
        PlaintextResolver(),
    )

    df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert reader.get_dataset_uri() == "seed_reader_directory_rows"
    assert list(df["relative_path"]) == ["alpha.txt", "nested/beta.txt"]
    assert list(df["file_name"]) == ["alpha.txt", "beta.txt"]


def test_plugin_style_filesystem_seed_reader_can_fan_out_rows(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha-0\nalpha-1", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("beta-0", encoding="utf-8")

    reader = LineFanoutDirectorySeedReader(include_file_name=True)
    reader.attach(
        DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt"),
        PlaintextResolver(),
    )

    df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert reader.get_seed_dataset_size() == 2
    assert list(df["relative_path"]) == ["alpha.txt", "alpha.txt", "beta.txt"]
    assert list(df["line_index"]) == [0, 1, 0]
    assert list(df["line"]) == ["alpha-0", "alpha-1", "beta-0"]


def test_directory_seed_reader_matches_files_recursively(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "beta.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "nested" / "gamma.md").write_text("gamma", encoding="utf-8")

    reader = DirectorySeedReader()
    reader.attach(
        DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt"),
        PlaintextResolver(),
    )

    df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert list(df["relative_path"]) == ["alpha.txt", "nested/beta.txt"]
    assert list(df["source_kind"]) == ["directory_file", "directory_file"]
    assert list(df["file_name"]) == ["alpha.txt", "beta.txt"]


def test_directory_seed_reader_can_disable_recursive_walk(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "beta.txt").write_text("beta", encoding="utf-8")

    reader = DirectorySeedReader()
    reader.attach(
        DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt", recursive=False),
        PlaintextResolver(),
    )

    df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert list(df["relative_path"]) == ["alpha.txt"]


def test_directory_seed_reader_raises_for_no_matches(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")

    reader = DirectorySeedReader()
    reader.attach(
        DirectorySeedSource(path=str(tmp_path), file_pattern="*.md"),
        PlaintextResolver(),
    )

    with pytest.raises(SeedReaderError, match="No files matched file_pattern '\\*\\.md'"):
        reader.get_seed_dataset_size()


def test_file_contents_seed_reader_reads_text_files(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "beta.txt").write_text("beta", encoding="utf-8")

    reader = FileContentsSeedReader()
    reader.attach(
        FileContentsSeedSource(path=str(tmp_path), file_pattern="*.txt"),
        PlaintextResolver(),
    )

    df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert list(df["relative_path"]) == ["alpha.txt", "nested/beta.txt"]
    assert list(df["content"]) == ["alpha", "beta"]
    assert list(df["source_kind"]) == ["file_contents", "file_contents"]


def test_file_contents_seed_reader_respects_encoding(tmp_path: Path) -> None:
    file_path = tmp_path / "latin1.txt"
    file_path.write_bytes("café".encode("latin-1"))

    reader = FileContentsSeedReader()
    reader.attach(
        FileContentsSeedSource(path=str(tmp_path), file_pattern="*.txt", encoding="latin-1"),
        PlaintextResolver(),
    )

    df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert list(df["content"]) == ["café"]


def test_file_contents_seed_reader_wraps_unknown_encoding_errors(tmp_path: Path) -> None:
    file_path = tmp_path / "alpha.txt"
    file_path.write_text("alpha", encoding="utf-8")

    source = FileContentsSeedSource.model_construct(
        seed_type="file_contents",
        path=str(tmp_path),
        file_pattern="*.txt",
        recursive=True,
        encoding="utf-999",
    )
    reader = FileContentsSeedReader()
    reader.attach(source, PlaintextResolver())

    with pytest.raises(SeedReaderError, match="Failed to decode file .* using encoding 'utf-999'"):
        reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()


def test_file_contents_seed_reader_hydrates_only_selected_manifest_rows(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "gamma.txt").write_text("gamma", encoding="utf-8")

    reader = TrackingFileContentsSeedReader()
    reader.attach(
        FileContentsSeedSource(path=str(tmp_path), file_pattern="*.txt"),
        PlaintextResolver(),
    )

    batch_reader = reader.create_batch_reader(
        batch_size=1,
        index_range=IndexRange(start=1, end=1),
        shuffle=False,
    )
    batch_df = batch_reader.read_next_batch().to_pandas()

    assert list(batch_df["relative_path"]) == ["beta.txt"]
    assert list(batch_df["content"]) == ["beta"]
    assert reader.hydrated_relative_paths == ["beta.txt"]


def test_filesystem_seed_reader_fanout_keeps_manifest_based_index_selection(
    write_alpha_beta_text_files: Callable[[str, str], Path],
) -> None:
    seed_dir = write_alpha_beta_text_files("alpha-0\nalpha-1", "beta-0\nbeta-1")

    reader = LineFanoutDirectorySeedReader(include_file_name=True)
    reader.attach(
        DirectorySeedSource(path=str(seed_dir), file_pattern="*.txt"),
        PlaintextResolver(),
    )

    batch_reader = reader.create_batch_reader(
        batch_size=1,
        index_range=IndexRange(start=1, end=1),
        shuffle=False,
    )
    batch_df = batch_reader.read_next_batch().to_pandas()

    assert list(batch_df["relative_path"]) == ["beta.txt", "beta.txt"]
    assert list(batch_df["line"]) == ["beta-0", "beta-1"]
    assert reader.hydrated_relative_paths == ["beta.txt"]


def test_filesystem_seed_reader_batch_reader_raises_for_selected_manifest_rows_with_empty_fanout(
    write_alpha_beta_text_files: Callable[[str, str], Path],
) -> None:
    seed_dir = write_alpha_beta_text_files("alpha-0", "")

    reader = LineFanoutDirectorySeedReader(include_file_name=True)
    reader.attach(
        DirectorySeedSource(path=str(seed_dir), file_pattern="*.txt"),
        PlaintextResolver(),
    )

    batch_reader = reader.create_batch_reader(
        batch_size=1,
        index_range=IndexRange(start=1, end=1),
        shuffle=False,
    )

    with pytest.raises(
        SeedReaderError,
        match="Selected manifest rows for seed source at .* did not produce any rows after hydration",
    ):
        batch_reader.read_next_batch()

    assert reader.hydrated_relative_paths == ["beta.txt"]


def test_filesystem_seed_reader_batch_reader_skips_empty_fanout_rows_before_returning_records(
    write_alpha_beta_text_files: Callable[[str, str], Path],
) -> None:
    seed_dir = write_alpha_beta_text_files("", "beta-0\nbeta-1")

    reader = LineFanoutDirectorySeedReader(include_file_name=True)
    reader.attach(
        DirectorySeedSource(path=str(seed_dir), file_pattern="*.txt"),
        PlaintextResolver(),
    )

    batch_reader = reader.create_batch_reader(
        batch_size=1,
        index_range=None,
        shuffle=False,
    )
    batch_df = batch_reader.read_next_batch().to_pandas()

    assert list(batch_df["relative_path"]) == ["beta.txt", "beta.txt"]
    assert list(batch_df["line"]) == ["beta-0", "beta-1"]
    assert reader.hydrated_relative_paths == ["alpha.txt", "beta.txt"]


def test_filesystem_seed_reader_batch_reader_stops_cleanly_after_emitting_records_when_only_empty_fanout_rows_remain(
    write_alpha_beta_text_files: Callable[[str, str], Path],
) -> None:
    seed_dir = write_alpha_beta_text_files("alpha-0", "")

    reader = LineFanoutDirectorySeedReader(include_file_name=True)
    reader.attach(
        DirectorySeedSource(path=str(seed_dir), file_pattern="*.txt"),
        PlaintextResolver(),
    )

    batch_reader = reader.create_batch_reader(
        batch_size=1,
        index_range=None,
        shuffle=False,
    )
    batch_df = batch_reader.read_next_batch().to_pandas()

    assert list(batch_df["relative_path"]) == ["alpha.txt"]
    assert list(batch_df["line"]) == ["alpha-0"]

    with pytest.raises(StopIteration):
        batch_reader.read_next_batch()

    assert reader.hydrated_relative_paths == ["alpha.txt", "beta.txt"]


def test_filesystem_seed_reader_full_output_raises_when_all_manifest_rows_fan_out_to_empty(
    write_alpha_beta_text_files: Callable[[str, str], Path],
) -> None:
    seed_dir = write_alpha_beta_text_files("", "")

    reader = LineFanoutDirectorySeedReader(include_file_name=True)
    reader.attach(
        DirectorySeedSource(path=str(seed_dir), file_pattern="*.txt"),
        PlaintextResolver(),
    )

    with pytest.raises(SeedReaderError, match="Seed source at .* did not produce any rows"):
        reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()


def test_local_file_seed_reader_uses_load_time_runtime_path_when_cwd_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_root = tmp_path / "initial"
    later_root = tmp_path / "later"
    initial_root.mkdir()
    later_root.mkdir()

    lazy.pd.DataFrame({"value": [1]}).to_parquet(initial_root / "seed.parquet", index=False)
    lazy.pd.DataFrame({"value": [2]}).to_parquet(later_root / "seed.parquet", index=False)

    monkeypatch.chdir(initial_root)
    source = LocalFileSeedSource(path="seed.parquet")
    reader = LocalFileSeedReader()

    monkeypatch.chdir(later_root)
    reader.attach(source, PlaintextResolver())
    df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert source.path == "seed.parquet"
    assert reader.get_dataset_uri() == str((initial_root / "seed.parquet").resolve())
    assert list(df["value"]) == [1]


def test_directory_seed_reader_uses_load_time_runtime_path_when_cwd_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_root = tmp_path / "initial"
    later_root = tmp_path / "later"
    initial_seed_dir = initial_root / "seed-dir"
    later_seed_dir = later_root / "seed-dir"
    initial_seed_dir.mkdir(parents=True)
    later_seed_dir.mkdir(parents=True)
    (initial_seed_dir / "alpha.txt").write_text("alpha", encoding="utf-8")
    (later_seed_dir / "beta.txt").write_text("beta", encoding="utf-8")

    monkeypatch.chdir(initial_root)
    source = DirectorySeedSource(path="seed-dir", file_pattern="*.txt")
    reader = DirectorySeedReader()

    monkeypatch.chdir(later_root)
    reader.attach(source, PlaintextResolver())
    df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert source.path == "seed-dir"
    assert list(df["relative_path"]) == ["alpha.txt"]
    assert list(df["source_path"]) == [str((initial_seed_dir / "alpha.txt").resolve())]


def test_filesystem_seed_reader_on_attach_requires_no_super_and_resets_state(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    first_dir.mkdir()
    (first_dir / "alpha.txt").write_text("alpha", encoding="utf-8")
    (first_dir / "beta.txt").write_text("beta", encoding="utf-8")

    second_dir = tmp_path / "second"
    second_dir.mkdir()
    (second_dir / "gamma.txt").write_text("gamma", encoding="utf-8")

    reader = OnAttachDirectorySeedReader(label_prefix="plugin")

    reader.attach(DirectorySeedSource(path=str(first_dir), file_pattern="*.txt"), PlaintextResolver())
    first_df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert reader.attach_call_count == 1
    assert reader.get_seed_dataset_size() == 2
    assert list(first_df["label"]) == ["plugin:alpha.txt", "plugin:beta.txt"]

    reader.attach(DirectorySeedSource(path=str(second_dir), file_pattern="*.txt"), PlaintextResolver())
    second_df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()

    assert reader.attach_call_count == 2
    assert reader.get_seed_dataset_size() == 1
    assert list(second_df["relative_path"]) == ["gamma.txt"]
    assert list(second_df["label"]) == ["plugin:gamma.txt"]


@pytest.mark.parametrize("use_batch_reader", [False, True], ids=["full-output", "batch-reader"])
def test_filesystem_seed_reader_raises_for_undeclared_hydrated_columns(
    tmp_path: Path,
    use_batch_reader: bool,
) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")

    reader = UndeclaredHydrationColumnSeedReader()
    reader.attach(DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt"), PlaintextResolver())

    with pytest.raises(SeedReaderError, match="Undeclared columns: \\['content'\\]"):
        if use_batch_reader:
            batch_reader = reader.create_batch_reader(
                batch_size=1,
                index_range=IndexRange(start=0, end=0),
                shuffle=False,
            )
            batch_reader.read_next_batch().to_pandas()
        else:
            reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()


@pytest.mark.parametrize(
    ("hydrated_return", "error_pattern"),
    [
        (None, "Manifest row index 0 returned NoneType"),
        (123, "Manifest row index 0 returned int"),
        (["not-a-record"], "Manifest row index 0 returned an iterable containing str"),
    ],
    ids=["none", "scalar", "iterable-of-invalid-records"],
)
def test_filesystem_seed_reader_rejects_invalid_hydrate_row_returns(
    tmp_path: Path,
    hydrated_return: Any,
    error_pattern: str,
) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")

    reader = ConfigurableHydrationDirectorySeedReader(hydrated_return=hydrated_return)
    reader.attach(DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt"), PlaintextResolver())

    with pytest.raises(SeedReaderError, match=error_pattern):
        reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()


@pytest.mark.parametrize(
    ("output_columns", "hydrated_return", "error_pattern"),
    [
        (
            ["relative_path", "content"],
            [
                {"relative_path": "alpha.txt", "content": "alpha"},
                {"relative_path": "alpha.txt"},
            ],
            "Hydrated record at index 1 .* Missing columns: \\['content'\\]",
        ),
        (
            ["relative_path"],
            [
                {"relative_path": "alpha.txt"},
                {"relative_path": "alpha.txt", "content": "alpha"},
            ],
            "Hydrated record at index 1 .* Undeclared columns: \\['content'\\]",
        ),
    ],
    ids=["missing-column", "undeclared-column"],
)
def test_filesystem_seed_reader_validates_each_fanout_record_against_output_columns(
    tmp_path: Path,
    output_columns: list[str],
    hydrated_return: list[dict[str, str]],
    error_pattern: str,
) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")

    reader = ConfigurableHydrationDirectorySeedReader(
        output_columns=output_columns,
        hydrated_return=hydrated_return,
    )
    reader.attach(DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt"), PlaintextResolver())

    with pytest.raises(SeedReaderError, match=error_pattern):
        reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()


@pytest.mark.parametrize("use_batch_reader", [False, True], ids=["full-output", "batch-reader"])
def test_filesystem_seed_reader_raises_for_missing_declared_hydrated_columns(
    tmp_path: Path,
    use_batch_reader: bool,
) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("beta", encoding="utf-8")

    reader = MissingHydrationColumnSeedReader()
    reader.attach(DirectorySeedSource(path=str(tmp_path), file_pattern="*.txt"), PlaintextResolver())

    with pytest.raises(SeedReaderError, match="Missing columns: \\['content'\\]"):
        if use_batch_reader:
            batch_reader = reader.create_batch_reader(
                batch_size=2,
                index_range=IndexRange(start=0, end=1),
                shuffle=False,
            )
            batch_reader.read_next_batch().to_pandas()
        else:
            reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()


def test_filesystem_seed_reader_reuses_filesystem_context_until_reattach(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    first_dir.mkdir()
    (first_dir / "alpha.txt").write_text("alpha", encoding="utf-8")
    (first_dir / "beta.txt").write_text("beta", encoding="utf-8")

    second_dir = tmp_path / "second"
    second_dir.mkdir()
    (second_dir / "gamma.txt").write_text("gamma", encoding="utf-8")

    reader = ContextCountingDirectorySeedReader()

    reader.attach(DirectorySeedSource(path=str(first_dir), file_pattern="*.txt"), PlaintextResolver())

    assert reader.get_seed_dataset_size() == 2
    assert reader.filesystem_context_calls == 1

    batch_reader = reader.create_batch_reader(
        batch_size=1,
        index_range=IndexRange(start=0, end=0),
        shuffle=False,
    )
    assert list(batch_reader.read_next_batch().to_pandas()["relative_path"]) == ["alpha.txt"]
    assert reader.filesystem_context_calls == 1

    first_df = reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()
    assert list(first_df["relative_path"]) == ["alpha.txt", "beta.txt"]
    assert reader.filesystem_context_calls == 1

    reader.attach(DirectorySeedSource(path=str(second_dir), file_pattern="*.txt"), PlaintextResolver())

    assert reader.get_seed_dataset_size() == 1
    assert reader.filesystem_context_calls == 2


def test_seed_reader_reuses_cached_duckdb_connection_until_reattach() -> None:
    reader = CountingDataFrameSeedReader()
    reader.attach(DataFrameSeedSource(df=lazy.pd.DataFrame({"value": [1, 2, 3]})), PlaintextResolver())

    assert reader.get_seed_dataset_size() == 3
    assert reader.get_column_names() == ["value"]
    batch_reader = reader.create_batch_reader(
        batch_size=2,
        index_range=IndexRange(start=0, end=1),
        shuffle=False,
    )

    assert list(batch_reader.read_next_batch().to_pandas()["value"]) == [1, 2]
    assert reader.create_duckdb_connection_calls == 1

    reader.attach(DataFrameSeedSource(df=lazy.pd.DataFrame({"value": [9]})), PlaintextResolver())

    assert reader.get_seed_dataset_size() == 1
    assert reader.create_duckdb_connection_calls == 2


def test_agent_rollout_seed_reader_manifest_returns_file_count(tmp_path: Path, write_jsonl: WriteJsonl) -> None:
    _write_claude_trace_directory(tmp_path, write_jsonl)

    reader = AgentRolloutSeedReader()
    reader.attach(
        AgentRolloutSeedSource(
            path=str(tmp_path),
            format=AgentRolloutFormat.CLAUDE_CODE,
        ),
        PlaintextResolver(),
    )

    assert reader.get_seed_dataset_size() == 2


def test_agent_rollout_seed_reader_hydrates_to_record_count(tmp_path: Path, write_jsonl: WriteJsonl) -> None:
    _write_claude_trace_directory(tmp_path, write_jsonl)

    reader = TrackingAgentRolloutSeedReader()
    reader.attach(
        AgentRolloutSeedSource(
            path=str(tmp_path),
            format=AgentRolloutFormat.CLAUDE_CODE,
        ),
        PlaintextResolver(),
    )

    batch_reader = reader.create_batch_reader(
        batch_size=10,
        index_range=None,
        shuffle=False,
    )
    batch_df = batch_reader.read_next_batch().to_pandas()

    assert len(batch_df) == 2
    assert sorted(reader.hydrated_relative_paths) == ["project-a/session-1.jsonl", "project-a/session-2.jsonl"]


def test_agent_rollout_seed_reader_hydrates_atif_json_files(tmp_path: Path) -> None:
    _write_atif_trace_directory(tmp_path)

    reader = TrackingAgentRolloutSeedReader()
    reader.attach(
        AgentRolloutSeedSource(
            path=str(tmp_path),
            format=AgentRolloutFormat.ATIF,
        ),
        PlaintextResolver(),
    )

    batch_reader = reader.create_batch_reader(batch_size=10, index_range=None, shuffle=False)
    batch_df = batch_reader.read_next_batch().to_pandas().sort_values("trace_id").reset_index(drop=True)

    assert list(batch_df["trace_id"]) == ["atif-session-1", "atif-session-2"]
    assert list(batch_df["source_kind"]) == ["atif", "atif"]
    assert list(batch_df["final_assistant_message"]) == ["Repo inspected", "Tests checked"]
    assert sorted(reader.hydrated_relative_paths) == ["project-a/session-1.json", "project-a/session-2.json"]


def test_agent_rollout_seed_reader_hydration_laziness(tmp_path: Path, write_jsonl: WriteJsonl) -> None:
    _write_claude_trace_directory(tmp_path, write_jsonl)

    reader = AgentRolloutSeedReader()
    reader.attach(
        AgentRolloutSeedSource(
            path=str(tmp_path),
            format=AgentRolloutFormat.CLAUDE_CODE,
        ),
        PlaintextResolver(),
    )

    with patch("data_designer.engine.resources.agent_rollout.claude_code.load_jsonl_rows") as mock_load:
        reader.get_seed_dataset_size()
        mock_load.assert_not_called()


def test_agent_rollout_seed_reader_supports_hermes_json_and_jsonl(
    tmp_path: Path,
    write_jsonl: WriteJsonl,
) -> None:
    _write_hermes_trace_directory(tmp_path, write_jsonl)

    reader = TrackingAgentRolloutSeedReader()
    reader.attach(
        AgentRolloutSeedSource(
            path=str(tmp_path),
            format=AgentRolloutFormat.HERMES_AGENT,
        ),
        PlaintextResolver(),
    )

    assert reader.get_seed_dataset_size() == 2

    batch_reader = reader.create_batch_reader(batch_size=10, index_range=None, shuffle=False)
    batch_df = batch_reader.read_next_batch().to_pandas().sort_values("trace_id").reset_index(drop=True)

    assert list(batch_df["trace_id"]) == ["20260407_092759_baeaac", "gateway-session-1"]
    assert list(batch_df["source_kind"]) == ["hermes_agent", "hermes_agent"]
    assert sorted(reader.hydrated_relative_paths) == [
        "gateway-session-1.jsonl",
        "session_20260407_092759_baeaac.json",
    ]


def test_agent_rollout_seed_reader_wraps_os_errors_as_seed_reader_error(
    tmp_path: Path, write_jsonl: WriteJsonl
) -> None:
    session_dir = tmp_path / "project-a"
    write_jsonl(session_dir / "session.jsonl", [{"type": "user", "message": {"content": "Hi"}}])

    reader = AgentRolloutSeedReader()
    source = AgentRolloutSeedSource.model_construct(
        seed_type="agent_rollout",
        path=str(tmp_path),
        file_pattern=None,
        recursive=True,
        format=AgentRolloutFormat.CLAUDE_CODE,
    )
    reader.attach(source, PlaintextResolver())

    with patch(
        "data_designer.engine.resources.agent_rollout.claude_code.load_jsonl_rows",
        side_effect=OSError("permission denied"),
    ):
        with pytest.raises(SeedReaderError, match="Failed to read agent rollout file"):
            reader.create_duckdb_connection().execute(f"SELECT * FROM '{reader.get_dataset_uri()}'").df()


def test_agent_rollout_seed_reader_uses_resolved_file_pattern_when_model_construct_skips_validation(
    tmp_path: Path,
    write_jsonl: WriteJsonl,
) -> None:
    _write_claude_trace_directory(tmp_path, write_jsonl)

    source = AgentRolloutSeedSource.model_construct(
        seed_type="agent_rollout",
        path=str(tmp_path),
        file_pattern=None,
        recursive=True,
        format=AgentRolloutFormat.CLAUDE_CODE,
    )
    reader = AgentRolloutSeedReader()
    reader.attach(source, PlaintextResolver())

    assert reader.get_seed_dataset_size() == 2


def test_claude_session_index_scanning_respects_recursive_false(tmp_path: Path, write_jsonl: WriteJsonl) -> None:
    write_jsonl(
        tmp_path / "top-session.jsonl",
        [
            {"type": "user", "sessionId": "top-sess", "message": {"content": "Hello"}},
            {"type": "assistant", "sessionId": "top-sess", "message": {"content": [{"type": "text", "text": "Hi"}]}},
        ],
    )
    (tmp_path / "sessions-index.json").write_text(
        json.dumps({"entries": [{"sessionId": "top-sess", "projectPath": "/from-top-index"}]}),
        encoding="utf-8",
    )

    nested_dir = tmp_path / "project-a"
    write_jsonl(
        nested_dir / "session.jsonl",
        [
            {"type": "user", "sessionId": "nested-sess", "message": {"content": "Bye"}},
            {
                "type": "assistant",
                "sessionId": "nested-sess",
                "message": {"content": [{"type": "text", "text": "Goodbye"}]},
            },
        ],
    )
    (nested_dir / "sessions-index.json").write_text(
        json.dumps({"entries": [{"sessionId": "nested-sess", "projectPath": "/from-nested-index"}]}),
        encoding="utf-8",
    )

    reader = AgentRolloutSeedReader()
    reader.attach(
        AgentRolloutSeedSource(
            path=str(tmp_path),
            format=AgentRolloutFormat.CLAUDE_CODE,
            file_pattern="*.jsonl",
            recursive=True,
        ),
        PlaintextResolver(),
    )
    batch_reader = reader.create_batch_reader(batch_size=10, index_range=None, shuffle=False)
    recursive_df = batch_reader.read_next_batch().to_pandas()
    assert sorted(recursive_df["project_path"]) == ["/from-nested-index", "/from-top-index"]

    reader_non_recursive = AgentRolloutSeedReader()
    reader_non_recursive.attach(
        AgentRolloutSeedSource(
            path=str(tmp_path),
            format=AgentRolloutFormat.CLAUDE_CODE,
            file_pattern="*.jsonl",
            recursive=False,
        ),
        PlaintextResolver(),
    )
    batch_reader_nr = reader_non_recursive.create_batch_reader(batch_size=10, index_range=None, shuffle=False)
    non_recursive_df = batch_reader_nr.read_next_batch().to_pandas()
    assert list(non_recursive_df["project_path"]) == ["/from-top-index"]


def test_agent_rollout_seed_reader_gracefully_skips_malformed_files(tmp_path: Path, write_jsonl: WriteJsonl) -> None:
    session_dir = tmp_path / "project"
    write_jsonl(
        session_dir / "good.jsonl",
        [
            {"type": "user", "sessionId": "s1", "message": {"content": "Hello"}},
            {"type": "assistant", "sessionId": "s1", "message": {"content": [{"type": "text", "text": "Hi"}]}},
        ],
    )
    malformed_path = session_dir / "bad.jsonl"
    malformed_path.parent.mkdir(parents=True, exist_ok=True)
    malformed_path.write_text('{"type": "assistant", "message": "not-a-dict"}\n', encoding="utf-8")

    reader = AgentRolloutSeedReader()
    reader.attach(
        AgentRolloutSeedSource(
            path=str(tmp_path),
            format=AgentRolloutFormat.CLAUDE_CODE,
            file_pattern="*.jsonl",
        ),
        PlaintextResolver(),
    )

    batch_reader = reader.create_batch_reader(batch_size=10, index_range=None, shuffle=False)
    batch_df = batch_reader.read_next_batch().to_pandas()

    assert len(batch_df) == 1
    assert batch_df.iloc[0]["root_session_id"] == "s1"
