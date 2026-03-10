# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

import data_designer.interface.data_designer as dd_mod
import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import ExpressionColumnConfig, SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.errors import InvalidConfigError
from data_designer.config.models import ModelProvider
from data_designer.config.processors import DropColumnsProcessorConfig
from data_designer.config.run_config import RunConfig
from data_designer.config.sampler_params import CategorySamplerParams, SamplerType
from data_designer.config.seed_source import HuggingFaceSeedSource, TraceSeedFormat, TraceSeedSource
from data_designer.engine.secret_resolver import CompositeResolver, EnvironmentResolver, PlaintextResolver
from data_designer.engine.testing.stubs import StubHuggingFaceSeedReader
from data_designer.interface.data_designer import DataDesigner
from data_designer.interface.errors import DataDesignerGenerationError, DataDesignerProfilingError


@pytest.fixture
def stub_artifact_path(tmp_path):
    """Temporary directory for artifacts."""
    return tmp_path / "artifacts"


@pytest.fixture
def stub_managed_assets_path(tmp_path):
    """Temporary directory for managed assets."""
    managed_path = tmp_path / "managed-assets"
    managed_path.mkdir(parents=True, exist_ok=True)
    return managed_path


@pytest.fixture
def stub_model_providers():
    return [
        ModelProvider(
            name="stub-model-provider",
            endpoint="https://api.stub-model-provider.com/v1",
            api_key="stub-model-provider-api-key",
        )
    ]


@pytest.fixture
def stub_seed_reader():
    return StubHuggingFaceSeedReader()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(f"{json.dumps(row)}\n")


def _write_invalid_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{invalid-json}\n", encoding="utf-8")


def _write_empty_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _write_claude_trace_directory(root_path: Path) -> None:
    session_dir = root_path / "project-a"
    subagents_dir = session_dir / "subagents"
    subagents_dir.mkdir(parents=True)

    _write_jsonl(
        session_dir / "session-1.jsonl",
        [
            {
                "type": "system",
                "sessionId": "session-1",
                "cwd": "/repo",
                "gitBranch": "main",
                "version": "2.1.7",
                "timestamp": "2026-01-01T00:00:00Z",
            },
            {
                "type": "user",
                "sessionId": "session-1",
                "cwd": "/repo",
                "gitBranch": "main",
                "timestamp": "2026-01-01T00:00:01Z",
                "message": {"role": "user", "content": "Inspect the repo"},
            },
            {
                "type": "assistant",
                "sessionId": "session-1",
                "cwd": "/repo",
                "gitBranch": "main",
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Need to inspect"},
                        {"type": "tool_use", "id": "toolu_1", "name": "ReadFile", "input": {"path": "README.md"}},
                    ],
                },
            },
            {
                "type": "user",
                "sessionId": "session-1",
                "cwd": "/repo",
                "gitBranch": "main",
                "timestamp": "2026-01-01T00:00:03Z",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "README contents"}],
                },
            },
            {
                "type": "assistant",
                "sessionId": "session-1",
                "cwd": "/repo",
                "gitBranch": "main",
                "timestamp": "2026-01-01T00:00:04Z",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Repo inspected"}]},
            },
        ],
    )
    _write_jsonl(
        subagents_dir / "agent-a.jsonl",
        [
            {
                "type": "user",
                "sessionId": "session-1",
                "agentId": "agent-a",
                "isSidechain": True,
                "cwd": "/repo",
                "gitBranch": "main",
                "timestamp": "2026-01-01T00:01:00Z",
                "message": {"role": "user", "content": "Check tests"},
            },
            {
                "type": "assistant",
                "sessionId": "session-1",
                "agentId": "agent-a",
                "isSidechain": True,
                "cwd": "/repo",
                "gitBranch": "main",
                "timestamp": "2026-01-01T00:01:01Z",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Tests checked"}]},
            },
        ],
    )
    (session_dir / "sessions-index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [
                    {
                        "sessionId": "session-1",
                        "projectPath": "/repo-from-index",
                        "summary": "Investigate repository",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_codex_trace_directory(root_path: Path) -> None:
    codex_dir = root_path / "sessions" / "2026" / "03" / "10"
    codex_dir.mkdir(parents=True)
    _write_jsonl(
        codex_dir / "rollout-2026-03-10T00-00-00-session.jsonl",
        [
            {
                "timestamp": "2026-03-10T00:00:00Z",
                "type": "session_meta",
                "payload": {
                    "id": "codex-session",
                    "timestamp": "2026-03-10T00:00:00Z",
                    "cwd": "/workspace",
                    "cli_version": "0.108.0",
                    "originator": "codex_cli_rs",
                    "model_provider": "openai",
                    "source": "api",
                },
            },
            {
                "timestamp": "2026-03-10T00:00:01Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Follow repo rules"}],
                },
            },
            {
                "timestamp": "2026-03-10T00:00:02Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "List files"}],
                },
            },
            {
                "timestamp": "2026-03-10T00:00:03Z",
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Need to run ls"}],
                },
            },
            {
                "timestamp": "2026-03-10T00:00:04Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"cmd":"ls"}',
                    "call_id": "call_1",
                },
            },
            {
                "timestamp": "2026-03-10T00:00:05Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "README.md\nsrc",
                },
            },
            {
                "timestamp": "2026-03-10T00:00:06Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Listed files"}],
                },
            },
        ],
    )


def _write_chat_completion_trace_directory(root_path: Path) -> None:
    _write_jsonl(
        root_path / "chat.jsonl",
        [
            {
                "trace_id": "row-1",
                "session_id": "sess-1",
                "split": "train",
                "file_line": 99,
                "messages": [
                    {"role": "developer", "content": "Use tools if needed"},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ],
            },
            {"prompt": "Question?", "completion": "Answer.", "file_line": 100},
        ],
    )


def _write_claude_trace_directory_with_skipped_files(root_path: Path) -> None:
    _write_claude_trace_directory(root_path)
    session_dir = root_path / "project-a"
    _write_empty_jsonl(session_dir / "empty.jsonl")
    _write_invalid_jsonl(session_dir / "malformed.jsonl")


def _write_codex_trace_directory_with_skipped_files(root_path: Path) -> None:
    _write_codex_trace_directory(root_path)
    codex_dir = root_path / "sessions" / "2026" / "03" / "10"
    _write_empty_jsonl(codex_dir / "rollout-empty.jsonl")
    _write_invalid_jsonl(codex_dir / "rollout-malformed.jsonl")


def _write_chat_completion_trace_directory_with_skipped_files(root_path: Path) -> None:
    _write_chat_completion_trace_directory(root_path)
    _write_empty_jsonl(root_path / "empty.jsonl")
    _write_jsonl(root_path / "unsupported.jsonl", [{"unexpected": "shape"}])


def test_init_with_custom_secret_resolver(stub_artifact_path, stub_model_providers):
    """Test DataDesigner initialization with custom secret resolver."""
    designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
    )
    assert designer is not None


def test_init_with_default_composite_secret_resolver(stub_artifact_path, stub_model_providers):
    """Test DataDesigner initialization with default composite secret resolver."""
    designer = DataDesigner(artifact_path=stub_artifact_path, model_providers=stub_model_providers)
    assert designer is not None
    assert isinstance(designer.secret_resolver, CompositeResolver)
    # Verify the composite resolver is properly configured with the expected resolvers
    resolvers = designer.secret_resolver.resolvers
    assert len(resolvers) == 2
    assert isinstance(resolvers[0], EnvironmentResolver)
    assert isinstance(resolvers[1], PlaintextResolver)


def test_init_with_string_path(stub_artifact_path, stub_model_providers):
    """Test DataDesigner accepts string paths."""
    designer = DataDesigner(artifact_path=str(stub_artifact_path), model_providers=stub_model_providers)
    assert designer is not None
    assert isinstance(designer._artifact_path, Path)


def test_init_with_path_object(stub_artifact_path, stub_model_providers):
    """Test DataDesigner accepts Path objects."""
    designer = DataDesigner(artifact_path=stub_artifact_path, model_providers=stub_model_providers)
    assert designer is not None


def test_run_config_setting_persists(stub_artifact_path, stub_model_providers):
    """Test that run config setting persists across multiple calls."""
    data_designer = DataDesigner(artifact_path=stub_artifact_path, model_providers=stub_model_providers)

    # Test default values
    assert data_designer._run_config.disable_early_shutdown is False
    assert data_designer._run_config.shutdown_error_rate == 0.5
    assert data_designer._run_config.shutdown_error_window == 10
    assert data_designer._run_config.buffer_size == 1000
    assert data_designer._run_config.max_conversation_restarts == 5
    assert data_designer._run_config.max_conversation_correction_steps == 0

    # Test setting custom values
    data_designer.set_run_config(
        RunConfig(
            disable_early_shutdown=True,
            shutdown_error_rate=0.8,
            shutdown_error_window=25,
            buffer_size=500,
            max_conversation_restarts=7,
            max_conversation_correction_steps=2,
        )
    )
    assert data_designer._run_config.disable_early_shutdown is True
    assert data_designer._run_config.shutdown_error_rate == 1.0  # normalized when disabled
    assert data_designer._run_config.shutdown_error_window == 25
    assert data_designer._run_config.buffer_size == 500
    assert data_designer._run_config.max_conversation_restarts == 7
    assert data_designer._run_config.max_conversation_correction_steps == 2

    # Test updating values
    data_designer.set_run_config(
        RunConfig(
            disable_early_shutdown=False,
            shutdown_error_rate=0.3,
            shutdown_error_window=5,
            buffer_size=750,
            max_conversation_restarts=9,
            max_conversation_correction_steps=1,
        )
    )
    assert data_designer._run_config.disable_early_shutdown is False
    assert data_designer._run_config.shutdown_error_rate == 0.3
    assert data_designer._run_config.shutdown_error_window == 5
    assert data_designer._run_config.buffer_size == 750
    assert data_designer._run_config.max_conversation_restarts == 9
    assert data_designer._run_config.max_conversation_correction_steps == 1


def test_run_config_normalizes_error_rate_when_disabled(stub_artifact_path, stub_model_providers):
    """Test that shutdown_error_rate is normalized to 1.0 when disabled."""
    data_designer = DataDesigner(artifact_path=stub_artifact_path, model_providers=stub_model_providers)

    # When enabled (default), shutdown_error_rate should use the configured value
    data_designer.set_run_config(
        RunConfig(
            disable_early_shutdown=False,
            shutdown_error_rate=0.7,
        )
    )
    assert data_designer._run_config.shutdown_error_rate == 0.7

    # When disabled, shutdown_error_rate should be normalized to 1.0
    data_designer.set_run_config(
        RunConfig(
            disable_early_shutdown=True,
            shutdown_error_rate=0.7,
        )
    )
    assert data_designer._run_config.shutdown_error_rate == 1.0


def test_run_config_rejects_invalid_buffer_size() -> None:
    with pytest.raises(ValidationError, match="buffer_size"):
        RunConfig(buffer_size=0)


def test_create_dataset_e2e_using_only_sampler_columns(
    stub_sampler_only_config_builder, stub_artifact_path, stub_model_providers, stub_managed_assets_path
):
    column_names = [config.name for config in stub_sampler_only_config_builder.get_column_configs()]

    num_records = 3

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    results = data_designer.create(stub_sampler_only_config_builder, num_records=num_records)

    df = results.load_dataset()
    assert len(df) == num_records
    assert set(df.columns) == set(column_names)

    # cycle through with no errors
    for _ in range(num_records + 2):
        results.display_sample_record()

    analysis = results.load_analysis()
    assert analysis.target_num_records == num_records

    # display report with no errors
    analysis.to_report()


@pytest.mark.parametrize(
    ("trace_format", "writer", "expected_trace_ids", "expected_messages", "expected_tool_counts"),
    [
        (
            TraceSeedFormat.CLAUDE_CODE_DIR,
            _write_claude_trace_directory,
            ["session-1", "session-1:agent-a"],
            ["Repo inspected", "Tests checked"],
            [1, 0],
        ),
        (
            TraceSeedFormat.CODEX_DIR,
            _write_codex_trace_directory,
            ["codex-session"],
            ["Listed files"],
            [1],
        ),
        (
            TraceSeedFormat.CHAT_COMPLETION_JSONL_DIR,
            _write_chat_completion_trace_directory,
            ["chat:2", "row-1"],
            ["Answer.", "Hello"],
            [0, 0],
        ),
    ],
    ids=["claude-code", "codex", "chat-completion-jsonl"],
)
def test_create_dataset_e2e_with_trace_seed_source(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
    trace_format: TraceSeedFormat,
    writer: Any,
    expected_trace_ids: list[str],
    expected_messages: list[str],
    expected_tool_counts: list[int],
) -> None:
    trace_dir = tmp_path / trace_format.value
    writer(trace_dir)

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(TraceSeedSource(path=str(trace_dir), format=trace_format))
    builder.add_column(ExpressionColumnConfig(name="assistant_copy", expr="{{ final_assistant_message }}"))
    builder.add_column(ExpressionColumnConfig(name="trace_label", expr="{{ source_kind }}::{{ trace_id }}"))

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    results = data_designer.create(
        builder,
        num_records=len(expected_trace_ids),
        dataset_name=f"trace-{trace_format.value}",
    )
    df = results.load_dataset().sort_values("trace_id").reset_index(drop=True)

    assert list(df["trace_id"]) == expected_trace_ids
    assert list(df["assistant_copy"]) == expected_messages
    assert list(df["tool_call_count"]) == expected_tool_counts
    assert list(df["trace_label"]) == [
        f"{source_kind}::{trace_id}"
        for source_kind, trace_id in df[["source_kind", "trace_id"]].itertuples(index=False)
    ]
    assert "messages" in df.columns
    assert "_internal_row_id" not in df.columns

    if trace_format == TraceSeedFormat.CLAUDE_CODE_DIR:
        assert list(df["source_kind"]) == ["claude_code", "claude_code"]
        assert lazy.pd.isna(df.iloc[0]["agent_id"])
        assert df.iloc[1]["agent_id"] == "agent-a"
        assert list(df["project_path"]) == ["/repo-from-index", "/repo-from-index"]
        assert list(df["is_sidechain"]) == [False, True]
    elif trace_format == TraceSeedFormat.CODEX_DIR:
        assert list(df["source_kind"]) == ["codex"]
        assert list(df["cwd"]) == ["/workspace"]
    else:
        assert list(df["source_kind"]) == ["chat_completion_jsonl", "chat_completion_jsonl"]
        assert list(df["root_session_id"]) == ["chat:2", "sess-1"]
        row_source_meta = dict(df[df["trace_id"] == "row-1"].iloc[0]["source_meta"])
        assert row_source_meta["file_line"] == "99"
        assert row_source_meta["source_file_line"] == "1"


@pytest.mark.parametrize(
    ("trace_format", "writer", "expected_trace_ids"),
    [
        (
            TraceSeedFormat.CLAUDE_CODE_DIR,
            _write_claude_trace_directory_with_skipped_files,
            ["session-1", "session-1:agent-a"],
        ),
        (
            TraceSeedFormat.CODEX_DIR,
            _write_codex_trace_directory_with_skipped_files,
            ["codex-session"],
        ),
        (
            TraceSeedFormat.CHAT_COMPLETION_JSONL_DIR,
            _write_chat_completion_trace_directory_with_skipped_files,
            ["chat:2", "row-1"],
        ),
    ],
    ids=["claude-code", "codex", "chat-completion-jsonl"],
)
def test_create_dataset_skips_empty_and_malformed_trace_files(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    trace_format: TraceSeedFormat,
    writer: Any,
    expected_trace_ids: list[str],
) -> None:
    trace_dir = tmp_path / f"{trace_format.value}-with-skips"
    writer(trace_dir)
    caplog.set_level(logging.WARNING)

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(TraceSeedSource(path=str(trace_dir), format=trace_format))
    builder.add_column(ExpressionColumnConfig(name="assistant_copy", expr="{{ final_assistant_message }}"))

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    results = data_designer.create(builder, num_records=len(expected_trace_ids), dataset_name="trace-skip-test")
    df = results.load_dataset().sort_values("trace_id").reset_index(drop=True)

    assert list(df["trace_id"]) == expected_trace_ids
    assert "Skipping empty" in caplog.text
    assert "Skipping malformed" in caplog.text


def test_create_raises_error_when_all_trace_files_are_skipped(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "invalid-traces"
    _write_empty_jsonl(trace_dir / "empty.jsonl")
    _write_jsonl(trace_dir / "unsupported.jsonl", [{"unexpected": "shape"}])

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(TraceSeedSource(path=str(trace_dir), format=TraceSeedFormat.CHAT_COMPLETION_JSONL_DIR))
    builder.add_column(ExpressionColumnConfig(name="assistant_copy", expr="{{ final_assistant_message }}"))

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    with pytest.raises(DataDesignerGenerationError, match="No valid chat-completion JSONL files found"):
        data_designer.create(builder, num_records=1, dataset_name="invalid-trace-seed")


def test_create_raises_error_when_builder_fails(
    stub_artifact_path, stub_model_providers, stub_sampler_only_config_builder, stub_managed_assets_path
):
    """Test that create method raises DataDesignerCreateError when builder.build fails."""
    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    with patch.object(data_designer, "_create_dataset_builder") as mock_builder_method:
        mock_builder = MagicMock()
        mock_builder.build.side_effect = RuntimeError("Builder failed")
        mock_builder_method.return_value = mock_builder

        with pytest.raises(DataDesignerGenerationError, match="🛑 Error generating dataset: Builder failed"):
            data_designer.create(stub_sampler_only_config_builder, num_records=3)


def test_create_raises_error_when_profiler_fails(
    stub_artifact_path, stub_model_providers, stub_sampler_only_config_builder, stub_managed_assets_path
):
    """Test that create method raises DataDesignerCreateError when profiler.profile_dataset fails."""
    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    with (
        patch.object(data_designer, "_create_dataset_builder") as mock_builder_method,
        patch.object(data_designer, "_create_dataset_profiler") as mock_profiler_method,
    ):
        # Mock builder to succeed
        mock_builder = MagicMock()
        mock_builder.build.return_value = None
        mock_builder.artifact_storage.load_dataset_with_dropped_columns.return_value = lazy.pd.DataFrame(
            {"col": [1, 2, 3]}
        )
        mock_builder_method.return_value = mock_builder

        # Mock profiler to fail
        mock_profiler = MagicMock()
        mock_profiler.profile_dataset.side_effect = ValueError("Profiler failed")
        mock_profiler_method.return_value = mock_profiler

        with pytest.raises(DataDesignerProfilingError, match="🛑 Error profiling dataset: Profiler failed"):
            data_designer.create(stub_sampler_only_config_builder, num_records=3)


def test_preview_raises_error_when_builder_fails(
    stub_artifact_path, stub_model_providers, stub_sampler_only_config_builder, stub_managed_assets_path
):
    """Test that preview method raises DataDesignerPreviewError when builder.build_preview fails."""
    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    with patch.object(data_designer, "_create_dataset_builder") as mock_builder_method:
        mock_builder = MagicMock()
        mock_builder.build_preview.side_effect = RuntimeError("Builder preview failed")
        mock_builder_method.return_value = mock_builder

        with pytest.raises(
            DataDesignerGenerationError, match="🛑 Error generating preview dataset: Builder preview failed"
        ):
            data_designer.preview(stub_sampler_only_config_builder, num_records=3)


def test_preview_raises_error_when_profiler_fails(
    stub_artifact_path, stub_model_providers, stub_sampler_only_config_builder, stub_managed_assets_path
):
    """Test that preview method raises DataDesignerPreviewError when profiler.profile_dataset fails."""
    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    with (
        patch.object(data_designer, "_create_dataset_builder") as mock_builder_method,
        patch.object(data_designer, "_create_dataset_profiler") as mock_profiler_method,
    ):
        # Mock builder to succeed
        mock_builder = MagicMock()
        mock_builder.build_preview.return_value = lazy.pd.DataFrame({"col": [1, 2, 3]})
        mock_builder.process_preview.return_value = lazy.pd.DataFrame({"col": [1, 2, 3]})
        mock_builder_method.return_value = mock_builder

        # Mock profiler to fail
        mock_profiler = MagicMock()
        mock_profiler.profile_dataset.side_effect = ValueError("Profiler failed in preview")
        mock_profiler_method.return_value = mock_profiler

        with pytest.raises(
            DataDesignerProfilingError, match="🛑 Error profiling preview dataset: Profiler failed in preview"
        ):
            data_designer.preview(stub_sampler_only_config_builder, num_records=3)


def test_create_raises_generation_error_when_dataset_is_empty(
    stub_artifact_path, stub_model_providers, stub_sampler_only_config_builder, stub_managed_assets_path
):
    """When all records are dropped during generation, create should raise
    DataDesignerGenerationError with a clear message instead of a misleading profiler error.
    """
    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    with patch(
        "data_designer.engine.storage.artifact_storage.ArtifactStorage.load_dataset_with_dropped_columns",
        return_value=lazy.pd.DataFrame(),
    ):
        with pytest.raises(DataDesignerGenerationError, match="Dataset is empty"):
            data_designer.create(stub_sampler_only_config_builder, num_records=1)


def test_create_raises_generation_error_when_load_dataset_fails(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_sampler_only_config_builder: DataDesignerConfigBuilder,
    stub_managed_assets_path: Path,
) -> None:
    """When no parquet was written (e.g. all records dropped), load_dataset_with_dropped_columns
    raises an exception. create() should surface this as DataDesignerGenerationError, not
    DataDesignerProfilingError.
    """
    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    with patch(
        "data_designer.engine.storage.artifact_storage.ArtifactStorage.load_dataset_with_dropped_columns",
        side_effect=FileNotFoundError("No parquet files found"),
    ):
        with pytest.raises(DataDesignerGenerationError, match="Failed to load generated dataset"):
            data_designer.create(stub_sampler_only_config_builder, num_records=1)


def test_preview_raises_generation_error_when_dataset_is_empty(
    stub_artifact_path, stub_model_providers, stub_sampler_only_config_builder, stub_managed_assets_path
):
    """When all records are dropped during generation, preview should raise
    DataDesignerGenerationError with a clear message instead of a misleading profiler error.
    """
    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    with patch(
        "data_designer.engine.dataset_builders.column_wise_builder.ColumnWiseDatasetBuilder.process_preview",
        return_value=lazy.pd.DataFrame(),
    ):
        with pytest.raises(DataDesignerGenerationError, match="Dataset is empty"):
            data_designer.preview(stub_sampler_only_config_builder, num_records=1)


def test_preview_with_dropped_columns(
    stub_artifact_path, stub_model_providers, stub_model_configs, stub_managed_assets_path
):
    """Test that preview correctly handles dropped columns and maintains consistency."""
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(
        SamplerColumnConfig(
            name="uuid", sampler_type="uuid", params={"prefix": "id_", "short_form": True, "uppercase": False}
        )
    )
    config_builder.add_column(
        SamplerColumnConfig(name="category", sampler_type="category", params={"values": ["a", "b", "c"]})
    )
    config_builder.add_column(
        SamplerColumnConfig(name="uniform", sampler_type="uniform", params={"low": 1, "high": 100})
    )

    config_builder.add_processor(DropColumnsProcessorConfig(name="drop_columns_processor", column_names=["category"]))

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    num_records = 5
    preview_results = data_designer.preview(config_builder, num_records=num_records)

    preview_dataset = preview_results.dataset

    assert "category" not in preview_dataset.columns, "Dropped column 'category' should not be in preview dataset"

    assert "uuid" in preview_dataset.columns, "Column 'uuid' should be in preview dataset"
    assert "uniform" in preview_dataset.columns, "Column 'uniform' should be in preview dataset"

    assert len(preview_dataset) == num_records, f"Preview dataset should have {num_records} records"

    analysis = preview_results.analysis
    assert analysis is not None, "Analysis should be generated"

    column_names_in_analysis = [stat.column_name for stat in analysis.column_statistics]
    assert "uuid" in column_names_in_analysis, "Column 'uuid' should be in analysis"
    assert "uniform" in column_names_in_analysis, "Column 'uniform' should be in analysis"
    assert "category" not in column_names_in_analysis, "Dropped column 'category' should not be in analysis statistics"

    assert analysis.side_effect_column_names is not None, "Side effect column names should be tracked"
    assert "category" in analysis.side_effect_column_names, (
        "Dropped column 'category' should be tracked in side_effect_column_names"
    )


def test_validate_raises_error_when_seed_collides(
    stub_artifact_path,
    stub_model_providers,
    stub_model_configs,
    stub_managed_assets_path,
    stub_seed_reader,
):
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))
    config_builder.add_column(
        SamplerColumnConfig(
            name="city",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["new york", "los angeles"]),
        )
    )

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
        seed_readers=[stub_seed_reader],
    )

    with pytest.raises(InvalidConfigError):
        data_designer.validate(config_builder)


def test_initialize_interface_runtime_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """_initialize_interface_runtime only runs initialization once."""
    monkeypatch.setattr(dd_mod, "_interface_runtime_initialized", False)

    with (
        patch("data_designer.interface.data_designer.configure_logging") as mock_logging,
        patch("data_designer.interface.data_designer.resolve_seed_default_model_settings") as mock_resolve,
    ):
        dd_mod._initialize_interface_runtime()
        dd_mod._initialize_interface_runtime()
        mock_logging.assert_called_once()
        mock_resolve.assert_called_once()
