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
from data_designer.config.seed import IndexRange, PartitionBlock, SamplingStrategy
from data_designer.config.seed_source import (
    AgentRolloutFormat,
    AgentRolloutSeedSource,
    DirectorySeedSource,
    FileContentsSeedSource,
    HuggingFaceSeedSource,
)
from data_designer.engine.resources.seed_reader import (
    FileSystemSeedReader,
    SeedReaderError,
    SeedReaderFileSystemContext,
)
from data_designer.engine.secret_resolver import CompositeResolver, EnvironmentResolver, PlaintextResolver
from data_designer.engine.testing.seed_readers import LineFanoutDirectorySeedReader
from data_designer.engine.testing.stubs import StubHuggingFaceSeedReader
from data_designer.interface.data_designer import DataDesigner
from data_designer.interface.errors import DataDesignerGenerationError, DataDesignerProfilingError


class CustomDirectorySeedReader(FileSystemSeedReader[DirectorySeedSource]):
    output_columns = ["relative_path", "file_name", "decorated_path"]

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

    def hydrate_row(
        self,
        *,
        manifest_row: dict[str, Any],
        context: SeedReaderFileSystemContext,
    ) -> dict[str, str]:
        del context
        return {
            "relative_path": str(manifest_row["relative_path"]),
            "file_name": str(manifest_row["file_name"]),
            "decorated_path": f"custom::{manifest_row['relative_path']}",
        }


def _add_irrelevant_sampler_column(builder: DataDesignerConfigBuilder) -> None:
    builder.add_column(
        SamplerColumnConfig(
            name="irrelevant",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["irrelevant"]),
        )
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(f"{json.dumps(row)}\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
            {"type": "user", "sessionId": "session-1", "message": {"content": "Inspect the repo"}},
            {
                "type": "assistant",
                "sessionId": "session-1",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "Need to inspect"},
                        {"type": "tool_use", "id": "toolu_1", "name": "ReadFile", "input": {"path": "README.md"}},
                    ],
                },
            },
            {
                "type": "user",
                "sessionId": "session-1",
                "message": {
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "README contents"}],
                },
            },
            {
                "type": "assistant",
                "sessionId": "session-1",
                "message": {"content": [{"type": "text", "text": "Repo inspected"}]},
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
                "message": {"content": "Check tests"},
            },
            {
                "type": "assistant",
                "sessionId": "session-1",
                "agentId": "agent-a",
                "isSidechain": True,
                "message": {"content": [{"type": "text", "text": "Tests checked"}]},
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
                "type": "session_meta",
                "payload": {"id": "codex-session", "cwd": "/workspace"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "List files"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"cmd":"ls"}',
                    "call_id": "call_1",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "README.md\nsrc",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Listed files"}],
                },
            },
        ],
    )


def _write_atif_trace_directory(root_path: Path) -> None:
    trace_dir = root_path / "sessions"
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "session-1.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.6",
                "session_id": "atif-session-1",
                "agent": {
                    "name": "harbor-agent",
                    "model_name": "gpt-5",
                    "extra": {"cwd": "/workspace/project", "git_branch": "main"},
                },
                "steps": [
                    {
                        "step_id": 1,
                        "timestamp": "2026-04-06T12:00:00Z",
                        "source": "user",
                        "message": "Inspect the repository.",
                    },
                    {
                        "step_id": 2,
                        "timestamp": "2026-04-06T12:00:04Z",
                        "source": "agent",
                        "message": [{"type": "text", "text": "Repository inspected"}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_hermes_trace_directory(root_path: Path) -> None:
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
                "origin": {
                    "platform": "slack",
                    "chat_id": "C123",
                    "chat_name": "deployments",
                    "chat_type": "thread",
                    "thread_id": "thread-1",
                },
            }
        },
    )
    _write_jsonl(
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


def _write_claude_trace_directory_with_unhandled_files(root_path: Path) -> None:
    _write_claude_trace_directory(root_path)
    _write_jsonl(root_path / "project-a" / "history.jsonl", [{"type": "system"}])
    _write_jsonl(root_path / "project-a" / "tool-results" / "ignored.jsonl", [{"type": "system"}])


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

        with pytest.raises(
            DataDesignerGenerationError, match="🛑 Error generating dataset: Builder failed"
        ) as exc_info:
            data_designer.create(stub_sampler_only_config_builder, num_records=3)
        assert isinstance(exc_info.value.__cause__, RuntimeError)


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

        with pytest.raises(DataDesignerProfilingError, match="🛑 Error profiling dataset: Profiler failed") as exc_info:
            data_designer.create(stub_sampler_only_config_builder, num_records=3)
        assert isinstance(exc_info.value.__cause__, ValueError)


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
        ) as exc_info:
            data_designer.preview(stub_sampler_only_config_builder, num_records=3)
        assert isinstance(exc_info.value.__cause__, RuntimeError)


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
        ) as exc_info:
            data_designer.preview(stub_sampler_only_config_builder, num_records=3)
        assert isinstance(exc_info.value.__cause__, ValueError)


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
        with pytest.raises(DataDesignerGenerationError, match="Failed to load generated dataset") as exc_info:
            data_designer.create(stub_sampler_only_config_builder, num_records=1)
        assert isinstance(exc_info.value.__cause__, FileNotFoundError)


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
        "data_designer.engine.dataset_builders.dataset_builder.DatasetBuilder.process_preview",
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


def test_create_dataset_e2e_with_directory_seed_source(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "directory-seed"
    (seed_dir / "subdir").mkdir(parents=True)
    (seed_dir / "alpha.txt").write_text("alpha", encoding="utf-8")
    (seed_dir / "subdir" / "beta.md").write_text("beta", encoding="utf-8")

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(DirectorySeedSource(path=str(seed_dir)))
    builder.add_column(ExpressionColumnConfig(name="path_label", expr="{{ source_kind }}::{{ relative_path }}"))

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    results = data_designer.create(builder, num_records=2, dataset_name="directory-seed-test")
    df = results.load_dataset().sort_values("relative_path").reset_index(drop=True)

    assert list(df["source_kind"]) == ["directory_file", "directory_file"]
    assert list(df["relative_path"]) == ["alpha.txt", "subdir/beta.md"]
    assert list(df["file_name"]) == ["alpha.txt", "beta.md"]
    assert list(df["path_label"]) == [
        "directory_file::alpha.txt",
        "directory_file::subdir/beta.md",
    ]


def test_preview_dataset_e2e_with_directory_seed_source(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "directory-preview-seed"
    (seed_dir / "subdir").mkdir(parents=True)
    (seed_dir / "alpha.txt").write_text("alpha", encoding="utf-8")
    (seed_dir / "subdir" / "beta.txt").write_text("beta", encoding="utf-8")
    (seed_dir / "subdir" / "gamma.md").write_text("gamma", encoding="utf-8")

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(DirectorySeedSource(path=str(seed_dir), file_pattern="*.txt"))
    builder.add_column(ExpressionColumnConfig(name="path_label", expr="{{ source_kind }}::{{ relative_path }}"))

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    preview_results = data_designer.preview(builder, num_records=2)
    df = preview_results.dataset.sort_values("relative_path").reset_index(drop=True)

    assert list(df["source_kind"]) == ["directory_file", "directory_file"]
    assert list(df["relative_path"]) == ["alpha.txt", "subdir/beta.txt"]
    assert list(df["path_label"]) == [
        "directory_file::alpha.txt",
        "directory_file::subdir/beta.txt",
    ]


def test_create_dataset_e2e_with_file_contents_seed_source(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "file-contents-seed"
    seed_dir.mkdir(parents=True)
    (seed_dir / "alpha.txt").write_text("alpha", encoding="utf-8")
    (seed_dir / "beta.txt").write_text("beta", encoding="utf-8")

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(FileContentsSeedSource(path=str(seed_dir), file_pattern="*.txt"))
    builder.add_column(ExpressionColumnConfig(name="content_label", expr="{{ file_name }}::{{ content }}"))

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    results = data_designer.create(builder, num_records=2, dataset_name="file-contents-seed-test")
    df = results.load_dataset().sort_values("file_name").reset_index(drop=True)

    assert list(df["source_kind"]) == ["file_contents", "file_contents"]
    assert list(df["file_name"]) == ["alpha.txt", "beta.txt"]
    assert list(df["content"]) == ["alpha", "beta"]
    assert list(df["content_label"]) == [
        "alpha.txt::alpha",
        "beta.txt::beta",
    ]


def test_preview_dataset_e2e_with_file_contents_seed_source(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "file-contents-preview-seed"
    seed_dir.mkdir(parents=True)
    (seed_dir / "alpha.txt").write_text("alpha", encoding="utf-8")
    (seed_dir / "beta.txt").write_text("beta", encoding="utf-8")

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(FileContentsSeedSource(path=str(seed_dir), file_pattern="*.txt"))
    builder.add_column(ExpressionColumnConfig(name="content_label", expr="{{ file_name }}::{{ content }}"))

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    preview_results = data_designer.preview(builder, num_records=2)
    df = preview_results.dataset.sort_values("file_name").reset_index(drop=True)

    assert list(df["source_kind"]) == ["file_contents", "file_contents"]
    assert list(df["file_name"]) == ["alpha.txt", "beta.txt"]
    assert list(df["content_label"]) == [
        "alpha.txt::alpha",
        "beta.txt::beta",
    ]


def test_create_dataset_e2e_with_directory_seed_source_index_range_cycles_within_selection(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "directory-index-range-seed"
    seed_dir.mkdir(parents=True)
    for index in range(4):
        (seed_dir / f"file-{index}.txt").write_text(f"value-{index}", encoding="utf-8")

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(
        DirectorySeedSource(path=str(seed_dir), file_pattern="*.txt"),
        selection_strategy=IndexRange(start=1, end=2),
    )
    _add_irrelevant_sampler_column(builder)

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    results = data_designer.create(builder, num_records=5, dataset_name="directory-index-range-test")
    df = results.load_dataset().reset_index(drop=True)

    assert list(df["relative_path"]) == [
        "file-1.txt",
        "file-2.txt",
        "file-1.txt",
        "file-2.txt",
        "file-1.txt",
    ]


def test_create_dataset_e2e_with_file_contents_seed_source_partition_block_cycles_within_selection(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "file-contents-partition-seed"
    seed_dir.mkdir(parents=True)
    for index in range(6):
        (seed_dir / f"file-{index}.txt").write_text(f"value-{index}", encoding="utf-8")

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(
        FileContentsSeedSource(path=str(seed_dir), file_pattern="*.txt"),
        selection_strategy=PartitionBlock(index=1, num_partitions=3),
    )
    _add_irrelevant_sampler_column(builder)

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    results = data_designer.create(builder, num_records=5, dataset_name="file-contents-partition-test")
    df = results.load_dataset().reset_index(drop=True)

    assert list(df["relative_path"]) == [
        "file-2.txt",
        "file-3.txt",
        "file-2.txt",
        "file-3.txt",
        "file-2.txt",
    ]
    assert list(df["content"]) == [
        "value-2",
        "value-3",
        "value-2",
        "value-3",
        "value-2",
    ]


def test_create_dataset_e2e_with_file_contents_seed_source_shuffle_within_selection(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "file-contents-shuffle-seed"
    seed_dir.mkdir(parents=True)
    for index in range(6):
        (seed_dir / f"file-{index}.txt").write_text(f"value-{index}", encoding="utf-8")

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(
        FileContentsSeedSource(path=str(seed_dir), file_pattern="*.txt"),
        sampling_strategy=SamplingStrategy.SHUFFLE,
        selection_strategy=IndexRange(start=0, end=4),
    )
    _add_irrelevant_sampler_column(builder)

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    results = data_designer.create(builder, num_records=15, dataset_name="file-contents-shuffle-test")
    df = results.load_dataset().reset_index(drop=True)

    expected_paths = [f"file-{index}.txt" for index in range(5)]
    assert len(df) == 15
    assert set(df["relative_path"]) == set(expected_paths)
    assert list(df["relative_path"]) != expected_paths * 3


def test_preview_dataset_e2e_with_custom_filesystem_seed_reader_via_seed_readers_argument(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "custom-directory-reader"
    seed_dir.mkdir(parents=True)
    (seed_dir / "alpha.txt").write_text("alpha", encoding="utf-8")
    (seed_dir / "beta.txt").write_text("beta", encoding="utf-8")

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(DirectorySeedSource(path=str(seed_dir), file_pattern="*.txt"))
    _add_irrelevant_sampler_column(builder)

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
        seed_readers=[CustomDirectorySeedReader()],
    )

    preview_results = data_designer.preview(builder, num_records=2)
    df = preview_results.dataset.sort_values("relative_path").reset_index(drop=True)

    assert list(df["decorated_path"]) == [
        "custom::alpha.txt",
        "custom::beta.txt",
    ]


def test_create_dataset_e2e_with_custom_filesystem_seed_reader_fanout_partition_block_selection(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "custom-fanout-directory-reader"
    seed_dir.mkdir(parents=True)
    (seed_dir / "alpha.txt").write_text("alpha-0\nalpha-1", encoding="utf-8")
    (seed_dir / "beta.txt").write_text("beta-0\nbeta-1", encoding="utf-8")

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(
        DirectorySeedSource(path=str(seed_dir), file_pattern="*.txt"),
        selection_strategy=PartitionBlock(index=1, num_partitions=2),
    )
    _add_irrelevant_sampler_column(builder)

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
        seed_readers=[LineFanoutDirectorySeedReader()],
    )

    results = data_designer.create(builder, num_records=3, dataset_name="custom-fanout-directory-reader-test")
    df = results.load_dataset().reset_index(drop=True)

    assert list(df["relative_path"]) == ["beta.txt", "beta.txt", "beta.txt"]
    assert list(df["line_index"]) == [0, 1, 0]
    assert list(df["line"]) == ["beta-0", "beta-1", "beta-0"]


def test_create_dataset_e2e_with_custom_filesystem_seed_reader_selected_empty_fanout_raises_generation_error(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "custom-fanout-empty-selection-directory-reader"
    seed_dir.mkdir(parents=True)
    (seed_dir / "alpha.txt").write_text("alpha-0", encoding="utf-8")
    (seed_dir / "beta.txt").write_text("", encoding="utf-8")

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(
        DirectorySeedSource(path=str(seed_dir), file_pattern="*.txt"),
        selection_strategy=PartitionBlock(index=1, num_partitions=2),
    )
    _add_irrelevant_sampler_column(builder)

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
        seed_readers=[LineFanoutDirectorySeedReader()],
    )

    with pytest.raises(
        DataDesignerGenerationError,
        match="Selected manifest rows for seed source at .* did not produce any rows after hydration",
    ) as exc_info:
        data_designer.create(builder, num_records=1, dataset_name="custom-fanout-empty-selection-directory-reader")

    assert exc_info.value.__cause__ is not None
    assert "Selected manifest rows for seed source at" in str(exc_info.value.__cause__)


def test_create_dataset_e2e_with_directory_seed_source_no_matches_raises_generation_error(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "directory-no-matches-seed"
    seed_dir.mkdir(parents=True)
    (seed_dir / "alpha.txt").write_text("alpha", encoding="utf-8")

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(DirectorySeedSource(path=str(seed_dir), file_pattern="*.md"))
    _add_irrelevant_sampler_column(builder)

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    with pytest.raises(DataDesignerGenerationError, match="No files matched file_pattern '\\*\\.md'") as exc_info:
        data_designer.create(builder, num_records=1, dataset_name="directory-no-matches-test")
    assert isinstance(exc_info.value.__cause__, SeedReaderError)


def test_preview_dataset_e2e_with_directory_seed_source_no_matches_raises_generation_error(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "directory-preview-no-matches-seed"
    seed_dir.mkdir(parents=True)
    (seed_dir / "alpha.txt").write_text("alpha", encoding="utf-8")

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(DirectorySeedSource(path=str(seed_dir), file_pattern="*.md"))
    _add_irrelevant_sampler_column(builder)

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    with pytest.raises(DataDesignerGenerationError, match="No files matched file_pattern '\\*\\.md'") as exc_info:
        data_designer.preview(builder, num_records=1)
    assert isinstance(exc_info.value.__cause__, SeedReaderError)


def test_create_dataset_e2e_with_file_contents_seed_source_decode_failure_raises_generation_error(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "file-contents-decode-error-seed"
    seed_dir.mkdir(parents=True)
    (seed_dir / "latin1.txt").write_bytes("café".encode("latin-1"))

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(FileContentsSeedSource(path=str(seed_dir), file_pattern="*.txt"))
    _add_irrelevant_sampler_column(builder)

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    with pytest.raises(DataDesignerGenerationError, match="Failed to decode file"):
        data_designer.create(builder, num_records=1, dataset_name="file-contents-decode-error-test")


def test_create_dataset_e2e_with_file_contents_seed_source_unreadable_file_raises_generation_error(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "file-contents-permissions-seed"
    seed_dir.mkdir(parents=True)
    unreadable_path = seed_dir / "blocked.txt"
    unreadable_path.write_text("blocked", encoding="utf-8")
    unreadable_path.chmod(0)

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(FileContentsSeedSource(path=str(seed_dir), file_pattern="*.txt"))
    _add_irrelevant_sampler_column(builder)

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    try:
        with pytest.raises(DataDesignerGenerationError, match="Failed to read file"):
            data_designer.create(builder, num_records=1, dataset_name="file-contents-permissions-test")
    finally:
        unreadable_path.chmod(0o644)


@pytest.mark.parametrize(
    ("dir_name", "seed_source_factory", "writer", "expected_trace_ids", "expected_messages", "expected_tool_counts"),
    [
        (
            "atif",
            lambda path: AgentRolloutSeedSource(path=str(path), format=AgentRolloutFormat.ATIF),
            _write_atif_trace_directory,
            ["atif-session-1"],
            ["Repository inspected"],
            [0],
        ),
        (
            "claude-code",
            lambda path: AgentRolloutSeedSource(
                path=str(path),
                format=AgentRolloutFormat.CLAUDE_CODE,
            ),
            _write_claude_trace_directory,
            ["session-1", "session-1:agent-a"],
            ["Repo inspected", "Tests checked"],
            [1, 0],
        ),
        (
            "codex",
            lambda path: AgentRolloutSeedSource(path=str(path), format=AgentRolloutFormat.CODEX),
            _write_codex_trace_directory,
            ["codex-session"],
            ["Listed files"],
            [1],
        ),
        (
            "hermes-agent",
            lambda path: AgentRolloutSeedSource(path=str(path), format=AgentRolloutFormat.HERMES_AGENT),
            _write_hermes_trace_directory,
            ["20260407_092759_baeaac", "gateway-session-1"],
            ["Done.", "I'll inspect the logs."],
            [1, 1],
        ),
    ],
    ids=["atif", "claude-code", "codex", "hermes-agent"],
)
def test_create_dataset_e2e_with_trace_seed_sources(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
    dir_name: str,
    seed_source_factory: Any,
    writer: Any,
    expected_trace_ids: list[str],
    expected_messages: list[str],
    expected_tool_counts: list[int],
) -> None:
    trace_dir = tmp_path / dir_name
    writer(trace_dir)

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(seed_source_factory(trace_dir))
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
        dataset_name=f"trace-{dir_name}",
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

    if dir_name == "claude-code":
        assert list(df["source_kind"]) == ["claude_code", "claude_code"]
        assert lazy.pd.isna(df.iloc[0]["agent_id"])
        assert df.iloc[1]["agent_id"] == "agent-a"
        assert list(df["project_path"]) == ["/repo-from-index", "/repo-from-index"]
        assert list(df["is_sidechain"]) == [False, True]
    elif dir_name == "codex":
        assert list(df["source_kind"]) == ["codex"]
        assert list(df["cwd"]) == ["/workspace"]
    elif dir_name == "atif":
        assert list(df["source_kind"]) == ["atif"]
        assert list(df["cwd"]) == ["/workspace/project"]
        assert list(df["git_branch"]) == ["main"]
    elif dir_name == "hermes-agent":
        assert list(df["source_kind"]) == ["hermes_agent", "hermes_agent"]
        assert list(df["started_at"]) == ["2026-04-07T09:39:07.028463", "2026-04-07T08:00:00"]
        assert list(df["ended_at"]) == ["2026-04-07T09:51:07.905570", "2026-04-07T08:05:00"]
        assert list(df["source_meta"].map(lambda meta: meta["session_format"])) == [
            "cli_session_log",
            "gateway_transcript",
        ]


def test_create_dataset_warns_for_unhandled_transform_files(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    trace_dir = tmp_path / "claude-code-with-unhandled"
    _write_claude_trace_directory_with_unhandled_files(trace_dir)
    caplog.set_level(logging.WARNING)

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(AgentRolloutSeedSource(path=str(trace_dir), format=AgentRolloutFormat.CLAUDE_CODE))
    builder.add_column(ExpressionColumnConfig(name="assistant_copy", expr="{{ final_assistant_message }}"))

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    results = data_designer.create(builder, num_records=2, dataset_name="trace-unhandled-test")
    df = results.load_dataset().sort_values("trace_id").reset_index(drop=True)

    assert list(df["trace_id"]) == ["session-1", "session-1:agent-a"]
    assert "Skipping unhandled claude_code file" in caplog.text


def test_create_raises_error_when_all_trace_files_are_skipped(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "invalid-traces"
    session_dir = trace_dir / "project-a"
    _write_empty_jsonl(session_dir / "empty-1.jsonl")
    _write_empty_jsonl(session_dir / "empty-2.jsonl")

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(
        AgentRolloutSeedSource(
            path=str(trace_dir),
            format=AgentRolloutFormat.CLAUDE_CODE,
        )
    )
    builder.add_column(ExpressionColumnConfig(name="assistant_copy", expr="{{ final_assistant_message }}"))

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    with pytest.raises(DataDesignerGenerationError, match="did not produce any rows"):
        data_designer.create(builder, num_records=1, dataset_name="invalid-trace-seed")
