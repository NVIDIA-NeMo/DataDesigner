# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution-oriented session for the web UI.

Loads configs from files, runs validate/preview/create via the DataDesigner
interface, and caches results for the frontend to consume.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any

from data_designer.cli.utils.config_loader import load_config_builder
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.utils.io_helpers import serialize_data
from data_designer.config.utils.trace_type import TraceType

logger = logging.getLogger(__name__)

MAX_LOG_LINES = 500


class _LogCaptureHandler(logging.Handler):
    """Captures log records into a deque for the web UI to read."""

    def __init__(self, buffer: deque) -> None:
        super().__init__()
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._buffer.append({
                "ts": record.created,
                "level": record.levelname,
                "name": record.name.split(".")[-1],
                "message": self.format(record),
            })
        except Exception:
            pass


class ExecutionState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class ExecutionSession:
    """Manages config loading, SDG execution, and result caching for the web UI."""

    def __init__(self, *, config_dir: Path | None = None, config_path: Path | None = None) -> None:
        self._config_dir = config_dir or Path.cwd()
        self._config_path: Path | None = config_path
        self._builder: DataDesignerConfigBuilder | None = None

        self._exec_state = ExecutionState.IDLE
        self._exec_error: str | None = None
        self._exec_type: str | None = None

        self._preview_dataset: list[dict[str, Any]] | None = None
        self._preview_columns: list[str] | None = None
        self._preview_analysis: dict[str, Any] | None = None
        self._create_result: dict[str, Any] | None = None

        self._lock = threading.Lock()
        self._log_buffer: deque[dict[str, Any]] = deque(maxlen=MAX_LOG_LINES)
        self._log_handler: _LogCaptureHandler | None = None

        if config_path and config_path.exists():
            try:
                self.load_config(str(config_path))
            except Exception as e:
                logger.warning(f"Could not load config at startup: {e}")
                self._config_path = config_path

    # -- Config discovery & loading -----------------------------------------

    def list_configs(self) -> list[dict[str, str]]:
        """List config files (yaml/yml/json) in the config directory."""
        configs = []
        for ext in ("*.yaml", "*.yml", "*.json"):
            for p in sorted(self._config_dir.glob(ext)):
                if p.is_file():
                    configs.append({
                        "name": p.name,
                        "path": str(p),
                        "active": self._config_path is not None and p.resolve() == self._config_path.resolve(),
                    })
        return configs

    def load_config(self, path: str) -> dict[str, Any]:
        """Load a config from a file path. Returns the config as a dict."""
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = self._config_dir / resolved
        self._builder = load_config_builder(str(resolved))
        self._config_path = resolved
        self._preview_dataset = None
        self._preview_columns = None
        self._preview_analysis = None
        self._create_result = None
        self._exec_state = ExecutionState.IDLE
        self._exec_error = None
        return self.get_config_dict()

    def save_config_yaml(self, yaml_content: str) -> dict[str, Any]:
        """Validate YAML, write to disk, and reload the builder.

        Validates the config can be parsed BEFORE writing to disk so a
        typo doesn't corrupt the file.  On validation failure the file
        and builder are left untouched.
        """
        if not self._config_path:
            raise RuntimeError("No config file path set")

        import tempfile

        # Write to a temp file and try to load it first
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", dir=self._config_path.parent, delete=False
        )
        try:
            tmp.write(yaml_content)
            tmp.close()
            new_builder = load_config_builder(tmp.name)
        except Exception:
            Path(tmp.name).unlink(missing_ok=True)
            raise
        finally:
            Path(tmp.name).unlink(missing_ok=True)

        # Validation passed -- safe to write the real file
        self._config_path.write_text(yaml_content)
        self._builder = new_builder
        self._preview_dataset = None
        self._preview_columns = None
        self._preview_analysis = None
        self._create_result = None
        self._exec_state = ExecutionState.IDLE
        self._exec_error = None
        return self.get_config_dict()

    def get_raw_yaml(self) -> str:
        """Read the raw YAML from the config file on disk."""
        if self._config_path and self._config_path.exists():
            return self._config_path.read_text()
        return self.get_config_yaml()

    @property
    def is_loaded(self) -> bool:
        return self._builder is not None

    @property
    def config_path(self) -> str | None:
        return str(self._config_path) if self._config_path else None

    def get_config_dict(self) -> dict[str, Any]:
        if not self._builder:
            return {}
        return self._builder.get_builder_config().to_dict()

    def get_config_yaml(self) -> str:
        if not self._builder:
            return ""
        return self._builder.get_builder_config().to_yaml() or ""

    def get_config_json(self) -> str:
        if not self._builder:
            return "{}"
        return serialize_data(self.get_config_dict(), indent=2)

    def list_columns(self) -> list[dict[str, Any]]:
        if not self._builder:
            return []
        return [
            {"name": c.name, "column_type": c.column_type, "drop": getattr(c, "drop", False)}
            for c in self._builder.get_column_configs()
        ]

    def list_models(self) -> list[dict[str, Any]]:
        if not self._builder:
            return []
        return [mc.model_dump(mode="json") for mc in self._builder.model_configs]

    # -- Execution ----------------------------------------------------------

    @property
    def execution_status(self) -> dict[str, Any]:
        return {
            "state": self._exec_state.value,
            "type": self._exec_type,
            "error": self._exec_error,
            "has_preview": self._preview_dataset is not None,
            "has_create": self._create_result is not None,
        }

    def validate(self) -> dict[str, Any]:
        """Validate the loaded config via static compilation (no LLM calls)."""
        if not self._builder:
            raise RuntimeError("No config loaded")

        try:
            self._builder.build()
            return {"valid": True, "message": "Configuration is valid"}
        except Exception as e:
            return {"valid": False, "message": str(e)}

    def _start_log_capture(self) -> None:
        self._log_buffer.clear()
        handler = _LogCaptureHandler(self._log_buffer)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.INFO)
        root = logging.getLogger("data_designer")
        root.addHandler(handler)
        self._log_handler = handler

    def _stop_log_capture(self) -> None:
        if self._log_handler:
            root = logging.getLogger("data_designer")
            root.removeHandler(self._log_handler)
            self._log_handler = None

    def get_logs(self, since: float = 0) -> list[dict[str, Any]]:
        """Return log entries newer than `since` (unix timestamp)."""
        return [entry for entry in self._log_buffer if entry["ts"] > since]

    def run_preview(self, num_records: int = 10, debug_mode: bool = False) -> None:
        """Run preview in a background thread."""
        if not self._builder:
            raise RuntimeError("No config loaded")
        if self._exec_state == ExecutionState.RUNNING:
            raise RuntimeError("An execution is already in progress")

        self._exec_state = ExecutionState.RUNNING
        self._exec_type = "preview"
        self._exec_error = None
        self._start_log_capture()

        def _run():
            try:
                from data_designer.interface import DataDesigner

                builder = self._builder
                if debug_mode and builder:
                    builder = self._enable_traces(builder)

                dd = DataDesigner()
                results = dd.preview(builder, num_records=num_records)

                dataset = results.dataset
                if dataset is not None and len(dataset) > 0:
                    self._preview_columns = list(dataset.columns)
                    self._preview_dataset = dataset.to_dict(orient="records")
                else:
                    self._preview_columns = []
                    self._preview_dataset = []

                if results.analysis is not None:
                    self._preview_analysis = {
                        "column_statistics": [
                            s.model_dump(mode="json") for s in results.analysis.column_statistics
                        ]
                    }
                else:
                    self._preview_analysis = None

                self._exec_state = ExecutionState.DONE
            except Exception as e:
                logger.error(f"Preview failed: {e}")
                self._exec_state = ExecutionState.ERROR
                self._exec_error = str(e)
            finally:
                self._stop_log_capture()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def run_create(
        self,
        num_records: int = 100,
        dataset_name: str = "dataset",
        artifact_path: str | None = None,
    ) -> None:
        """Run full create in a background thread."""
        if not self._builder:
            raise RuntimeError("No config loaded")
        if self._exec_state == ExecutionState.RUNNING:
            raise RuntimeError("An execution is already in progress")

        self._exec_state = ExecutionState.RUNNING
        self._exec_type = "create"
        self._exec_error = None
        self._start_log_capture()

        def _run():
            try:
                from data_designer.interface import DataDesigner

                resolved_path = Path(artifact_path) if artifact_path else Path.cwd() / "artifacts"
                dd = DataDesigner(artifact_path=resolved_path)
                results = dd.create(
                    self._builder,
                    num_records=num_records,
                    dataset_name=dataset_name,
                )
                dataset = results.load_dataset()
                self._create_result = {
                    "num_records": len(dataset),
                    "artifact_path": str(results.artifact_storage.base_dataset_path),
                    "columns": list(dataset.columns),
                }
                self._exec_state = ExecutionState.DONE
            except Exception as e:
                logger.error(f"Create failed: {e}")
                self._exec_state = ExecutionState.ERROR
                self._exec_error = str(e)
            finally:
                self._stop_log_capture()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    # -- Results access -----------------------------------------------------

    def get_preview_results(self) -> dict[str, Any]:
        return {
            "columns": self._preview_columns or [],
            "rows": self._preview_dataset or [],
            "analysis": self._preview_analysis,
            "row_count": len(self._preview_dataset) if self._preview_dataset else 0,
        }

    def get_preview_trace(self, row: int, column: str) -> list[dict[str, Any]]:
        """Get trace data for a specific row/column from preview results."""
        if not self._preview_dataset or row >= len(self._preview_dataset):
            return []
        record = self._preview_dataset[row]
        trace_key = f"{column}__trace"
        trace = record.get(trace_key)
        if isinstance(trace, list):
            return trace
        return []

    def get_create_result(self) -> dict[str, Any]:
        return self._create_result or {}

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _enable_traces(builder: DataDesignerConfigBuilder) -> DataDesignerConfigBuilder:
        """Clone the builder with TraceType.ALL_MESSAGES on all LLM columns."""
        from data_designer.config.column_configs import (
            LLMCodeColumnConfig,
            LLMJudgeColumnConfig,
            LLMStructuredColumnConfig,
            LLMTextColumnConfig,
        )

        LLM_TYPES = (LLMTextColumnConfig, LLMCodeColumnConfig, LLMStructuredColumnConfig, LLMJudgeColumnConfig)

        for col in builder.get_column_configs():
            if isinstance(col, LLM_TYPES):
                col.with_trace = TraceType.ALL_MESSAGES
        return builder
