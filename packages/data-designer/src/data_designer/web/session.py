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

from data_designer.cli.repositories.mcp_provider_repository import MCPProviderRegistry, MCPProviderRepository
from data_designer.cli.utils.config_loader import load_config_builder
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider, MCPProviderT
from data_designer.config.utils.constants import DATA_DESIGNER_HOME
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


_REVIEW_SYSTEM_PROMPT = """\
You are a Data Designer config reviewer. Data Designer is a framework for generating synthetic datasets using samplers and LLMs.

Analyze the YAML config and return a JSON array of improvement tips. Each tip is an object with:
- "category": one of "prompt_design", "model_selection", "performance", "quality_gates", "pipeline_structure", "general"
- "severity": one of "info", "suggestion", "warning"
- "column": column name this applies to, or null if config-wide
- "tip": concise actionable suggestion (1-2 sentences)

Best practices to check against:

PROMPT DESIGN:
- Prompts should reference other columns via {{ column_name }} for row-level diversity
- Static prompts (no references) produce identical rows — always vary the input
- Avoid mega-prompts — decompose complex generation into multiple focused columns
- System prompts should set the role/constraints; user prompts should carry the variable content

MODEL SELECTION:
- Use reasoning models for logic-heavy tasks (math, code analysis, multi-step reasoning)
- Use text models for creative generation (stories, questions, paraphrasing)
- Temperature 0.0-0.3 for structured/deterministic output; 0.7-1.0 for creative diversity
- If no model_configs are specified, default models are used — this is fine for getting started

PERFORMANCE:
- Set max_tokens appropriate to expected output length — too high wastes tokens, too low truncates
- Increase max_parallel_requests (default 4) for throughput on large datasets
- Keep structured output schemas flat — deeply nested JSON schemas are model-sensitive

QUALITY GATES:
- Code generation columns should have a validation column to check syntax/execution
- Important outputs benefit from an llm-judge column for quality scoring
- Consider adding expression columns to combine or post-process outputs

PIPELINE STRUCTURE:
- Build hierarchical diversity: category → subcategory → content
- Downstream LLM columns should reference upstream column values
- Use sampler columns for controlled variability (topics, difficulty levels, personas)
- Expression columns are free (no LLM call) — use them for formatting and combining

Return ONLY the JSON array, no other text. If the config looks good, return an empty array [].\
"""


def _parse_llm_tips(text: str) -> list[dict[str, Any]]:
    """Parse LLM response text into a list of tip dicts."""
    import json

    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        tips = json.loads(text)
        if isinstance(tips, list):
            return [
                {
                    "category": t.get("category", "general"),
                    "severity": t.get("severity", "suggestion"),
                    "column": t.get("column"),
                    "tip": t.get("tip", ""),
                }
                for t in tips
                if isinstance(t, dict) and t.get("tip")
            ]
    except json.JSONDecodeError:
        pass

    # Fallback: return the raw text as a single tip
    if text:
        return [{"category": "general", "severity": "info", "column": None, "tip": text}]
    return []


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
        self._annotations: dict[int, dict[str, Any]] = {}

        self._mcp_repo = MCPProviderRepository(config_dir=DATA_DESIGNER_HOME)
        self._mcp_providers: list[MCPProviderT] = self._load_mcp_providers()

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
        self._annotations = {}
        self._exec_state = ExecutionState.IDLE
        self._exec_error = None
        self._load_annotations_from_disk()
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

    def get_output_schema(self) -> list[dict[str, Any]]:
        """Return the expected output schema: columns that will appear in the final dataset."""
        if not self._builder:
            return []
        schema = []
        for col in self._builder.get_column_configs():
            drop = getattr(col, "drop", False)
            schema.append({
                "name": col.name,
                "column_type": col.column_type,
                "drop": drop,
                "in_output": not drop,
            })
            for side_col in col.side_effect_columns:
                schema.append({
                    "name": side_col,
                    "column_type": col.column_type,
                    "drop": drop,
                    "in_output": not drop,
                    "side_effect_of": col.name,
                })
        return schema

    def list_models(self) -> list[dict[str, Any]]:
        if not self._builder:
            return []
        used_aliases = self._get_used_model_aliases()
        result = []
        for mc in self._builder.model_configs:
            d = mc.model_dump(mode="json")
            d["_used"] = mc.alias in used_aliases
            result.append(d)
        return result

    def _get_used_model_aliases(self) -> set[str]:
        """Return the set of model aliases actually referenced by columns."""
        if not self._builder:
            return set()
        aliases = set()
        for col in self._builder.get_column_configs():
            alias = getattr(col, "model_alias", None)
            if alias:
                aliases.add(alias)
        return aliases

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
        self._annotations = {}
        self._start_log_capture()

        def _run():
            try:
                from data_designer.interface import DataDesigner

                builder = self._builder
                if debug_mode and builder:
                    builder = self._enable_traces(builder)

                dd = DataDesigner(mcp_providers=self._mcp_providers or None)
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
                dd = DataDesigner(artifact_path=resolved_path, mcp_providers=self._mcp_providers or None)
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

    # -- Annotations --------------------------------------------------------

    def annotate_row(self, row: int, rating: str | None, note: str) -> None:
        """Set or update an annotation for a preview row."""
        self._annotations[row] = {"rating": rating, "note": note}
        self._save_annotations_to_disk()

    def get_annotations(self) -> dict[str, Any]:
        """Return all annotations keyed by row index (as strings for JSON)."""
        return {str(k): v for k, v in self._annotations.items()}

    def get_annotations_summary(self) -> dict[str, int]:
        total = len(self._preview_dataset) if self._preview_dataset else 0
        good = sum(1 for a in self._annotations.values() if a.get("rating") == "good")
        bad = sum(1 for a in self._annotations.values() if a.get("rating") == "bad")
        return {"good": good, "bad": bad, "unreviewed": total - good - bad, "total": total}

    def _annotations_file_path(self) -> Path | None:
        if not self._config_path:
            return None
        reviews_dir = self._config_dir / "reviews"
        return reviews_dir / f"{self._config_path.stem}_annotations.json"

    def _save_annotations_to_disk(self) -> None:
        path = self._annotations_file_path()
        if not path or not self._annotations:
            return
        try:
            import json
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._annotations, indent=2, default=str))
        except Exception as e:
            logger.warning(f"Failed to save annotations: {e}")

    def _load_annotations_from_disk(self) -> None:
        path = self._annotations_file_path()
        if not path or not path.exists():
            return
        try:
            import json
            data = json.loads(path.read_text())
            self._annotations = {int(k): v for k, v in data.items()}
        except Exception as e:
            logger.warning(f"Failed to load annotations: {e}")

    # -- MCP Provider Management --------------------------------------------

    def _load_mcp_providers(self) -> list[MCPProviderT]:
        registry = self._mcp_repo.load()
        return list(registry.providers) if registry else []

    def _save_mcp_providers(self) -> None:
        registry = MCPProviderRegistry(providers=self._mcp_providers)
        DATA_DESIGNER_HOME.mkdir(parents=True, exist_ok=True)
        self._mcp_repo.save(registry)

    def list_mcp_providers(self) -> list[dict[str, Any]]:
        return [p.model_dump(mode="json") for p in self._mcp_providers]

    def add_mcp_provider(self, data: dict[str, Any]) -> dict[str, Any]:
        provider_type = data.get("provider_type", "sse")
        if provider_type == "stdio":
            provider = LocalStdioMCPProvider.model_validate(data)
        else:
            provider = MCPProvider.model_validate(data)

        self._mcp_providers = [p for p in self._mcp_providers if p.name != provider.name]
        self._mcp_providers.append(provider)
        self._save_mcp_providers()
        return provider.model_dump(mode="json")

    def delete_mcp_provider(self, name: str) -> None:
        self._mcp_providers = [p for p in self._mcp_providers if p.name != name]
        self._save_mcp_providers()

    def get_required_providers(self) -> list[str]:
        if not self._builder:
            return []
        names: list[str] = []
        for tc in self._builder.tool_configs:
            names.extend(tc.providers)
        return sorted(set(names))

    def get_mcp_status(self) -> dict[str, Any]:
        required = self.get_required_providers()
        configured_names = {p.name for p in self._mcp_providers}
        providers_status = []
        for name in required:
            providers_status.append({
                "name": name,
                "configured": name in configured_names,
            })
        return {
            "required": providers_status,
            "configured": self.list_mcp_providers(),
            "all_satisfied": all(s["configured"] for s in providers_status),
        }

    # -- Config Review ------------------------------------------------------

    def review_config(self, model_alias: str) -> dict[str, Any]:
        """Run static analysis + LLM review of the loaded config."""
        if not self._builder:
            raise RuntimeError("No config loaded")

        static_issues = self._run_static_analysis()
        llm_tips = self._run_llm_review(model_alias)

        return {
            "static_issues": static_issues,
            "llm_tips": llm_tips,
            "model_used": model_alias,
        }

    def _run_static_analysis(self) -> list[dict[str, Any]]:
        from data_designer.engine.validation import validate_data_designer_config

        columns = self._builder.get_column_configs()
        processors = self._builder.get_processor_configs()
        allowed_refs = self._builder.allowed_references

        violations = validate_data_designer_config(
            columns=columns,
            processor_configs=processors,
            allowed_references=allowed_refs,
        )
        return [
            {
                "level": v.level.value,
                "type": v.type.value,
                "column": v.column,
                "message": v.message,
            }
            for v in violations
        ]

    def _run_llm_review(self, model_alias: str) -> list[dict[str, Any]]:
        try:
            from data_designer.interface import DataDesigner

            dd = DataDesigner()
            models = dd.get_models([model_alias])
            facade = models[model_alias]

            from data_designer.engine.models.utils import ChatMessage

            config_yaml = self.get_raw_yaml()
            system_msg = ChatMessage(role="system", content=_REVIEW_SYSTEM_PROMPT)
            user_msg = ChatMessage(
                role="user",
                content=f"Review this Data Designer config and provide improvement tips as a JSON array:\n\n```yaml\n{config_yaml}\n```",
            )

            response = facade.completion([system_msg, user_msg])
            text = response.choices[0].message.content or ""

            return _parse_llm_tips(text)
        except Exception as e:
            logger.error(f"LLM review failed: {e}")
            return [{"category": "error", "severity": "warning", "column": None, "tip": f"LLM review failed: {e}"}]

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
