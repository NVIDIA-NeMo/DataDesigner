# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import types
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from data_designer.cli.main import app

runner = CliRunner()

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class _AlwaysTTY:
    def isatty(self) -> bool:
        return True


def _normalize_text(text: str) -> str:
    without_ansi = ANSI_ESCAPE_RE.sub("", text)
    return re.sub(r"\s+", " ", without_ansi).strip().lower()


def _write_usage_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "usage_config.py"
    config_path.write_text(
        """from __future__ import annotations

import data_designer.config as dd


def load_config_builder() -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder()
    builder.add_column(
        dd.SamplerColumnConfig(
            name="record_id",
            sampler_type=dd.SamplerType.UUID,
            params=dd.UUIDSamplerParams(),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="category",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["A", "B", "C"]),
        )
    )
    builder.add_column(
        dd.ExpressionColumnConfig(
            name="summary",
            expr="{{ category }}::{{ record_id }}",
        )
    )
    return builder
""",
        encoding="utf-8",
    )
    return config_path


def test_usage_preview_non_interactive_shows_records(tmp_path: Path) -> None:
    config_path = _write_usage_config(tmp_path)
    result = runner.invoke(
        app,
        ["preview", str(config_path), "--num-records", "3", "--non-interactive"],
        color=False,
    )

    normalized = _normalize_text(result.output)
    assert result.exit_code == 0
    assert "record 1 of 3" in normalized
    assert "record 3 of 3" in normalized
    assert "preview complete" in normalized


def test_usage_interactive_preview_navigation(tmp_path: Path) -> None:
    config_path = _write_usage_config(tmp_path)
    fake_sys = types.SimpleNamespace(stdin=_AlwaysTTY(), stdout=_AlwaysTTY())

    with (
        patch("data_designer.cli.controllers.generation_controller.sys", fake_sys),
        patch(
            "data_designer.cli.controllers.generation_controller.wait_for_navigation_key",
            side_effect=["n", "p", "q"],
        ),
    ):
        result = runner.invoke(
            app,
            ["preview", str(config_path), "--num-records", "3"],
            color=False,
        )

    normalized = _normalize_text(result.output)
    assert result.exit_code == 0
    assert "record 1 of 3" in normalized
    assert "record 2 of 3" in normalized
    assert "done browsing." in normalized


def test_usage_validate_unsupported_extension_is_actionable(tmp_path: Path) -> None:
    bad_config = tmp_path / "config.txt"
    bad_config.write_text("not supported", encoding="utf-8")

    result = runner.invoke(app, ["validate", str(bad_config)], color=False)
    normalized = _normalize_text(result.output)

    assert result.exit_code == 1
    assert "unsupported file extension" in normalized
    assert "supported extensions" in normalized


def test_usage_introspect_unknown_type_error_is_actionable() -> None:
    result = runner.invoke(app, ["inspect", "column", "nonexistent"], color=False)
    normalized = _normalize_text(result.output)

    assert result.exit_code == 1
    assert "error: unknown column_type" in normalized
    assert "available types:" in normalized
