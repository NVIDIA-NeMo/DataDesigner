# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pyarrow as pa
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
RECIPE_PATH = REPO_ROOT / "docs" / "assets" / "recipes" / "translation" / "indic_document_translation.py"
PLUGIN_PACKAGE = "data_designer_nemotron_indic_translation"
PLUGIN_INSTALLED = importlib.util.find_spec(PLUGIN_PACKAGE) is not None

requires_plugin = pytest.mark.skipif(not PLUGIN_INSTALLED, reason=f"needs the {PLUGIN_PACKAGE} plugin installed")


def _stub_document_translation_config(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**{"enable_windowing": False, **kwargs})


def _load_recipe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("indic_document_translation_recipe", RECIPE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    if PLUGIN_INSTALLED:
        spec.loader.exec_module(module)
        return module
    # CI has no plugin installed: load the recipe against a stub so its logic is still tested.
    package = ModuleType(PLUGIN_PACKAGE)
    package.__path__ = []
    config_module = ModuleType(f"{PLUGIN_PACKAGE}.config")
    config_module.DocumentTranslationConfig = _stub_document_translation_config  # type: ignore[attr-defined]
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, PLUGIN_PACKAGE, package)
        patch.setitem(sys.modules, f"{PLUGIN_PACKAGE}.config", config_module)
        spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recipe() -> ModuleType:
    return _load_recipe()


@pytest.fixture
def seed_path(tmp_path: Path) -> Path:
    path = tmp_path / "seed.jsonl"
    path.write_text('{"text": "The cat sat on the mat."}\n', encoding="utf-8")
    return path


class _StubBuilder:
    """Records what the recipe adds, since the package's pytest env disables plugin column registration."""

    def __init__(self, aliases: list[str]) -> None:
        self.model_configs = [SimpleNamespace(alias=alias) for alias in aliases]
        self.seed_source: Any = None
        self.columns: list[Any] = []

    def with_seed_dataset(self, seed_source: Any) -> _StubBuilder:
        self.seed_source = seed_source
        return self

    def add_column(self, column: Any) -> _StubBuilder:
        self.columns.append(column)
        return self


def _register_aliases(recipe: ModuleType, monkeypatch: pytest.MonkeyPatch, aliases: list[str]) -> None:
    monkeypatch.setattr(recipe.dd, "DataDesignerConfigBuilder", lambda: _StubBuilder(aliases))


@pytest.fixture
def registered_aliases(recipe: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    _register_aliases(recipe, monkeypatch, ["gemma", "bodhan"])


def _translation_column(recipe: ModuleType, seed_path: Path, **kwargs: Any) -> Any:
    # With the plugin installed, the real DocumentTranslationConfig validates every route and plugin warnings fail the test.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        builder = recipe.build_config(seed_path=str(seed_path), **kwargs)
    assert builder.seed_source.path == str(seed_path)
    assert len(builder.columns) == 1
    assert builder.columns[0].name == recipe.OUTPUT_COLUMN
    return builder.columns[0]


@pytest.mark.usefixtures("registered_aliases")
def test_default_build_uses_gemma_for_translation_and_judge(recipe: ModuleType, seed_path: Path) -> None:
    column = _translation_column(recipe, seed_path)

    assert column.model_alias == "gemma"
    assert column.evaluator_model_alias == "gemma"
    assert column.translation_mode == "generic"
    assert column.source_language == "en-Latn-IN"
    assert column.target_language == "hi-Deva-IN"
    assert column.text_column == "text"
    assert column.max_rewrites == 2
    assert column.rewrite_threshold == 75
    assert column.enable_windowing is False


@pytest.mark.usefixtures("registered_aliases")
def test_specialist_route_judges_with_gemma_and_runs_a_single_pass(recipe: ModuleType, seed_path: Path) -> None:
    column = _translation_column(
        recipe,
        seed_path,
        target_language="brx-Deva-IN",
        translation_mode="bodhan",
        model_alias="bodhan",
        judge_model_alias="gemma",
    )

    assert column.translation_mode == "bodhan"
    assert column.model_alias == "bodhan"
    assert column.evaluator_model_alias == "gemma"
    assert column.max_rewrites == 0


@pytest.mark.usefixtures("registered_aliases")
def test_windowing_route_merges_windows_with_the_judge_alias(recipe: ModuleType, seed_path: Path) -> None:
    column = _translation_column(
        recipe,
        seed_path,
        target_language="sat-Olck-IN",
        translation_mode="bodhan",
        model_alias="bodhan",
        judge_model_alias="gemma",
        enable_windowing=True,
        window_tokens=1024,
    )

    assert column.enable_windowing is True
    assert column.window_tokens == 1024
    assert column.window_merge_model_alias == "gemma"
    assert column.max_rewrites == 0


@pytest.mark.usefixtures("registered_aliases")
def test_script_remap_route_passes_the_target_script(recipe: ModuleType, seed_path: Path) -> None:
    column = _translation_column(
        recipe,
        seed_path,
        target_language="mni-Beng-IN",
        translation_mode="bodhan",
        model_alias="bodhan",
        judge_model_alias="gemma",
        target_language_script="Mtei",
    )

    assert column.target_language == "mni-Beng-IN"
    assert column.target_language_script == "Mtei"


@pytest.mark.usefixtures("registered_aliases")
def test_romanization_route_mirrors_rewrite_settings_on_the_intermediate_stage(
    recipe: ModuleType, seed_path: Path
) -> None:
    column = _translation_column(
        recipe,
        seed_path,
        target_language="hi-Latn-IN",
        intermediate_language="hi-Deva-IN",
        max_rewrites=1,
        rewrite_threshold=80,
    )

    assert column.intermediate_language == "hi-Deva-IN"
    assert column.max_rewrites_intermediate == 1
    assert column.rewrite_threshold_intermediate == 80


@requires_plugin
@pytest.mark.usefixtures("registered_aliases")
def test_recipe_builds_the_plugins_real_config_class(recipe: ModuleType, seed_path: Path) -> None:
    column = _translation_column(recipe, seed_path)

    assert type(column).__module__.startswith(PLUGIN_PACKAGE)
    assert column.column_type == "document-translation"


@requires_plugin
@pytest.mark.usefixtures("registered_aliases")
def test_plugin_validation_rejects_an_unsupported_script_remap(recipe: ModuleType, seed_path: Path) -> None:
    with pytest.raises(ValueError, match="No script remapper"):
        recipe.build_config(seed_path=str(seed_path), target_language="hi-Deva-IN", target_language_script="Mtei")


@pytest.mark.usefixtures("registered_aliases")
def test_native_numeral_probability_reaches_the_column(recipe: ModuleType, seed_path: Path) -> None:
    column = _translation_column(recipe, seed_path, native_numeral_probability=0.5)

    assert column.native_numeral_probability == 0.5


@pytest.mark.parametrize(
    ("max_rewrites", "translation_mode", "enable_windowing", "expected"),
    [
        (None, "generic", False, 2),
        (None, "bodhan", False, 0),
        (None, "generic", True, 0),
        (3, "generic", False, 3),
    ],
)
def test_resolve_max_rewrites(
    recipe: ModuleType, max_rewrites: int | None, translation_mode: str, enable_windowing: bool, expected: int
) -> None:
    assert recipe.resolve_max_rewrites(max_rewrites, translation_mode, enable_windowing) == expected


def test_unregistered_alias_lists_the_registered_ones(
    recipe: ModuleType, seed_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_aliases(recipe, monkeypatch, ["gemma"])

    with pytest.raises(ValueError, match="Register these model alias") as exc_info:
        recipe.build_config(seed_path=str(seed_path), model_alias="bodhan", judge_model_alias="gemma")

    assert "bodhan" in str(exc_info.value)
    assert "Registered aliases: gemma" in str(exc_info.value)


@pytest.mark.usefixtures("registered_aliases")
def test_missing_seed_file_is_rejected(recipe: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(Exception, match="seed"):
        recipe.build_config(seed_path=str(tmp_path / "missing.parquet"))


def test_arg_parser_defaults(recipe: ModuleType) -> None:
    args = recipe.build_arg_parser().parse_args(["--seed-path", "docs.parquet"])

    assert args.model_alias == "gemma"
    assert args.judge_model_alias is None
    assert args.target_language == "hi-Deva-IN"
    assert args.translation_mode == "generic"
    assert args.max_rewrites is None
    assert args.output_path is None


def test_arg_parser_requires_a_seed_path(recipe: ModuleType) -> None:
    with pytest.raises(SystemExit) as exc_info:
        recipe.build_arg_parser().parse_args([])

    assert exc_info.value.code == 2


def test_create_dataset_forwards_record_count_and_artifact_path(
    recipe: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_results = MagicMock()
    data_designer = MagicMock()
    data_designer.create.return_value = dataset_results
    data_designer_class = MagicMock(return_value=data_designer)
    monkeypatch.setattr(recipe, "DataDesigner", data_designer_class)
    config_builder = MagicMock()

    result = recipe.create_dataset(config_builder, num_records=7, artifact_path="artifacts")

    assert result is dataset_results
    data_designer_class.assert_called_once_with(artifact_path="artifacts")
    data_designer.create.assert_called_once_with(
        config_builder, num_records=7, dataset_name="indic_document_translation"
    )


# Data Designer returns judge output as Arrow struct columns; pandas cannot read these back
# from a parquet file that carries them in its pandas metadata.
EVALUATION_DTYPE = pd.ArrowDtype(
    pa.struct(
        [
            ("analysis", pa.string()),
            ("error_analysis", pa.list_(pa.null())),
            ("justification", pa.string()),
            ("overall_score", pa.int64()),
        ]
    )
)


class _FakeResults:
    def load_dataset(self) -> pd.DataFrame:
        evaluation = {"analysis": "ok", "error_analysis": [], "justification": "good", "overall_score": 92}
        return pd.DataFrame(
            {
                "text": ["hello"],
                "translation": ["नमस्ते"],
                "translation_evaluation": pd.Series([evaluation], dtype=EVALUATION_DTYPE),
                "translation_overall_score": [92],
            }
        )


def test_save_output_writes_parquet_and_creates_parent_directories(recipe: ModuleType, tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "out.parquet"

    recipe.save_output(_FakeResults(), output_path)

    written = pd.read_parquet(output_path)
    assert list(written["translation"]) == ["नमस्ते"]
    assert list(written["translation_overall_score"]) == [92]
    assert written["translation_evaluation"][0]["overall_score"] == 92


def test_save_output_writes_readable_indic_text_to_jsonl(recipe: ModuleType, tmp_path: Path) -> None:
    output_path = tmp_path / "out.jsonl"

    recipe.save_output(_FakeResults(), output_path)

    raw = output_path.read_text(encoding="utf-8")
    assert "नमस्ते" in raw
    assert "\\u" not in raw
    assert list(pd.read_json(output_path, lines=True)["translation"]) == ["नमस्ते"]


def test_main_rejects_unsupported_output_extension_before_generating(
    recipe: ModuleType, seed_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    create_dataset = MagicMock()
    monkeypatch.setattr(recipe, "create_dataset", create_dataset)
    monkeypatch.setattr(sys, "argv", ["recipe", "--seed-path", str(seed_path), "--output-path", "out.csv"])

    with pytest.raises(SystemExit) as exc_info:
        recipe.main()

    assert exc_info.value.code == 2
    assert "--output-path must end in one of .parquet, .jsonl" in capsys.readouterr().err
    create_dataset.assert_not_called()
