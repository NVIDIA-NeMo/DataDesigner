# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
RECIPE_PATH = REPO_ROOT / "docs" / "assets" / "recipes" / "workflow_chaining" / "document_review_gate.py"
SPEC = importlib.util.spec_from_file_location("document_review_gate", RECIPE_PATH)
assert SPEC is not None
document_review_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = document_review_gate
assert SPEC.loader is not None
SPEC.loader.exec_module(document_review_gate)


@pytest.fixture
def generated_pages(tmp_path: Path) -> pd.DataFrame:
    return document_review_gate.generate_sample_pages(tmp_path, count=4, seed=3)


def test_generate_sample_pages_creates_images(generated_pages: pd.DataFrame) -> None:
    for image_path in generated_pages["image_path"]:
        path = Path(image_path)
        assert path.exists()
        with Image.open(path) as image:
            assert image.size == (document_review_gate.PAGE_WIDTH, document_review_gate.PAGE_HEIGHT)


def test_generate_sample_pages_writes_valid_metadata(tmp_path: Path, generated_pages: pd.DataFrame) -> None:
    metadata_path = document_review_gate.metadata_path(tmp_path)

    assert metadata_path.exists()
    assert len(pd.read_parquet(metadata_path)) == len(generated_pages)
    document_review_gate.validate_metadata_rows(generated_pages)


def test_generate_sample_pages_ground_truth_boxes_fit_image_bounds(generated_pages: pd.DataFrame) -> None:
    for row in generated_pages.to_dict(orient="records"):
        with Image.open(row["image_path"]) as image:
            width, height = image.size
        for box in document_review_gate.parse_boxes(row["ground_truth_boxes"]):
            document_review_gate.validate_box(box, width, height)


def test_generate_sample_pages_labels_match_supported_set(generated_pages: pd.DataFrame) -> None:
    supported = set(document_review_gate.SUPPORTED_LABELS)

    for boxes in generated_pages["ground_truth_boxes"].map(document_review_gate.parse_boxes):
        assert {box["label"] for box in boxes} <= supported


def test_validate_metadata_rows_rejects_empty_ground_truth_boxes(generated_pages: pd.DataFrame) -> None:
    generated_pages.loc[0, "ground_truth_boxes"] = "[]"

    with pytest.raises(ValueError, match="Missing ground-truth boxes for page: synthetic-page-000"):
        document_review_gate.validate_metadata_rows(generated_pages)


def test_write_simulated_review_artifact_fills_selected_rows(tmp_path: Path) -> None:
    document_review_gate.run_to_review_stage(tmp_path, count=4, seed=3, review_pages=2)

    reviewed_path = document_review_gate.write_simulated_review_artifact(tmp_path)
    reviewed = pd.read_parquet(reviewed_path)
    selected = reviewed[reviewed["selected_for_review"]]
    skipped = reviewed[~reviewed["selected_for_review"]]

    assert len(selected) == 2
    assert all(selected["human_boxes"] == selected["ground_truth_boxes"])
    assert all(skipped["human_boxes"] == "[]")


def test_run_recipe_exports_final_dataset(tmp_path: Path) -> None:
    final_path = document_review_gate.run_recipe(tmp_path, count=4, seed=3, review_pages=2)

    final = pd.read_parquet(final_path)

    assert final_path == document_review_gate.final_dataset_path(tmp_path)
    assert len(final) == 4
    assert final["selected_for_review"].sum() == 2
    assert (final["source"] == "human_review").sum() == 2
    assert document_review_gate.review_candidates_path(tmp_path).exists()
    assert document_review_gate.reviewed_candidates_path(tmp_path).exists()


def test_run_recipe_rejects_existing_artifacts_without_overwrite(tmp_path: Path) -> None:
    base_dir = tmp_path / "existing"
    document_review_gate.generate_sample_pages(base_dir, count=1, seed=3)

    with pytest.raises(FileExistsError, match="Use --overwrite"):
        document_review_gate.run_recipe(base_dir, count=2, seed=4, review_pages=1)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_arg_parser_rejects_non_positive_num_records(value: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        document_review_gate.build_arg_parser().parse_args(["--num-records", value])

    assert exc_info.value.code == 2
