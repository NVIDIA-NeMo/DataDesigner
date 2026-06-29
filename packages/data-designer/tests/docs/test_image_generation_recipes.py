# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
IMAGE_RECIPE_NAMES = [
    "agriculture_crop_imagery",
    "airport_security_scans",
    "drone_aerial_inspection",
    "funny_pet_image_edits",
    "humanoid_robot_scene_understanding",
    "medical_extremity_xrays",
    "product_image_variations",
    "rich_document_images",
    "traffic_scenarios",
]


def _load_image_recipe(recipe_name: str) -> ModuleType:
    recipe_path = REPO_ROOT / f"fern/assets/recipes/image_generation/{recipe_name}.py"
    spec = importlib.util.spec_from_file_location(f"{recipe_name}_recipe", recipe_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("recipe_name", IMAGE_RECIPE_NAMES)
def test_image_generation_recipes_use_openrouter_chat_image_request_shape(recipe_name: str) -> None:
    recipe = _load_image_recipe(recipe_name)

    config = recipe.build_config().build()

    extra_body = config.model_configs[0].inference_parameters.extra_body
    assert set(extra_body) == {"modalities", "image_config"}
    assert extra_body["modalities"] == ["image", "text"]
    assert set(extra_body["image_config"]) == {"aspect_ratio", "image_size"}
    assert extra_body["image_config"]["image_size"] == "1K"
    assert "generationConfig" not in extra_body


@pytest.mark.parametrize(
    ("recipe_name", "subcategory_columns"),
    [
        ("agriculture_crop_imagery", {"severity": "disease_or_condition"}),
        ("drone_aerial_inspection", {"severity": "defect_or_event"}),
        ("product_image_variations", {"composition": "accessibility_context"}),
        ("traffic_scenarios", {"vehicle_mix": "traffic_density"}),
    ],
)
def test_image_generation_conditional_recipe_columns_use_subcategories(
    recipe_name: str,
    subcategory_columns: dict[str, str],
) -> None:
    recipe = _load_image_recipe(recipe_name)

    config = recipe.build_config().build()
    columns = {column.name: column for column in config.columns}

    assert "severity_sample" not in columns
    assert "composition_sample" not in columns
    assert "vehicle_mix_sample" not in columns
    for column_name, source_column_name in subcategory_columns.items():
        column = columns[column_name]
        assert column.sampler_type == "subcategory"
        assert column.params.category == source_column_name
        assert column.drop is False


def test_rich_document_export_handles_parquet_backed_image_lists(tmp_path: Path) -> None:
    recipe = _load_image_recipe("rich_document_images")
    dataset_path = tmp_path / "dataset"
    image_ref = Path("images/document_image/example.jpg")
    image_path = dataset_path / image_ref
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 6), color="red").save(image_path, format="JPEG")

    source_dataset = pd.DataFrame(
        {
            "document_image": [[str(image_ref)]],
            "document_type": ["quarterly business review"],
            "primary_visual": ["line chart"],
            "secondary_visual": ["KPI cards"],
            "layout_style": ["dashboard export"],
            "document_condition": ["pristine exported PDF screenshot"],
        }
    )
    roundtrip_path = tmp_path / "roundtrip.parquet"
    source_dataset.to_parquet(roundtrip_path, index=False)
    roundtripped_dataset = pd.read_parquet(roundtrip_path)
    results = SimpleNamespace(
        load_dataset=lambda: roundtripped_dataset,
        artifact_storage=SimpleNamespace(base_dataset_path=dataset_path),
    )

    output_path = tmp_path / "seed.parquet"
    recipe.export_seed_parquet(results, output_path)

    exported = pd.read_parquet(output_path)
    assert "image_base64" in exported.columns
    assert "png_base64" not in exported.columns
    assert base64.b64decode(exported.loc[0, "image_base64"]) == image_path.read_bytes()
    assert exported.loc[0, "image_format"] == "JPEG"
    assert exported.loc[0, "image_mime_type"] == "image/jpeg"
    assert exported.loc[0, "image_width"] == 8
    assert exported.loc[0, "image_height"] == 6
    assert exported.loc[0, "document_type"] == "quarterly business review"
