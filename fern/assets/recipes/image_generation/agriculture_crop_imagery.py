# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
# ]
# ///
"""Agriculture Crop Disease Detection Image Recipe

Generate synthetic crop disease detection images with controlled variation over
crop type, growth stage, viewpoint, disease or confounding condition, severity,
weather, irrigation, and field condition. The objective is to create examples
where the expected crop-health label is known, including healthy negatives and
hard confounders, so teams can evaluate detection prompts, build labeling
rubrics, calibrate reviewers, and prototype crop-disease workflows before using
governed field imagery.

Prerequisites:
    - An image-generation provider key for the selected model. The defaults use
      OpenRouter, so set OPENROUTER_API_KEY before running.

Run:
    uv run agriculture_crop_imagery.py --num-records 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults

DEFAULT_MODEL_PROVIDER = "openrouter"
DEFAULT_MODEL_ID = "google/gemini-3.1-flash-image-preview"
DEFAULT_MODEL_ALIAS = "agriculture-image-model"


def build_model_configs(
    *,
    model_provider: str,
    model_id: str,
    model_alias: str,
    image_size: str,
    aspect_ratio: str,
    max_parallel_requests: int,
) -> list[dd.ModelConfig]:
    return [
        dd.ModelConfig(
            alias=model_alias,
            model=model_id,
            provider=model_provider,
            inference_parameters=dd.ImageInferenceParams(
                extra_body={
                    "modalities": ["image", "text"],
                    "image_config": {
                        "aspect_ratio": aspect_ratio,
                        "image_size": image_size,
                    },
                },
                max_parallel_requests=max_parallel_requests,
            ),
            skip_health_check=True,
        )
    ]


def add_category(config_builder: dd.DataDesignerConfigBuilder, name: str, values: list[str]) -> None:
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name=name,
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=values),
        )
    )


def add_visual_variation_id(config_builder: dd.DataDesignerConfigBuilder) -> None:
    """Add a unique row-level key that discourages duplicate image generations."""
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="visual_variation_id",
            sampler_type=dd.SamplerType.UUID,
            params=dd.UUIDSamplerParams(prefix="crop-", short_form=True),
        )
    )


def build_config(
    *,
    model_provider: str = DEFAULT_MODEL_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    model_alias: str = DEFAULT_MODEL_ALIAS,
    image_size: str = "1K",
    aspect_ratio: str = "4:3",
    max_parallel_requests: int = 10,
) -> dd.DataDesignerConfigBuilder:
    model_configs = build_model_configs(
        model_provider=model_provider,
        model_id=model_id,
        model_alias=model_alias,
        image_size=image_size,
        aspect_ratio=aspect_ratio,
        max_parallel_requests=max_parallel_requests,
    )
    config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)
    add_visual_variation_id(config_builder)

    add_category(
        config_builder,
        "crop_type",
        [
            "corn",
            "soybean",
            "wheat",
            "rice",
            "tomato",
            "grape vineyard",
            "apple orchard",
            "lettuce",
            "potato",
            "strawberry",
        ],
    )
    add_category(
        config_builder,
        "growth_stage",
        [
            "seedling",
            "vegetative growth",
            "flowering",
            "fruiting",
            "grain fill",
            "near harvest",
        ],
    )
    add_category(
        config_builder,
        "viewpoint",
        [
            "close-up leaf-level scouting photo",
            "row-level field photo",
            "drone oblique field view",
            "top-down drone crop-row view",
            "greenhouse bench view",
            "orchard row view",
        ],
    )
    add_category(
        config_builder,
        "disease_or_condition",
        [
            "healthy crop with no visible disease",
            "powdery mildew on leaves",
            "rust-colored fungal pustules on leaf surfaces",
            "early blight with concentric brown leaf spots",
            "late blight with irregular dark lesions",
            "bacterial leaf spot with small dark speckles",
            "downy mildew patches on leaf undersides",
            "leaf curl with mosaic discoloration",
            "insect feeding damage as a disease confounder",
            "nutrient deficiency yellowing as a disease confounder",
        ],
    )
    disease_severity_values = [
        "low severity affecting isolated plants",
        "moderate severity affecting patches",
        "high severity affecting large field sections",
    ]
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="severity",
            sampler_type=dd.SamplerType.SUBCATEGORY,
            params=dd.SubcategorySamplerParams(
                category="disease_or_condition",
                values={
                    "healthy crop with no visible disease": ["none - healthy negative"],
                    "powdery mildew on leaves": disease_severity_values,
                    "rust-colored fungal pustules on leaf surfaces": disease_severity_values,
                    "early blight with concentric brown leaf spots": disease_severity_values,
                    "late blight with irregular dark lesions": disease_severity_values,
                    "bacterial leaf spot with small dark speckles": disease_severity_values,
                    "downy mildew patches on leaf undersides": disease_severity_values,
                    "leaf curl with mosaic discoloration": disease_severity_values,
                    "insect feeding damage as a disease confounder": ["confounder - not a disease severity label"],
                    "nutrient deficiency yellowing as a disease confounder": [
                        "confounder - not a disease severity label"
                    ],
                },
            ),
        )
    )
    add_category(
        config_builder,
        "field_condition",
        [
            "uniform crop stand",
            "patchy emergence",
            "uneven row spacing",
            "visible irrigation lines",
            "muddy soil after rain",
            "dry cracked soil",
            "mulched bed system",
        ],
    )
    add_category(
        config_builder,
        "weather_lighting",
        [
            "bright midday sun",
            "soft overcast light",
            "golden hour light",
            "after-rain humid conditions",
            "hazy smoky sky",
            "greenhouse diffuse lighting",
        ],
    )

    config_builder.add_column(
        dd.ImageColumnConfig(
            name="crop_image",
            prompt=AGRICULTURE_IMAGE_PROMPT,
            model_alias=model_alias,
        )
    )

    return config_builder


def create_dataset(
    config_builder: dd.DataDesignerConfigBuilder,
    *,
    num_records: int,
    dataset_name: str,
    artifact_path: Path | str | None = None,
) -> DatasetCreationResults:
    data_designer = DataDesigner(artifact_path=artifact_path)
    data_designer.validate(config_builder)
    return data_designer.create(config_builder, num_records=num_records, dataset_name=dataset_name)


AGRICULTURE_IMAGE_PROMPT = """\
Create a realistic crop disease detection image.

Scene requirements:
- Visual variation ID, for internal diversity only: {{ visual_variation_id }}
- Crop type: {{ crop_type }}
- Growth stage: {{ growth_stage }}
- Viewpoint: {{ viewpoint }}
- Disease or condition: {{ disease_or_condition }}
- Severity: {{ severity }}
- Field condition: {{ field_condition }}
- Weather and lighting: {{ weather_lighting }}

Make the image useful for crop disease detection, visual QA, reviewer
calibration, and data-labeling experiments. The requested crop, condition,
severity, and field context should be visually inspectable. Show realistic
plant structure, leaves, rows, soil, and disease symptoms when requested. For
healthy examples, show clear healthy leaves or canopy with no visible disease.
For confounders, make the non-disease condition plausible enough to test a
classifier or VLM prompt. Do not include real farm names, readable license
plates, watermarks, or people as the primary subject. Generate exactly one
final crop image for this row. Do not return alternate versions, a grid, a pair
of examples, before/after panels, or multiple frames. Use the visual variation
ID only as an internal diversity key; never render it as text.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic crop disease detection imagery.")
    parser.add_argument("--num-records", type=int, default=10, help="Number of crop images to generate.")
    parser.add_argument("--dataset-name", default="crop-disease-detection-images", help="Output dataset name.")
    parser.add_argument("--artifact-path", type=Path, default=None, help="Optional Data Designer artifact directory.")
    parser.add_argument("--model-provider", default=DEFAULT_MODEL_PROVIDER, help="Image model provider name.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Provider model ID.")
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS, help="Alias used by image columns.")
    parser.add_argument("--image-size", default="1K", help="OpenRouter image size tier, such as 1K, 2K, or 4K.")
    parser.add_argument("--aspect-ratio", default="4:3", help="Provider-specific aspect ratio value.")
    parser.add_argument("--max-parallel-requests", type=int, default=10, help="Maximum parallel image requests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_builder = build_config(
        model_provider=args.model_provider,
        model_id=args.model_id,
        model_alias=args.model_alias,
        image_size=args.image_size,
        aspect_ratio=args.aspect_ratio,
        max_parallel_requests=args.max_parallel_requests,
    )
    results = create_dataset(
        config_builder,
        num_records=args.num_records,
        dataset_name=args.dataset_name,
        artifact_path=args.artifact_path,
    )
    dataset = results.load_dataset()
    print(f"Generated {len(dataset)} crop disease detection image rows.")
    print(f"Dataset artifacts: {results.artifact_storage.base_dataset_path}")


if __name__ == "__main__":
    main()
