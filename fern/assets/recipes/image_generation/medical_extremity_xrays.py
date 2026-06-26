# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
# ]
# ///
"""Synthetic Extremity X-ray Image Generation Recipe

Generate synthetic extremity X-ray style images with controlled variation over
anatomical region, view, imaging context, technical quality, and musculoskeletal
findings.

Medical disclaimer:
    These generated images are synthetic and intended only for AI research,
    education, data-pipeline prototyping, and evaluation workflows. They are not
    real medical images and must not be used for diagnosis, treatment planning,
    clinical decision-making, or as a substitute for real clinical validation.

Prerequisites:
    - An image-generation provider key for the selected model. The defaults use
      OpenRouter, so set OPENROUTER_API_KEY before running.

Run:
    uv run medical_extremity_xrays.py --num-records 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults

DEFAULT_MODEL_PROVIDER = "openrouter"
DEFAULT_MODEL_ID = "google/gemini-3.1-flash-image-preview"
DEFAULT_MODEL_ALIAS = "medical-image-model"


def build_model_configs(
    *,
    model_provider: str,
    model_id: str,
    model_alias: str,
    image_size: str,
    aspect_ratio: str,
    max_parallel_requests: int,
) -> list[dd.ModelConfig]:
    """Build a provider-agnostic image-generation model config."""
    return [
        dd.ModelConfig(
            alias=model_alias,
            model=model_id,
            provider=model_provider,
            inference_parameters=dd.ImageInferenceParams(
                extra_body={
                    "n": 1,
                    "generationConfig": {
                        "imageConfig": {
                            "aspectRatio": aspect_ratio,
                            "imageSize": image_size,
                        }
                    },
                },
                max_parallel_requests=max_parallel_requests,
            ),
            skip_health_check=True,
        )
    ]


def add_category(config_builder: dd.DataDesignerConfigBuilder, name: str, values: list[str]) -> None:
    """Add a categorical sampler column."""
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
            params=dd.UUIDSamplerParams(prefix="xray-", short_form=True),
        )
    )


def build_config(
    *,
    model_provider: str = DEFAULT_MODEL_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    model_alias: str = DEFAULT_MODEL_ALIAS,
    image_size: str = "1024",
    aspect_ratio: str = "1:1",
    max_parallel_requests: int = 10,
) -> dd.DataDesignerConfigBuilder:
    """Build a synthetic extremity X-ray image-generation pipeline."""
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
        "patient_age_group",
        [
            "young adult",
            "adult",
            "middle-aged adult",
            "older adult",
            "geriatric adult",
        ],
    )

    add_category(
        config_builder,
        "patient_sex",
        [
            "female",
            "male",
        ],
    )

    add_category(
        config_builder,
        "body_habitus",
        [
            "thin build",
            "athletic build",
            "average build",
            "overweight build",
            "obese build",
        ],
    )

    add_category(
        config_builder,
        "anatomical_region",
        [
            "right shoulder",
            "left shoulder",
            "right humerus",
            "left humerus",
            "right elbow",
            "left elbow",
            "right forearm with radius and ulna",
            "left forearm with radius and ulna",
            "right wrist",
            "left wrist",
            "right hand and fingers",
            "left hand and fingers",
            "right hip",
            "left hip",
            "right femur",
            "left femur",
            "right knee",
            "left knee",
            "right tibia and fibula",
            "left tibia and fibula",
            "right ankle",
            "left ankle",
            "right foot and toes",
            "left foot and toes",
        ],
    )

    add_category(
        config_builder,
        "equipment_type",
        [
            "fixed radiography unit",
            "portable X-ray machine",
            "digital radiography system",
            "computed radiography system",
        ],
    )

    add_category(
        config_builder,
        "imaging_context",
        [
            "emergency department acute trauma",
            "emergency department fall injury",
            "emergency department sports injury",
            "orthopedic clinic routine follow-up",
            "post-operative hardware check",
            "pre-operative planning",
            "urgent care pain evaluation",
        ],
    )

    add_category(
        config_builder,
        "xray_view",
        [
            "anteroposterior (AP)",
            "lateral",
            "oblique internal rotation",
            "oblique external rotation",
            "weight-bearing AP",
            "stress view",
        ],
    )

    add_category(
        config_builder,
        "exposure_quality",
        [
            "underexposed with cortical margins poorly defined",
            "optimal exposure with clear cortical and trabecular detail",
            "overexposed with washed out bone detail",
            "low kVp technique with high bone contrast",
            "high kVp technique with better soft tissue visualization",
        ],
    )

    add_category(
        config_builder,
        "positioning",
        [
            "well-positioned true AP or lateral",
            "slightly rotated",
            "oblique positioning",
            "splint or cast in place",
            "traction device visible",
            "suboptimal because the patient could not cooperate due to pain",
        ],
    )

    add_category(
        config_builder,
        "primary_finding",
        [
            "normal with no acute osseous abnormality",
            "nondisplaced fracture through the imaged bone",
            "displaced fracture through the imaged bone",
            "comminuted fracture involving the imaged bone",
            "stress fracture line in the imaged bone",
            "joint dislocation or subluxation in the imaged region",
            "degenerative osteoarthritis in the imaged joint",
            "suspected osteomyelitis with focal cortical destruction",
            "soft tissue swelling with no acute fracture identified",
        ],
    )

    add_category(
        config_builder,
        "secondary_findings",
        [
            "none",
            "osteopenia",
            "degenerative joint changes at adjacent joints",
            "old healed fracture with callus formation",
            "orthopedic plate and screws",
            "intramedullary nail",
            "joint effusion",
            "soft tissue calcifications",
            "vascular calcifications",
        ],
    )

    add_category(
        config_builder,
        "image_quality",
        [
            "excellent sharp cortical margins and clear trabecular pattern",
            "good adequate visualization of all bony structures",
            "fair with mild motion artifact",
            "fair with mild noise or graininess",
            "fair with cast or splint partially obscuring detail",
            "limited portable technique with technical limitations",
            "limited by patient body habitus",
        ],
    )

    config_builder.add_column(
        dd.ImageColumnConfig(
            name="extremity_xray",
            prompt=EXTREMITY_XRAY_PROMPT,
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


EXTREMITY_XRAY_PROMPT = """\
Create a synthetic research-only grayscale X-ray style radiograph of the
{{ anatomical_region }}, {{ xray_view }} view.

Patient and acquisition context:
- Visual variation ID, for internal diversity only: {{ visual_variation_id }}
- Patient age group: {{ patient_age_group }}
- Patient sex: {{ patient_sex }}
- Body habitus: {{ body_habitus }}
- Equipment: {{ equipment_type }}
- Context: {{ imaging_context }}
- Technical quality: {{ exposure_quality }}
- Positioning: {{ positioning }}
- Image quality: {{ image_quality }}

Findings to depict:
- Primary finding: {{ primary_finding }}
- Secondary findings: {{ secondary_findings }}

Use a realistic educational radiograph style with visible bones, joints, cortex,
trabecular pattern, and soft-tissue silhouette. Include standard left/right
markers where appropriate. Make the image look synthetic but useful for AI
research and data-pipeline prototyping. Do not include real patient names, real
medical record numbers, hospital logos, or any real protected health information.
Generate exactly one final radiograph for this row. Do not return alternate
versions, a two-view panel, a grid, a before/after image, duplicated views, or
multiple image candidates. Use the visual variation ID only as an internal
diversity key for anatomy framing, rotation, exposure texture, and soft-tissue
background; never render it as text. Do not add diagnostic captions or
explanatory text overlays.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic extremity X-ray style images.")
    parser.add_argument("--num-records", type=int, default=5, help="Number of synthetic X-ray images to generate.")
    parser.add_argument("--dataset-name", default="synthetic-extremity-xrays", help="Output dataset name.")
    parser.add_argument("--artifact-path", type=Path, default=None, help="Optional Data Designer artifact directory.")
    parser.add_argument("--model-provider", default=DEFAULT_MODEL_PROVIDER, help="Image model provider name.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Provider model ID.")
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS, help="Alias used by image columns.")
    parser.add_argument("--image-size", default="1024", help="Provider-specific image size value.")
    parser.add_argument("--aspect-ratio", default="1:1", help="Provider-specific aspect ratio value.")
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
    print(f"Generated {len(dataset)} synthetic extremity X-ray rows.")
    print(f"Dataset artifacts: {results.artifact_storage.base_dataset_path}")


if __name__ == "__main__":
    main()
