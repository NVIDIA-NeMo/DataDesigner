# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
# ]
# ///
"""Humanoid Robot Scene Understanding Image Generation Recipe

Generate synthetic egocentric humanoid robot images with controlled variation
over indoor environment, robot viewpoint, task goal, object set, scene state,
safety condition, lighting, and adult human presence.

Use the generated images for embodied-AI scene understanding, visual QA,
reviewer calibration, safety review, and robotics demos where the image should
look like a frame captured from the robot's own camera in a controlled setting.

Prerequisites:
    - An image-generation provider key for the selected model. The defaults use
      OpenRouter, so set OPENROUTER_API_KEY before running.

Run:
    uv run humanoid_robot_scene_understanding.py --num-records 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults

DEFAULT_MODEL_PROVIDER = "openrouter"
DEFAULT_MODEL_ID = "google/gemini-3.1-flash-image-preview"
DEFAULT_MODEL_ALIAS = "humanoid-scene-model"


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
            params=dd.UUIDSamplerParams(prefix="humanoid-", short_form=True),
        )
    )


def build_config(
    *,
    model_provider: str = DEFAULT_MODEL_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    model_alias: str = DEFAULT_MODEL_ALIAS,
    image_size: str = "1024",
    aspect_ratio: str = "16:9",
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
        "environment",
        [
            "teaching kitchen with counters, cabinets, and everyday objects",
            "mock apartment living room arranged for assistive robotics",
            "assisted living bedroom with bedside table and mobility aids",
            "robotics lab workbench with tools and calibration objects",
            "retail stockroom with shelves, totes, and handheld items",
            "hospital supply room with carts, bins, and sealed supplies",
            "office break room with appliances, tableware, and waste bins",
            "laundry room with baskets, detergent, shelves, and folded towels",
            "tool bench training area with bins, fasteners, and hand tools",
            "grocery practice aisle with shelves, baskets, and fallen items",
        ],
    )
    add_category(
        config_builder,
        "robot_viewpoint",
        [
            "head-mounted camera at standing adult height",
            "chest-mounted camera with both robot hands barely visible at the bottom edge",
            "slightly downward gaze toward a tabletop work surface",
            "close manipulation view with one robot hand near the target object",
            "wide room scan from a doorway before entering the scene",
            "low crouched inspection angle looking under a table or cart",
        ],
    )
    add_category(
        config_builder,
        "task_goal",
        [
            "locate the requested object before reaching",
            "judge whether the path is safe to walk through",
            "identify which objects are reachable from the current pose",
            "verify that a cleanup task is complete",
            "prepare a clear handoff area for an adult user",
            "find the missing tool or supply item",
            "inspect a spill or obstacle before moving closer",
            "decide whether fragile items are too close to an edge",
        ],
    )
    add_category(
        config_builder,
        "object_set",
        [
            "mug, kettle, sponge, dish towel, and cereal bowl",
            "water glass, medication organizer, tissue box, and walking cane",
            "pipette rack, beaker, nitrile gloves, and small screwdriver",
            "barcode scanner, tote, tape dispenser, folded shirt, and box cutter",
            "laundry basket, detergent bottle, folded towels, and loose sock",
            "pliers, hex keys, small bolts, tape measure, and plastic bins",
            "shopping basket, cereal boxes, soup cans, and fallen fruit",
            "meal tray, sealed supplies, clipboard, and rolling cart",
        ],
    )
    add_category(
        config_builder,
        "scene_state",
        [
            "organized and ready for the task",
            "moderately cluttered but navigable",
            "target object partly occluded by other items",
            "target object moved to an unexpected location",
            "container open with mixed contents visible",
            "fragile item near the table edge",
            "object stack unstable but still standing",
            "task area partly blocked by a chair or cart",
        ],
    )
    add_category(
        config_builder,
        "safety_condition",
        [
            "no visible hazard",
            "small liquid spill on the floor",
            "power cable crossing the walking path",
            "sharp tool exposed on the work surface",
            "hot appliance indicator light visible",
            "glass object on the floor near the path",
            "drawer left open at knee height",
            "rolling cart partially blocking the doorway",
        ],
    )
    add_category(
        config_builder,
        "human_presence",
        [
            "no person visible",
            "adult worker's gloved hands visible at a safe distance",
            "adult caregiver standing in the background with face turned away",
            "adult shopper passing through the background, not identifiable",
            "adult lab worker partially visible from shoulders down",
            "adult office worker's arm visible near the handoff area",
        ],
    )
    add_category(
        config_builder,
        "lighting",
        [
            "bright even lab lighting",
            "warm apartment lighting",
            "overcast window light",
            "mixed overhead and task lighting",
            "dim hallway light with localized task lamp",
            "high-contrast backlighting from a nearby window",
        ],
    )

    config_builder.add_column(
        dd.ImageColumnConfig(
            name="humanoid_scene_image",
            prompt=HUMANOID_SCENE_PROMPT,
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


HUMANOID_SCENE_PROMPT = """\
Create a realistic egocentric humanoid robot scene-understanding image.

The frame must look like it was captured from the humanoid robot's own camera
inside a controlled indoor environment. Show the robot's viewpoint clearly:
camera height, reachable workspace, path geometry, task-relevant objects,
obstacles, and safety condition should all be visible enough for visual QA or
embodied-AI scene understanding. If the viewpoint mentions robot hands, show at
most one or two simple robot hands at the image edge; do not make the robot the
main subject.

Scene requirements:
- Visual variation ID, for internal diversity only: {{ visual_variation_id }}
- Environment: {{ environment }}
- Robot viewpoint: {{ robot_viewpoint }}
- Task goal: {{ task_goal }}
- Object set: {{ object_set }}
- Scene state: {{ scene_state }}
- Safety condition: {{ safety_condition }}
- Human presence: {{ human_presence }}
- Lighting: {{ lighting }}

Make the requested task goal, object set, scene state, and safety condition
visually legible without adding labels or annotation graphics. Use realistic
materials, clutter, occlusion, reachability cues, shadows, and indoor scale.

Do not include children, identifiable faces, readable personal names, real
company logos, surveillance UI, bounding boxes, arrows, captions, labels,
watermarks, subtitles, HUD overlays, or diagnostic text. Generate exactly one
final camera frame for this row. Do not return alternate versions, a grid, a
pair of examples, before/after panels, or multiple frames. Use the visual
variation ID only as an internal diversity key; never render it as text.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic humanoid robot scene-understanding images.")
    parser.add_argument("--num-records", type=int, default=10, help="Number of humanoid scene images to generate.")
    parser.add_argument("--dataset-name", default="humanoid-robot-scene-understanding", help="Output dataset name.")
    parser.add_argument("--artifact-path", type=Path, default=None, help="Optional Data Designer artifact directory.")
    parser.add_argument("--model-provider", default=DEFAULT_MODEL_PROVIDER, help="Image model provider name.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Provider model ID.")
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS, help="Alias used by image columns.")
    parser.add_argument("--image-size", default="1024", help="Provider-specific image size value.")
    parser.add_argument("--aspect-ratio", default="16:9", help="Provider-specific aspect ratio value.")
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
    print(f"Generated {len(dataset)} humanoid robot scene-understanding rows.")
    print(f"Dataset artifacts: {results.artifact_storage.base_dataset_path}")


if __name__ == "__main__":
    main()
