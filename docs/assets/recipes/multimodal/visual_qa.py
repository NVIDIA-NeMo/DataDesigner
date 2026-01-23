# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from data_designer.essentials import (
    CategorySamplerParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    ImageContext,
    ImageFormat,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
    LocalFileSeedSource,
    ModalityDataType,
    SamplerColumnConfig,
    SamplerType,
)
from data_designer.interface.results import DatasetCreationResults
from data_designer.lazy_heavy_imports import pd


class Question(BaseModel):
    """Schema for generated questions."""

    question: str = Field(description="The question to be generated")


class Options(BaseModel):
    """Schema for multiple choice options."""

    option_a: str = Field(description="The first answer choice")
    option_b: str = Field(description="The second answer choice")
    option_c: str = Field(description="The third answer choice")
    option_d: str = Field(description="The fourth answer choice")


class Answer(BaseModel):
    """Schema for question answers."""

    answer: Literal["option_a", "option_b", "option_c", "option_d"] = Field(
        description="The correct answer to the question"
    )


def create_sample_image_dataset(num_images: int = 10) -> Path:
    """Create a sample dataset with placeholder images.

    Args:
        num_images: Number of sample images to generate

    Returns:
        Path to the temporary CSV file containing base64-encoded images
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError(
            "Pillow is required for creating sample images. Install it with: pip install Pillow"
        ) from None

    images_data = []
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]

    for i in range(num_images):
        # Create a simple colored image with text
        img = Image.new("RGB", (400, 300), color=colors[i % len(colors)])
        draw = ImageDraw.Draw(img)

        # Add text to image
        text = f"Sample Image #{i + 1}\nColor: {colors[i % len(colors)]}"
        try:
            font = ImageFont.truetype("Arial", 40)
        except OSError:
            font = ImageFont.load_default()

        # Draw text in center
        draw.text((50, 120), text, fill="white", font=font)

        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        images_data.append({"base64_image": img_base64})

    df = pd.DataFrame(images_data)

    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()

    return Path(temp_file.name)


def build_config(
    model_alias: str, seed_dataset_path: Path | str, image_column: str = "base64_image"
) -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder()

    # Add seed dataset with images
    seed_source = LocalFileSeedSource(path=str(seed_dataset_path))
    config_builder.with_seed_dataset(seed_source)

    config_builder.add_column(
        SamplerColumnConfig(
            name="difficulty",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["easy", "medium", "hard"]),
        )
    )

    config_builder.add_column(
        LLMTextColumnConfig(
            name="summary",
            model_alias=model_alias,
            prompt=(
                "Provide a detailed summary of the content in this image in Markdown format. "
                "Start from the top of the image and then describe it from top to bottom. "
                "Place a summary at the bottom."
            ),
            multi_modal_context=[
                ImageContext(
                    column_name=image_column,
                    data_type=ModalityDataType.BASE64,
                    image_format=ImageFormat.PNG,
                )
            ],
        )
    )

    config_builder.add_column(
        LLMStructuredColumnConfig(
            name="question",
            model_alias=model_alias,
            prompt=(
                "Based on this image summary, generate a {{ difficulty }} level question about the content:\n\n"
                "{{ summary }}"
            ),
            output_format=Question,
        )
    )

    config_builder.add_column(
        LLMStructuredColumnConfig(
            name="options",
            model_alias=model_alias,
            prompt=(
                "Generate four multiple choice options for this question. One should be correct, "
                "and three should be plausible but incorrect distractors:\n\n"
                "Question: {{ question }}\n"
                "Summary: {{ summary }}"
            ),
            output_format=Options,
        )
    )

    config_builder.add_column(
        LLMStructuredColumnConfig(
            name="answer",
            model_alias=model_alias,
            prompt=(
                "Based on the image summary and question, identify which option is correct:\n\n"
                "Summary: {{ summary }}\n"
                "Question: {{ question }}\n"
                "Options: {{ options }}"
            ),
            output_format=Answer,
        )
    )

    return config_builder


def create_dataset(
    config_builder: DataDesignerConfigBuilder,
    num_records: int,
    artifact_path: Path | str | None = None,
) -> DatasetCreationResults:
    data_designer = DataDesigner(artifact_path=artifact_path)
    results = data_designer.create(config_builder, num_records=num_records)
    return results


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model-alias", type=str, default="nvidia-vision")
    parser.add_argument("--num-records", type=int, default=10)
    parser.add_argument("--artifact-path", type=str, default=None)
    parser.add_argument(
        "--image-dataset-path",
        type=str,
        default=None,
        help="Path to the seed dataset containing images (CSV/Parquet with base64-encoded images). "
        "If not provided, sample placeholder images will be generated.",
    )
    parser.add_argument(
        "--image-column",
        type=str,
        default="base64_image",
        help="Name of the column containing base64-encoded images (default: base64_image)",
    )
    args = parser.parse_args()

    # Use provided dataset or create sample images
    if args.image_dataset_path:
        image_dataset_path = args.image_dataset_path
        print(f"Using image dataset: {image_dataset_path}")
    else:
        print("No image dataset provided. Generating sample placeholder images...")
        image_dataset_path = create_sample_image_dataset(num_images=args.num_records)
        print(f"Created sample image dataset: {image_dataset_path}")

    config_builder = build_config(
        model_alias=args.model_alias,
        seed_dataset_path=image_dataset_path,
        image_column=args.image_column,
    )
    results = create_dataset(config_builder, num_records=args.num_records, artifact_path=args.artifact_path)

    print(f"Dataset saved to: {results.artifact_storage.final_dataset_path}")

    results.load_analysis().to_report()
