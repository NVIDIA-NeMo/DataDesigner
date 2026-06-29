# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
#     "pandas",
#     "pyarrow",
# ]
# ///
"""Rich Document Image Generation Recipe

Generate synthetic business-document page images with controlled variation.
Each generated row pairs an image with the metadata that produced it, making
the output useful as seed data for visual QA, OCR robustness, multimodal
judging, and document-understanding experiments.

Prerequisites:
    - An image-generation provider key for the selected model. The defaults use
      OpenRouter, so set OPENROUTER_API_KEY before running.

Run:
    # Generate 5 rich document images with the default OpenRouter model.
    uv run rich_document_images.py --num-records 5

    # Export a VQA-ready seed parquet with base64 image bytes and image metadata.
    uv run rich_document_images.py --num-records 25 --export-seed rich_document_seed.parquet

    # Use a different provider or image model.
    uv run rich_document_images.py --model-provider openrouter --model-id google/gemini-3.1-flash-image-preview
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from PIL import Image

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults

DEFAULT_MODEL_PROVIDER = "openrouter"
DEFAULT_MODEL_ID = "google/gemini-3.1-flash-image-preview"
DEFAULT_MODEL_ALIAS = "document-generation-model"

SEED_METADATA_COLUMNS = [
    "document_type",
    "primary_visual",
    "secondary_visual",
    "layout_style",
    "document_condition",
]


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


def add_category(
    config_builder: dd.DataDesignerConfigBuilder,
    name: str,
    values: list[str],
    weights: list[float] | None = None,
) -> None:
    """Add a categorical sampler column."""
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name=name,
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=values, weights=weights),
        )
    )


def add_visual_variation_id(config_builder: dd.DataDesignerConfigBuilder) -> None:
    """Add a unique row-level key that discourages duplicate image generations."""
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="visual_variation_id",
            sampler_type=dd.SamplerType.UUID,
            params=dd.UUIDSamplerParams(prefix="doc-", short_form=True),
        )
    )


def build_config(
    *,
    model_provider: str = DEFAULT_MODEL_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    model_alias: str = DEFAULT_MODEL_ALIAS,
    image_size: str = "1K",
    aspect_ratio: str = "2:3",
    max_parallel_requests: int = 10,
) -> dd.DataDesignerConfigBuilder:
    """Build a rich document image-generation pipeline."""
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
        "document_type",
        [
            "quarterly business review",
            "market research brief",
            "operations dashboard export",
            "clinical trial status report",
            "sustainability impact report",
            "financial variance memo",
            "customer support incident review",
            "supply chain risk assessment",
            "product launch readiness plan",
            "employee engagement summary",
        ],
        weights=[0.12, 0.10, 0.14, 0.08, 0.08, 0.12, 0.12, 0.10, 0.12, 0.12],
    )

    add_category(
        config_builder,
        "organization_name",
        [
            "Aster Analytics",
            "Blue Ridge Health",
            "CedarWorks Manufacturing",
            "DeltaGrid Energy",
            "Evergreen Mobility",
            "Harborlight Retail",
            "Northstar Robotics",
            "Redwood BioSystems",
            "Summit Cloud Services",
            "Valley Forge Logistics",
        ],
    )

    add_category(
        config_builder,
        "document_owner",
        [
            "Maya Chen",
            "Jonas Patel",
            "Elena Garcia",
            "Noah Williams",
            "Amara Okafor",
            "Theo Martin",
            "Priya Raman",
            "Sofia Rossi",
            "Lena Fischer",
            "Caleb Brooks",
        ],
    )

    add_category(
        config_builder,
        "owner_role",
        [
            "VP Operations",
            "Finance Director",
            "Clinical Program Manager",
            "Customer Success Lead",
            "Risk Officer",
            "Product Launch Owner",
            "People Analytics Partner",
        ],
    )

    add_category(
        config_builder,
        "audience",
        [
            "executive leadership",
            "finance review committee",
            "field operations managers",
            "clinical program leads",
            "board audit committee",
            "customer success directors",
        ],
    )

    add_category(
        config_builder,
        "content_theme",
        [
            "quarterly revenue performance and forecast variance",
            "regional customer adoption and churn risk",
            "service-level agreement compliance and incident aging",
            "inventory throughput, backorders, and supplier delays",
            "trial enrollment, site activation, and adverse event counts",
            "energy consumption, emissions, and sustainability targets",
            "hiring funnel conversion, offer acceptance, and attrition",
            "product launch milestones, owners, and readiness status",
        ],
    )

    add_category(
        config_builder,
        "primary_visual",
        [
            "clustered bar chart comparing three regions across four quarters",
            "line chart with two series, annotated inflection points, and a target band",
            "stacked area chart showing category mix over six months",
            "waterfall chart showing contributors to budget variance",
            "scatter plot with labeled outliers and a trend line",
            "Gantt-style timeline with milestones and owner initials",
            "heatmap matrix with risk severity by team and region",
            "donut chart with callout labels and percentages",
        ],
    )

    add_category(
        config_builder,
        "secondary_visual",
        [
            "dense financial table with subtotals and variance arrows",
            "KPI card row with current value, target, delta, and traffic-light status",
            "two-column risk register with owner, due date, and mitigation note",
            "small process diagram with arrows between four labeled stages",
            "ranked list table with sparklines in the final column",
            "compact map inset with region labels and numeric badges",
            "executive callout box with three bullet conclusions",
            "signature block plus approval checklist",
        ],
    )

    add_category(
        config_builder,
        "layout_style",
        [
            "clean consulting report page with narrow margins and section dividers",
            "dashboard export with a top filter bar and grid of panels",
            "formal memo with letterhead, dense paragraphs, and one embedded chart",
            "board-pack page with title ribbon, footnotes, and small-print source notes",
            "compliance form with checkboxes, tables, and stamped approval",
            "research brief with abstract, sidebar definitions, and figure captions",
            "operations one-pager with color-coded status chips and action table",
        ],
    )

    add_category(
        config_builder,
        "document_condition",
        [
            "pristine exported PDF screenshot",
            "high-resolution office scanner output",
            "faded photocopy with mild paper texture",
            "creased printout with a clipped corner",
            "low-contrast scan with light shadow near the binding edge",
        ],
    )

    add_category(
        config_builder,
        "annotation_layer",
        [
            "no manual annotations",
            "yellow highlights over two key numbers",
            "red pen circle around one chart outlier",
            "blue sticky note partially covering the lower right table",
            "handwritten margin note asking for follow-up",
            "rubber stamp reading DRAFT across the header",
        ],
    )

    add_category(
        config_builder,
        "numeric_context",
        [
            "include values in thousands with one decimal place",
            "include percentages, basis-point deltas, and small footnotes",
            "include dates across the next six months",
            "include currency values, totals, and year-over-year deltas",
            "include counts by region plus a total row",
        ],
    )

    config_builder.add_column(
        dd.ImageColumnConfig(
            name="document_image",
            prompt=RICH_DOCUMENT_IMAGE_PROMPT,
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


def export_seed_parquet(results: DatasetCreationResults, output_path: Path) -> None:
    """Export generated images as format-neutral base64 seed rows for VLM pipelines."""
    dataset = results.load_dataset()
    base_path = results.artifact_storage.base_dataset_path
    rows: list[dict[str, str | int]] = []

    for row in dataset.itertuples(index=False):
        image_ref = _first_image_ref(row.document_image)
        image_path = base_path / image_ref
        image_format, image_mime_type, image_width, image_height = _image_metadata(image_path)
        output_row = {
            "image_base64": base64.b64encode(image_path.read_bytes()).decode("utf-8"),
            "image_format": image_format,
            "image_mime_type": image_mime_type,
            "image_width": image_width,
            "image_height": image_height,
        }
        output_row.update({column: getattr(row, column) for column in SEED_METADATA_COLUMNS})
        rows.append(output_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)


def _first_image_ref(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        first = next(iter(value), None)
        if isinstance(first, str):
            return first
    raise ValueError(f"Expected document_image to be a string path or non-empty iterable, got {type(value)!r}")


def _image_metadata(image_path: Path) -> tuple[str, str, int, int]:
    with Image.open(image_path) as image:
        image_format = image.format or image_path.suffix.lstrip(".").upper() or "UNKNOWN"
        image_mime_type = Image.MIME.get(image_format, "application/octet-stream")
        image_width, image_height = image.size
    return image_format, image_mime_type, image_width, image_height


RICH_DOCUMENT_IMAGE_PROMPT = """\
Create a realistic single-page business document image with rich visual information.

Document requirements:
- Visual variation ID, for internal diversity only: {{ visual_variation_id }}
- Document type: {{ document_type }}
- Organization: {{ organization_name }}
- Document owner: {{ document_owner }}, {{ owner_role }}
- Intended audience: {{ audience }}
- Theme: {{ content_theme }}
- Layout style: {{ layout_style }}
- Physical/rendering condition: {{ document_condition }}
- Annotation layer: {{ annotation_layer }}
- Numeric style: {{ numeric_context }}

Required visual content:
- Primary visual: {{ primary_visual }}
- Secondary visual: {{ secondary_visual }}
- At least one readable table with row and column labels
- At least one chart, timeline, heatmap, diagram, or KPI-card cluster
- A clear title, date, organization name, document owner, section headings, and small source note
- Enough readable text to ask visual QA questions about exact values, trends, labels, owners, dates, and relationships

Make the page visually dense but professionally designed. Use realistic fonts,
alignment, legends, axis labels, table borders, captions, and spacing. The text
and numbers should be legible. Avoid blank areas, generic placeholder blocks,
or lorem ipsum. Generate exactly one final document page for this row. Do not
return alternate versions, a grid, a pair of examples, before/after panels, or
multiple pages. Use the visual variation ID only as an internal diversity key;
never render it as text. Do not include real company logos or real personal
data.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rich synthetic business-document images.")
    parser.add_argument("--num-records", type=int, default=5, help="Number of document images to generate.")
    parser.add_argument("--dataset-name", default="rich-document-images", help="Output dataset name.")
    parser.add_argument("--artifact-path", type=Path, default=None, help="Optional Data Designer artifact directory.")
    parser.add_argument("--model-provider", default=DEFAULT_MODEL_PROVIDER, help="Image model provider name.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Provider model ID.")
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS, help="Alias used by image columns.")
    parser.add_argument("--image-size", default="1K", help="OpenRouter image size tier, such as 1K, 2K, or 4K.")
    parser.add_argument("--aspect-ratio", default="2:3", help="Provider-specific aspect ratio value.")
    parser.add_argument("--max-parallel-requests", type=int, default=10, help="Maximum parallel image requests.")
    parser.add_argument(
        "--export-seed",
        type=Path,
        default=None,
        help="Optional parquet path for a VQA-ready seed with base64 image bytes and image metadata.",
    )
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
    print(f"Generated {len(dataset)} rich document image rows.")
    print(f"Dataset artifacts: {results.artifact_storage.base_dataset_path}")

    if args.export_seed is not None:
        export_seed_parquet(results, args.export_seed)
        print(f"Exported VQA seed parquet: {args.export_seed}")


if __name__ == "__main__":
    main()
