# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer>=0.5.6",
# ]
# ///
"""Long-Document Understanding Nemotron-Parse OCR Recipe

Run Nemotron-Parse v1.1 OCR over document images from a seed parquet file.
Each record produces:
  - `transcribed_texts`: clean text extracted from the OCR output
  - `transcribed_texts__metadata`: bounding-box coordinates and class labels

Prerequisites:
    - A seed parquet file containing a `png_images_base64` column with a JSON
      array of base64-encoded PNG images (one element per page; single-page
      seeds have a one-element array).
    - A vLLM-compatible deployment of nvidia/NVIDIA-Nemotron-Parse-v1.1.
      The vLLM server must be launched with a chat template that injects the
      Nemotron-Parse special tokens. Save the following as a .jinja file and
      pass it via --chat-template:

        {% for message in messages %}{% if message["role"] == "user" %}{{ "</s><s><predict_bbox><predict_classes><output_markdown>" }}{% endif %}{% endfor %}

      Example launch script for 1× H100:
        docker run -d --gpus all \
            -p 8000:8000 \
            --entrypoint bash \
            vllm/vllm-openai:v0.14.1 \
            -c "pip install open-clip-torch albumentations timm && vllm serve nvidia/NVIDIA-Nemotron-Parse-v1.1 \
            --tensor-parallel-size 1 \
            --max-model-len 9000 \
            --gpu-memory-utilization 0.85 \
            --max-num-seqs 128 \
            --chat-template /chat_template.jinja \
            --trust-remote-code"

Run:
    # Basic usage (processes 5 records by default)
    uv run 02-nemotron-parse-ocr-sdg.py --vllm-endpoint http://localhost:8000/v1 --seed-path seed_data/seed_per_page.parquet

    # Custom record count
    uv run 02-nemotron-parse-ocr-sdg.py --vllm-endpoint http://localhost:8000/v1 --seed-path seed_data/seed_per_page.parquet --num-records 100

    # For help message and available options
    uv run 02-nemotron-parse-ocr-sdg.py --help
"""

import re
from pathlib import Path

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults

NEMOTRON_PARSE_MODEL = "nvidia/NVIDIA-Nemotron-Parse-v1.1"
VLLM_PROVIDER_NAME = "vllm"

_STRUCTURED_ELEMENT_PATTERN = re.compile(
    r"<x_([\d.]+)><y_([\d.]+)>(.*?)<x_([\d.]+)><y_([\d.]+)><class_([^>]+)>",
    re.DOTALL,
)


def _extract_structured_elements(text: str) -> list[dict]:
    """Parse Nemotron-Parse bbox markup into structured dicts.

    Input format: <x_START><y_START>TEXT<x_END><y_END><class_LABEL>

    Returns list of dicts with keys: bbox ({x1,y1,x2,y2}), class_label, text.
    """
    elements = []
    for match in _STRUCTURED_ELEMENT_PATTERN.finditer(text):
        x1, y1, content, x2, y2, class_label = match.groups()
        elements.append(
            {
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                },
                "class_label": class_label,
                "text": content.strip(),
            }
        )
    return elements


@dd.custom_column_generator(
    required_columns=["raw_ocr_output"],
    side_effect_columns=["transcribed_texts__metadata"],
)
def parse_ocr_output(row: dict) -> dict:
    """Extract clean text and bbox metadata from raw Nemotron-Parse output."""
    raw = row["raw_ocr_output"]
    elements = _extract_structured_elements(raw)
    row["transcribed_texts"] = "\n".join(el["text"] for el in elements)
    row["transcribed_texts__metadata"] = [{"bbox": el["bbox"], "class_label": el["class_label"]} for el in elements]
    return row


def build_config(
    seed_path: str = "seed.parquet",
    model_alias: str = "ocr",
) -> dd.DataDesignerConfigBuilder:
    model_configs = [
        dd.ModelConfig(
            alias=model_alias,
            model=NEMOTRON_PARSE_MODEL,
            provider=VLLM_PROVIDER_NAME,
            # Health check sends a text-only probe; this model requires image
            # input, so the check would fail. Skip it.
            skip_health_check=True,
            inference_parameters=dd.ChatCompletionInferenceParams(
                temperature=0,
                timeout=60,
                max_parallel_requests=32,
                extra_body={
                    "skip_special_tokens": False,
                    "top_k": 1,
                    "repetition_penalty": 1.1,
                },
            ),
        ),
    ]

    config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

    config_builder.with_seed_dataset(
        dd.LocalFileSeedSource(path=seed_path),
        sampling_strategy=dd.SamplingStrategy.ORDERED,
    )

    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="raw_ocr_output",
            model_alias=model_alias,
            prompt="",
            multi_modal_context=[
                dd.ImageContext(
                    # Expects a single-element JSON array from the per-page seed.
                    column_name="png_images_base64",
                    data_type=dd.ModalityDataType.BASE64,
                    image_format=dd.ImageFormat.PNG,
                ),
            ],
            drop=True,
        )
    )

    config_builder.add_column(
        dd.CustomColumnConfig(
            name="transcribed_texts",
            generator_function=parse_ocr_output,
        )
    )

    return config_builder


def create_dataset(
    config_builder: dd.DataDesignerConfigBuilder,
    num_records: int,
    vllm_endpoint: str,
    artifact_path: Path | str | None = None,
) -> DatasetCreationResults:
    model_providers = [
        dd.ModelProvider(
            name=VLLM_PROVIDER_NAME,
            endpoint=vllm_endpoint,
        ),
    ]
    data_designer = DataDesigner(
        artifact_path=artifact_path,
        model_providers=model_providers,
    )
    data_designer.set_run_config(dd.RunConfig(progress_bar=True, disable_early_shutdown=True))
    results = data_designer.create(config_builder, num_records=num_records, dataset_name="nemotron_parse_ocr")
    return results


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--vllm-endpoint",
        type=str,
        required=True,
        help="Base URL of the vLLM server hosting nemotron-parse (e.g. http://localhost:8000/v1)",
    )
    parser.add_argument("--seed-path", type=str, required=True, help="Path to the seed parquet file")
    parser.add_argument("--model-alias", type=str, default="ocr")
    parser.add_argument("--num-records", type=int, default=5)
    parser.add_argument("--artifact-path", type=str, default=None)
    args = parser.parse_args()

    config_builder = build_config(
        seed_path=args.seed_path,
        model_alias=args.model_alias,
    )
    results = create_dataset(
        config_builder,
        num_records=args.num_records,
        vllm_endpoint=args.vllm_endpoint,
        artifact_path=args.artifact_path,
    )

    print(f"Dataset saved to: {results.artifact_storage.final_dataset_path}")

    results.load_analysis().to_report()
