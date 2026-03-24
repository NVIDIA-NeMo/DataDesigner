"""Regenerate assistant responses for CLEVR VQA data using OpenAI via DataDesigner.

Reads 5 examples from LLaVA-OneVision CLEVR dataset, sends each image + user
prompt to OpenAI GPT-4o, and outputs the dataset in the original conversation
JSON format with the new assistant responses.

Prerequisites:
    export OPENAI_API_KEY=...

Run:
    uv run python -m elorian.clevr_regen
"""

from __future__ import annotations

import base64
import json
import os

import pandas as pd
from datasets import load_dataset

import data_designer.config as dd
from data_designer.interface import DataDesigner

from elorian.image_utils import pil_to_base64, resize_image
from elorian.models import PROVIDERS
from elorian.pipeline import _patched_format_base64_context  # noqa: F401 — triggers the monkey-patch


def load_clevr_seed(num_examples: int = 5) -> tuple[pd.DataFrame, list[dict]]:
    """Load CLEVR examples from HuggingFace and prepare a seed DataFrame.

    Returns:
        seed_df: DataFrame with columns [id, base64_image, user_prompt]
        raw_records: Original records for reconstructing the output JSON.
    """
    dataset = load_dataset(
        "mvp-lab/LLaVA-OneVision-1.5-Instruct-Data",
        "CLEVR",
        split="train",
        streaming=True,
    )

    raw_records = []
    seed_rows = []

    for example in dataset.take(num_examples):
        image = example["image"].convert("RGB")
        image = resize_image(image, 512)

        # Extract user prompt (strip the <image>\n prefix)
        user_msg = next(
            c["content"] for c in example["conversations"] if c["role"] == "user"
        )
        user_prompt = user_msg.replace("<image>\n", "").strip()

        seed_rows.append({
            "id": example["id"],
            "base64_image": pil_to_base64(image),
            "user_prompt": user_prompt,
        })
        raw_records.append(example)

    return pd.DataFrame(seed_rows), raw_records


def build_and_run(seed_df: pd.DataFrame, num_records: int) -> pd.DataFrame:
    """Build a DataDesigner config and generate new assistant responses."""
    os.environ["DATA_DESIGNER_MODEL_BACKEND"] = "litellm_bridge"

    model_configs = [
        dd.ModelConfig(
            alias="gpt4o",
            model="gpt-4o",
            provider="openai",
            inference_parameters=dd.ChatCompletionInferenceParams(
                max_tokens=1024,
                temperature=0.7,
            ),
            skip_health_check=True,
        ),
    ]

    builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)
    builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_df))

    builder.add_column(
        dd.LLMTextColumnConfig(
            name="new_response",
            model_alias="gpt4o",
            prompt="{{ user_prompt }}",
            multi_modal_context=[
                dd.ImageContext(column_name="base64_image"),
            ],
        )
    )

    designer = DataDesigner(
        artifact_path="artifacts",
        model_providers=[PROVIDERS["openai"]],
    )
    designer.validate(builder)
    preview = designer.preview(builder, num_records=num_records)
    return preview.dataset


def format_output(
    result_df: pd.DataFrame, raw_records: list[dict],
) -> list[dict]:
    """Merge new responses back into the original conversation format.

    Output matches the input schema: {id, image, conversations, data_source}.
    The ``image`` field is stored as a base64-encoded PNG string so it can be
    serialised to JSON and converted to bytes for parquet.
    """
    output = []
    for _, row in result_df.iterrows():
        raw = next((r for r in raw_records if r["id"] == row["id"]), None)
        if raw is None:
            continue

        user_msg = next(
            c["content"] for c in raw["conversations"] if c["role"] == "user"
        )

        # Keep the original image as base64 PNG
        image_b64 = pil_to_base64(raw["image"].convert("RGB"))

        output.append({
            "id": raw["id"],
            "image": image_b64,
            "conversations": [
                {"content": user_msg, "role": "user"},
                {"content": row["new_response"], "role": "assistant"},
            ],
            "data_source": raw.get("data_source", "CLEVR"),
        })
    return output


def main() -> None:
    num_examples = 5

    print("Loading CLEVR examples from HuggingFace...")
    seed_df, raw_records = load_clevr_seed(num_examples)
    print(f"Loaded {len(seed_df)} examples")
    print(f"Sample prompt: {seed_df['user_prompt'].iloc[0][:80]}...")

    print("\nGenerating new responses with GPT-4o...")
    result_df = build_and_run(seed_df, num_records=num_examples)
    print(f"Generated {len(result_df)} responses")

    print("\nFormatting output...")
    output = format_output(result_df, raw_records)

    # Parquet output (image as raw PNG bytes, conversations as JSON string)
    parquet_path = "clevr_regen_output.parquet"
    parquet_df = pd.DataFrame(output)
    parquet_df["image"] = parquet_df["image"].apply(
        lambda b64: base64.b64decode(b64)
    )
    parquet_df["conversations"] = parquet_df["conversations"].apply(json.dumps)
    parquet_df.to_parquet(parquet_path, index=False)
    print(f"Wrote {len(parquet_df)} records to {parquet_path}")

    # Print first record as preview
    if output:
        preview_record = {k: v for k, v in output[0].items() if k != "image"}
        preview_record["image"] = f"<base64 PNG, {len(output[0]['image'])} chars>"
        print("\n--- First record preview ---")
        print(json.dumps(preview_record, indent=2))


if __name__ == "__main__":
    main()
