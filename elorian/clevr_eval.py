"""CLEVR evaluation: load examples, generate with Claude + GPT-4o, judge, output best.

Loads CLEVR VQA examples from HuggingFace, sends each image + user prompt to both
Claude and GPT-4o, uses an LLM judge to score each response independently, picks
the best response per record, and writes a parquet file in the original conversation
format.

Prerequisites:
    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...

Run:
    uv run python -m elorian.clevr_eval
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
from elorian.models import get_default_model_registry
from elorian.pipeline import _patched_format_base64_context  # noqa: F401 — triggers the monkey-patch

# Score dimensions for per-model evaluation
EVAL_SCORES: list[dd.Score] = [
    dd.Score(
        name="accuracy",
        description="How accurately does the response describe the image content?",
        options={
            1: "Mostly inaccurate or hallucinated",
            2: "Some correct details but significant errors",
            3: "Generally accurate with minor errors",
            4: "Accurate with very few issues",
            5: "Perfectly accurate",
        },
    ),
    dd.Score(
        name="detail",
        description="How detailed and comprehensive is the response?",
        options={
            1: "Extremely sparse",
            2: "Covers only obvious elements",
            3: "Moderate detail",
            4: "Good detail",
            5: "Exceptionally detailed",
        },
    ),
    dd.Score(
        name="coherence",
        description="How well-structured and coherent is the response?",
        options={
            1: "Incoherent",
            2: "Somewhat readable",
            3: "Reasonably clear",
            4: "Well-structured",
            5: "Exceptionally clear and organized",
        },
    ),
]

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


JUDGE_PROMPT_TEMPLATE = (
    "You are an expert evaluator of vision-language model outputs.\n\n"
    "Given the image and the following response, evaluate it on the scoring "
    "dimensions provided.\n\n"
    "User question: {{{{ user_prompt }}}}\n\n"
    "Response to evaluate:\n{{{{ {response_col} }}}}\n\n"
    "Score this response carefully."
)


def build_and_run(seed_df: pd.DataFrame, num_records: int) -> pd.DataFrame:
    """Build a DataDesigner config with both models + per-model judges, then run."""
    os.environ["DATA_DESIGNER_MODEL_BACKEND"] = "litellm_bridge"

    model_registry = get_default_model_registry()
    model_aliases = model_registry.aliases  # ["claude_vision", "gpt4o_vision"]

    # Collect all model configs: VLM models + one judge per model
    model_configs = list(model_registry.to_model_configs())
    model_configs.append(
        dd.ModelConfig(
            alias="judge",
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            inference_parameters=dd.ChatCompletionInferenceParams(
                max_tokens=2048,
                temperature=0.3,
            ),
            skip_health_check=True,
        )
    )

    builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)
    builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_df))

    # One response column per VLM model
    for alias in model_aliases:
        builder.add_column(
            dd.LLMTextColumnConfig(
                name=f"response_{alias}",
                model_alias=alias,
                prompt="{{ user_prompt }}",
                multi_modal_context=[
                    dd.ImageContext(column_name="base64_image"),
                ],
            )
        )

    # One judge column per model response (independent scoring)
    for alias in model_aliases:
        builder.add_column(
            dd.LLMJudgeColumnConfig(
                name=f"eval_{alias}",
                model_alias="judge",
                prompt=JUDGE_PROMPT_TEMPLATE.format(response_col=f"response_{alias}"),
                scores=EVAL_SCORES,
                multi_modal_context=[
                    dd.ImageContext(column_name="base64_image"),
                ],
            )
        )

    providers = model_registry.get_unique_providers()

    designer = DataDesigner(
        artifact_path="artifacts",
        model_providers=providers,
    )
    designer.validate(builder)
    preview = designer.preview(builder, num_records=num_records)
    return preview.dataset


def pick_best_response(result_df: pd.DataFrame, model_aliases: list[str]) -> pd.DataFrame:
    """Add a 'best_response' column by comparing judge scores across models.

    For each row, sums the three score dimensions per model and picks the model
    with the highest total. Ties go to the first model in the list.
    """
    score_dims = ["accuracy", "detail", "coherence"]

    best_responses = []
    best_models = []

    for _, row in result_df.iterrows():
        best_alias = model_aliases[0]
        best_total = -1

        for alias in model_aliases:
            total = 0
            for dim in score_dims:
                col = f"eval_{alias}_{dim}"
                if col in row and pd.notna(row[col]):
                    total += int(row[col])
            if total > best_total:
                best_total = total
                best_alias = alias

        best_responses.append(row[f"response_{best_alias}"])
        best_models.append(best_alias)

    result_df = result_df.copy()
    result_df["best_response"] = best_responses
    result_df["best_model"] = best_models
    return result_df


def format_output(result_df: pd.DataFrame, raw_records: list[dict]) -> list[dict]:
    """Merge best responses back into the original CLEVR conversation format.

    Output schema: {id, image, conversations, data_source, best_model}.
    """
    output = []
    for _, row in result_df.iterrows():
        raw = next((r for r in raw_records if r["id"] == row["id"]), None)
        if raw is None:
            continue

        user_msg = next(
            c["content"] for c in raw["conversations"] if c["role"] == "user"
        )
        image_b64 = pil_to_base64(raw["image"].convert("RGB"))

        output.append({
            "id": raw["id"],
            "image": image_b64,
            "conversations": [
                {"content": user_msg, "role": "user"},
                {"content": row["best_response"], "role": "assistant"},
            ],
            "data_source": raw.get("data_source", "CLEVR"),
            "best_model": row["best_model"],
        })
    return output


def main() -> None:
    num_examples = 5

    print("Loading CLEVR examples from HuggingFace...")
    seed_df, raw_records = load_clevr_seed(num_examples)
    print(f"Loaded {len(seed_df)} examples")
    print(f"Sample prompt: {seed_df['user_prompt'].iloc[0][:80]}...")

    print("\nGenerating responses with Claude + GPT-4o and judging...")
    result_df = build_and_run(seed_df, num_records=num_examples)
    print(f"Generated {len(result_df)} records")

    # Show available columns
    print(f"\nResult columns: {list(result_df.columns)}")

    model_aliases = get_default_model_registry().aliases
    result_df = pick_best_response(result_df, model_aliases)

    # Print score summary
    print("\n--- Score Summary ---")
    for _, row in result_df.iterrows():
        print(f"\nID: {row['id']}")
        for alias in model_aliases:
            scores = {
                dim: row.get(f"eval_{alias}_{dim}", "N/A")
                for dim in ["accuracy", "detail", "coherence"]
            }
            print(f"  {alias}: {scores}")
        print(f"  Winner: {row['best_model']}")

    print("\nFormatting output...")
    output = format_output(result_df, raw_records)

    # Write parquet
    parquet_path = "clevr_eval_output.parquet"
    parquet_df = pd.DataFrame(output)
    parquet_df["image"] = parquet_df["image"].apply(lambda b64: base64.b64decode(b64))
    parquet_df["conversations"] = parquet_df["conversations"].apply(json.dumps)
    parquet_df.to_parquet(parquet_path, index=False)
    print(f"Wrote {len(parquet_df)} records to {parquet_path}")

    # Preview first record
    if output:
        preview = {k: v for k, v in output[0].items() if k != "image"}
        preview["image"] = f"<base64 PNG, {len(output[0]['image'])} chars>"
        print("\n--- First record preview ---")
        print(json.dumps(preview, indent=2))


if __name__ == "__main__":
    main()
