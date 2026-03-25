"""Example: evaluate Claude vs GPT-4o on pet images using an LLM judge.

Prerequisites:
    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...

Run:
    uv run python -m elorian.run_example
"""

from __future__ import annotations

import data_designer.config as dd

from elorian.image_utils import load_images_from_hf_dataset
from elorian.judges import JudgeRegistry, JudgeSpec, get_default_judge_registry
from elorian.models import ModelRegistry, ModelSpec, get_default_model_registry
from elorian.pipeline import MultimodalEvalPipeline


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Load images                                                      #
    # ------------------------------------------------------------------ #
    print("Loading images from HuggingFace...")
    seed_df = load_images_from_hf_dataset(
        "zh-plus/tiny-imagenet",
        split="valid",
        max_images=5,          # small for demo; increase for real eval
        target_height=512,
    )
    print(f"Loaded {len(seed_df)} images (columns: {list(seed_df.columns)})")

    # ------------------------------------------------------------------ #
    # 2. Configure models (defaults: Claude + GPT-4o)                     #
    # ------------------------------------------------------------------ #
    model_registry = get_default_model_registry()

    # Example: add a third model (uncomment when you have access)
    # from elorian.models import register_provider
    # register_provider(dd.ModelProvider(
    #     name="google", endpoint="https://generativelanguage.googleapis.com/v1",
    #     provider_type="gemini", api_key="GEMINI_API_KEY",
    # ))
    # model_registry.register(
    #     ModelSpec(
    #         alias="gemini_vision",
    #         model_id="gemini-2.0-flash",
    #         provider="google",
    #         description="Google Gemini 2.0 Flash (vision)",
    #     )
    # )

    # ------------------------------------------------------------------ #
    # 3. Configure judges (default: Claude judge)                         #
    # ------------------------------------------------------------------ #
    judge_registry = get_default_judge_registry()

    # Example: add a second judge (uncomment to use)
    # judge_registry.register(
    #     JudgeSpec(
    #         alias="judge-gpt4o",
    #         model_id="openai/gpt-4o",
    #         description="GPT-4o as LLM judge",
    #     )
    # )

    # ------------------------------------------------------------------ #
    # 4. Build and run the pipeline                                       #
    # ------------------------------------------------------------------ #
    pipeline = MultimodalEvalPipeline(
        seed_df=seed_df,
        model_registry=model_registry,
        judge_registry=judge_registry,
    )

    # Show what's configured
    print("\n" + pipeline.describe())

    # Preview a couple of records first
    print("\n--- Running preview (2 records) ---")
    preview = pipeline.preview(num_records=2)
    print("\nPreview dataset columns:", list(preview.dataset.columns))
    print(preview.dataset.head())

    # Full run
    # print("\n--- Running full evaluation ---")
    # results = pipeline.run(num_records=5, dataset_name="pet-eval")
    # dataset = results.load_dataset()
    # print(f"\nGenerated {len(dataset)} records")
    # print(dataset.head())


if __name__ == "__main__":
    main()
