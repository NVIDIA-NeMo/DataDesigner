"""Multimodal evaluation pipeline built on DataDesigner.

Orchestrates:
1. Seed dataset (images) loading
2. One response column per registered VLM model
3. One judge column per registered LLM judge
4. Winner selection via expression column

Usage:
    pipeline = MultimodalEvalPipeline(seed_df=df)
    # optionally customise registries before running
    pipeline.model_registry.register(ModelSpec(...))
    preview = pipeline.preview(num_records=3)
    results = pipeline.run(num_records=50)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import pandas as pd

import data_designer.config as dd
from data_designer.config.models import ImageContext
from data_designer.config.utils.image_helpers import decode_base64_image, detect_image_format
from data_designer.interface import DataDesigner

from elorian.judges import JudgeRegistry, get_default_judge_registry
from elorian.models import ModelRegistry, get_default_model_registry, get_provider

logger = logging.getLogger(__name__)


def _patched_format_base64_context(self: ImageContext, base64_data: str) -> dict[str, str]:
    """Patch for ImageContext._format_base64_context that omits the ``format`` key.

    The upstream method includes ``"format": "png"`` which Anthropic's API
    (via LiteLLM) rejects — it expects ``"image/png"`` as media_type. The
    media type is already encoded in the data URL, so the key is redundant.
    """
    image_format = self.image_format
    if image_format is None:
        image_bytes = decode_base64_image(base64_data)
        image_format = detect_image_format(image_bytes)
    return {
        "url": f"data:image/{image_format.value};base64,{base64_data}",
    }


# Apply the patch once at import time.
ImageContext._format_base64_context = _patched_format_base64_context  # type: ignore[assignment]


class MultimodalEvalPipeline:
    """End-to-end pipeline for multimodal VLM evaluation.

    Args:
        seed_df: DataFrame containing at least a ``base64_image`` column
            (and optionally ``uuid``, ``label``, etc.).
        image_column: Name of the column holding base64-encoded images.
        prompt: Vision prompt sent to every VLM model.
        model_registry: Registry of VLM models to evaluate. Defaults to
            Claude + GPT-4o.
        judge_registry: Registry of LLM judges. Defaults to Claude judge.
        artifact_path: Where DataDesigner stores generated artifacts.
    """

    def __init__(
        self,
        seed_df: pd.DataFrame,
        *,
        image_column: str = "base64_image",
        prompt: str = (
            "Provide a detailed description of the content in this image "
            "in Markdown format. Describe the main subject, background, "
            "colors, and any notable details."
        ),
        model_registry: ModelRegistry | None = None,
        judge_registry: JudgeRegistry | None = None,
        artifact_path: str | Path = "artifacts",
    ) -> None:
        self.seed_df = seed_df
        self.image_column = image_column
        self.prompt = prompt
        self.model_registry = model_registry or get_default_model_registry()
        self.judge_registry = judge_registry or get_default_judge_registry()
        self.artifact_path = Path(artifact_path)

    def _build_config(self) -> dd.DataDesignerConfigBuilder:
        """Build the DataDesigner config with all models and judges."""
        model_configs = (
            self.model_registry.to_model_configs()
            + [spec.to_model_config() for spec in self.judge_registry.specs]
        )
        builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

        # Seed dataset
        builder.with_seed_dataset(dd.DataFrameSeedSource(df=self.seed_df))

        # One response column per VLM model
        model_aliases = self.model_registry.aliases
        for alias in model_aliases:
            builder.add_column(
                dd.LLMTextColumnConfig(
                    name=f"response_{alias}",
                    model_alias=alias,
                    prompt=self.prompt,
                    multi_modal_context=[
                        dd.ImageContext(column_name=self.image_column),
                    ],
                )
            )

        # One judge column per judge
        for judge_spec in self.judge_registry.specs:
            judge_prompt = judge_spec.build_prompt(model_aliases)
            builder.add_column(
                dd.LLMJudgeColumnConfig(
                    name=f"eval_{judge_spec.alias}",
                    model_alias=judge_spec.alias,
                    prompt=judge_prompt,
                    scores=judge_spec.scores,
                    multi_modal_context=[
                        dd.ImageContext(column_name=self.image_column),
                    ],
                )
            )

        return builder

    def _collect_providers(self) -> list[dd.ModelProvider]:
        """Collect all unique ModelProviders from model and judge registries."""
        seen: dict[str, dd.ModelProvider] = {}
        for spec in self.model_registry.specs:
            if spec.provider not in seen:
                seen[spec.provider] = get_provider(spec.provider)
        for spec in self.judge_registry.specs:
            if spec.provider not in seen:
                seen[spec.provider] = get_provider(spec.provider)
        return list(seen.values())

    def _get_data_designer(self) -> DataDesigner:
        """Create a DataDesigner interface instance with the required providers."""
        # Use LiteLLM bridge so provider_type is used as the routing prefix.
        os.environ["DATA_DESIGNER_MODEL_BACKEND"] = "litellm_bridge"

        return DataDesigner(
            artifact_path=self.artifact_path,
            model_providers=self._collect_providers(),
        )

    def preview(self, num_records: int = 2) -> object:
        """Generate a small preview to iterate on prompt quality.

        Args:
            num_records: Number of records to preview.

        Returns:
            DataDesigner PreviewResults object.
        """
        builder = self._build_config()
        designer = self._get_data_designer()
        designer.validate(builder)
        return designer.preview(builder, num_records=num_records)

    def run(
        self,
        num_records: int = 10,
        dataset_name: str = "elorian-eval",
    ) -> object:
        """Run full-scale evaluation.

        Args:
            num_records: Number of records to generate.
            dataset_name: Name for the output dataset.

        Returns:
            DataDesigner DatasetCreationResults object.
        """
        builder = self._build_config()
        designer = self._get_data_designer()
        designer.validate(builder)
        return designer.create(
            builder,
            num_records=num_records,
            dataset_name=dataset_name,
        )

    def describe(self) -> str:
        """Return a human-readable summary of the pipeline configuration."""
        lines = ["Multimodal Eval Pipeline Configuration", "=" * 40]
        lines.append(f"\nImage column: {self.image_column}")
        lines.append(f"Prompt: {self.prompt[:80]}...")
        lines.append(f"\nVLM Models ({len(self.model_registry.specs)}):")
        for spec in self.model_registry.specs:
            lines.append(f"  - {spec.alias}: {spec.model_id} ({spec.description})")
        lines.append(f"\nJudges ({len(self.judge_registry.specs)}):")
        for spec in self.judge_registry.specs:
            score_dims = ", ".join(s.name for s in spec.scores)
            lines.append(f"  - {spec.alias}: {spec.model_id} (scores: {score_dims})")
        return "\n".join(lines)
