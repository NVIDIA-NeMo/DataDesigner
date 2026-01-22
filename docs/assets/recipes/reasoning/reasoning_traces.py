# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from data_designer.essentials import (
    CategorySamplerParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    SamplerType,
    SubcategorySamplerParams,
)
from data_designer.interface.results import DatasetCreationResults


def build_config(model_alias: str) -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder()

    config_builder.add_column(
        SamplerColumnConfig(
            name="domain",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=[
                    "Family Dynamics",
                    "Workplace Challenges",
                    "Friendship Moments",
                    "Community Interactions",
                    "Personal Well-being",
                    "Unexpected Encounters",
                ]
            ),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="theme",
            sampler_type=SamplerType.SUBCATEGORY,
            params=SubcategorySamplerParams(
                category="domain",
                values={
                    "Family Dynamics": ["Parenting Dilemmas", "Sibling Rivalries"],
                    "Workplace Challenges": [
                        "Communication Breakdowns",
                        "Leadership Dilemmas",
                    ],
                    "Friendship Moments": [
                        "Support & Understanding",
                        "Misunderstandings & Reconciliations",
                    ],
                    "Community Interactions": [
                        "Neighborhood Support",
                        "Cultural Celebrations",
                    ],
                    "Personal Well-being": ["Mental Health", "Self-care & Reflection"],
                    "Unexpected Encounters": [
                        "Serendipitous Meetings",
                        "Moments of Realization",
                    ],
                },
            ),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="complexity",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["Basic", "Intermediate", "Advanced"]),
        )
    )

    config_builder.add_column(
        LLMTextColumnConfig(
            name="scenario",
            model_alias=model_alias,
            system_prompt=(
                "You are an empathic reasoning agent. Your task is to generate realistic and compassionate reasoning traces for "
                "common day-to-day situations. Adopt a caring and supportive tone as you provide detailed insights into human experiences and emotions."
            ),
            prompt=(
                "Generate a clear and concise everyday scenario for the {{ domain }} domain, theme {{ theme }}, and "
                "complexity {{ complexity }}, where empathy and understanding play a crucial role. Focus on a situation that "
                "highlights emotional challenges or opportunities for compassionate support, and include a specific "
                "question or request for help that clearly outlines a problem or challenge needing resolution."
            ),
        )
    )

    config_builder.add_column(
        LLMTextColumnConfig(
            name="reasoning_trace",
            model_alias=model_alias,
            system_prompt=(
                "You are an empathic reasoning agent. Provide thoughtful, step-by-step reasoning that demonstrates "
                "emotional intelligence and compassion. Use <think> tags to show your reasoning process, then provide "
                "a final compassionate response in <answer> tags."
            ),
            prompt=(
                "Given this scenario: {{ scenario }}\n\n"
                "Provide a detailed reasoning trace that:\n"
                "1. Analyzes the emotional dynamics at play\n"
                "2. Considers multiple perspectives\n"
                "3. Shows empathy and understanding\n"
                "4. Offers constructive guidance\n\n"
                "Format: <think>your reasoning process</think><answer>your compassionate response</answer>"
            ),
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
    parser.add_argument("--model-alias", type=str, default="openai-text")
    parser.add_argument("--num-records", type=int, default=5)
    parser.add_argument("--artifact-path", type=str, default=None)
    args = parser.parse_args()

    config_builder = build_config(model_alias=args.model_alias)
    results = create_dataset(config_builder, num_records=args.num_records, artifact_path=args.artifact_path)

    print(f"Dataset saved to: {results.artifact_storage.final_dataset_path}")

    results.load_analysis().to_report()

