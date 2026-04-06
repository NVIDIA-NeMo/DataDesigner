# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
#     "pydantic",
# ]
# ///
"""Conditional Generation with skip.when Recipe

Generate a customer support ticket dataset where expensive downstream columns
(escalation plan, detailed risk analysis) are only generated when the ticket
actually needs them. Tickets that don't require escalation get None for the
escalation columns, saving LLM calls without post-hoc filtering.

The pipeline demonstrates three skip patterns:
  1. Expression gate — skip a column when a Jinja2 condition is truthy.
  2. Skip propagation — downstream columns auto-skip when their dependency
     was skipped (default behavior via propagate_skip=True).
  3. Propagation opt-out — a column that depends on a skippable upstream
     but handles None gracefully by setting propagate_skip=False.

Prerequisites:
    - OPENAI_API_KEY environment variable for OpenAI provider model aliases
      (default model alias is "openai-text").

Run:
    # Basic usage (generates 10 records by default)
    uv run conditional_generation_with_skip.py

    # For help message and available options
    uv run conditional_generation_with_skip.py --help
"""

from pathlib import Path

from pydantic import BaseModel, Field

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults


class TicketClassification(BaseModel):
    risk_level: str = Field(description="One of: low, medium, high")
    category: str = Field(description="Support category, e.g. billing, technical, account-security, general")
    requires_escalation: bool = Field(description="Whether the ticket needs escalation to a senior agent")


def build_config(model_alias: str) -> dd.DataDesignerConfigBuilder:
    config_builder = dd.DataDesignerConfigBuilder()

    # --- Seed columns ---

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="domain",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=[
                    "Cloud Infrastructure",
                    "E-Commerce",
                    "Banking",
                    "Healthcare IT",
                    "Telecom",
                ]
            ),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="needs_escalation",
            sampler_type=dd.SamplerType.BERNOULLI,
            params=dd.BernoulliSamplerParams(p=0.4),
        )
    )

    # --- LLM columns ---

    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="scenario",
            model_alias=model_alias,
            prompt=(
                "You are a support ticket generator for the {{ domain }} industry.\n"
                "{% if needs_escalation == 1 %}"
                "Create a CRITICAL support scenario that clearly requires senior agent escalation "
                "(e.g. data breach, service outage, compliance violation).\n"
                "{% else %}"
                "Create a ROUTINE support scenario that can be resolved by a frontline agent "
                "(e.g. password reset, billing question, feature inquiry).\n"
                "{% endif %}"
                "Write 2-3 sentences describing the customer's issue."
            ),
        )
    )

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="classification",
            model_alias=model_alias,
            prompt=(
                "Classify this support ticket:\n\n"
                "{{ scenario }}\n\n"
                "Determine the risk level, category, and whether it requires escalation."
            ),
            output_format=TicketClassification,
        )
    )

    # --- Conditionally generated columns ---

    # Pattern 1: Expression gate — only generate an escalation plan when the
    # ticket needs escalation. Rows where needs_escalation == 0 get value=None.
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="escalation_plan",
            model_alias=model_alias,
            prompt=(
                "A customer in the {{ domain }} industry reported:\n"
                "{{ scenario }}\n\n"
                "Classification: {{ classification }}\n\n"
                "Write a concise escalation plan: who to notify, what actions to take, "
                "and the expected resolution timeline."
            ),
            skip=dd.SkipConfig(when="{{ needs_escalation == 0 }}"),
        )
    )

    # Pattern 2: Skip propagation (default) — resolution_notes depends on
    # escalation_plan. When escalation_plan is skipped, resolution_notes
    # auto-skips too because propagate_skip defaults to True.
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="resolution_notes",
            model_alias=model_alias,
            prompt=(
                "Given this escalation plan:\n"
                "{{ escalation_plan }}\n\n"
                "Write internal resolution notes summarizing the steps taken "
                "and the final outcome."
            ),
        )
    )

    # Pattern 3: Propagation opt-out — customer_summary also depends on
    # escalation_plan, but it sets propagate_skip=False so it always generates.
    # The prompt handles the None case with a Jinja2 conditional.
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="customer_summary",
            model_alias=model_alias,
            propagate_skip=False,
            prompt=(
                "Write a brief customer-facing summary for this ticket:\n"
                "Scenario: {{ scenario }}\n"
                "Classification: {{ classification }}\n"
                "{% if escalation_plan %}"
                "Escalation plan: {{ escalation_plan }}\n"
                "{% endif %}"
                "Keep it professional and under 3 sentences."
            ),
        )
    )

    # Expression gate on a structured output field — only generate a detailed
    # risk analysis for high-risk tickets.
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="detailed_risk_analysis",
            model_alias=model_alias,
            prompt=(
                "Perform a detailed risk analysis for this {{ domain }} support ticket:\n"
                "{{ scenario }}\n\n"
                "Classification: {{ classification }}\n\n"
                "Cover: root cause hypothesis, blast radius, mitigation steps, "
                "and preventive measures."
            ),
            skip=dd.SkipConfig(when='{{ classification.risk_level == "low" }}'),
        )
    )

    return config_builder


def create_dataset(
    config_builder: dd.DataDesignerConfigBuilder,
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
    parser.add_argument("--num-records", type=int, default=10)
    parser.add_argument("--artifact-path", type=str, default=None)
    args = parser.parse_args()

    config_builder = build_config(model_alias=args.model_alias)
    results = create_dataset(config_builder, num_records=args.num_records, artifact_path=args.artifact_path)

    print(f"Dataset saved to: {results.artifact_storage.final_dataset_path}")
