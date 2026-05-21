# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import data_designer.config as dd

PROVIDER = "nvidia-internal"

MODEL_CONFIGS = [
    dd.ModelConfig(
        alias="gpt-oss-20b",
        model="nvidia/openai/gpt-oss-20b",
        provider=PROVIDER,
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=0.2,
            top_p=0.8,
            max_tokens=32,
            max_parallel_requests=2,
        ),
    ),
    dd.ModelConfig(
        alias="nemotron-31b",
        model="nvidia/nvidia/nemotron-nano-31b-v3",
        provider=PROVIDER,
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=0.2,
            top_p=0.8,
            max_tokens=32,
            max_parallel_requests=2,
        ),
    ),
    dd.ModelConfig(
        alias="gpt-oss-120b",
        model="nvcf/openai/gpt-oss-120b",
        provider=PROVIDER,
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=0.2,
            top_p=0.8,
            max_tokens=32,
            max_parallel_requests=2,
        ),
    ),
]

SYSTEM_PROMPT = (
    "You generate compact customer-support metadata. Answer in one short phrase and do not include markdown."
)


def load_config_builder() -> dd.DataDesignerConfigBuilder:
    config_builder = dd.DataDesignerConfigBuilder(model_configs=MODEL_CONFIGS)

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="topic",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=[
                    "GPU quota increase",
                    "Training job timeout",
                    "Invoice dispute",
                    "Model quality regression",
                    "Dataset permission issue",
                    "Deployment latency spike",
                    "Account recovery",
                    "Documentation clarification",
                ],
            ),
        )
    )
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="customer_tone",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["calm", "urgent", "frustrated", "curious"]),
        )
    )
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="gpt_oss_20b_intent",
            model_alias="gpt-oss-20b",
            system_prompt=SYSTEM_PROMPT,
            prompt=("Classify the likely support intent for a {{ customer_tone }} customer asking about: {{ topic }}."),
        )
    )
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="nemotron_31b_next_action",
            model_alias="nemotron-31b",
            system_prompt=SYSTEM_PROMPT,
            prompt=("Suggest the first support action for a {{ customer_tone }} customer asking about: {{ topic }}."),
        )
    )
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="gpt_oss_120b_risk_signal",
            model_alias="gpt-oss-120b",
            system_prompt=SYSTEM_PROMPT,
            prompt=("Name the main operational risk for a {{ customer_tone }} customer asking about: {{ topic }}."),
        )
    )

    return config_builder
