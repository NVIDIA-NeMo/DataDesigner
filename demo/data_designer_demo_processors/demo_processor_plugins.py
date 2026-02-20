# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Processor Plugins Demo
#
# Demonstrates the `regex-filter` and `semantic-dedup` processor plugins
# with a simple product review pipeline.

# %%
import pandas as pd
from data_designer_demo_processors.regex_filter.config import RegexFilterProcessorConfig
from data_designer_demo_processors.semantic_dedup.config import SemanticDedupProcessorConfig

import data_designer.config as dd
from data_designer.interface import DataDesigner

# %% [markdown]
# ### Setup

# %%
MODEL_ALIAS = "openai-text"

data_designer = DataDesigner()

model_configs = [
    dd.ModelConfig(
        alias=MODEL_ALIAS,
        model="nvidia/nemotron-3-nano-30b-a3b",
        provider="nvidia",
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=1.0,
            max_tokens=512,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
    )
]

# %% [markdown]
# ### Seed data
#
# A simple CSV with topics and languages. The regex filter will keep only English rows.

# %%
seed_data = pd.DataFrame(
    {
        "topic": [
            "machine learning",
            "aprendizaje automático",
            "cloud computing",
            "computación en la nube",
            "quantum computing",
            "web development",
            "desarrollo web",
            "cybersecurity",
        ],
        "language": ["en", "es", "en", "es", "en", "en", "es", "en"],
    }
)
print(f"Seed data: {len(seed_data)} rows")
seed_data

# %% [markdown]
# ### Build the config

# %%
config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

config_builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_data))

config_builder.add_column(
    dd.LLMTextColumnConfig(
        name="explanation",
        prompt=("Write a short (2-3 sentence) explanation of {{ topic }}. Be concise and informative."),
        model_alias=MODEL_ALIAS,
    )
)

# %% [markdown]
# ### Add processor plugins
#
# 1. **RegexFilter** (process_before_batch): keep only English rows
# 2. **SemanticDedup** (process_after_generation): remove near-duplicate explanations

# %%
config_builder.add_processor(
    RegexFilterProcessorConfig(
        name="english_only",
        column="language",
        pattern="^en$",
    )
)

config_builder.add_processor(
    SemanticDedupProcessorConfig(
        name="dedup_explanations",
        column="explanation",
        similarity_threshold=0.9,
    )
)

data_designer.validate(config_builder)

# %% [markdown]
# ### Preview

# %%
preview = data_designer.preview(config_builder, num_records=4)
preview.dataset

# %% [markdown]
# ### Full run

# %%
results = data_designer.create(config_builder, num_records=10, dataset_name="processor-plugins-demo")
dataset = results.load_dataset()
print(f"Final dataset: {len(dataset)} rows")
dataset
