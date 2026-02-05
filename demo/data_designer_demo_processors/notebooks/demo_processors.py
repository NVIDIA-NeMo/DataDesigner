# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demo: Processor Plugins with PRE_GENERATION and POST_GENERATION stages.

This notebook demonstrates:
1. RegexFilterProcessor (PRE_GENERATION) - filters seed data before generation
2. SemanticDedupProcessor (POST_GENERATION) - deduplicates final dataset

Run cells with `#%%` markers in VS Code or PyCharm.
"""

# %% Imports
import tempfile
from pathlib import Path

import pandas as pd
from data_designer_demo_processors.regex_filter import RegexFilterProcessorConfig
from data_designer_demo_processors.semantic_dedup import SemanticDedupProcessorConfig

import data_designer.config as dd
from data_designer.interface import DataDesigner

# %% Create seed data with some rows we want to filter out
seed_data = pd.DataFrame(
    {
        "topic": [
            "Python programming",
            "Machine learning",
            "SPAM: Buy now!",  # Will be filtered by regex
            "Data science",
            "SPAM: Click here",  # Will be filtered by regex
            "Natural language processing",
            "Computer vision",
        ],
        "difficulty": ["beginner", "advanced", "N/A", "intermediate", "N/A", "advanced", "advanced"],
    }
)

print("Seed data before PRE_GENERATION filtering:")
print(seed_data)
print(f"Total rows: {len(seed_data)}")

# %% Setup temporary directory and save seed data
output_dir = Path(tempfile.mkdtemp())
seed_path = output_dir / "seed.parquet"
seed_data.to_parquet(seed_path, index=False)

# %% Build the Data Designer configuration (uses default openai-text model)
config_builder = dd.DataDesignerConfigBuilder()

# Add seed dataset
config_builder.with_seed_dataset(dd.LocalFileSeedSource(path=str(seed_path)))

# Add LLM column to generate explanations
config_builder.add_column(
    dd.LLMTextColumnConfig(
        name="explanation",
        prompt="""Write a brief one-sentence explanation of the topic: {{ topic }}
Difficulty level: {{ difficulty }}

Keep it concise and educational.""",
        model_alias="openai-text",
    )
)

# Add PRE_GENERATION processor to filter out spam rows
config_builder.add_processor(
    RegexFilterProcessorConfig(
        name="filter_spam",
        column="topic",
        pattern=r"^SPAM:",
        invert=True,  # Keep rows that do NOT match (i.e., filter out spam)
    )
)

# Add POST_GENERATION processor to deduplicate similar explanations
config_builder.add_processor(
    SemanticDedupProcessorConfig(
        name="dedup_explanations",
        column="explanation",
        similarity_threshold=0.85,
    )
)

print("Configuration created successfully!")
processor_configs = config_builder.get_processor_configs()
print(f"Processors configured: {[p.name for p in processor_configs]}")

# %% Run preview to test with a few records
data_designer = DataDesigner()

print("\nRunning preview (3 records)...")
preview = data_designer.preview(config_builder, num_records=3)

print("\nPreview dataset:")
print(preview.dataset)

# %% Run full generation
print("\nRunning full generation...")
results = data_designer.create(
    config_builder,
    num_records=5,
    dataset_name="processor-demo",
)

# Load the final dataset
final_dataset = results.load_dataset()

print("\nFinal dataset after all processors:")
print(final_dataset)
print(f"\nTotal rows in final dataset: {len(final_dataset)}")

# %% Summary
print("\n" + "=" * 60)
print("DEMO SUMMARY")
print("=" * 60)
print(f"Original seed rows: {len(seed_data)}")
print("After PRE_GENERATION (regex filter): Expected ~5 rows (SPAM removed)")
print(f"After POST_GENERATION (semantic dedup): {len(final_dataset)} rows")
