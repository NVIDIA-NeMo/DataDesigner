"""Writer-Editor Dataset — custom column generator with multi-model orchestration.

PATTERN REFERENCE ONLY — copy the structure, not the domain-specific values.

Demonstrates:
- Default model aliases (nvidia-text, nvidia-reasoning) — no manual ModelConfig needed
- @custom_column_generator decorator with model_aliases and side_effect_columns
- Multi-model orchestration (writer + editor) within a single custom column
- GenerationStrategy.CELL_BY_CELL for row-based LLM access
- CATEGORY sampler for topic diversity
- preview() + create() workflow
"""

import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner()

config_builder = dd.DataDesignerConfigBuilder()

# --- Sampler columns ---

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="topic",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=[
                "renewable energy",
                "space exploration",
                "artificial intelligence",
                "ocean conservation",
                "quantum computing",
            ],
        ),
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="audience",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["general public", "technical experts", "students"],
            weights=[0.4, 0.3, 0.3],
        ),
    )
)


# --- Custom column: writer-editor pattern ---


@dd.custom_column_generator(
    required_columns=["topic", "audience"],
    side_effect_columns=["draft", "editorial_feedback"],
    model_aliases=["nvidia-text", "nvidia-reasoning"],
)
def writer_editor(row: dict, generator_params: None, models: dict) -> dict:
    # Step 1: Writer drafts the article
    draft, _ = models["nvidia-text"].generate(
        prompt=(f"Write a 200-word article about {row['topic']} for a {row['audience']} audience."),
    )

    # Step 2: Editor provides critique
    feedback, _ = models["nvidia-reasoning"].generate(
        prompt=(f"Review this article and provide 3 specific improvement suggestions:\n\n{draft}"),
    )

    # Step 3: Writer revises based on feedback
    final, _ = models["nvidia-text"].generate(
        prompt=(
            f"Revise this article based on the editorial feedback.\n\n"
            f"Original draft:\n{draft}\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Write the improved version."
        ),
    )

    row["final_article"] = final
    row["draft"] = draft
    row["editorial_feedback"] = feedback
    return row


config_builder.add_column(
    dd.CustomColumnConfig(
        name="final_article",
        generator_function=writer_editor,
        generation_strategy=dd.GenerationStrategy.CELL_BY_CELL,
    )
)

# --- Preview then create ---

preview = data_designer.preview(config_builder, num_records=3)
preview.display_sample_record()

results = data_designer.create(config_builder, num_records=50, dataset_name="writer-editor")
dataset = results.load_dataset()
analysis = results.load_analysis()
