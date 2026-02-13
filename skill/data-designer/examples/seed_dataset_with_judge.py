"""Clinical Notes Dataset — seed data + judge scoring patterns.

PATTERN REFERENCE ONLY — copy the structure, not the domain-specific values.

Demonstrates:
- Default model aliases (nvidia-text) — no manual ModelConfig needed
- LocalFileSeedSource with SamplingStrategy.SHUFFLE
- UUID and DATETIME samplers
- PERSON_FROM_FAKER sampler
- LLMTextColumnConfig referencing seed columns ({{ diagnosis }}, {{ symptoms }})
- LLMJudgeColumnConfig with two Score rubrics
- preview() + create() workflow
"""

import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner()

config_builder = dd.DataDesignerConfigBuilder()

# --- Seed dataset (columns: diagnosis, symptoms) ---

seed_source = dd.LocalFileSeedSource(path="medical_conditions.csv")
config_builder.with_seed_dataset(seed_source, sampling_strategy=dd.SamplingStrategy.SHUFFLE)

# --- Sampler columns ---

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="patient",
        sampler_type=dd.SamplerType.PERSON_FROM_FAKER,
        params=dd.PersonFromFakerSamplerParams(locale="en_US"),
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="patient_id",
        sampler_type=dd.SamplerType.UUID,
        params=dd.UUIDSamplerParams(prefix="PT-", short_form=True, uppercase=True),
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="visit_date",
        sampler_type=dd.SamplerType.DATETIME,
        params=dd.DatetimeSamplerParams(start="2024-01-01", end="2024-12-31"),
    )
)

# --- LLM text column referencing seed + sampler columns ---

config_builder.add_column(
    dd.LLMTextColumnConfig(
        name="clinical_notes",
        prompt=(
            "You are a physician writing clinical notes for patient "
            "{{ patient.first_name }} {{ patient.last_name }} "
            "(ID: {{ patient_id }}).\n\n"
            "Visit date: {{ visit_date }}\n"
            "Diagnosis: {{ diagnosis }}\n"
            "Presenting symptoms: {{ symptoms }}\n\n"
            "Write detailed clinical notes including assessment and treatment plan."
        ),
        model_alias="nvidia-text",
    )
)

# --- LLM judge column with two scoring rubrics ---

config_builder.add_column(
    dd.LLMJudgeColumnConfig(
        name="quality_scores",
        prompt=(
            "Evaluate the following clinical notes:\n\n{{ clinical_notes }}\n\nRate the notes on the criteria below."
        ),
        scores=[
            dd.Score(
                name="Medical Accuracy",
                description=(
                    "Are the assessment and treatment plan consistent with the stated diagnosis and symptoms?"
                ),
                options={
                    1: "Major inaccuracies or contradictions",
                    2: "Some inaccuracies",
                    3: "Mostly accurate with minor issues",
                    4: "Accurate and clinically sound",
                    5: "Exemplary accuracy and clinical reasoning",
                },
            ),
            dd.Score(
                name="Completeness",
                description=("Does the note cover history, assessment, and treatment plan?"),
                options={
                    1: "Missing most required sections",
                    2: "Incomplete — key sections missing",
                    3: "Adequate but could be more thorough",
                    4: "Thorough with all major sections present",
                    5: "Comprehensive and publication-ready",
                },
            ),
        ],
        model_alias="nvidia-text",
    )
)

# --- Preview then create ---

preview = data_designer.preview(config_builder, num_records=3)
preview.display_sample_record()

results = data_designer.create(config_builder, num_records=100, dataset_name="clinical-notes")
dataset = results.load_dataset()
analysis = results.load_analysis()
