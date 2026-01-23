# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from data_designer.essentials import (
    CategorySamplerParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    ExpressionColumnConfig,
    GaussianSamplerParams,
    LLMTextColumnConfig,
    PersonSamplerParams,
    SamplerColumnConfig,
    SamplerType,
    SubcategorySamplerParams,
)
from data_designer.interface.results import DatasetCreationResults


def build_config(model_alias: str) -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder()

    config_builder.add_column(
        SamplerColumnConfig(
            name="patient",
            sampler_type=SamplerType.PERSON,
            params=PersonSamplerParams(locale="en_US", age_range=[18, 85]),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="physician",
            sampler_type=SamplerType.PERSON,
            params=PersonSamplerParams(locale="en_US", age_range=[30, 70]),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="specialty",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=[
                    "Internal Medicine",
                    "Cardiology",
                    "Orthopedics",
                    "Neurology",
                    "Pediatrics",
                ],
            ),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="visit_type",
            sampler_type=SamplerType.SUBCATEGORY,
            params=SubcategorySamplerParams(
                category="specialty",
                values={
                    "Internal Medicine": ["Annual Physical", "Follow-up", "Acute Care"],
                    "Cardiology": ["Chest Pain Evaluation", "Hypertension Management", "Post-MI Follow-up"],
                    "Orthopedics": ["Joint Pain", "Sports Injury", "Post-Surgical"],
                    "Neurology": ["Headache Evaluation", "Seizure Management", "Memory Issues"],
                    "Pediatrics": ["Well-Child Visit", "Sick Visit", "Vaccination"],
                },
            ),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="vital_signs_systolic",
            sampler_type=SamplerType.GAUSSIAN,
            params=GaussianSamplerParams(mean=120, stddev=15),
            convert_to="int",
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="vital_signs_diastolic",
            sampler_type=SamplerType.GAUSSIAN,
            params=GaussianSamplerParams(mean=80, stddev=10),
            convert_to="int",
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="heart_rate",
            sampler_type=SamplerType.GAUSSIAN,
            params=GaussianSamplerParams(mean=75, stddev=12),
            convert_to="int",
        )
    )

    config_builder.add_column(
        ExpressionColumnConfig(
            name="patient_info",
            expr="Patient: {{ patient.first_name }} {{ patient.last_name }}, DOB: {{ patient.birth_date }}, MRN: {{ patient.national_id }}",
        )
    )

    config_builder.add_column(
        LLMTextColumnConfig(
            name="chief_complaint",
            model_alias=model_alias,
            prompt=(
                "Generate a realistic chief complaint for a {{ specialty }} visit of type {{ visit_type }}. "
                "Keep it brief (1-2 sentences) as it would appear in a medical record."
            ),
        )
    )

    config_builder.add_column(
        LLMTextColumnConfig(
            name="physician_notes",
            model_alias=model_alias,
            prompt=(
                "Write detailed physician notes for:\n"
                "Patient: {{ patient.first_name }} {{ patient.last_name }}, DOB: {{ patient.birth_date }}\n"
                "Physician: Dr. {{ physician.last_name }} ({{ specialty }})\n"
                "Visit Type: {{ visit_type }}\n"
                "Chief Complaint: {{ chief_complaint }}\n"
                "Vital Signs: BP {{ vital_signs_systolic }}/{{ vital_signs_diastolic }}, HR {{ heart_rate }}\n\n"
                "Include: History of Present Illness, Physical Examination, Assessment, and Plan. "
                "Use appropriate medical terminology and SOAP note format."
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
