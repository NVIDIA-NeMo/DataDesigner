# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from data_designer.essentials import (
    BernoulliSamplerParams,
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
    UniformSamplerParams,
    UUIDSamplerParams,
)
from data_designer.interface.results import DatasetCreationResults


def build_config(model_alias: str) -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder()

    config_builder.add_column(
        SamplerColumnConfig(
            name="participant",
            sampler_type=SamplerType.PERSON,
            params=PersonSamplerParams(locale="en_US"),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="investigator",
            sampler_type=SamplerType.PERSON,
            params=PersonSamplerParams(locale="en_US"),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="study_id",
            sampler_type=SamplerType.UUID,
            params=UUIDSamplerParams(prefix="CT-", short_form=True, uppercase=True),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="trial_phase",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=["Phase I", "Phase II", "Phase III", "Phase IV"],
                weights=[0.2, 0.3, 0.4, 0.1],
            ),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="therapeutic_area",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=[
                    "Oncology",
                    "Cardiology",
                    "Neurology",
                    "Immunology",
                    "Infectious Disease",
                ],
                weights=[0.3, 0.2, 0.2, 0.15, 0.15],
            ),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="study_design",
            sampler_type=SamplerType.SUBCATEGORY,
            params=SubcategorySamplerParams(
                category="trial_phase",
                values={
                    "Phase I": [
                        "Single Arm",
                        "Dose Escalation",
                        "First-in-Human",
                        "Safety Assessment",
                    ],
                    "Phase II": [
                        "Randomized",
                        "Double-Blind",
                        "Proof of Concept",
                        "Open-Label Extension",
                    ],
                    "Phase III": [
                        "Randomized Controlled",
                        "Double-Blind Placebo-Controlled",
                        "Multi-Center",
                        "Pivotal",
                    ],
                    "Phase IV": [
                        "Post-Marketing Surveillance",
                        "Real-World Evidence",
                        "Long-Term Safety",
                        "Expanded Access",
                    ],
                },
            ),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="baseline_measurement",
            sampler_type=SamplerType.GAUSSIAN,
            params=GaussianSamplerParams(mean=100, stddev=15),
            convert_to="float",
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="final_measurement",
            sampler_type=SamplerType.GAUSSIAN,
            params=GaussianSamplerParams(mean=85, stddev=20),
            convert_to="float",
        )
    )

    config_builder.add_column(
        ExpressionColumnConfig(
            name="percent_change",
            expr="{{ (final_measurement - baseline_measurement) / baseline_measurement * 100 }}",
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="has_adverse_event",
            sampler_type=SamplerType.BERNOULLI,
            params=BernoulliSamplerParams(p=0.3),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="adverse_event_type",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=[
                    "Headache",
                    "Nausea",
                    "Fatigue",
                    "Rash",
                    "Dizziness",
                    "Pain at injection site",
                    "Other",
                ],
                weights=[0.2, 0.15, 0.15, 0.1, 0.1, 0.2, 0.1],
            ),
            conditional_params={"has_adverse_event == 0": CategorySamplerParams(values=["None"])},
        )
    )

    config_builder.add_column(
        LLMTextColumnConfig(
            name="medical_observations",
            model_alias=model_alias,
            prompt=(
                "Write medical observations for participant {{ participant.first_name }} {{ participant.last_name }} "
                "in the clinical trial for {{ therapeutic_area }} (Study ID: {{ study_id }}). "
                "Baseline measurement was {{ baseline_measurement }} and final measurement was {{ final_measurement }}, "
                "representing a change of {{ percent_change }}%. "
                "Include reference to the site investigator, Dr. {{ investigator.last_name }}."
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

