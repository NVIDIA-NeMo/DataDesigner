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
    UUIDSamplerParams,
)
from data_designer.interface.results import DatasetCreationResults


def build_config(model_alias: str) -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder()

    config_builder.add_column(
        SamplerColumnConfig(
            name="policyholder",
            sampler_type=SamplerType.PERSON,
            params=PersonSamplerParams(locale="en_US"),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="claimant",
            sampler_type=SamplerType.PERSON,
            params=PersonSamplerParams(locale="en_US"),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="adjuster",
            sampler_type=SamplerType.PERSON,
            params=PersonSamplerParams(locale="en_US"),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="policy_number",
            sampler_type=SamplerType.UUID,
            params=UUIDSamplerParams(prefix="POL-", short_form=True, uppercase=True),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="claim_number",
            sampler_type=SamplerType.UUID,
            params=UUIDSamplerParams(prefix="CLM-", short_form=True, uppercase=True),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="policy_type",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=["Auto", "Home", "Health", "Life"],
                weights=[0.4, 0.3, 0.2, 0.1],
            ),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="claim_type",
            sampler_type=SamplerType.SUBCATEGORY,
            params=SubcategorySamplerParams(
                category="policy_type",
                values={
                    "Auto": ["Collision", "Comprehensive", "Liability"],
                    "Home": ["Fire", "Theft", "Water Damage", "Storm"],
                    "Health": ["Surgery", "Emergency", "Prescription"],
                    "Life": ["Death Benefit", "Critical Illness"],
                },
            ),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="claim_amount",
            sampler_type=SamplerType.GAUSSIAN,
            params=GaussianSamplerParams(mean=5000, stddev=3000),
            convert_to="float",
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="claim_status",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=["Pending", "Approved", "Denied", "Under Review"],
                weights=[0.3, 0.4, 0.2, 0.1],
            ),
        )
    )

    config_builder.add_column(
        ExpressionColumnConfig(
            name="policyholder_info",
            expr="{{ policyholder.first_name }} {{ policyholder.last_name }}, SSN: {{ policyholder.national_id }}, DOB: {{ policyholder.birth_date }}",
        )
    )

    config_builder.add_column(
        LLMTextColumnConfig(
            name="claim_description",
            model_alias=model_alias,
            prompt=(
                "Write a detailed insurance claim description for a {{ policy_type }} policy claim of type {{ claim_type }}. "
                "The claim is filed by {{ claimant.first_name }} {{ claimant.last_name }} "
                "({{ claimant.email_address }}) for policy {{ policy_number }}. "
                "The claim amount is ${{ claim_amount | round(2) }}. "
                "Include specific details about the incident, date, location, and circumstances."
            ),
        )
    )

    config_builder.add_column(
        LLMTextColumnConfig(
            name="adjuster_notes",
            model_alias=model_alias,
            prompt=(
                "Write adjuster notes for claim {{ claim_number }} reviewed by {{ adjuster.first_name }} {{ adjuster.last_name }}. "
                "The claim is {{ claim_status }}. "
                "Claim description: {{ claim_description }}\n\n"
                "Include assessment of the claim, any concerns, and recommendations."
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
