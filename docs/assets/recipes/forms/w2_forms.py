# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from data_designer.essentials import (
    BernoulliMixtureSamplerParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    ExpressionColumnConfig,
    GaussianSamplerParams,
    LLMTextColumnConfig,
    PersonSamplerParams,
    SamplerColumnConfig,
    SamplerType,
)
from data_designer.interface.results import DatasetCreationResults


def build_config(model_alias: str) -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder()

    config_builder.add_column(
        SamplerColumnConfig(
            name="taxpayer",
            sampler_type=SamplerType.PERSON,
            params=PersonSamplerParams(locale="en_US", age_range=[18, 75]),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="employer",
            sampler_type=SamplerType.PERSON,
            params=PersonSamplerParams(
                locale="en_US",
            ),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="box_1_wages_tips_other_compensation",
            sampler_type=SamplerType.BERNOULLI_MIXTURE,
            params=BernoulliMixtureSamplerParams(p=0.994, dist_name="expon", dist_params={"scale": 35891.49}),
            convert_to="int",
        )
    )

    config_builder.add_column(
        ExpressionColumnConfig(
            name="standard_deduction",
            expr="{% if taxpayer.marital_status == 'married_present' %}25900{% else %}12950{% endif %}",
            dtype="float",
        ),
    )

    config_builder.add_column(
        ExpressionColumnConfig(
            name="taxable_income",
            expr="{{ [0, box_1_wages_tips_other_compensation - standard_deduction]|max }}",
            dtype="float",
        )
    )

    BRACKETS = [
        {"name": "bracket1", "rate": 0.10, "max": 10275, "min": 0},
        {"name": "bracket2", "rate": 0.12, "max": 41775, "min": 10275},
        {"name": "bracket3", "rate": 0.22, "max": 89075, "min": 41775},
        {"name": "bracket4", "rate": 0.24, "max": 170050, "min": 89075},
        {"name": "bracket5", "rate": 0.32, "max": 215950, "min": 170050},
        {"name": "bracket6", "rate": 0.35, "max": 539900, "min": 215950},
        {"name": "bracket7", "rate": 0.37, "max": 10000000000000, "min": 539900},
    ]
    for bracket in BRACKETS:
        expression = f"{bracket['rate']}*([[taxable_income,{bracket['max']}]|min - {bracket['min']}, 0] | max)"
        config_builder.add_column(
            ExpressionColumnConfig(
                name=bracket["name"],
                expr="{{ " + expression + " }}",
                dtype="float",
            )
        )

    config_builder.add_column(
        ExpressionColumnConfig(
            name="mean_tax_liability",
            expr="{{ bracket1 + bracket2 + bracket3 + bracket4 + bracket5 + bracket6 + bracket7 }}",
            dtype="int",
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="tax_liability_noise",
            sampler_type=SamplerType.GAUSSIAN,
            params=GaussianSamplerParams(mean=1, stddev=0.1),
        )
    )

    config_builder.add_column(
        ExpressionColumnConfig(
            name="box_2_federal_income_tax_withheld",
            expr="{{ (mean_tax_liability * tax_liability_noise) | int }}",
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="social_security_wages_ratio",
            sampler_type=SamplerType.GAUSSIAN,
            params=GaussianSamplerParams(mean=0.8216, stddev=0.2),
        )
    )

    config_builder.add_column(
        ExpressionColumnConfig(
            name="box_3_social_security_wages",
            expr="{{ (box_1_wages_tips_other_compensation * social_security_wages_ratio) | int }}",
        )
    )

    config_builder.add_column(
        ExpressionColumnConfig(
            name="box_4_social_security_tax_withheld",
            expr="{{ (([box_3_social_security_wages, 147000]|min) * 0.062) | int }}",
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="medicare_wages_and_tips_ratio",
            sampler_type=SamplerType.GAUSSIAN,
            params=GaussianSamplerParams(mean=1.038, stddev=0.2),
        )
    )

    config_builder.add_column(
        ExpressionColumnConfig(
            name="box_5_medicare_wages_and_tips",
            expr="{{ (box_1_wages_tips_other_compensation * medicare_wages_and_tips_ratio) | int }}",
        )
    )

    config_builder.add_column(
        ExpressionColumnConfig(
            name="box_6_medicare_tax_withheld",
            expr="{{ ((box_5_medicare_wages_and_tips * 0.0145) + (([box_5_medicare_wages_and_tips - 200000, 0]|max) * 0.009)) | int }}",
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="box_7_social_security_tips",
            sampler_type=SamplerType.BERNOULLI_MIXTURE,
            params=BernoulliMixtureSamplerParams(p=0.0454, dist_name="expon", dist_params={"scale": 4428.91}),
        )
    )

    config_builder.add_column(
        ExpressionColumnConfig(
            name="box_a_employee_ssn",
            expr="{{ taxpayer.national_id }}",
        )
    )

    config_builder.add_column(
        LLMTextColumnConfig(
            name="employer_business",
            model_alias=model_alias,
            system_prompt=(
                "You are assisting a user generate synthetic W-2 forms. "
                "You must generate a realistic industry category for the employer "
                "eg: software, health insurance, shoe store, restaurant, plumbing"
            ),
            prompt=(
                "Generate the industry category for the employer. Ensure it is consistent with the employer location "
                "City: {{ employer.city }}\nState: {{ employer.state }}"
            ),
        )
    )

    config_builder.add_column(
        LLMTextColumnConfig(
            name="employer_name",
            model_alias=model_alias,
            prompt="Generate an original name for a {{ employer_business }} business in {{ employer.city }}.",
        )
    )

    config_builder.add_column(
        ExpressionColumnConfig(
            name="box_c_employer_name_address_zip",
            expr="{{ employer_name }}\n{{ employer.street_number }} {{ employer.street_name }}\n{{ employer.city }}, {{ employer.state }} {{ employer.postcode }}",
        )
    )

    config_builder.add_column(
        ExpressionColumnConfig(
            name="box_e_employee_first_name_initial_last_name",
            expr="{{ taxpayer.first_name }} {{ taxpayer.middle_name[:1] }} {{ taxpayer.last_name }}",
        )
    )

    config_builder.add_column(
        ExpressionColumnConfig(
            name="box_f_employee_address_zip",
            expr="{{ taxpayer.street_number }} {{ taxpayer.street_name }}\n{{ taxpayer.city }}, {{ taxpayer.state }} {{ taxpayer.postcode }}",
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
