import string
from pathlib import Path

from pydantic import BaseModel, Field

from data_designer.essentials import (
    BernoulliSamplerParams,
    CategorySamplerParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    ExpressionColumnConfig,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    SamplerType,
    Score,
    UniformSamplerParams,
)
from data_designer.interface.results import DatasetCreationResults


def build_config(model_alias: str) -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder()
    config_builder.add_column(
        SamplerColumnConfig(
            name="category",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=[
                    "Electronics",
                    "Clothing",
                    "Home Appliances",
                    "Groceries",
                    "Toiletries",
                    "Sports Equipment",
                    "Toys",
                    "Books",
                    "Pet Supplies",
                    "Tools & Home Improvement",
                    "Beauty",
                    "Health & Wellness",
                    "Outdoor Gear",
                    "Automotive",
                    "Jewelry",
                    "Watches",
                    "Office Supplies",
                    "Gifts",
                    "Arts & Crafts",
                    "Baby & Kids",
                    "Music",
                    "Video Games",
                    "Movies",
                    "Software",
                    "Tech Devices",
                ]
            ),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="price_tens_of_dollars",
            sampler_type=SamplerType.UNIFORM,
            params=UniformSamplerParams(low=1, high=200),
        )
    )

    config_builder.add_column(
        ExpressionColumnConfig(
            name="product_price",
            expr="{{ (price_tens_of_dollars * 10) - 0.01 | round(2) }}",
            dtype="float",
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="first_letter",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=list(string.ascii_uppercase)),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="is_hallucination",
            sampler_type=SamplerType.BERNOULLI,
            params=BernoulliSamplerParams(p=0.5),
        )
    )

    config_builder.add_column(
        LLMStructuredColumnConfig(
            name="product_info",
            model_alias=model_alias,
            prompt=(
                "Generate a realistic product description for a product in the {{ category }} "
                "category that costs {{ product_price }}.\n"
                "The name of the product MUST start with the letter {{ first_letter }}.\n"
            ),
            output_format=ProductInfo,
        )
    )

    config_builder.add_column(
        LLMTextColumnConfig(
            name="question",
            model_alias=model_alias,
            prompt=("Ask a question about the following product:\n\n {{ product_info }}"),
        )
    )

    config_builder.add_column(
        LLMTextColumnConfig(
            name="answer",
            model_alias=model_alias,
            prompt=(
                "{%- if is_hallucination == 0 -%}\n"
                "<product_info>\n"
                "{{ product_info }}\n"
                "</product_info>\n"
                "{%- endif -%}\n"
                "User Question: {{ question }}\n"
                "Directly and succinctly answer the user's question.\n"
                "{%- if is_hallucination == 1 -%}\n"
                "Make up whatever information you need to in order to answer the user's request.\n"
                "{%- endif -%}"
            ),
        )
    )

    # Evaluate answer quality
    config_builder.add_column(
        LLMJudgeColumnConfig(
            name="llm_answer_metrics",
            model_alias=model_alias,
            prompt=(
                "<product_info>\n"
                "{{ product_info }}\n"
                "</product_info>\n"
                "User Question: {{question }}\n"
                "AI Assistant Answer: {{ answer }}\n"
                "Judge the AI assistant's response to the user's question about the product described in <product_info>."
            ),
            scores=answer_quality_scores,
        )
    )

    config_builder.add_column(
        ExpressionColumnConfig(
            name="completeness_result",
            expr="{{ llm_answer_metrics.completeness.score }}",
        )
    )

    config_builder.add_column(
        ExpressionColumnConfig(
            name="accuracy_result",
            expr="{{ llm_answer_metrics.accuracy.score }}",
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


class ProductInfo(BaseModel):
    product_name: str = Field(..., description="A realistic product name for the market.")
    key_features: list[str] = Field(..., min_length=1, max_length=3, description="Key product features.")
    description: str = Field(
        ...,
        description="A short, engaging description of what the product does, highlighting a unique but believable feature.",
    )
    price_usd: float = Field(..., description="The price of the product", ge=10, le=1000, decimal_places=2)


completeness_score = Score(
    name="Completeness",
    description="Evaluation of AI assistant's thoroughness in addressing all aspects of the user's query.",
    options={
        "Complete": "The response thoroughly covers all key points requested in the question, providing sufficient detail to satisfy the user's information needs.",
        "PartiallyComplete": "The response addresses the core question but omits certain important details or fails to elaborate on relevant aspects that were requested.",
        "Incomplete": "The response significantly lacks necessary information, missing major components of what was asked and leaving the query largely unanswered.",
    },
)

accuracy_score = Score(
    name="Accuracy",
    description="Evaluation of how factually correct the AI assistant's response is relative to the product information.",
    options={
        "Accurate": "The information provided aligns perfectly with the product specifications without introducing any misleading or incorrect details.",
        "PartiallyAccurate": "While some information is correctly stated, the response contains minor factual errors or potentially misleading statements about the product.",
        "Inaccurate": "The response presents significantly wrong information about the product, with claims that contradict the actual product details.",
    },
)

answer_quality_scores = [completeness_score, accuracy_score]


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
