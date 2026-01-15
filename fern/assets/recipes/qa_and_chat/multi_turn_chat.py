from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from data_designer.essentials import (
    CategorySamplerParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    SamplerType,
    Score,
    SubcategorySamplerParams,
)
from data_designer.interface.results import DatasetCreationResults


def build_config(model_alias: str) -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder()

    config_builder.add_column(
        SamplerColumnConfig(
            name="domain",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["Tech Support", "Personal Finances", "Educational Guidance"]),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="topic",
            sampler_type=SamplerType.SUBCATEGORY,
            params=SubcategorySamplerParams(
                category="domain",
                values={
                    "Tech Support": [
                        "Troubleshooting a Laptop",
                        "Setting Up a Home Wi-Fi Network",
                        "Installing Software Updates",
                    ],
                    "Personal Finances": [
                        "Budgeting Advice",
                        "Understanding Taxes",
                        "Investment Strategies",
                    ],
                    "Educational Guidance": [
                        "Choosing a College Major",
                        "Effective Studying Techniques",
                        "Learning a New Language",
                    ],
                },
            ),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="complexity",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["Basic", "Intermediate", "Advanced"]),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="conversation_length",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=[2, 4, 6, 8]),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="user_mood",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["happy", "silly", "sarcastic", "combative", "disappointed", "toxic"]),
        )
    )

    config_builder.add_column(
        LLMTextColumnConfig(
            name="assistant_system_prompt",
            prompt=(
                "Write a reasonable system prompt for a helpful AI assistant with expertise in "
                "{{domain}} and {{topic}}. The AI assistant must not engage in harmful behaviors."
            ),
            model_alias=model_alias,
        )
    )

    config_builder.add_column(
        LLMTextColumnConfig(
            name="user_task",
            prompt="Define a simple task related to {{topic}} of {{complexity}} complexity for the user.",
            model_alias=model_alias,
        )
    )

    config_builder.add_column(
        LLMStructuredColumnConfig(
            name="conversation",
            prompt=(
                "<task>\n{{user_task}}\n</task>\n\n"
                "<system_prompt>{{assistant_system_prompt}}</system_prompt>\n\n"
                "Generate a conversation between a user and an AI assistant with <system_prompt> about <task>.\n"
                "User is asking the assistant for advice and is in a {{user_mood}} mood.\n"
                "The conversation must be {{conversation_length}} messages in length.\n"
                "The conversation must come to a natural end in {{conversation_length}} messages and if the assistant is unable "
                "to solve the user's needs by then, they should offer to continue the conversation later or redirect them to additional resources.\n"
                "The conversation must be realistic and natural, incorporating details from both the user's and assistant's profiles.\n"
                "The AI assistant role has the system prompt defined in <system_prompt>. All of its responses must conform to <system_prompt>:\n"
            ),
            output_format=ChatConversation,
            model_alias=model_alias,
        )
    )

    config_builder.add_column(
        LLMJudgeColumnConfig(
            name="toxicity_evaluation",
            prompt=TOXICITY_PROMPT_TEMPLATE,
            scores=[toxicity_score],
            model_alias=model_alias,
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


toxicity_score = Score(
    name="Toxicity",
    description="Evaluation of harmful, harassing, or hateful language in conversations",
    options={
        "4": "No toxicity detected; conversation is respectful and appropriate.",
        "3": "Mild toxicity; slightly rude or sarcastic but not hateful or harmful.",
        "2": "Moderate toxicity; contains disrespectful or harassing language.",
        "1": "Severe toxicity; contains overt hate, harassment, or harmful content.",
        "0": "Extreme toxicity; contains dangerous, threatening, or abusive content.",
    },
)

TOXICITY_PROMPT_TEMPLATE = (
    "You are an expert in content moderation and digital communication ethics. "
    "You are evaluating the toxicity level in user messages.\n"
    "Take a deep breath and carefully analyze the conversation provided. \n"
    "Use the rubric to score the level of toxicity in the USER'S messages only.\n"
    "Consider:\n"
    "- Focus primarily on harassment, hate, or harmful language\n"
    "- Evaluate the full context of the conversation\n"
    "- Be objective and consistent in your assessment\n"
    "- If any single message exhibits toxicity, it should influence the final score\n"
    "## CONVERSATION\n"
    "{{ conversation }}"
)


class Message(BaseModel):
    """A single message turn in the conversation."""

    role: Literal["user", "assistant"] = Field(..., description="Which role is writing the message.")
    content: str = Field(..., description="Message contents.")


class ChatConversation(BaseModel):
    """A chat conversation between a specific user and an AI assistant.
    * All conversations are initiated by the user role.
    * The assistant role always responds to the user message.
    * Turns alternate between user and assistant roles.
    * The last message is always from the assistant role.
    * Message content can be long or short.
    * All assistant messages are faithful responses and must be answered fully.
    """

    conversation: list[Message] = Field(..., description="List of all messages in the conversation.")


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
