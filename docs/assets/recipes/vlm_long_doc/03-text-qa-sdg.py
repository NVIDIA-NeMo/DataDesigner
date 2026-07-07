# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer>=0.5.6",
# ]
# ///
"""Long-Document Understanding Text Question-Answering Recipe

Generate question-answer pairs from OCR-transcribed document text using a
reasoning LLM. For each seed record the pipeline:

  1. Samples a question type (multiple choice, true/false, short answer, numerical)
  2. Generates a structured question + answer pair grounded in the transcribed text
  3. Evaluates question relevance against the source text
  4. Evaluates answer correctness against the source text

Prerequisites:
    - A seed parquet file containing a `transcribed_texts` column with the
      OCR-transcribed document text (e.g. output of 02-nemotron-parse-ocr-sdg.py).
    - A vLLM-compatible deployment of the reasoning LLM
      (default: openai/gpt-oss-120b).
      Recommended vLLM launch flags:
        --tensor-parallel-size 2
        --reasoning-parser openai_gptoss

      Example launch script for 2× H100:
        docker run --gpus all \
            -p 8000:8000 \
            vllm/vllm-openai:latest \
            --model openai/gpt-oss-120b \
            --tensor-parallel-size 2 \
            --reasoning-parser openai_gptoss \
            --gpu-memory-utilization 0.80 \
            --max-model-len 32768

Run:
    # Basic usage (seed-path should point to the output of 02-nemotron-parse-ocr-sdg.py)
    uv run 03-text-qa-sdg.py --vllm-endpoint http://localhost:8000/v1 --seed-path artifacts/nemotron_parse_ocr/parquet-files/*.parquet

    # Custom model and record count
    uv run 03-text-qa-sdg.py --vllm-endpoint http://localhost:8000/v1 --seed-path artifacts/nemotron_parse_ocr/parquet-files/*.parquet --num-records 100

    # For help message and available options
    uv run 03-text-qa-sdg.py --help
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults

DEFAULT_REASONER_MODEL = "openai/gpt-oss-120b"
VLLM_PROVIDER_NAME = "vllm"

# =============================================================================
# Structured output schemas
# =============================================================================


class QuestionAnswer(BaseModel):
    question: str = Field(..., description="The question to be answered.")
    answer: str = Field(..., description="The correct answer to the question.")


class QuestionRelevance(BaseModel):
    is_relevant: Literal["Relevant", "Irrelevant"] = Field(
        ...,
        description="The relevance of the question to the document content provided.",
    )


class AnswerCorrectness(BaseModel):
    is_correct: Literal["Correct", "Incorrect"] = Field(..., description="Whether the answer is correct.")


# =============================================================================
# Prompt templates
# =============================================================================

PROMPT_QUESTION_ANSWER = """\
<question-type>
{{question_type}}
</question-type>

<context>
{{ transcribed_texts }}
</context>

You are an expert in creating challenging reasoning questions that require deep analysis \
and critical thinking. Your task is to examine the provided pages information and create a \
question that can only be answered by reviewing <context>.

Create a question & answer pair using <context> of type <question-type>.\
"""

PROMPT_QUESTION_RELEVANCE = """\
<context>
{{ transcribed_texts }}
</context>

<question>
{{ question_and_answer.question }}
</question>

Determine if the <question> is relevant to the <context>.\
"""

PROMPT_ANSWER_CORRECTNESS = """\
<context>
{{ transcribed_texts }}
</context>

<question>
{{ question_and_answer.question }}
</question>

<answer>
{{ question_and_answer.answer }}
</answer>

Determine if the <answer> to <question> is correct given <context>.\
"""


# =============================================================================
# Pipeline configuration
# =============================================================================


def build_config(
    seed_path: str = "seed.parquet",
    model_alias: str = "reasoner",
    model_id: str = DEFAULT_REASONER_MODEL,
) -> dd.DataDesignerConfigBuilder:
    model_configs = [
        dd.ModelConfig(
            alias=model_alias,
            model=model_id,
            provider=VLLM_PROVIDER_NAME,
            inference_parameters=dd.ChatCompletionInferenceParams(
                max_tokens=32768,
                timeout=1200,
                extra_body={"reasoning_effort": "high"},
                max_parallel_requests=32,
            ),
        ),
    ]

    config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

    config_builder.with_seed_dataset(
        dd.LocalFileSeedSource(path=seed_path),
        sampling_strategy=dd.SamplingStrategy.ORDERED,
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="question_type",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=[
                    "multiple choice",
                    "true or false",
                    "short answer",
                    "numerical question",
                ],
            ),
        )
    )

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="question_and_answer",
            model_alias=model_alias,
            prompt=PROMPT_QUESTION_ANSWER,
            output_format=QuestionAnswer,
        )
    )

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="question_relevance",
            model_alias=model_alias,
            prompt=PROMPT_QUESTION_RELEVANCE,
            output_format=QuestionRelevance,
        )
    )

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="answer_correctness",
            model_alias=model_alias,
            prompt=PROMPT_ANSWER_CORRECTNESS,
            output_format=AnswerCorrectness,
        )
    )

    return config_builder


def create_dataset(
    config_builder: dd.DataDesignerConfigBuilder,
    num_records: int,
    vllm_endpoint: str,
    artifact_path: Path | str | None = None,
) -> DatasetCreationResults:
    model_providers = [
        dd.ModelProvider(
            name=VLLM_PROVIDER_NAME,
            endpoint=vllm_endpoint,
        ),
    ]
    data_designer = DataDesigner(
        artifact_path=artifact_path,
        model_providers=model_providers,
    )
    data_designer.set_run_config(dd.RunConfig(display_tui=True, disable_early_shutdown=True))
    results = data_designer.create(config_builder, num_records=num_records, dataset_name="text_qa")
    return results


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--vllm-endpoint",
        type=str,
        required=True,
        help="Base URL of the vLLM server hosting the reasoning LLM (e.g. http://localhost:8000/v1)",
    )
    parser.add_argument("--seed-path", type=str, required=True, help="Path to the seed parquet file")
    parser.add_argument("--model-alias", type=str, default="reasoner")
    parser.add_argument("--model-id", type=str, default=DEFAULT_REASONER_MODEL)
    parser.add_argument("--num-records", type=int, default=5)
    parser.add_argument("--artifact-path", type=str, default=None)
    args = parser.parse_args()

    config_builder = build_config(
        seed_path=args.seed_path,
        model_alias=args.model_alias,
        model_id=args.model_id,
    )
    results = create_dataset(
        config_builder,
        num_records=args.num_records,
        vllm_endpoint=args.vllm_endpoint,
        artifact_path=args.artifact_path,
    )

    print(f"Dataset saved to: {results.artifact_storage.final_dataset_path}")

    results.load_analysis().to_report()
