# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from data_designer.essentials import (
    CategorySamplerParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LocalFileSeedSource,
    SamplerColumnConfig,
    SamplerType,
    Score,
)
from data_designer.interface.results import DatasetCreationResults
from data_designer.lazy_heavy_imports import pd


class QAPair(BaseModel):
    """Question-answer pair with reasoning for RAG evaluation."""

    question: str = Field(..., description="A specific question related to the domain of the context")
    answer: str = Field(
        ...,
        description="Either a context-supported answer or explanation of why the question cannot be answered",
    )
    reasoning: str = Field(
        ...,
        description="A clear and traceable explanation of the reasoning behind the answer",
    )


def chunk_pdf_to_dataset(pdf_path: Path | str, chunk_size: int = 1000, overlap: int = 200) -> Path:
    """
    Extract text from a PDF and chunk it into a temporary CSV file.

    Args:
        pdf_path: Path to the PDF file
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        Path to the temporary CSV file containing chunks
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "pypdf is required for PDF processing. Install it with: pip install pypdf"
        ) from None

    pdf_path = Path(pdf_path)
    reader = PdfReader(pdf_path)

    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()

    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        if chunk.strip():
            chunks.append({"text": chunk.strip()})
        start = end - overlap

    df = pd.DataFrame(chunks)

    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()

    return Path(temp_file.name)


def build_config(model_alias: str, seed_dataset_path: Path | str) -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder()

    # Add seed dataset - columns from CSV (like 'text') will be automatically available
    seed_source = LocalFileSeedSource(path=str(seed_dataset_path))
    config_builder.with_seed_dataset(seed_source)

    config_builder.add_column(
        SamplerColumnConfig(
            name="difficulty",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=["easy", "medium", "hard"],
            ),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="reasoning_type",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=[
                    "factual recall",
                    "inferential reasoning",
                    "comparative analysis",
                    "procedural understanding",
                    "cause and effect",
                ],
            ),
        )
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="question_type",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=["answerable", "unanswerable"],
                weights=[10, 1],
            ),
        )
    )

    config_builder.add_column(
        LLMStructuredColumnConfig(
            name="qa_pair",
            model_alias=model_alias,
            prompt=(
                "{{ text }}\n"
                "\n"
                "Generate a {{ difficulty }} {{ reasoning_type }} question-answer pair.\n"
                "The question should be {{ question_type }} using the provided context.\n"
                "\n"
                "For answerable questions:\n"
                "- Ensure the answer is fully supported by the context\n"
                "\n"
                "For unanswerable questions:\n"
                "- Keep the question topically relevant\n"
                "- Make it clearly beyond the context's scope\n"
            ),
            output_format=QAPair,
        )
    )

    config_builder.add_column(
        LLMJudgeColumnConfig(
            name="eval_metrics",
            model_alias=model_alias,
            prompt=EVAL_METRICS_PROMPT_TEMPLATE,
            scores=[
                context_relevance_rubric,
                answer_precision_rubric,
                answer_completeness_rubric,
                hallucination_avoidance_rubric,
            ],
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


context_relevance_rubric = Score(
    name="Context Relevance",
    description="Evaluates how relevant the answer is to the provided context",
    options={
        5: "Perfect relevance to context with no extraneous information",
        4: "Highly relevant with minor deviations from context",
        3: "Moderately relevant but includes some unrelated information",
        2: "Minimally relevant with significant departure from context",
        1: "Almost entirely irrelevant to the provided context",
    },
)

answer_precision_rubric = Score(
    name="Answer Precision",
    description="Evaluates the accuracy and specificity of the answer",
    options={
        5: "Extremely precise with exact, specific information",
        4: "Very precise with minor imprecisions",
        3: "Adequately precise but could be more specific",
        2: "Imprecise with vague or ambiguous information",
        1: "Completely imprecise or inaccurate",
    },
)

answer_completeness_rubric = Score(
    name="Answer Completeness",
    description="Evaluates how thoroughly the answer addresses all aspects of the question",
    options={
        5: "Fully complete, addressing all aspects of the question",
        4: "Mostly complete with minor omissions",
        3: "Adequately complete but missing some details",
        2: "Substantially incomplete, missing important aspects",
        1: "Severely incomplete, barely addresses the question",
    },
)

hallucination_avoidance_rubric = Score(
    name="Hallucination Avoidance",
    description="Evaluates the absence of made-up or incorrect information",
    options={
        5: "No hallucinations, all information is factual and verifiable",
        4: "Minimal hallucinations that don't impact the core answer",
        3: "Some hallucinations that partially affect the answer quality",
        2: "Significant hallucinations that undermine the answer",
        1: "Severe hallucinations making the answer entirely unreliable",
    },
)

EVAL_METRICS_PROMPT_TEMPLATE = (
    "You are an expert evaluator of question-answer pairs. Analyze the following Q&A pair and evaluate it objectively.\n\n"
    "Context:\n"
    "{{ text }}\n\n"
    "For this {{ difficulty }} {{ reasoning_type }} Q&A pair:\n"
    "{{ qa_pair }}\n\n"
    "Take a deep breath and carefully evaluate each criterion based on the provided rubrics, considering the "
    "difficulty level and reasoning type indicated."
)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model-alias", type=str, default="openai-text")
    parser.add_argument("--num-records", type=int, default=5)
    parser.add_argument("--artifact-path", type=str, default=None)
    parser.add_argument(
        "--pdf-path",
        type=str,
        default=None,
        help="Path to PDF file to process (default: databricks-state-of-data-ai-report.pdf in same directory)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of characters per chunk (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Number of overlapping characters between chunks (default: 200)",
    )
    args = parser.parse_args()

    # Use the default PDF in the same directory if not specified
    if args.pdf_path is None:
        script_dir = Path(__file__).parent
        default_pdf = script_dir / "databricks-state-of-data-ai-report.pdf"
        if not default_pdf.exists():
            raise FileNotFoundError(
                f"Default PDF not found at {default_pdf}. "
                "Please specify a PDF path using --pdf-path"
            )
        pdf_path = default_pdf
    else:
        pdf_path = Path(args.pdf_path)

    print(f"Processing PDF: {pdf_path}")
    seed_dataset_path = chunk_pdf_to_dataset(
        pdf_path,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    print(f"Created chunked dataset: {seed_dataset_path}")

    config_builder = build_config(model_alias=args.model_alias, seed_dataset_path=seed_dataset_path)
    results = create_dataset(config_builder, num_records=args.num_records, artifact_path=args.artifact_path)

    print(f"Dataset saved to: {results.artifact_storage.final_dataset_path}")

    results.load_analysis().to_report()

