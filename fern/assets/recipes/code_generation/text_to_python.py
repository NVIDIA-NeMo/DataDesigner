from pathlib import Path

from data_designer.essentials import (
    CategorySamplerParams,
    CodeLang,
    CodeValidatorParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    SamplerType,
    Score,
    SubcategorySamplerParams,
    ValidationColumnConfig,
    ValidatorType,
)
from data_designer.interface.results import DatasetCreationResults


def build_config(model_alias: str) -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder()

    config_builder.add_column(
        SamplerColumnConfig(
            name="industry_sector",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=[
                    "Healthcare",
                    "Finance",
                    "Technology",
                ],
            ),
        ),
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="topic",
            sampler_type=SamplerType.SUBCATEGORY,
            params=SubcategorySamplerParams(
                category="industry_sector",
                values={
                    "Healthcare": [
                        "Electronic Health Records (EHR) Systems",
                        "Telemedicine Platforms",
                        "AI-Powered Diagnostic Tools",
                    ],
                    "Finance": [
                        "Fraud Detection Software",
                        "Automated Trading Systems",
                        "Personal Finance Apps",
                    ],
                    "Technology": [
                        "Cloud Computing Platforms",
                        "Artificial Intelligence and Machine Learning Platforms",
                        "DevOps and CI/CD Tools",
                    ],
                },
            ),
        ),
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="code_complexity",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=[
                    "Beginner",
                    "Intermediate",
                    "Advanced",
                ],
            ),
        ),
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="code_concept",
            sampler_type=SamplerType.SUBCATEGORY,
            params=SubcategorySamplerParams(
                category="code_complexity",
                values={
                    "Beginner": [
                        "Variables",
                        "Data Types",
                        "Functions",
                        "Loops",
                        "Classes",
                    ],
                    "Intermediate": [
                        "List Comprehensions",
                        "Object-oriented programming",
                        "Lambda Functions",
                        "Web frameworks",
                        "Pandas",
                    ],
                    "Advanced": [
                        "Multithreading",
                        "Context Managers",
                        "Generators",
                    ],
                },
            ),
        ),
    )

    config_builder.add_column(
        SamplerColumnConfig(
            name="instruction_phrase",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=[
                    "Write a function that",
                    "Create a class that",
                    "Implement a script",
                    "Can you create a function",
                    "Develop a module that",
                ],
            ),
        ),
    )

    config_builder.add_column(
        LLMTextColumnConfig(
            name="instruction",
            model_alias=model_alias,
            system_prompt=("You are an expert at generating clear and specific programming tasks."),
            prompt=(
                "Generate an instruction to create Python code that solves a specific problem.\n"
                "Each instruction should begin with one of the following phrases: {{ instruction_phrase }}.\n\n"
                "Important Guidelines:\n"
                "* Industry Relevance: Ensure the instruction pertains to the {{ industry_sector }} sector and {{ topic }} topic.\n"
                "* Code Complexity: Tailor the instruction to the {{ code_complexity }} level. Utilize relevant {{ code_concept }} where appropriate to match the complexity level.\n"
                "* Clarity and Specificity: Make the problem statement clear and unambiguous. Provide sufficient context to understand the requirements without being overly verbose.\n"
                "* Response Formatting: Do not include any markers such as ### Response ### in the instruction.\n"
            ),
        )
    )

    config_builder.add_column(
        LLMCodeColumnConfig(
            name="code_implementation",
            model_alias=model_alias,
            code_lang=CodeLang.PYTHON,
            system_prompt=(
                "You are an expert Python programmer who writes clean, efficient, and well-documented code."
            ),
            prompt=(
                "Write Python code for the following instruction:\n"
                "Instruction: {{ instruction }}\n\n"
                "Important Guidelines:\n"
                "* Code Quality: Your code should be clean, complete, self-contained, and accurate.\n"
                "* Code Validity: Please ensure that your Python code is executable and does not contain any errors.\n"
                "* Packages: Remember to import any necessary libraries, and to use all libraries you import.\n"
                "* Complexity & Concepts: The code should be written at a {{ code_complexity }} level, making use of concepts such as {{code_concept}}.\n"
            ),
        )
    )

    config_builder.add_column(
        LLMTextColumnConfig(
            name="instruction",
            model_alias=model_alias,
            system_prompt=("You are an expert at generating clear and specific programming tasks."),
            prompt=(
                "Generate an instruction to create Python code that solves a specific problem.\n"
                "Each instruction should begin with one of the following phrases: {{ instruction_phrase }}.\n\n"
                "Important Guidelines:\n"
                "* Industry Relevance: Ensure the instruction pertains to the {{ industry_sector }} sector and {{ topic }} topic.\n"
                "* Code Complexity: Tailor the instruction to the {{ code_complexity }} level. Utilize relevant {{ code_concept }} where appropriate to match the complexity level.\n"
                "* Clarity and Specificity: Make the problem statement clear and unambiguous. Provide sufficient context to understand the requirements without being overly verbose.\n"
                "* Response Formatting: Do not include any markers such as ### Response ### in the instruction.\n"
            ),
        )
    )

    config_builder.add_column(
        LLMCodeColumnConfig(
            name="code_implementation",
            model_alias=model_alias,
            code_lang=CodeLang.PYTHON,
            system_prompt=(
                "You are an expert Python programmer who writes clean, efficient, and well-documented code."
            ),
            prompt=(
                "Write Python code for the following instruction:\n"
                "Instruction: {{ instruction }}\n\n"
                "Important Guidelines:\n"
                "* Code Quality: Your code should be clean, complete, self-contained, and accurate.\n"
                "* Code Validity: Please ensure that your Python code is executable and does not contain any errors.\n"
                "* Packages: Remember to import any necessary libraries, and to use all libraries you import.\n"
                "* Complexity & Concepts: The code should be written at a {{ code_complexity }} level, making use of concepts such as {{ code_concept }}.\n"
            ),
        )
    )

    config_builder.add_column(
        LLMJudgeColumnConfig(
            name="code_judge_result",
            model_alias=model_alias,
            prompt=TEXT_TO_PYTHON_JUDGE_TEMPLATE,
            scores=python_scoring,
        )
    )

    config_builder.add_column(
        ValidationColumnConfig(
            name="code_validity_result",
            validator_type=ValidatorType.CODE,
            target_columns=["code_implementation"],
            validator_params=CodeValidatorParams(
                code_lang=CodeLang.PYTHON,
            ),
            batch_size=100,
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


TEXT_TO_PYTHON_JUDGE_TEMPLATE = """\
You are an expert in Python programming, with specialized knowledge in software engineering, data science, and algorithmic problem-solving.

You think about potential flaws and errors in the code. You are a tough critic, but a fair one.

Take a deep breath and use the Python Code Quality Rubric below to score the **Generated Python Code** based on the INSTRUCTIONS.

#### INSTRUCTIONS
The Generated Python Code should be a valid response to the Natural Language Prompt below

Natural Language Prompt:
{{ instruction }}

Generated Python Code
{{ code_implementation }}
"""


python_scoring = [
    Score(
        name="Relevance",
        description="Adherence to INSTRUCTIONS and CONTEXT",
        options={
            4: "Perfectly meets all specified requirements.",
            3: "Meets most requirements with minor deviations.",
            2: "Moderate deviation from the instructions.",
            1: "Significant deviations from the instructions.",
            0: "Does not adhere to the instructions.",
        },
    ),
    Score(
        name="Pythonic",
        description="Pythonic Code and Best Practices (Does the code follow Python conventions and best practices?)",
        options={
            4: "The code exemplifies Pythonic principles, making excellent use of Python-specific constructs, standard library modules and programming idioms; follows all relevant PEPs.",
            3: "The code closely follows Python conventions and adheres to many best practices; good use of Python-specific constructs, standard library modules and programming idioms.",
            2: "The code generally follows Python conventions but has room for better alignment with Pythonic practices.",
            1: "The code loosely follows Python conventions, with several deviations from best practices.",
            0: "The code does not follow Python conventions or best practices, using non-Pythonic approaches.",
        },
    ),
    Score(
        name="Readability",
        description="Readability and Maintainability (Is the Python code easy to understand and maintain?)",
        options={
            4: (
                "The code is excellently formatted, follows PEP 8 guidelines, is elegantly concise and clear, uses meaningful variable names, "
                "ensuring high readability and ease of maintenance; organizes complex logic well. Docstrings are given in a Google Docstring format."
            ),
            3: "The code is well-formatted in the sense of code-as-documentation, making it relatively easy to understand and maintain; uses descriptive names and organizes logic clearly.",
            2: "The code is somewhat readable with basic formatting and some comments, but improvements are needed; needs better use of descriptive names and organization.",
            1: "The code has minimal formatting, making it hard to understand; lacks meaningful names and organization.",
            0: "The code is unreadable, with no attempt at formatting or description.",
        },
    ),
    Score(
        name="Efficiency",
        description="Efficiency and Performance (Is the code optimized for performance?)",
        options={
            4: "The solution is highly efficient, using appropriate data structures and algorithms; avoids unnecessary computations and optimizes for both time and space complexity.",
            3: "The solution is efficient, with good use of Python's built-in functions and libraries; minor areas for optimization.",
            2: "The solution is moderately efficient, but misses some opportunities for optimization; uses some inefficient patterns.",
            1: "The solution shows poor efficiency, with notable performance issues; lacks effective optimization techniques.",
            0: "The solution is highly inefficient; overlooks fundamental optimization practices, resulting in significant performance issues.",
        },
    ),
]


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
