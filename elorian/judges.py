"""LLM judge registry for evaluating VLM responses.

Each JudgeSpec defines a judge model, the scoring rubric (list of Score
dimensions), and the prompt template used for evaluation. The pipeline
creates one LLMJudgeColumnConfig per judge automatically.

To add a new judge:
    registry = JudgeRegistry()
    registry.register(JudgeSpec(
        alias="my-judge",
        model_id="anthropic/claude-sonnet-4-20250514",
        scores=[...],
        prompt_template="...",
    ))
"""

from __future__ import annotations

from dataclasses import dataclass, field

import data_designer.config as dd

DEFAULT_JUDGE_PROMPT = (
    "You are an expert evaluator of vision-language model outputs.\n\n"
    "Given an image and multiple model responses describing it, evaluate EACH "
    "response on the scoring dimensions provided.\n\n"
    "Responses to evaluate:\n"
    "{responses_block}\n\n"
    "Evaluate each response carefully. Consider accuracy, detail, coherence, "
    "and how well it captures the visual content."
)

DEFAULT_SCORES: list[dd.Score] = [
    dd.Score(
        name="accuracy",
        description=(
            "How accurately does the response describe the actual content "
            "of the image? Penalize hallucinations or incorrect details."
        ),
        options={
            1: "Mostly inaccurate or hallucinated content",
            2: "Some correct details but significant errors",
            3: "Generally accurate with minor errors",
            4: "Accurate with very few minor issues",
            5: "Perfectly accurate description of all visual elements",
        },
    ),
    dd.Score(
        name="detail",
        description=(
            "How detailed and comprehensive is the response? Does it cover "
            "the main subject, background, colors, and notable features?"
        ),
        options={
            1: "Extremely sparse, missing most details",
            2: "Covers only the most obvious elements",
            3: "Moderate detail, covers key elements",
            4: "Good detail, covers most visual elements",
            5: "Exceptionally detailed and comprehensive",
        },
    ),
    dd.Score(
        name="coherence",
        description=(
            "How well-structured and coherent is the response? Is it easy "
            "to read and logically organized?"
        ),
        options={
            1: "Incoherent or very poorly structured",
            2: "Somewhat readable but disorganized",
            3: "Reasonably clear and organized",
            4: "Well-structured and easy to follow",
            5: "Exceptionally clear, logical, and well-organized",
        },
    ),
]


@dataclass
class JudgeSpec:
    """Specification for an LLM judge.

    Attributes:
        alias: Unique short name for this judge (used as column prefix).
        model_id: LiteLLM model identifier for the judge model.
        description: Human-readable description.
        scores: List of Score dimensions for the rubric.
        prompt_template: Prompt template with {responses_block} placeholder.
        max_tokens: Max tokens for judge generation.
        temperature: Sampling temperature (lower = more deterministic).
        skip_health_check: Whether to skip the DataDesigner health check.
    """

    alias: str
    model_id: str
    provider: str = "anthropic"
    description: str = ""
    scores: list[dd.Score] = field(default_factory=lambda: list(DEFAULT_SCORES))
    prompt_template: str = DEFAULT_JUDGE_PROMPT
    max_tokens: int = 2048
    temperature: float = 0.3
    skip_health_check: bool = True

    def to_model_config(self) -> dd.ModelConfig:
        """Convert to a DataDesigner ModelConfig for the judge model."""
        return dd.ModelConfig(
            alias=self.alias,
            model=self.model_id,
            provider=self.provider,
            inference_parameters=dd.ChatCompletionInferenceParams(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
            skip_health_check=self.skip_health_check,
        )

    def build_prompt(self, model_aliases: list[str]) -> str:
        """Build the judge prompt with response column references.

        Args:
            model_aliases: List of model aliases whose response columns
                will be referenced in the prompt via Jinja2 {{ }} syntax.

        Returns:
            Prompt string with Jinja2 column references.
        """
        lines = []
        for alias in model_aliases:
            col_name = f"response_{alias}"
            lines.append(f"--- Response from {alias} ---\n{{{{ {col_name} }}}}")
        responses_block = "\n\n".join(lines)
        return self.prompt_template.format(responses_block=responses_block)


class JudgeRegistry:
    """Registry of LLM judges available for evaluation."""

    def __init__(self) -> None:
        self._specs: dict[str, JudgeSpec] = {}

    def register(self, spec: JudgeSpec) -> None:
        """Register a judge spec (overwrites if alias already exists)."""
        self._specs[spec.alias] = spec

    def unregister(self, alias: str) -> None:
        """Remove a judge by alias."""
        self._specs.pop(alias, None)

    def get(self, alias: str) -> JudgeSpec:
        """Get a judge spec by alias."""
        return self._specs[alias]

    @property
    def specs(self) -> list[JudgeSpec]:
        """All registered judge specs."""
        return list(self._specs.values())

    @property
    def aliases(self) -> list[str]:
        """All registered judge aliases."""
        return list(self._specs.keys())


def get_default_judge_registry() -> JudgeRegistry:
    """Create a registry pre-loaded with a Claude judge."""
    registry = JudgeRegistry()
    registry.register(
        JudgeSpec(
            alias="judge_claude",
            model_id="claude-sonnet-4-20250514",
            description="Claude Sonnet 4 as LLM judge",
        )
    )
    return registry
