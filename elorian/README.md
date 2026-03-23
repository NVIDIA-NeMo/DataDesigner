# Elorian — Multimodal VLM Evaluation Pipeline

Evaluate and compare vision-language models on image understanding tasks using LLM-as-a-judge, built on top of [NVIDIA DataDesigner](https://nvidia-nemo.github.io/DataDesigner/).

## Quick Start

```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

uv run python -m elorian.run_example
```

This loads 5 images from HuggingFace, generates descriptions with Claude and GPT-4o, then scores both using a Claude judge on accuracy, detail, and coherence.

## Architecture

```
elorian/
├── models.py        # VLM model registry + provider config
├── judges.py        # LLM judge registry + scoring rubrics
├── pipeline.py      # Orchestrates: images → models → judges
├── image_utils.py   # Load images from directories, PIL, or HuggingFace
└── run_example.py   # Working example
```

The pipeline generates one DataDesigner column per model (for responses) and one per judge (for evaluation), then runs them through DataDesigner's DAG-based execution engine.

## Usage

### Basic

```python
from elorian.image_utils import load_images_from_hf_dataset
from elorian.pipeline import MultimodalEvalPipeline

seed_df = load_images_from_hf_dataset("zh-plus/tiny-imagenet", split="valid", max_images=10)

pipeline = MultimodalEvalPipeline(seed_df=seed_df)
preview = pipeline.preview(num_records=3)
print(preview.dataset.head())
```

### Full run

```python
results = pipeline.run(num_records=50, dataset_name="my-eval")
dataset = results.load_dataset()
```

## Configuring Models

Default models: **Claude Sonnet 4** and **GPT-4o**. Both use the built-in `anthropic` and `openai` providers.

### Add a model from an existing provider

```python
from elorian.models import ModelSpec, get_default_model_registry

model_registry = get_default_model_registry()
model_registry.register(
    ModelSpec(
        alias="gpt4o_mini_vision",   # must use underscores, not hyphens
        model_id="gpt-4o-mini",
        provider="openai",
        description="GPT-4o Mini (vision)",
        max_tokens=1024,
        temperature=0.7,
    )
)
```

### Add a model from a new provider

```python
import data_designer.config as dd
from elorian.models import ModelSpec, register_provider, get_default_model_registry

# 1. Register the provider (once)
register_provider(
    dd.ModelProvider(
        name="google",
        endpoint="https://generativelanguage.googleapis.com/v1",
        provider_type="gemini",        # LiteLLM routing prefix
        api_key="GEMINI_API_KEY",      # env var name
    )
)

# 2. Register the model
model_registry = get_default_model_registry()
model_registry.register(
    ModelSpec(
        alias="gemini_flash",
        model_id="gemini-2.0-flash",
        provider="google",
        description="Google Gemini 2.0 Flash",
    )
)
```

### Remove a default model

```python
model_registry = get_default_model_registry()
model_registry.unregister("gpt4o_vision")  # keep only Claude
```

### Build from scratch

```python
from elorian.models import ModelRegistry, ModelSpec

model_registry = ModelRegistry()
model_registry.register(ModelSpec(alias="claude_vision", model_id="claude-sonnet-4-20250514", provider="anthropic"))
model_registry.register(ModelSpec(alias="gpt4o_vision", model_id="gpt-4o", provider="openai"))
```

## Configuring Judges

Default judge: **Claude Sonnet 4** scoring on accuracy, detail, and coherence (1–5 scale each).

### Add a second judge

```python
from elorian.judges import JudgeSpec, get_default_judge_registry

judge_registry = get_default_judge_registry()
judge_registry.register(
    JudgeSpec(
        alias="judge_gpt4o",
        model_id="gpt-4o",
        provider="openai",
        description="GPT-4o as judge",
    )
)
```

### Custom scoring rubric

```python
import data_designer.config as dd
from elorian.judges import JudgeSpec, JudgeRegistry

custom_scores = [
    dd.Score(
        name="faithfulness",
        description="Does the response only describe what is actually in the image?",
        options={
            1: "Major hallucinations",
            2: "Some hallucinated details",
            3: "Mostly faithful with minor liberties",
            4: "Faithful to the image content",
            5: "Perfectly faithful, no hallucinations",
        },
    ),
    dd.Score(
        name="usefulness",
        description="How useful is this description for someone who cannot see the image?",
        options={1: "Not useful", 3: "Moderately useful", 5: "Highly useful"},
    ),
]

judge_registry = JudgeRegistry()
judge_registry.register(
    JudgeSpec(
        alias="judge_claude_custom",
        model_id="claude-sonnet-4-20250514",
        provider="anthropic",
        scores=custom_scores,
    )
)
```

### Custom judge prompt

```python
JudgeSpec(
    alias="judge_strict",
    model_id="claude-sonnet-4-20250514",
    provider="anthropic",
    prompt_template=(
        "You are a strict image description evaluator.\n\n"
        "Penalize any detail not directly visible in the image.\n\n"
        "Responses:\n{responses_block}\n\n"
        "Score each response on the provided dimensions."
    ),
)
```

The `{responses_block}` placeholder is replaced with Jinja2 references to each model's response column.

## Configuring the Pipeline

```python
pipeline = MultimodalEvalPipeline(
    seed_df=seed_df,
    image_column="base64_image",       # column name in seed_df
    prompt="Describe this image in detail.",
    model_registry=model_registry,
    judge_registry=judge_registry,
    artifact_path="artifacts",         # where DataDesigner stores results
)

# Inspect configuration before running
print(pipeline.describe())
```

## Loading Images

### From a local directory

```python
from elorian.image_utils import load_images_from_directory

seed_df = load_images_from_directory(
    "path/to/images/",
    max_images=100,
    target_height=512,
)
```

### From PIL images

```python
from elorian.image_utils import load_images_from_pil

seed_df = load_images_from_pil(
    images=[img1, img2, img3],
    labels=["cat", "dog", "bird"],
    target_height=512,
)
```

### From HuggingFace

```python
from elorian.image_utils import load_images_from_hf_dataset

seed_df = load_images_from_hf_dataset(
    "zh-plus/tiny-imagenet",
    split="valid",
    image_column="image",
    label_column="label",
    max_images=50,
    target_height=512,
)
```

## Output Format

The preview/run result contains a DataFrame with these columns:

| Column | Source | Description |
|--------|--------|-------------|
| `uuid` | seed | Unique image identifier |
| `base64_image` | seed | Base64-encoded image |
| `label` | seed | Image label (if provided) |
| `response_{model_alias}` | VLM | Model's description of the image |
| `eval_{judge_alias}` | Judge | JSON with per-dimension scores and reasoning |

Example judge output (one cell):

```json
{
  "accuracy": {"reasoning": "Claude accurately described...", "score": 4},
  "detail": {"reasoning": "Good coverage of main elements...", "score": 4},
  "coherence": {"reasoning": "Well-structured markdown...", "score": 5}
}
```

## Provider Reference

Built-in providers (env var must be set):

| Provider | `provider_type` | API Key Env Var |
|----------|----------------|-----------------|
| `anthropic` | `anthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `openai` | `OPENAI_API_KEY` |

The `provider_type` is the LiteLLM routing prefix — it gets prepended to `model_id` when calling the API. See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for the full list.

## Alias Naming Rule

All aliases (model and judge) **must use underscores**, not hyphens. Column names with hyphens break Jinja2 template rendering (e.g. `response_claude-vision` is parsed as `response_claude` minus `vision`).
