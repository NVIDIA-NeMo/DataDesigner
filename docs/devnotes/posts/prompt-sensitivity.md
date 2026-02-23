---
date: 2026-02-18
authors:
  - dnathawani
---

# **Mitigating Prompt Sensitivity: Manufacturing Robustness Through Diverse Preambles**

Models behave differently based on how a question is phrased --- a "cynical senior dev" and a "curious student" get different answers to the same problem. Using NeMo Data Designer, we built a pipeline that generates hundreds of diverse prompt preambles with controlled variation across tone, strictness, verbosity, and answer format, then validates each one for compliance. These preambles feed into a YAML-driven training mixture pipeline that prepends diverse instructions to existing SFT data at scale. This work directly improved Nemotron model robustness on evaluation benchmarks where prompt format varies.

<!-- more -->

---

## **Why This Matters: The 5-15 Point Swing**

When we evaluated early Nemotron checkpoints on [LiveBench](https://livebench.ai/) and internal STEM benchmarks, we noticed a troubling pattern: the model's accuracy swung by **5-15 percentage points** depending solely on how the question was phrased. The underlying reasoning was identical --- the model could solve the problem --- but small variations in the preamble caused it to format its response differently, triggering scoring failures.

```
"Select the best answer"           → 82% accuracy
"Choose the correct option"        → 78% accuracy
"Which of the following is true?"  → 74% accuracy
```

Same questions. Same model. Same knowledge. Different scores.

This is the **prompt sensitivity problem**, and it's pervasive across the industry. The root cause is simple: **the training data lacks prompt diversity**. If every STEM MCQ in your SFT dataset starts with "Answer the following question and place your answer in \boxed{}", the model learns that specific format perfectly but becomes brittle to anything else.

The fix is equally simple in principle --- expose the model to the same problems with many different phrasings --- but doing this manually at the scale of thousands of training examples is impractical. We needed to generate preambles that span a wide diversity space:

- **Sentence types:** imperative ("Select the answer"), interrogative ("Which option is correct?"), declarative ("The correct answer is to be placed in...")
- **Tones:** formal, neutral, concise, encouraging, strict
- **Strictness levels:** from "here's a question" to "you MUST follow this exact format"
- **Verbosity:** one-liners vs. detailed multi-sentence instructions
- **Answer formats:** `\boxed{}`, `\boxed{LETTER}`, `Answer: A/B/C/D`, `((X))`, `<final_answer>X</final_answer>`, and dozens more

Manually writing hundreds of preambles covering all combinations is tedious and inevitably misses regions of the diversity space. Data Designer's sampler-driven approach solves this systematically.

---

## **Pipeline Architecture**

```
                                    PROMPT SENSITIVITY PIPELINE
                                    ==========================

             ┌─────────────────────────────────────────────────────────────────────────────────────┐
             │                             STAGE 1: SEED EXAMPLES                                  │
             │                                                                                     │
             │   5 curated MCQ preambles as style anchors (not templates to copy)                  │
             │   Loaded via DataFrameSeedSource with SamplingStrategy.SHUFFLE                      │
             └─────────────────────────────────────────┬───────────────────────────────────────────┘
                                                       │
                                                       ▼
             ┌─────────────────────────────────────────────────────────────────────────────────────┐
             │                          STAGE 2: DIVERSITY SAMPLERS                                │
             │                                                                                     │
             │   sentence_type (3)    tone (5)         strictness_level (3)                        │
             │   imperative           formal            low                                        │
             │   interrogative        neutral           medium                                     │
             │   declarative          concise           high                                       │
             │                        encouraging                                                  │
             │                        strict            verbosity_level (3)                        │
             │                                          concise / standard / detailed              │
             │   domain_context (3)   answer_format (3)                                            │
             │   general              \boxed{}                                                     │
             │   STEM                 \boxed{LETTER}    Combinatorial space:                       │
             │   humanities           \boxed{<letter>}  3 × 5 × 3 × 3 × 3 × 3 = 1,215            │
             └─────────────────────────────────────────┬───────────────────────────────────────────┘
                                                       │
                                                       ▼
             ┌─────────────────────────────────────────────────────────────────────────────────────┐
             │                         STAGE 3: LLM PREAMBLE GENERATION                            │
             │                                                                                     │
             │   Single LLMTextColumnConfig conditioned on all 6 samplers + seed example           │
             │   "Generate a preamble instruction for a multiple-choice question..."               │
             │   1,000+ unique preambles generated per domain (STEM, Math, Code)                   │
             └─────────────────────────────────────────┬───────────────────────────────────────────┘
                                                       │
                                                       ▼
             ┌─────────────────────────────────────────────────────────────────────────────────────┐
             │                          STAGE 4: DUAL QUALITY JUDGES                               │
             │                                                                                     │
             │   format_compliance (binary 0/1):                                                   │
             │     Does the preamble mention the correct answer format?                            │
             │     ~15-20% of preambles fail this without the judge gate                           │
             │                                                                                     │
             │   preamble_quality (rubric 0-3):                                                    │
             │     Does the preamble match requested tone, verbosity, clarity?                     │
             │     Filter to score ≥ 2 (Good or Excellent)                                         │
             └─────────────────────────────────────────┬───────────────────────────────────────────┘
                                                       │
                                                       ▼
             ┌─────────────────────────────────────────────────────────────────────────────────────┐
             │                    STAGE 5: INTEGRATION INTO TRAINING MIXTURES                      │
             │                                                                                     │
             │   YAML-driven pipeline: add_prompt_variations.py                                    │
             │   ├─ Load base SFT dataset (STEM MCQ, Math, Code)                                  │
             │   ├─ Sample preambles from generated pool                                           │
             │   ├─ Prepend preamble to each problem's user prompt                                 │
             │   ├─ majority_percentage: 25% (original format) / 75% (diverse variations)          │
             │   └─ Pack sequences to 128k tokens for training                                     │
             └─────────────────────────────────────────────────────────────────────────────────────┘
```

The 6 samplers create a combinatorial diversity space of **1,215 unique combinations**. Even generating 1,000 records covers a substantial fraction of this space, ensuring the training data doesn't cluster around a few dominant styles.

---

## **Step 1: Curated Seed Examples**

We provide 5 hand-written preambles as reference examples. These aren't used as templates to copy --- they're provided as *style anchors* so the LLM understands what a preamble looks like:

```python
import pandas as pd
import data_designer.config as dd
from data_designer.interface import DataDesigner

seed_data = pd.DataFrame([
    {"example_preamble": "Choose the correct answer. Place your final answer in \\boxed{}."},
    {"example_preamble": "Read carefully and select the best option. Write your answer as \\boxed{LETTER}."},
    {"example_preamble": "Answer the following MCQ. Put your final answer in \\boxed{}."},
    {"example_preamble": "Consider each option and pick the correct one. Format: \\boxed{your answer}."},
    {"example_preamble": "Solve the problem and select the right choice. Enclose your answer in \\boxed{}."},
])

config = dd.DataDesignerConfigBuilder(model_configs=[
    dd.ModelConfig(alias="preamble-gen", model="qwen/qwen3-235b-a22b", provider="nvidia"),
])

config.with_seed_dataset(
    dd.DataFrameSeedSource(df=seed_data),
    sampling_strategy=dd.SamplingStrategy.SHUFFLE,
)
```

With `SamplingStrategy.SHUFFLE`, each generated record sees a random seed example alongside its sampled style parameters.

---

## **Step 2: Six Dimensions of Diversity**

Each sampler controls one axis of variation:

```python
config.add_column(dd.SamplerColumnConfig(
    name="sentence_type",
    sampler_type=dd.SamplerType.CATEGORY,
    params=dd.CategorySamplerParams(values=["imperative", "interrogative", "declarative"]),
))

config.add_column(dd.SamplerColumnConfig(
    name="tone",
    sampler_type=dd.SamplerType.CATEGORY,
    params=dd.CategorySamplerParams(values=["formal", "neutral", "concise", "encouraging", "strict"]),
))

config.add_column(dd.SamplerColumnConfig(
    name="strictness_level",
    sampler_type=dd.SamplerType.CATEGORY,
    params=dd.CategorySamplerParams(values=["low", "medium", "high"]),
))

config.add_column(dd.SamplerColumnConfig(
    name="verbosity_level",
    sampler_type=dd.SamplerType.CATEGORY,
    params=dd.CategorySamplerParams(values=["concise", "standard", "detailed"]),
))

config.add_column(dd.SamplerColumnConfig(
    name="domain_context",
    sampler_type=dd.SamplerType.CATEGORY,
    params=dd.CategorySamplerParams(values=["general", "STEM", "humanities"]),
))

config.add_column(dd.SamplerColumnConfig(
    name="answer_format",
    sampler_type=dd.SamplerType.CATEGORY,
    params=dd.CategorySamplerParams(values=["\\boxed{}", "\\boxed{LETTER}", "\\boxed{<letter>}"]),
))
```

The generation prompt references all six via Jinja2 templates:

```python
config.add_column(dd.LLMTextColumnConfig(
    name="preamble",
    model_alias="preamble-gen",
    prompt=(
        "Generate a preamble instruction for a multiple-choice question.\n"
        "Style requirements:\n"
        "- Sentence type: {{ sentence_type }}\n"
        "- Tone: {{ tone }}\n"
        "- Strictness: {{ strictness_level }}\n"
        "- Verbosity: {{ verbosity_level }}\n"
        "- Domain: {{ domain_context }}\n"
        "- Answer format: the student MUST place their final answer in {{ answer_format }}\n\n"
        "Reference example (for style, not to copy): {{ example_preamble }}\n\n"
        "Requirements:\n"
        "1. Do NOT include the actual question or options\n"
        "2. The preamble MUST mention the exact format {{ answer_format }}\n"
        "3. Keep it to 1-3 sentences based on verbosity level\n"
        "4. Match the requested tone and sentence type\n\n"
        "Return ONLY the preamble text."
    ),
))
```

This is the power of Data Designer's sampler + template approach: you define the diversity *dimensions*, and the framework handles the *combinatorics*.

---

## **Step 3: Dual Quality Judges**

Two separate judges evaluate each generated preamble:

**Format compliance (binary):** Does the preamble actually mention the required answer format? Without this gate, roughly 15-20% of generated preambles mention the wrong format or omit it entirely.

```python
config.add_column(dd.LLMJudgeColumnConfig(
    name="format_compliance",
    model_alias="preamble-gen",
    prompt=(
        "Check if this preamble instructs the user to format their answer as "
        "{{ answer_format }}.\n\n"
        "Preamble: {{ preamble }}\n"
        "Expected format: {{ answer_format }}"
    ),
    scores=[dd.Score(
        name="format_match",
        description="Does the preamble mention the correct answer format?",
        options={1: "Yes, format is correctly specified", 0: "No, format is missing or wrong"},
    )],
))
```

**Quality (0-3 rubric):** Does the preamble match the requested tone and verbosity? Is it clear and professional?

```python
config.add_column(dd.LLMJudgeColumnConfig(
    name="preamble_quality",
    model_alias="preamble-gen",
    prompt=(
        "Evaluate the quality of this MCQ preamble.\n\n"
        "Preamble: {{ preamble }}\n"
        "Tone requested: {{ tone }}\n"
        "Verbosity requested: {{ verbosity_level }}"
    ),
    scores=[dd.Score(
        name="quality",
        description="Overall preamble quality",
        options={3: "Excellent", 2: "Good", 1: "Fair", 0: "Poor"},
    )],
))
```

---

## **Step 4: Integration into Training Mixtures**

Generated preambles don't exist in isolation. They feed into a YAML-driven training mixture pipeline (`add_prompt_variations.py`) that operates at production scale:

```yaml
mixture:
  target: 1000000
  seed: 13
  files:
    - path: /path/to/openstem.jsonl
      percent: 70.6
    - path: /path/to/openstem_15p.jsonl
      percent: 10.6
    - path: /path/to/hle.jsonl
      percent: 16.3
    - path: /path/to/hle_15p.jsonl
      percent: 2.5

preamble:
  augment: true
  majority_preamble: |-
    Answer the following multiple choice question. The last line of your
    response should be in the following format: 'Answer: \boxed{A/B/C/D}'
    (e.g. 'Answer: \boxed{A}').

    {problem}
  majority_percentage: 25.0
  variations:
    path: prompts/stem_prompts_1000.jsonl
    field: preamble_text

pack:
  enabled: true
  shuffle_before: true
  shuffle_after: true
  max_seq_length: 128000
```

The pipeline:

1. **Builds the mixture** by reservoir-sampling from multiple JSONL files according to configured percentages.
2. **Applies preamble variations**: for each record, with 25% probability it uses the `majority_preamble` (the original canonical format); with 75% probability it samples from the 1,000 DD-generated variations. This ratio ensures the model still sees the canonical format frequently but is exposed to massive diversity.
3. **Detects MCQ-like prompts** using regex heuristics (looks for `\n(A)`, `\n(B)` patterns or `\boxed{}` in the assistant response). Non-MCQ records pass through unchanged.
4. **Packs sequences** to 128k tokens for efficient training.

The result: a 1M-record training mixture where each problem appears with one of 1,000+ different instruction phrasings.

---

## **Beyond SFT: Diverse RL Answer Formats**

For RL training, prompt sensitivity manifests differently. The model needs to produce answers in whatever format the evaluation harness expects, and the reward signal depends on parsing the answer correctly. We generated diverse answer format templates, each paired with an extraction regex:

```yaml
- prompt: 'End your response with ''Correct Option: A/B/C/D/...''.'
  output_regex: 'Correct Option:\s*([A-Za-z])'

- prompt: 'Put the chosen letter inside double brackets: ((X)).'
  output_regex: '\(\(([A-Za-z])\)\)'

- prompt: 'Wrap your final answer letter in XML-style tags: <final_answer>X</final_answer>.'
  output_regex: '<final_answer>\s*([A-Za-z])\s*</final_answer>'

- prompt: 'Finish by enclosing the correct option letter in double asterisks (like **A**).'
  output_regex: '\*\*([A-Za-z])\*\*'

- prompt: 'Conclude by stating ''Correct Answer >> A/B/C/D/...''.'
  output_regex: 'Correct Answer >> ([A-Za-z])'
```

We curated 25+ distinct format templates spanning brackets, parentheses, XML tags, markdown bold, arrows, and plain text. Each template includes the corresponding regex so the RL reward function can extract the answer regardless of format. This teaches the model to follow *arbitrary* formatting instructions, not just the one it saw most during SFT.

---

## **Production Scale**

We generated preambles for three domains, each with its own Data Designer pipeline:

| Domain | Preambles Generated | Mixture Size | Packing |
|--------|-------------------|--------------|---------|
| STEM MCQ | 1,000 | 1,000,000 records | 128k tokens |
| Math (boxed) | 1,000 | 1,000,000 records | 128k tokens |
| Code (Python/C++) | 1,000 | 1,000,000 records | 128k tokens |

Each domain has domain-specific answer formats (MCQ uses letter options, Math uses `\boxed{}` with numeric answers, Code uses code-only or function-signature formats) and domain-specific tone calibration.

---

## **Key Takeaways**

1. **Samplers make diversity systematic.** Six categorical samplers with 3-5 values each create a 1,215-combination space. No human annotator covers that surface area consistently.

2. **Seed examples are style anchors, not templates.** The LLM needs to see what a preamble *is*, but the samplers control what each preamble *says*. Without seeds, the LLM guesses at the format; without samplers, it converges to a narrow style.

3. **Format compliance is a hard gate.** A preamble that mentions `\boxed{LETTER}` when the answer should be `\boxed{}` will confuse the model during training. Binary judges catch this --- LLMs generate the wrong format ~15-20% of the time.

4. **The value is in the pipeline, not the individual records.** Any single preamble is easy to write by hand. The value is generating 1,000+ diverse, validated preambles automatically and integrating them into million-record training mixtures with controlled majority/variation ratios.

5. **RL needs format diversity too.** The reward function parses answers using regex. If the model only sees `\boxed{}` during training, it can't follow "put your answer in double brackets: ((X))" at evaluation time. Paired prompt-regex templates solve this.

6. **Majority percentage controls the tradeoff.** Setting `majority_percentage: 25` means the model sees the canonical format 25% of the time and diverse variations 75% of the time. This ratio was tuned empirically --- too much diversity degrades canonical-format performance; too little doesn't build robustness.

---

Key Resources:

1. [NeMo Data Designer on GitHub](https://github.com/NVIDIA-NeMo/DataDesigner)
2. [LiveBench: A Challenging, Contamination-Free LLM Benchmark](https://livebench.ai/)

---

*Want to learn more about NeMo Data Designer? Check out our [documentation](https://github.com/NVIDIA-NeMo/DataDesigner) and start building your own synthetic data pipelines today.*
