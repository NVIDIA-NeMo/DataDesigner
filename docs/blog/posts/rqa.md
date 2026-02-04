---
date: 2026-02-04
authors:
  - dcorneil
  - etramel
---

# **Graduate-Level Science Reasoning Data with NeMo Data Designer**

Using NeMo Data Designer, we created the RQA (Reasoning Question-Answer) dataset: a massive collection of graduate-level, reasoning-heavy science samples designed to push the boundaries of model performance.

<!-- more -->

---

Inference-time reasoning has transformed LLM capabilities, boosting performance in difficult domains like math and science. While reasoning is introduced in the post-training phase using Reinforcement Learning (RL), it builds on patterns that the model has seen throughout pretraining. In fact, research from NVIDIA has shown that [front-loading examples of reasoning into the pretraining phase](https://research.nvidia.com/labs/adlr/Synergy/) can have a positive, compounding impact on the quality of the final model. When training Nemotron 3 Nano, our goal was to introduce rich and diverse examples of reasoning directly into pretraining, laying the groundwork for reasoning RL in post-training.

Using NeMo Data Designer, we created the RQA (Reasoning Question-Answer) dataset: a massive collection of graduate-level, reasoning-heavy science samples designed to push the boundaries of model performance. Each sample contains a question, a trace from a reasoning LLM attempting to answer that question, and the final resulting answer. As we’ll show in the results, introducing RQA into pretraining **didn’t just result in stronger scientific reasoning \- it improved math and coding performance as well**.

This blog post walks you through how we built it, and how you can adapt our approach for your own reasoning-intensive datasets.

![RQA Blog](../../images/rqa-blog.png)

---

## **Step 1: Curating High-Quality Science Seeds from Essential-Web**

For our reasoning dataset, we knew that both quality and diversity were critical. We wanted to show the model examples of reasoning through difficult scientific problems, and we wanted to make sure that those problems covered as wide a range of scientific domains as possible. Using seed passages from web text was an obvious choice, because it allowed us to use the seed data to control both quality and diversity.

We started with [Essential-Web](https://arxiv.org/abs/2506.14111), a Common Crawl (web text) dataset where each document has been labelled with respect to both quality and subject. For instance, documents are labelled with an estimated *Education Level*, where *Graduate Level* indicates that the text “requires graduate-level education or domain expertise. Assumes deep background knowledge and specialized training to comprehend”. These labels let us rapidly filter down the documents to the highest-quality seeds for our scientific reasoning dataset.

Starting from the [STEM subset of Essential-Web](https://huggingface.co/datasets/EssentialAI/eai-taxonomy-stem-w-dclm), we filtered to documents that were:

1. Undergraduate-to-graduate education level
2. Advanced reasoning depth
3. High technical correctness
4. Advanced [Bloom taxonomy levels](https://en.wikipedia.org/wiki/Bloom's_taxonomy) for both cognitive processes (Analyze, Evaluate or Create) and knowledge domains (Conceptual, Procedural or Metacognitive)
5. In the English language and over 1000 characters.

The resulting subset consisted of roughly 14 million documents, mostly academic. Since many of the documents were very long, we extracted random chunks of \<4096 characters in length.

Essential-AI also labelled the documents according to the [Free Decimal Correspondence (FDC) code](https://everybodyslibraries.com/about-the-free-decimal-correspondence/#:~:text=What%20is%20the%20Free%20Decimal,group%20of%20subjects%20and%20disciplines.), a public-domain analogue of the Dewey Decimal system. Using the FDC code, we could see that the topics weren’t equally balanced across scientific domains; for instance, Medicine & Health was heavily over-represented. Since we planned to generate \<14 million samples in total, we aimed to capture as broad a range of topics as possible in the subset of seeds we used.

To arrive at a smaller set of seed documents balanced by topic, we used a hierarchical round-robin approach. First, we rotated between selecting seed documents across 8 major domains (Biology, Chemistry, Computer Science, Engineering, Math, Medicine/Health, Physics, and Other). Within each high-level domain, we further rotated between seed documents based on their 3-digit FDC codes; for instance, given a Physics sample with code 535 (*Light*), the next Physics sample might be from code 536 (*Heat*), then 537 (*Electricity*) and so on, ensuring that no single subdomain dominates. We continued the round robin selection at the first and second decimal place of the FDC code, where they existed.

We tested approaches using both the first 4.5 million and the first 9 million seeds according to the round-robin approach described above.

---

## **Step 2: Generating Challenging Questions**

With our seed documents ready, we moved to NeMo Data Designer to design the actual dataset. While the seed documents ground our dataset in the types of advanced scientific topics we’re interested in, they don’t typically show the *active process* of thinking through a difficult scientific problem; instead, scientific papers usually show the polished end result of advanced reasoning. This is where LLMs come in.

We first needed examples of the type of tough questions that Nemotron might be asked by a user in the real world. To do this, we used Data Designer to prompt a reasoning-enabled LLM to generate a graduate-level question *inspired by* each seed passage:

```py
import data_designer.config as dd
from data_designer.interface import DataDesigner

QUESTION_PROMPT = """
Task: Generate a highly challenging, graduate-level reasoning question
inspired by the following passage.

Follow these instructions:
1. The text serves only as inspiration for the question. You *must not*
   reference the text directly in any way.
2. The question should be appropriate for an advanced graduate-level exam
   in a course specialized in this topic.
3. Ensure that the question requires higher-order reasoning beyond simple
   recall, such as mathematical reasoning, quantitative analysis, or synthesis.
4. Tag the question with "Question:".

Text:
{{ seed_passage }}

Question: [question]
"""

# Configure the workflow with a reasoning-enabled model
config = dd.DataDesignerConfigBuilder(model_configs=[
    dd.ModelConfig(
        alias="reasoning-model",
        model="qwen/qwen3-235b-a22b",
        provider="nvidia",
    ),
])

config.with_seed_dataset(
    dd.LocalFileSeedSource(path="path/to/seed_data.parquet"),
    sampling_strategy=dd.SamplingStrategy.SHUFFLE,
)

config.add_column(
    dd.LLMTextColumnConfig(
        name="question",
        prompt=QUESTION_PROMPT,
        model_alias="reasoning-model",
    )
)
```

Note that our prompt emphasizes that the question shouldn’t reference the source passage. We want questions that stand on their own, without including the source passage itself; since these are passages from Common Crawl, we can expect that they appear in the pretraining data already, and our focus here is on generating new tokens.

---

## **Step 3: Generating High-Quality Answers with Reasoning Traces**

If you’ve ever tried to read a teacher’s answer key before, you know that sometimes the person who *wrote* the question isn’t always the best at explaining how to *answer* it. In the real world, reasoning involves a lot of what-ifs, dead ends and backtracking \- the types of behavior we can only get from a model when it has never seen the question before. This is why we chose to decouple answer generation from question generation, ensuring that the model doesn’t have any context about how the question was generated or the source passage itself when it attempts to answer it.

Below, we prompt the LLM directly with the questions we generated above, then capture the resulting reasoning trace and final answer for our RQA samples.

```py
config.add_column(
    dd.LLMTextColumnConfig(
        name="answer",
        prompt="{{ question }}",  # Present just the question
        model_alias="reasoning-model",
        extract_reasoning_content=True,  # Extract reasoning into separate column
    )
)

# Combine question, reasoning trace, and answer into final sample
config.add_column(
    dd.ExpressionColumnConfig(
        name="rqa_sample",
        expr="{{ question }}\n\n{{ answer__reasoning_content }}\n\n{{ answer }}",
    )
)
```

In the resulting dataset, we see the following new columns concatenated to the seed data columns:

- `question`
- `answer`
- `answer__reasoning_content`
- `rqa_sample`

The `question` and `answer` columns are the final result of the calls to our reasoning LLM, while `answer__reasoning_content` is the reasoning trace generated by the LLM when generating the answer. Typically we discard the reasoning trace, but here it’s critical: we want to include the model’s chain-of-thought in the final training data, distilling strong priors in Nemotron Nano 3 about *how* to work through a challenging problem. The final column, `rqa_sample`, uses Jinja2 syntax to combine all three fields into the final sample for training.

We chose to use the same frontier reasoning model to answer the questions as we used to generate them \- leveraging the model’s advanced capabilities both for formulating a tough, well-formed question and for puzzling through the answer. But with Data Designer, this choice is up to you \- you can mix-and-match models any way you like.

---

## **Results: Measurable Improvements in STEM Reasoning**

To evaluate the impact of the RQA data, we ran continued pretraining experiments on an internal checkpoint of [Nemotron-H 8B](https://research.nvidia.com/labs/adlr/nemotronh/). Nemotron-H used a two-phase pretraining approach (you can read more about it in our white paper [here](https://arxiv.org/pdf/2504.03624)). We intervened at the Phase 2 training stage, comparing the result of replacing either 4% or 8% of the existing data blend with RQA samples (taking weight from high-quality Common Crawl data). We ran the intervention for 18k steps, between a checkpoint at 140k steps and a checkpoint at 158k steps.

| Data Blend | Validation Loss (↓) | MMLU-Pro (with CoT, ↑) | Math 500 (with CoT, ↑) | GSM8K (with CoT, ↑) | Humaneval+ (↑) | MBPP+ (↑) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Baseline data blend (140k steps)** | 1.309 | 36.99 | \- | 79.98 | 38.14 | 48.68 |
| **Baseline data blend (158k steps)** | 1.258 | 43.39 | 71.00 | 81.96 | 42.71 | 53.31 |
| **with RQA (4.5m @4%, 158k steps)** | 1.256 | 44.31 | **73.40** | 82.79 | **47.20** | **54.84** |
| **with RQA (9m @8%, 158k steps)** | **1.255** | **45.80** | **73.40** | **84.76** | 45.61 | 53.80 |

One of the most surprising (and exciting\!) results was that RQA didn’t just improve performance on tests of scientific reasoning like MMLU-Pro \- it also improved performance on benchmarks associated with math reasoning (Math 500, GSM8K) and coding capabilities (Humaneval+, MBPP+). This shows how early introduction of advanced reasoning capabilities can produce robust improvements across different domains.

You can check out the RQA dataset we generated for Nemotron 3 Nano [here](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1/viewer/Nemotron-Pretraining-RQA).

---

## **Get Started with Data Designer**

Apart from the seed data, the entire pipeline is reproducible using NeMo Data Designer. Note how Data Designer handles complex data formatting with ease, leveraging Jinja2 templates in prompt generation and built-in logic to extract reasoning traces from model responses.

```py
import data_designer.config as dd
from data_designer.interface import DataDesigner

# Configure your model
model_configs = [
    dd.ModelConfig(
        alias="reasoning-model",
        model="qwen/qwen3-235b-a22b",
        provider="nvidia",
        inference_parameters=dd.ChatCompletionInferenceParams(
            max_tokens=8192,
            timeout=300,  # 5 minute timeout for long reasoning chains
        ),
    ),
]

# Build the workflow
config = dd.DataDesignerConfigBuilder(model_configs=model_configs)
config.with_seed_dataset(
    dd.LocalFileSeedSource(path="path/to/your_seed_data.parquet"),
    sampling_strategy=dd.SamplingStrategy.SHUFFLE,
)

# Generate questions
config.add_column(
    dd.LLMTextColumnConfig(
        name="question",
        prompt=QUESTION_PROMPT,
        model_alias="reasoning-model",
    )
)

# Generate answers with reasoning trace
config.add_column(
    dd.LLMTextColumnConfig(
        name="answer",
        prompt="{{ question }}",
        model_alias="reasoning-model",
        extract_reasoning_content=True,  # Extract reasoning into separate column
    )
)

# Combine into final sample
config.add_column(
    dd.ExpressionColumnConfig(
        name="rqa_sample",
        expr="{{ question }}\n\n{{ answer__reasoning_content }}\n\n{{ answer }}",
    )
)

# Run generation and save to disk
data_designer = DataDesigner()
result = data_designer.create(
    config_builder=config,
    num_records=N_RECORDS,
    dataset_name="rqa_dataset",
)
```

---

## **Summary**

The RQA dataset demonstrates that targeted synthetic data generation can meaningfully improve advanced reasoning capabilities. By:

1. Curating high-quality scientific seed data
2. Generating challenging, standalone questions from those seeds
3. Using powerful reasoning models to reason through how to answer those questions

… we created a dataset that pushes models toward graduate-level science reasoning \- and generalizable improvements on math and code as well.

Key Resources:

1. [NeMo Data Designer on GitHub](https://github.com/NVIDIA-NeMo/DataDesigner)
2. [Nemotron 3 Nano Technical Report](https://arxiv.org/pdf/2512.20848)
3. [Essential-Web](https://arxiv.org/abs/2506.14111)

The workflow is fully configurable and extensible: swap in your own seed data, adjust the prompts, or add custom validators. Data Designer makes it possible to iterate rapidly on synthetic data pipelines, turning what used to be months of manual annotation into hours of programmable generation.

---

*Want to learn more about NeMo Data Designer? Check out our [documentation](https://github.com/NVIDIA-NeMo/DataDesigner) and start building your own high-fidelity synthetic datasets today.*
