# Workflow Chaining

!!! warning "Experimental Feature"
    Workflow chaining is currently **experimental** and under active development. The documentation, examples, workflow API, metadata schema, and artifact layout are subject to significant changes in future releases. If you encounter any issues, have questions, or have ideas for improvement, please consider starting [a discussion on GitHub](https://github.com/NVIDIA-NeMo/DataDesigner/discussions).

Workflow chaining lets you split a dataset build into named stages. Each stage runs a normal `DataDesigner.create()` call, writes its own artifact directory, and hands a selected parquet output to the next stage as a `LocalFileSeedSource`.

Use it when one generation step naturally depends on the cleaned or reshaped output of another step, especially when a processor-only stage is clearer than mixing all transformations into one config.

## Basic shape

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner()

drafts = (
    dd.DataDesignerConfigBuilder(model_configs=[fast_model])
    .with_seed_dataset(dd.LocalFileSeedSource(path="parsed_docs/*.parquet"))
    .add_column(
        name="chunk_summary",
        column_type="llm_text",
        model_alias="fast",
        prompt="Summarize this passage:\n\n{{ text }}",
    )
    .add_column(
        name="question",
        column_type="llm_text",
        model_alias="fast",
        prompt="Write a question about this passage:\n\n{{ chunk_summary }}",
    )
    .add_column(
        name="answer",
        column_type="llm_text",
        model_alias="fast",
        prompt="Answer {{ question }} using this passage:\n\n{{ text }}",
    )
)

chatml = dd.DataDesignerConfigBuilder().add_processor(
    dd.SchemaTransformProcessorConfig(
        name="chatml",
        template={
            "messages": [
                {"role": "user", "content": "{{ question }}"},
                {"role": "assistant", "content": "{{ answer }}"},
            ],
        },
    )
)

workflow = data_designer.compose_workflow(name="doc-qa")
workflow.add_stage(
    "drafts",
    drafts,
    num_records=100,
    output_processors=[
        dd.DropColumnsProcessorConfig(
            name="drop_scratch",
            column_names=["text", "chunk_summary"],
        )
    ],
)
workflow.add_stage("chatml", chatml, output="processor:chatml")

results = workflow.run()
training_rows = results.load_dataset()
results.export("chatml.jsonl")
```

## Stage outputs

A stage can expose different views of its data:

| Surface | What it returns |
|---------|-----------------|
| `results["stage_name"]` | The effective `DatasetCreationResults` for that stage. If the stage uses `output_processors`, this points at the output-processor run. |
| `results.load_stage_output("stage_name")` | The selected output handed to downstream stages. This follows `output="processor:<name>"` and `on_success`. |
| `results.load_dataset()` | The selected output from the final stage. |

Processors added with `config_builder.add_processor(...)` run inside the stage and usually create side artifacts. They do not automatically change what the next stage receives. Use `output_processors=[...]` when a processor should define the stage boundary output.

## Processor-only stages

Stages can be processor-only when they receive seed data from an upstream stage:

```python
cleanup = dd.DataDesignerConfigBuilder().add_processor(
    dd.DropColumnsProcessorConfig(
        name="drop_private_fields",
        column_names=["email", "raw_notes"],
    )
)

workflow.add_stage("cleanup", cleanup)
```

This is useful for final cleanup, schema transforms, and format-specific export preparation.

## Resume

Workflow names are durable artifact identities. Reusing the same name with `resume=ResumeMode.IF_POSSIBLE` reuses compatible completed stages, resumes a matching partial stage through `DataDesigner.create(..., resume=ResumeMode.ALWAYS)`, and reruns the first changed or missing stage plus its descendants.

```python
from data_designer.interface import ResumeMode

results = workflow.run(resume=ResumeMode.IF_POSSIBLE)
```

Use `ResumeMode.ALWAYS` for strict resume before the first recovered checkpoint. A changed stage or missing selected output raises instead of starting fresh. If a matching partial stage resumes successfully, descendants are recreated from that stage's current output.

## Current limits

- Stages are linear. DAGs, parallel branches, and joins are planned separately.
- `push_to_hub()` does not support selected processor or callback outputs yet. Use `export()` for the selected workflow output.
- `on_success` callbacks are trusted user code. If a callback returns a path, Data Designer reads that path as the next stage input.
- The artifact layout is intended for inspection, but it is not yet a stable public contract.
