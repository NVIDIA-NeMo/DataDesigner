---
name: new-sdg
description: Implement a new synthetic data generator using NeMo Data Designer by defining its configuration and executing a preview job. Use when the user wants to create a synthetic dataset, generate training data, or build an SDG pipeline.
---

# Your Goal

Implement a new synthetic data generator using NeMo Data Designer to match the user's specifications.

## Getting Exact Specifications

The user will provide a description, but you likely need more detail. Ask follow up questions to narrow down:

- IMPORTANT: What the "axes of diversity" are -- what should be well represented and diverse.
- The kind and nature of any input data.
- What variables should be randomized.
- The schema of the final dataset.
- The structure of any required structured output columns.
- What facets of the output dataset are important.

## Interactive, Iterative Design

> USER: Request
> YOU: Clarifying Questions
> YOU: Script Implementation (with preview)
> YOU: Script Execution
> YOU: Launch Review UI
> USER: Reviews in browser, returns with feedback
> YOU: Read annotations, edit config
> YOU: ...repeat...

Engage in an **iterative design loop**. Build a configuration, run a preview, present results via the review UI, read the user's annotations, and iterate. DO NOT disengage unless commanded by the user.

## Implementing the Script

- Write a new python script in the current working directory.
- Implement as a stand-alone, `uv`-executable script (https://docs.astral.sh/uv/guides/scripts/#creating-a-python-script).
- The script should depend on the latest version of `data-designer`.
- Model aliases are required when defining LLM generation columns.
- Before implementing, explore the src/ and docs/ to understand the API.
- Review available model aliases and providers.
- Ask the user what Model Provider they want to use.
- Use Web Search for real-world grounding when building the dataset.
- For large category lists, build a pandas DataFrame and use it as a Seed dataset.

### Model Aliases and Providers

```bash
uv run --with data-designer data-designer config list
```

### Real World Seed Data

Search for datasets on HuggingFace using the `datasets` library. Convert to Pandas DataFrames. Avoid large file transfers -- use streaming or small slices.

### Example Script

```python
# /// script
# dependencies = [
#   "data-designer",
# ]
# ///

from data_designer.config import DataDesignerConfigBuilder
from data_designer.interface import DataDesigner

def build_config() -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder()
    # config_builder.add_column(...)
    return config_builder

if __name__ == "__main__":
    config_builder = build_config()
    designer = DataDesigner()
    preview = designer.preview(config_builder=config_builder)

    # Save and launch review UI (opens browser automatically)
    preview.dataset.to_parquet("preview.parquet")
    print(f"Saved {len(preview.dataset)} records to preview.parquet")

    from data_designer.web.server import run_server
    run_server(data_file="preview.parquet", open_browser=True)
```

Run the script backgrounded: `uv run script.py &`

The browser opens automatically with the review UI. Tell the user: "I've opened the review UI in your browser. Review the records and annotate any that need improvement. Come back here when you're done."

## Reading Feedback

When the user returns, read their annotations:

```python
import json
from pathlib import Path

annotations = json.loads(Path("preview_annotations.json").read_text())

# annotations["summary"] -> {"good": 3, "bad": 2, ...}
# annotations["annotations"] -> [{"row": 2, "rating": "bad", "column": "answer", "note": "...", "data": {...}}]
```

Use the annotations to guide your next iteration:
- `column` field tells you which prompt/column to fix
- `note` tells you what's wrong
- `data` gives you the actual values

After editing and re-running preview, overwrite the parquet and reload:

```bash
curl -X POST http://127.0.0.1:8765/api/session/reload
```

The review UI picks up the new data automatically.

## When Done

When the user clicks "Finish Review" in the UI and returns satisfied, ask if they want to run a full `create` job for the final dataset.
