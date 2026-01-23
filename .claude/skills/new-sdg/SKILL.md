---
name: new-sdg 
description: Implement a new synthetic data generator using NeMo Data Designer by defining its configuration and executing a preview job.
argument-hint: <dataset-description>
---

# Your Goal

Implement a new synthetic data generator using NeMo Data Designer to match the user's specifications below.

<dataset-description>
 **$ARGUMENTS**
</dataset-description>

## Getting Exact Specifications

The user will provide you with some description, but it is likely that you
do not have enough information to precisely define what they want. It is hard
for a user to define everything up front. Ask follow up questions to the user
using the AskUser tool to narrow down on precisely what they want. 

Common things to make precise are:

- IMPORTANT: What the "axes of diversity" are -- e.g. what should be well represented and diverse in the resulting dataset.
- The kind an nature of any input data to the dataset.
- What variables should be randomized. 
- The schema of the final dataset.
- The structure of any required structured output columns.
- What facets of the output dataset are important to capture.

## Interactive, Iterative Design

> USER: Request
> YOU: Clarifying AskUser Questions
> YOU: Script Impelmentation (with preview)
> YOU: Script Execution
> YOU: Result Presentation
> YOU: Followup Questions
> USER: Respond
> YOU: ...repeat...

Very often, the initial implementation will not conform precisely to what the user wants. You are to engage in an **iterative design loop** with the user. As shown 
in the example below, you will construct a configuration, then review its outputs,
present those outputs to the user, and ask follow up questions. 

Depending on the user responses, you will then edit the script, re-run it, and present the user with the results and ask followups and so. When showing results to the user DO NOT SUMMARIZE content, it is *very important* that you show them the records as-is so they can make thoughtful decisions.

DO NOT disengage from this **iterative design loop** unless commanded by the user.


## Implementing a NeMo Data Designer Synthetic Data Generator 

- You will be writing a new python script for execution.
- The script should be made in the current working directory, so `$(pwd)/script-name.py`.
- Implement the script as a stand-alone, `uv`-executable script (https://docs.astral.sh/uv/guides/scripts/#creating-a-python-script).
- The script should depend on the latest version of `data-designer`.
- Include other third-party dependencies only if the job requires it. 
- Model aliases are required when definining LLM generation columns.
- Before implementing, make sure to use the Explore tool to understand the src/ and docs/.
- Review available model aliases and providers.
- You will need to ask the user what Model Provider they want to use via AskUser tool.
- You may use Web Search to find any information you need to help you construct the SDG, since real-world grounding is key to a good dataset.
- If you need to use a large number of categories for a sampler, just build a pandas DataFrame and use it as a Seed dataset.

### Model Alises and Providers

View known model aliases and providers with the following command. You will need a longer timeout on first run (package first-time boot).

```bash
uv run --with data-designer data-designer config list
```

### Real World Seed Data

Depending on user requirements, you may need to access real-world datasets to serve as Seed datasets for your Data Designer SDG. 
In these cases, you may use Web Search tools to search for datasets available on HuggingFace, and use the `datasets` python library
to load them. You will have to convert them to Pandas DataFrames in these cases.

If you do use real-world data, pay attention to file sizes and avoid large file transfers. Only download small sections of datasets or use a streaming option.

### Example

```python
# /// script
# dependencies = [
#   "data-designer",
# ]
# ///

# ... data designer config_builder implementation 

def build_config() -> DataDesignerConfigBuilder:
    """Implements the definition of the synthetic data generator.
    """
    config_builder = DataDesignerConfigBuilder()

    ## Add whatever columns need to be added
    # config_builder.add_column(...)
    # config_builder.add_column(...)
    # config_builder.add_column(...)

    return config_builder

if __name__ == "__main__":
    config_builder = build_config()
    designer = DataDesigner()
    preview = designer.preview(config_builder=config_builder)

    # The following command will print a random sample record
    # which you can present to the user
    preview.display_sample_record()

    # The raw data is located in this Pandas DataFrame object.
    # You can implenent code to display some or all of this 
    # to STDOUT so you can see the outputs and report to the user.
    preview.dataset
```