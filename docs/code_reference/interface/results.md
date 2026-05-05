# Dataset Creation Results

The `data_designer.interface.results` module defines [DatasetCreationResults](#data_designer.interface.results.DatasetCreationResults), the results container returned by [DataDesigner.create()](data_designer.md#data_designer.interface.data_designer.DataDesigner.create). It provides access to persisted creation artifacts, including the generated dataset, profiling analysis, processor outputs, task traces, dataset metadata, and Hugging Face Hub upload support.

Preview generation uses the in-memory `data_designer.config.preview_results.PreviewResults` object returned by [DataDesigner.preview()](data_designer.md#data_designer.interface.data_designer.DataDesigner.preview). This page documents persisted dataset creation results.

## `DatasetCreationResults` {#data_designer.interface.results.DatasetCreationResults}

::: data_designer.interface.results.DatasetCreationResults
    options:
      show_root_toc_entry: false
