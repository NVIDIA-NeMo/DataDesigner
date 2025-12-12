# Analysis

The `analysis` modules provide tools for profiling and analyzing generated datasets. It includes statistics tracking, column profiling, and reporting capabilities.

## Column Statistics

Column statistics are automatically computed for every column after generation. They provide basic metrics specific to the column type. For example, LLM columns track token usage statistics, sampler columns track distribution information, and validation columns track validation success rates.

The classes below are result objects that store the computed statistics for each column type and provide methods for formatting these results for display in reports.

::: data_designer.config.analysis.column_statistics

## Column Profilers

Column profilers are optional analysis tools that provide deeper insights into specific column types. Currently, the only column profiler available is the Judge Score Profiler.

The classes below are result objects that store the computed profiler results and provide methods for formatting these results for display in reports.

::: data_designer.config.analysis.column_profilers

## Dataset Profiler

The [DatasetProfilerResults](#data_designer.config.analysis.dataset_profiler.DatasetProfilerResults) class contains complete profiling results for a generated dataset. It aggregates column-level statistics, metadata, and profiler results, and provides methods to:

- Compute dataset-level metrics (completion percentage, column type summary)
- Filter statistics by column type
- Generate formatted analysis reports via the `to_report()` method

Reports can be displayed in the console or exported to HTML/SVG formats.

::: data_designer.config.analysis.dataset_profiler
