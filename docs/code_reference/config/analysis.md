# Analysis

Profiling result objects and report helpers returned after generation.

## Column Statistics

`DataDesigner.create()` and `DataDesigner.preview()` run the dataset profiler after generation. The profiler computes statistics for each configured column; side-effect columns are recorded separately in `DatasetProfilerResults.side_effect_column_names`.

Statistics result classes store computed metrics for each column type and format those metrics for reports.

::: data_designer.config.analysis.column_statistics

## Column Profilers

Column profilers are optional analysis tools that provide deeper insights into specific column types. Currently, the only column profiler available is the Judge Score Profiler.

Profiler result classes store computed profiler output and format it for reports.

::: data_designer.config.analysis.column_profilers

## Dataset Profiler

The [DatasetProfilerResults](#data_designer.config.analysis.dataset_profiler.DatasetProfilerResults) class stores profiling results for a generated dataset. It aggregates column-level statistics, side-effect column names, and optional profiler results, and provides methods to:

- Compute dataset-level metrics (completion percentage, column type summary)
- Filter statistics by column type
- Generate formatted analysis reports via the `to_report()` method

Reports can be displayed in the console or exported to HTML/SVG formats.

::: data_designer.config.analysis.dataset_profiler
