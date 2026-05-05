# Seeds

Seed configs declare existing data used as input during generation. A [SeedConfig](#data_designer.config.seed.SeedConfig) combines a seed source with optional row sampling and selection settings. Seed source objects declare where seed data comes from; the engine reads them through seed readers.

Use these objects with `DataDesignerConfigBuilder.with_seed_dataset()`. Related pages: [Seed Datasets](../../concepts/seed-datasets.md) and [seed readers](../engine/seed_readers.md).

Built-in seed sources include local files, Hugging Face paths, in-memory DataFrames, directories, file contents, and agent rollout traces. Plugin seed sources can extend the same discriminated union through the plugin system.

## Seed Config

::: data_designer.config.seed

## Built-In Seed Sources

::: data_designer.config.seed_source

## DataFrame Seed Source

::: data_designer.config.seed_source_dataframe
