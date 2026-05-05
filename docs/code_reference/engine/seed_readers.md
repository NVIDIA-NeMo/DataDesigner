# Seed Readers

Seed readers are engine-side adapters that turn a configured seed source into tabular seed rows. The engine attaches a `SeedSource` and secret resolver, asks the reader for column names and dataset size, then streams batches into generation.

!!! note "Engine"
    This page is for Data Designer developers and seed reader plugin authors. Config-side seed source objects live in `data_designer.config.seed_source`.

Plugin authors can use these APIs from the [Build Your Own](../../plugins/build_your_own.md) guide. For config-side seed source objects, see the seed dataset guides and `data_designer.config.seed_source`.

## Core Contracts

### `SeedReader` {#data_designer.engine.resources.seed_reader.SeedReader}

::: data_designer.engine.resources.seed_reader.SeedReader
    options:
      show_root_toc_entry: false

### `FileSystemSeedReader` {#data_designer.engine.resources.seed_reader.FileSystemSeedReader}

::: data_designer.engine.resources.seed_reader.FileSystemSeedReader
    options:
      show_root_toc_entry: false

### `SeedReaderFileSystemContext` {#data_designer.engine.resources.seed_reader.SeedReaderFileSystemContext}

::: data_designer.engine.resources.seed_reader.SeedReaderFileSystemContext
    options:
      show_root_toc_entry: false

### `SeedReaderBatch` {#data_designer.engine.resources.seed_reader.SeedReaderBatch}

::: data_designer.engine.resources.seed_reader.SeedReaderBatch
    options:
      show_root_toc_entry: false

### `SeedReaderBatchReader` {#data_designer.engine.resources.seed_reader.SeedReaderBatchReader}

::: data_designer.engine.resources.seed_reader.SeedReaderBatchReader
    options:
      show_root_toc_entry: false

### `PandasSeedReaderBatch` {#data_designer.engine.resources.seed_reader.PandasSeedReaderBatch}

::: data_designer.engine.resources.seed_reader.PandasSeedReaderBatch
    options:
      show_root_toc_entry: false

### `create_seed_reader_output_dataframe` {#data_designer.engine.resources.seed_reader.create_seed_reader_output_dataframe}

::: data_designer.engine.resources.seed_reader.create_seed_reader_output_dataframe
    options:
      show_root_toc_entry: false

## Built-In Readers

### `LocalFileSeedReader` {#data_designer.engine.resources.seed_reader.LocalFileSeedReader}

::: data_designer.engine.resources.seed_reader.LocalFileSeedReader
    options:
      show_root_toc_entry: false

### `HuggingFaceSeedReader` {#data_designer.engine.resources.seed_reader.HuggingFaceSeedReader}

::: data_designer.engine.resources.seed_reader.HuggingFaceSeedReader
    options:
      show_root_toc_entry: false

### `DataFrameSeedReader` {#data_designer.engine.resources.seed_reader.DataFrameSeedReader}

::: data_designer.engine.resources.seed_reader.DataFrameSeedReader
    options:
      show_root_toc_entry: false

### `DirectorySeedReader` {#data_designer.engine.resources.seed_reader.DirectorySeedReader}

::: data_designer.engine.resources.seed_reader.DirectorySeedReader
    options:
      show_root_toc_entry: false

### `FileContentsSeedReader` {#data_designer.engine.resources.seed_reader.FileContentsSeedReader}

::: data_designer.engine.resources.seed_reader.FileContentsSeedReader
    options:
      show_root_toc_entry: false

### `AgentRolloutSeedReader` {#data_designer.engine.resources.seed_reader.AgentRolloutSeedReader}

::: data_designer.engine.resources.seed_reader.AgentRolloutSeedReader
    options:
      show_root_toc_entry: false

## Registry and Errors

### `SeedReaderRegistry` {#data_designer.engine.resources.seed_reader.SeedReaderRegistry}

::: data_designer.engine.resources.seed_reader.SeedReaderRegistry
    options:
      show_root_toc_entry: false

### `SeedReaderError` {#data_designer.engine.resources.seed_reader.SeedReaderError}

::: data_designer.engine.resources.seed_reader.SeedReaderError
    options:
      show_root_toc_entry: false
