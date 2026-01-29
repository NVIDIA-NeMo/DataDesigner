# Run Config

The `run_config` module defines runtime settings that control dataset generation behavior,
including early shutdown thresholds, batch sizing, and non-inference worker concurrency.

## Usage

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner()
data_designer.set_run_config(dd.RunConfig(
    buffer_size=500,
    max_conversation_restarts=3,
))
```

## API Reference

::: data_designer.config.run_config
