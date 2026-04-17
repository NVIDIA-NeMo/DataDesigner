# Run Config

The `run_config` module defines runtime settings that control dataset generation behavior,
including early shutdown thresholds, batch sizing, non-inference worker concurrency,
and the Jinja rendering engine used by the runtime.

`JinjaRenderingEngine.SECURE` is the default. Set `JinjaRenderingEngine.NATIVE`
when you want Jinja2's broader built-in sandbox behavior instead of Data Designer's
hardened renderer.

## Usage

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner()
data_designer.set_run_config(dd.RunConfig(
    buffer_size=500,
    max_conversation_restarts=3,
    jinja_rendering_engine=dd.JinjaRenderingEngine.NATIVE,
))
```

## API Reference

::: data_designer.config.run_config
