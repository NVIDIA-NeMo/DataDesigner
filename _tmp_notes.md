# Plugin system updates

## Requirements

1. Plugins MUST support defining both a configuration object (a Pydantic model) and some `engine`-related implementation object (`ConfigurableTask`, `ColumnGenerator`, etc.).
1. The UX for making plugins discoverable MUST be simple. We should only require users define a single `Plugin` object that gets referenced in a single entry point.
1. The plugin system MUST NOT introduce a dependency chain that makes any `config` module depend on any `engine` module.
  a. Breaks "slim install" support, because `engine` code may include third-party deps that a `config`-only slim install will not include.
  b. Introduces a high risk of circular imports, because `engine` code depends on `config` modules.
1. A client using a slim-install of the library SHOULD be able to use plugins.


## Current state

The current plugin system violates REQ 3 (and by extension REQ 4):

```
config.column_types -> data_designer.plugin_manager -> data_designer.plugins.plugin -> data_designer.engine.configurable_task
```
(`->` means "imports" aka "depends on")


## Blessed engine modules?

One idea that was floated is to refactor existing `engine` code so that base classes exist in some "blessed" module(s) that we would ensure do not create circular imports with `config` modules,
but this seems...
- hard to enforce/guarantee
- potentially restricting (what if a plugin author wants to use objects from other parts of `engine`)
- conceptually not ideal (it's just simpler to say "`engine` can import/depend on `config`, but not vice-versa" full stop instead of carving out exceptions)
- potentially complicated with respect to however we restructure packages to support slim installs


## Idea proposed in this branch

Make the `Plugin` object "lazy" by defining the config and and task types as fully-qualified strings rather than objects.

By using strings in the `Plugin` object fields, **if** the plugin is structured with multiple files (e.g. `config.py` and `task.py`)*,
then the core library's `config` code that uses plugins (to extend discriminated union types) can load the plugin and resolve **only**
the config class type; it would not need to resolve/load/import the plugin's task-related module where `engine` base classes are imported and subclassed.

> *This multi-file setup wouldn't be **required** out of the box; see "Plugin development lifecycle" below.

Example:
```python
# src/my_plugin/config.py
from data_designer.config.column_types import SingleColumnConfig

class MyPluginConfig(SingleColumnConfig):
    foo: str



# src/my_plugin/generator.py
from data_designer.engine.column_generators.generators.base import ColumnGenerator
from my_plugin.config import MyPluginConfig

class MyPluginGenerator(ColumnGenerator[MyPluginConfig]):
    pass



# src/my_plugin/plugin.py
from data_designer.plugins.plugin import Plugin, PluginType

plugin = Plugin(
    config_cls="my_plugin.config.MyPluginConfig",
    task_cls="my_plugin.generator.MyPluginGenerator",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
```


### Strings instead of concrete types?

Yeah, a little sad, but seems a reasonable compromise given the benefits this unlocks.

To mitigate against dumb stuff like typos, I suggest we ship a test helper function that we'd encourage plugin authors use in their unit tests:
```python
# my_plugin/tests/test_plugin.py
from data_designer.plugins.test import is_valid_plugin
from my_plugin.plugin import plugin


def test_plugin_validity():
    assert is_valid_plugin(plugin)
```
(Similar to `pd.testing.assert_frame_equal`.)

To start, that test helper would ensure two things:
1. The string class names resolve to concrete types that do exist
2. The resolved concrete types are subclasses of the expected base classes

In the future, we could extend the helper to validate other things that are more complex than just Pydantic field type validations.

Remember: we can't implement this validation as a Pydantic validator because it would break the laziness.
We **can** at least validate that the module exists (and this branch does so), but only the test helper
can go further and actually fully resolve the two fields.


### Plugin development lifecycle

A plugin author _could_ continue defining everything in one Python file and things would still work in the library.
The limitation would be that a plugin defined that way would not support slim installs, and so clients like NMP would not be able to use it.
**This might be perfectly fine for many plugins**, especially in the early going.
A reasonable "plugin development lifecycle" might be:
1. Develop everything in one file and get it working with the library
2. Refactor the plugin to support slim installs (if ever desired)

Plugin authors would only need to do step 2 if/when we want to make the plugin available in NMP.
That step 2 refactor would involve breaking the plugin implementation up into multiple files _and_ (if necessary) making sure any heavyweight,
task-only third party dependencies are included under an `engine` extra.
