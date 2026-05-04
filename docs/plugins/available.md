# Catalog

The NVIDIA-maintained [DataDesignerPlugins](https://github.com/NVIDIA-NeMo/DataDesignerPlugins) repository is the home for first-party Data Designer plugin packages. It includes a generated catalog of available plugins, package templates, and validation tooling for plugin maintainers.

This documentation does not mirror every package from every plugin source. Install plugins from a trusted package index or source repository, then use the package's documented config classes in your builder code. Data Designer discovers installed plugin packages that expose a `Plugin` object through the `data_designer.plugins` entry point group.

## Common install sources

- **Local development packages**: Use `uv pip install -e .` from the plugin package directory.
- **NVIDIA-maintained plugins**: Review the [DataDesignerPlugins catalog](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/blob/main/docs/catalog.md) and install the package named by the plugin docs.
- **Internal package indexes**: Publish organization-specific plugins to the same package index your team already trusts.
- **Public package indexes**: Install community plugins only after reviewing their source, package metadata, and dependency footprint.

## Evaluating a plugin

Before adding a plugin to a production workflow, check:

- The plugin supports the Data Designer version you run.
- Its `pyproject.toml` registers entry points under `data_designer.plugins`.
- Its docs identify the config class users should import.
- Its discriminator value, such as `column_type`, `seed_type`, or `processor_type`, is unique in your environment.
- Its package dependencies fit your deployment and security requirements.

For authoring guidance, see the [plugin overview](overview.md), [Build Your Own](implement.md), [processor plugin guide](processor.md), and [FileSystemSeedReader plugin guide](filesystem_seed_reader.md).
