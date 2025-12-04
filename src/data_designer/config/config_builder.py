# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from typing_extensions import Self

from data_designer.config.analysis.column_profilers import ColumnProfilerConfigT
from data_designer.config.base import ExportableConfigBase
from data_designer.config.column_configs import SeedDatasetColumnConfig
from data_designer.config.column_types import (
    ColumnConfigT,
    DataDesignerColumnType,
    column_type_is_llm_generated,
    get_column_config_from_kwargs,
    get_column_display_order,
)
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.dataset_builders import BuildStage
from data_designer.config.datastore import DatastoreSettings, fetch_seed_dataset_column_names
from data_designer.config.default_model_settings import get_default_model_configs
from data_designer.config.errors import BuilderConfigurationError, InvalidColumnTypeError, InvalidConfigError
from data_designer.config.models import ModelConfig, load_model_configs
from data_designer.config.processors import ProcessorConfig, ProcessorType, get_processor_config_from_kwargs
from data_designer.config.sampler_constraints import (
    ColumnConstraintT,
    ColumnInequalityConstraint,
    ConstraintType,
    ScalarInequalityConstraint,
)
from data_designer.config.seed import (
    DatastoreSeedDatasetReference,
    IndexRange,
    LocalSeedDatasetReference,
    PartitionBlock,
    SamplingStrategy,
    SeedConfig,
    SeedDatasetReference,
)
from data_designer.config.utils.constants import DEFAULT_REPR_HTML_STYLE, REPR_HTML_TEMPLATE
from data_designer.config.utils.info import ConfigBuilderInfo
from data_designer.config.utils.io_helpers import serialize_data, smart_load_yaml
from data_designer.config.utils.misc import can_run_data_designer_locally, json_indent_list_of_strings, kebab_to_snake
from data_designer.config.utils.type_helpers import resolve_string_enum
from data_designer.config.utils.validation import ViolationLevel, rich_print_violations, validate_data_designer_config

logger = logging.getLogger(__name__)


class BuilderConfig(ExportableConfigBase):
    """Configuration container for Data Designer builder.

    This class holds the main Data Designer configuration along with optional
    datastore settings needed for seed dataset operations.

    Attributes:
        data_designer: The main Data Designer configuration containing columns,
            constraints, profilers, and other settings.
        datastore_settings: Optional datastore settings for accessing external
            datasets.
    """

    data_designer: DataDesignerConfig
    datastore_settings: Optional[DatastoreSettings]


class DataDesignerConfigBuilder:
    """Config builder for Data Designer configurations.

    This class provides a high-level interface for building Data Designer configurations.
    """

    @classmethod
    def from_config(cls, config: Union[dict, str, Path, BuilderConfig]) -> Self:
        """Create a DataDesignerConfigBuilder from an existing configuration.

        Args:
            config: Configuration source. Can be:
                - A dictionary containing the configuration
                - A string or Path to a YAML/JSON configuration file
                - A BuilderConfig object

        Returns:
            A new instance populated with the configuration from the provided source.

        Raises:
            ValueError: If the config format is invalid.
            ValidationError: If the builder config loaded from the config is invalid.
        """
        if isinstance(config, BuilderConfig):
            builder_config = config
        else:
            json_config = json.loads(serialize_data(smart_load_yaml(config)))
            builder_config = BuilderConfig.model_validate(json_config)

        builder = cls(model_configs=builder_config.data_designer.model_configs)
        config = builder_config.data_designer

        for col in config.columns:
            builder.add_column(col)

        for constraint in config.constraints or []:
            builder.add_constraint(constraint=constraint)

        if config.seed_config:
            if builder_config.datastore_settings is None:
                if can_run_data_designer_locally():
                    seed_dataset_reference = LocalSeedDatasetReference(dataset=config.seed_config.dataset)
                else:
                    raise BuilderConfigurationError("🛑 Datastore settings are required.")
            else:
                seed_dataset_reference = DatastoreSeedDatasetReference(
                    dataset=config.seed_config.dataset,
                    datastore_settings=builder_config.datastore_settings,
                )
                builder.set_seed_datastore_settings(builder_config.datastore_settings)
            builder.with_seed_dataset(
                seed_dataset_reference,
                sampling_strategy=config.seed_config.sampling_strategy,
                selection_strategy=config.seed_config.selection_strategy,
            )

        return builder

    def __init__(self, model_configs: Optional[Union[list[ModelConfig], str, Path]] = None):
        """Initialize a new DataDesignerConfigBuilder instance.

        Args:
            model_configs: Model configurations. Can be:
                - None to use default model configurations in local mode
                - A list of ModelConfig objects
                - A string or Path to a model configuration file
        """
        self._column_configs = {}
        self._model_configs = _load_model_configs(model_configs)
        self._processor_configs: list[ProcessorConfig] = []
        self._seed_config: Optional[SeedConfig] = None
        self._constraints: list[ColumnConstraintT] = []
        self._profilers: list[ColumnProfilerConfigT] = []
        self._datastore_settings: Optional[DatastoreSettings] = None

    @property
    def model_configs(self) -> list[ModelConfig]:
        """Get the model configurations for this builder.

        Returns:
            A list of ModelConfig objects used for data generation.
        """
        return self._model_configs

    @property
    def allowed_references(self) -> list[str]:
        """Get all referenceable variables allowed in prompt templates and expressions.

        This includes all column names and their side effect columns that can be
        referenced in prompt templates and expressions within the configuration.

        Returns:
            A list of variable names that can be referenced in templates and expressions.
        """
        side_effect_columns = sum([[c.name] + c.side_effect_columns for c in self._column_configs.values()], [])
        return list(self._column_configs.keys()) + list(set(side_effect_columns))

    @property
    def info(self) -> ConfigBuilderInfo:
        """Get the ConfigBuilderInfo object for this builder.

        Returns:
            An object containing information about the configuration.
        """
        return ConfigBuilderInfo(model_configs=self._model_configs)

    def add_model_config(self, model_config: ModelConfig) -> Self:
        """Add a model configuration to the current Data Designer configuration.

        Args:
            model_config: The model configuration to add.
        """
        if model_config.alias in [mc.alias for mc in self._model_configs]:
            raise BuilderConfigurationError(
                f"🛑 Model configuration with alias {model_config.alias} already exists. Please delete the existing model configuration or choose a different alias."
            )
        self._model_configs.append(model_config)
        return self

    def delete_model_config(self, alias: str) -> Self:
        """Delete a model configuration from the current Data Designer configuration by alias.

        Args:
            alias: The alias of the model configuration to delete.
        """
        self._model_configs = [mc for mc in self._model_configs if mc.alias != alias]
        if len(self._model_configs) == 0:
            logger.warning(
                f"⚠️ No model configurations found after deleting model configuration with alias {alias}. Please add a model configuration before building the configuration."
            )
        return self

    def add_column(
        self,
        column_config: Optional[ColumnConfigT] = None,
        *,
        name: Optional[str] = None,
        column_type: Optional[DataDesignerColumnType] = None,
        **kwargs,
    ) -> Self:
        """Add a Data Designer column configuration to the current Data Designer configuration.

        If no column config object is provided, you must provide the `name`, `column_type`, and any
        additional keyword arguments that are required by the column config constructor.

        Args:
            column_config: Data Designer column config object to add.
            name: Name of the column to add. This is only used if `column_config` is not provided.
            column_type: Column type to add. This is only used if `column_config` is not provided.
            **kwargs: Additional keyword arguments to pass to the column constructor.

        Returns:
            The current Data Designer config builder instance.
        """
        if column_config is None:
            if name is None or column_type is None:
                raise BuilderConfigurationError(
                    "🛑 You must provide either a 'column_config' object or 'name' *and* 'column_type' "
                    f"with additional keyword arguments. You provided {column_config=}, {name=}, and {column_type=}."
                )
            column_config = get_column_config_from_kwargs(name=name, column_type=column_type, **kwargs)

        allowed_column_configs = ColumnConfigT.__args__
        if not any(isinstance(column_config, t) for t in allowed_column_configs):
            raise InvalidColumnTypeError(
                f"🛑 Invalid column config object: '{column_config}'. Valid column config options are: "
                f"{', '.join([t.__name__ for t in allowed_column_configs])}"
            )

        self._column_configs[column_config.name] = column_config
        return self

    def add_constraint(
        self,
        constraint: Optional[ColumnConstraintT] = None,
        *,
        constraint_type: Optional[ConstraintType] = None,
        **kwargs,
    ) -> Self:
        """Add a constraint to the current Data Designer configuration.

        Currently, constraints are only supported for numerical samplers.

        You can either provide a constraint object directly, or provide a constraint type and
        additional keyword arguments to construct the constraint object. Valid constraint types are:
            - "scalar_inequality": Constraint between a column and a scalar value.
            - "column_inequality": Constraint between two columns.

        Args:
            constraint: Constraint object to add.
            constraint_type: Constraint type to add. Ignored when `constraint` is provided.
            **kwargs: Additional keyword arguments to pass to the constraint constructor.

        Returns:
            The current Data Designer config builder instance.
        """
        if constraint is None:
            if constraint_type is None:
                raise BuilderConfigurationError(
                    "🛑 You must provide either a 'constraint' object or 'constraint_type' "
                    "with additional keyword arguments."
                )
            try:
                constraint_type = ConstraintType(constraint_type)
            except Exception:
                raise BuilderConfigurationError(
                    f"🛑 Invalid constraint type: {constraint_type}. Valid options are: "
                    f"{', '.join([t.value for t in ConstraintType])}"
                )
            if constraint_type == ConstraintType.SCALAR_INEQUALITY:
                constraint = ScalarInequalityConstraint(**kwargs)
            elif constraint_type == ConstraintType.COLUMN_INEQUALITY:
                constraint = ColumnInequalityConstraint(**kwargs)

        allowed_constraint_types = ColumnConstraintT.__args__
        if not any(isinstance(constraint, t) for t in allowed_constraint_types):
            raise BuilderConfigurationError(
                "🛑 Invalid constraint object. Valid constraint options are: "
                f"{', '.join([t.__name__ for t in allowed_constraint_types])}"
            )

        self._constraints.append(constraint)
        return self

    def add_processor(
        self,
        processor_config: Optional[ProcessorConfig] = None,
        *,
        processor_type: Optional[ProcessorType] = None,
        **kwargs,
    ) -> Self:
        """Add a processor to the current Data Designer configuration.

        You can either provide a processor config object directly, or provide a processor type and
        additional keyword arguments to construct the processor config object.

        Args:
            processor_config: The processor configuration object to add.
            processor_type: The type of processor to add.
            **kwargs: Additional keyword arguments to pass to the processor constructor.

        Returns:
            The current Data Designer config builder instance.
        """
        if processor_config is None:
            if processor_type is None:
                raise BuilderConfigurationError(
                    "🛑 You must provide either a 'processor_config' object or 'processor_type' "
                    "with additional keyword arguments."
                )
            processor_config = get_processor_config_from_kwargs(processor_type=processor_type, **kwargs)

        # Checks elsewhere fail if DropColumnsProcessor drops a column but it is not marked for drop
        if processor_config.processor_type == ProcessorType.DROP_COLUMNS:
            for column in processor_config.column_names:
                if column in self._column_configs:
                    self._column_configs[column].drop = True

        self._processor_configs.append(processor_config)
        return self

    def add_profiler(self, profiler_config: ColumnProfilerConfigT) -> Self:
        """Add a profiler to the current Data Designer configuration.

        Args:
            profiler_config: The profiler configuration object to add.

        Returns:
            The current Data Designer config builder instance.

        Raises:
            BuilderConfigurationError: If the profiler configuration is of an invalid type.
        """
        if not isinstance(profiler_config, ColumnProfilerConfigT):
            if hasattr(ColumnProfilerConfigT, "__args__"):
                valid_options = ", ".join([t.__name__ for t in ColumnProfilerConfigT.__args__])
            else:
                valid_options = ColumnProfilerConfigT.__name__
            raise BuilderConfigurationError(f"🛑 Invalid profiler object. Valid profiler options are: {valid_options}")
        self._profilers.append(profiler_config)
        return self

    def get_profilers(self) -> list[ColumnProfilerConfigT]:
        """Get all profilers.

        Returns:
            A list of profiler configuration objects.
        """
        return self._profilers

    def build(self, *, skip_validation: bool = False, raise_exceptions: bool = False) -> DataDesignerConfig:
        """Build a DataDesignerConfig instance based on the current builder configuration.

        Args:
            skip_validation: Whether to skip validation of the configuration.
            raise_exceptions: Whether to raise an exception if the configuration is invalid.

        Returns:
            The current Data Designer config object.
        """
        if not skip_validation:
            self.validate(raise_exceptions=raise_exceptions)

        return DataDesignerConfig(
            model_configs=self._model_configs,
            seed_config=self._seed_config,
            columns=list(self._column_configs.values()),
            constraints=self._constraints or None,
            profilers=self._profilers or None,
            processors=self._processor_configs or None,
        )

    def delete_constraints(self, target_column: str) -> Self:
        """Delete all constraints for the given target column.

        Args:
            target_column: Name of the column to remove constraints for.

        Returns:
            The current Data Designer config builder instance.
        """
        self._constraints = [c for c in self._constraints if c.target_column != target_column]
        return self

    def delete_column(self, column_name: str) -> Self:
        """Delete the column with the given name.

        Args:
            column_name: Name of the column to delete.

        Returns:
            The current Data Designer config builder instance.

        Raises:
            BuilderConfigurationError: If trying to delete a seed dataset column.
        """
        if isinstance(self._column_configs.get(column_name), SeedDatasetColumnConfig):
            raise BuilderConfigurationError("Seed columns cannot be deleted. Please update the seed dataset instead.")
        self._column_configs.pop(column_name, None)
        return self

    def get_column_config(self, name: str) -> ColumnConfigT:
        """Get a column configuration by name.

        Args:
            name: Name of the column to retrieve the config for.

        Returns:
            The column configuration object.

        Raises:
            KeyError: If no column with the given name exists.
        """
        return self._column_configs[name]

    def get_column_configs(self) -> list[ColumnConfigT]:
        """Get all column configurations.

        Returns:
            A list of all column configuration objects.
        """
        return list(self._column_configs.values())

    def get_constraints(self, target_column: str) -> list[ColumnConstraintT]:
        """Get all constraints for the given target column.

        Args:
            target_column: Name of the column to get constraints for.

        Returns:
            A list of constraint objects targeting the specified column.
        """
        return [c for c in self._constraints if c.target_column == target_column]

    def get_llm_gen_columns(self) -> list[ColumnConfigT]:
        """Get all LLM-generated column configurations.

        Returns:
            A list of column configurations that use LLM generation.
        """
        return [c for c in self._column_configs.values() if column_type_is_llm_generated(c.column_type)]

    def get_columns_of_type(self, column_type: DataDesignerColumnType) -> list[ColumnConfigT]:
        """Get all column configurations of the specified type.

        Args:
            column_type: The type of columns to filter by.

        Returns:
            A list of column configurations matching the specified type.
        """
        column_type = resolve_string_enum(column_type, DataDesignerColumnType)
        return [c for c in self._column_configs.values() if c.column_type == column_type]

    def get_columns_excluding_type(self, column_type: DataDesignerColumnType) -> list[ColumnConfigT]:
        """Get all column configurations excluding the specified type.

        Args:
            column_type: The type of columns to exclude.

        Returns:
            A list of column configurations that do not match the specified type.
        """
        column_type = resolve_string_enum(column_type, DataDesignerColumnType)
        return [c for c in self._column_configs.values() if c.column_type != column_type]

    def get_processor_configs(self) -> dict[BuildStage, list[ProcessorConfig]]:
        """Get processor configuration objects.

        Returns:
            A dictionary of processor configuration objects by dataset builder stage.
        """
        return self._processor_configs

    def get_seed_config(self) -> Optional[SeedConfig]:
        """Get the seed config for the current Data Designer configuration.

        Returns:
            The seed config if configured, None otherwise.
        """
        return self._seed_config

    def get_seed_datastore_settings(self) -> Optional[DatastoreSettings]:
        """Get most recent datastore settings for the current Data Designer configuration.

        Returns:
            The datastore settings if configured, None otherwise.
        """
        return None if not self._datastore_settings else DatastoreSettings.model_validate(self._datastore_settings)

    def num_columns_of_type(self, column_type: DataDesignerColumnType) -> int:
        """Get the count of columns of the specified type.

        Args:
            column_type: The type of columns to count.

        Returns:
            The number of columns matching the specified type.
        """
        return len(self.get_columns_of_type(column_type))

    def set_seed_datastore_settings(self, datastore_settings: Optional[DatastoreSettings]) -> Self:
        """Set the datastore settings for the seed dataset.

        Args:
            datastore_settings: The datastore settings to use for the seed dataset.
        """
        self._datastore_settings = datastore_settings
        return self

    def validate(self, *, raise_exceptions: bool = False) -> Self:
        """Validate the current Data Designer configuration.

        Args:
            raise_exceptions: Whether to raise an exception if the configuration is invalid.

        Returns:
            The current Data Designer config builder instance.

        Raises:
            InvalidConfigError: If the configuration is invalid and raise_exceptions is True.
        """

        violations = validate_data_designer_config(
            columns=list(self._column_configs.values()),
            processor_configs=self._processor_configs,
            allowed_references=self.allowed_references,
        )
        rich_print_violations(violations)
        if raise_exceptions and len([v for v in violations if v.level == ViolationLevel.ERROR]) > 0:
            raise InvalidConfigError(
                "🛑 Your configuration contains validation errors. Please address the indicated issues and try again."
            )
        if len(violations) == 0:
            logger.info("✅ Validation passed")
        return self

    def with_seed_dataset(
        self,
        dataset_reference: SeedDatasetReference,
        *,
        sampling_strategy: SamplingStrategy = SamplingStrategy.ORDERED,
        selection_strategy: Optional[Union[IndexRange, PartitionBlock]] = None,
    ) -> Self:
        """Add a seed dataset to the current Data Designer configuration.

        This method sets the seed dataset for the configuration and automatically creates
        SeedDatasetColumnConfig objects for each column found in the dataset. The column
        names are fetched from the dataset source (Hugging Face Hub or NeMo Microservices Datastore).

        Args:
            dataset_reference: Seed dataset reference for fetching from the datastore.
            sampling_strategy: The sampling strategy to use when generating data from the seed dataset.
                Defaults to ORDERED sampling.

        Returns:
            The current Data Designer config builder instance.
        """
        self._seed_config = SeedConfig(
            dataset=dataset_reference.dataset,
            sampling_strategy=sampling_strategy,
            selection_strategy=selection_strategy,
        )
        self.set_seed_datastore_settings(
            dataset_reference.datastore_settings if hasattr(dataset_reference, "datastore_settings") else None
        )
        for column_name in fetch_seed_dataset_column_names(dataset_reference):
            self._column_configs[column_name] = SeedDatasetColumnConfig(name=column_name)
        return self

    def write_config(self, path: Union[str, Path], indent: Optional[int] = 2, **kwargs) -> None:
        """Write the current configuration to a file.

        Args:
            path: Path to the file to write the configuration to.
            indent: Indentation level for the output file (default: 2).
            **kwargs: Additional keyword arguments passed to the serialization methods used.

        Raises:
            BuilderConfigurationError: If the file format is unsupported.
        """
        cfg = self.get_builder_config()
        suffix = Path(path).suffix
        if suffix in {".yaml", ".yml"}:
            cfg.to_yaml(path, indent=indent, **kwargs)
        elif suffix == ".json":
            cfg.to_json(path, indent=indent, **kwargs)
        else:
            raise BuilderConfigurationError(f"🛑 Unsupported file type: {suffix}. Must be `.yaml`, `.yml` or `.json`.")

    def get_builder_config(self) -> BuilderConfig:
        """Get the builder config for the current Data Designer configuration.

        Returns:
            The builder config.
        """
        return BuilderConfig(data_designer=self.build(), datastore_settings=self._datastore_settings)

    def __repr__(self) -> str:
        """Generates a string representation of the DataDesignerConfigBuilder instance.

        Returns:
            A formatted string showing the builder's configuration including seed dataset and column information grouped by type.
        """
        if len(self._column_configs) == 0:
            return f"{self.__class__.__name__}()"

        props_to_repr = {
            "seed_dataset": (None if self._seed_config is None else f"'{self._seed_config.dataset}'"),
        }

        for column_type in get_column_display_order():
            columns = self.get_columns_of_type(column_type)
            if len(columns) > 0:
                column_label = f"{kebab_to_snake(column_type.value)}_columns"
                props_to_repr[column_label] = json_indent_list_of_strings([c.name for c in columns], indent=8)

        repr_string = f"{self.__class__.__name__}(\n"
        for k, v in props_to_repr.items():
            if v is not None:
                v_indented = v if "[" not in v else f"{v[:-1]}" + "    " + v[-1]
                repr_string += f"    {k}: {v_indented}\n"
        repr_string += ")"
        return repr_string

    def _repr_html_(self) -> str:
        """Return an HTML representation of the DataDesignerConfigBuilder instance..

        This method provides a syntax-highlighted HTML representation of the
        builder's string representation.

        Returns:
            HTML string with syntax highlighting for the builder representation.
        """
        repr_string = self.__repr__()
        formatter = HtmlFormatter(style=DEFAULT_REPR_HTML_STYLE, cssclass="code")
        highlighted_html = highlight(repr_string, PythonLexer(), formatter)
        css = formatter.get_style_defs(".code")
        return REPR_HTML_TEMPLATE.format(css=css, highlighted_html=highlighted_html)


def _load_model_configs(model_configs: Optional[Union[list[ModelConfig], str, Path]] = None) -> list[ModelConfig]:
    """Resolves the provided model_configs, which may be a string or Path to a model configuration file.
    If None or empty, returns default model configurations if possible, otherwise raises an error.
    """
    if model_configs:
        return load_model_configs(model_configs)
    elif can_run_data_designer_locally():
        return get_default_model_configs()
    else:
        raise BuilderConfigurationError("🛑 Model configurations are required!")
