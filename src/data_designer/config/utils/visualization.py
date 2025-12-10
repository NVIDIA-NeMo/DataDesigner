# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from collections import OrderedDict
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.pretty import Pretty
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from data_designer.config.base import ConfigBase
from data_designer.config.column_types import DataDesignerColumnType
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.config.sampler_params import SamplerType
from data_designer.config.utils.code_lang import code_lang_to_syntax_lexer
from data_designer.config.utils.constants import NVIDIA_API_KEY_ENV_VAR_NAME, OPENAI_API_KEY_ENV_VAR_NAME
from data_designer.config.utils.errors import DatasetSampleDisplayError

if TYPE_CHECKING:
    from data_designer.config.config_builder import DataDesignerConfigBuilder


console = Console()


def get_nvidia_api_key() -> Optional[str]:
    return os.getenv(NVIDIA_API_KEY_ENV_VAR_NAME)


def get_openai_api_key() -> Optional[str]:
    return os.getenv(OPENAI_API_KEY_ENV_VAR_NAME)


class ColorPalette(str, Enum):
    NVIDIA_GREEN = "#76b900"
    PURPLE = "#9525c6"
    YELLOW = "#f9c500"
    BLUE = "#0074df"
    RED = "#e52020"
    ORANGE = "#ef9100"
    MAGENTA = "#d2308e"
    TEAL = "#1dbba4"


class WithRecordSamplerMixin:
    _display_cycle_index: int = 0

    @cached_property
    def _record_sampler_dataset(self) -> pd.DataFrame:
        if hasattr(self, "dataset") and self.dataset is not None and isinstance(self.dataset, pd.DataFrame):
            return self.dataset
        elif (
            hasattr(self, "load_dataset")
            and callable(self.load_dataset)
            and (dataset := self.load_dataset()) is not None
            and isinstance(dataset, pd.DataFrame)
        ):
            return dataset
        else:
            raise DatasetSampleDisplayError("No valid dataset found in results object.")

    def _has_processor_artifacts(self) -> bool:
        return hasattr(self, "processor_artifacts") and self.processor_artifacts is not None

    def display_sample_record(
        self,
        index: Optional[int] = None,
        *,
        hide_seed_columns: bool = False,
        syntax_highlighting_theme: str = "dracula",
        background_color: Optional[str] = None,
        processors_to_display: Optional[list[str]] = None,
    ) -> None:
        """Display a sample record from the Data Designer dataset preview.

        Args:
            index: Index of the record to display. If None, the next record will be displayed.
                This is useful for running the cell in a notebook multiple times.
            hide_seed_columns: If True, the columns from the seed dataset (if any) will not be displayed.
            syntax_highlighting_theme: Theme to use for syntax highlighting. See the `Syntax`
                documentation from `rich` for information about available themes.
            background_color: Background color to use for the record. See the `Syntax`
                documentation from `rich` for information about available background colors.
            processors_to_display: List of processors to display the artifacts for. If None, all processors will be displayed.
        """
        i = index or self._display_cycle_index

        try:
            record = self._record_sampler_dataset.iloc[i]
            num_records = len(self._record_sampler_dataset)
        except IndexError:
            raise DatasetSampleDisplayError(f"Index {i} is out of bounds for dataset of length {num_records}.")

        processor_data_to_display = None
        if self._has_processor_artifacts() and len(self.processor_artifacts) > 0:
            if processors_to_display is None:
                processors_to_display = list(self.processor_artifacts.keys())

            if len(processors_to_display) > 0:
                processor_data_to_display = {}
                for processor in processors_to_display:
                    if (
                        isinstance(self.processor_artifacts[processor], list)
                        and len(self.processor_artifacts[processor]) == num_records
                    ):
                        processor_data_to_display[processor] = self.processor_artifacts[processor][i]
                    else:
                        processor_data_to_display[processor] = self.processor_artifacts[processor]

        display_sample_record(
            record=record,
            processor_data_to_display=processor_data_to_display,
            config_builder=self._config_builder,
            background_color=background_color,
            syntax_highlighting_theme=syntax_highlighting_theme,
            hide_seed_columns=hide_seed_columns,
            record_index=i,
        )
        if index is None:
            self._display_cycle_index = (self._display_cycle_index + 1) % num_records


def create_rich_histogram_table(
    data: dict[str, Union[int, float]],
    column_names: tuple[int, int],
    name_style: str = ColorPalette.BLUE.value,
    value_style: str = ColorPalette.TEAL.value,
    title: Optional[str] = None,
    **kwargs,
) -> Table:
    table = Table(title=title, **kwargs)
    table.add_column(column_names[0], justify="right", style=name_style)
    table.add_column(column_names[1], justify="left", style=value_style)

    max_count = max(data.values())
    for name, value in data.items():
        bar = "" if max_count <= 0 else "█" * int((value / max_count) * 20)
        table.add_row(str(name), f"{bar} {value:.1f}")

    return table


def display_sample_record(
    record: Union[dict, pd.Series, pd.DataFrame],
    config_builder: DataDesignerConfigBuilder,
    processor_data_to_display: Optional[dict[str, Union[list[str], str]]] = None,
    background_color: Optional[str] = None,
    syntax_highlighting_theme: str = "dracula",
    record_index: Optional[int] = None,
    hide_seed_columns: bool = False,
):
    if isinstance(record, (dict, pd.Series)):
        record = pd.DataFrame([record]).iloc[0]
    elif isinstance(record, pd.DataFrame):
        if record.shape[0] > 1:
            raise DatasetSampleDisplayError(
                f"The record must be a single record. You provided a DataFrame with {record.shape[0]} records."
            )
        record = record.iloc[0]
    else:
        raise DatasetSampleDisplayError(
            "The record must be a single record in a dictionary, pandas Series, "
            f"or pandas DataFrame. You provided: {type(record)}."
        )

    render_list = []
    table_kws = dict(show_lines=True, expand=True)

    seed_columns = config_builder.get_columns_of_type(DataDesignerColumnType.SEED_DATASET)
    if not hide_seed_columns and len(seed_columns) > 0:
        table = Table(title="Seed Columns", **table_kws)
        table.add_column("Name")
        table.add_column("Value")
        for col in seed_columns:
            if not col.drop:
                table.add_row(col.name, convert_to_row_element(record[col.name]))
        render_list.append(pad_console_element(table))

    non_code_columns = (
        config_builder.get_columns_of_type(DataDesignerColumnType.SAMPLER)
        + config_builder.get_columns_of_type(DataDesignerColumnType.EXPRESSION)
        + config_builder.get_columns_of_type(DataDesignerColumnType.LLM_TEXT)
        + config_builder.get_columns_of_type(DataDesignerColumnType.LLM_STRUCTURED)
    )
    if len(non_code_columns) > 0:
        table = Table(title="Generated Columns", **table_kws)
        table.add_column("Name")
        table.add_column("Value")
        for col in non_code_columns:
            if not col.drop:
                table.add_row(col.name, convert_to_row_element(record[col.name]))
        render_list.append(pad_console_element(table))

    for col in config_builder.get_columns_of_type(DataDesignerColumnType.LLM_CODE):
        panel = Panel(
            Syntax(
                record[col.name],
                lexer=code_lang_to_syntax_lexer(col.code_lang),
                theme=syntax_highlighting_theme,
                word_wrap=True,
                background_color=background_color,
            ),
            title=col.name,
            expand=True,
        )
        render_list.append(pad_console_element(panel))

    validation_columns = config_builder.get_columns_of_type(DataDesignerColumnType.VALIDATION)
    if len(validation_columns) > 0:
        table = Table(title="Validation", **table_kws)
        table.add_column("Name")
        table.add_column("Value", ratio=1)
        for col in validation_columns:
            if not col.drop:
                # Add is_valid before other fields
                if "is_valid" in record[col.name]:
                    value_to_display = {"is_valid": record[col.name].get("is_valid")} | record[col.name]
                else:  # if columns treated separately
                    value_to_display = {}
                    for col_name, validation_output in record[col.name].items():
                        value_to_display[col_name] = {
                            "is_valid": validation_output.get("is_valid", None)
                        } | validation_output

                table.add_row(col.name, convert_to_row_element(value_to_display))
        render_list.append(pad_console_element(table, (1, 0, 1, 0)))

    llm_judge_columns = config_builder.get_columns_of_type(DataDesignerColumnType.LLM_JUDGE)
    if len(llm_judge_columns) > 0:
        for col in llm_judge_columns:
            if col.drop:
                continue
            table = Table(title=f"LLM-as-a-Judge: {col.name}", **table_kws)
            row = []
            judge = record[col.name]

            for measure, results in judge.items():
                table.add_column(measure)
                row.append(f"score: {results['score']}\nreasoning: {results['reasoning']}")
            table.add_row(*row)
            render_list.append(pad_console_element(table, (1, 0, 1, 0)))

    if processor_data_to_display and len(processor_data_to_display) > 0:
        for processor_name, processor_data in processor_data_to_display.items():
            table = Table(title=f"Processor Outputs: {processor_name}", **table_kws)
            table.add_column("Name")
            table.add_column("Value")
            for col, value in processor_data.items():
                table.add_row(col, convert_to_row_element(value))
        render_list.append(pad_console_element(table, (1, 0, 1, 0)))

    if record_index is not None:
        index_label = Text(f"[index: {record_index}]", justify="center")
        render_list.append(index_label)

    console.print(Group(*render_list), markup=False)


def display_sampler_table(
    sampler_params: dict[SamplerType, ConfigBase],
    title: Optional[str] = None,
) -> None:
    table = Table(expand=True)
    table.add_column("Type")
    table.add_column("Parameter")
    table.add_column("Data Type")
    table.add_column("Required", justify="center")
    table.add_column("Constraints")

    for sampler_type, params in sampler_params.items():
        num = 0
        schema = params.model_json_schema()
        for param_name, field_info in schema["properties"].items():
            is_required = param_name in schema.get("required", [])
            table.add_row(
                sampler_type if num == 0 else "",
                param_name,
                _get_field_type(field_info),
                "✓" if is_required else "",
                _get_field_constraints(field_info, schema),
            )
            num += 1
        table.add_section()

    title = title or "NeMo Data Designer Samplers"

    group = Group(Rule(title, end="\n\n"), table)
    console.print(group)


def display_model_configs_table(model_configs: list[ModelConfig]) -> None:
    table_model_configs = Table(expand=True)
    table_model_configs.add_column("Alias")
    table_model_configs.add_column("Model")
    table_model_configs.add_column("Provider")
    table_model_configs.add_column("Temperature")
    table_model_configs.add_column("Top P")
    for model_config in model_configs:
        table_model_configs.add_row(
            model_config.alias,
            model_config.model,
            model_config.provider,
            str(model_config.inference_parameters.temperature),
            str(model_config.inference_parameters.top_p),
        )
    group_args: list = [Rule(title="Model Configs"), table_model_configs]
    if len(model_configs) == 0:
        subtitle = Text(
            "‼️ No model configs found. Please provide at least one model config to the config builder",
            style="dim",
            justify="center",
        )
        group_args.insert(1, subtitle)
    group = Group(*group_args)
    console.print(group)


def display_model_providers_table(model_providers: list[ModelProvider]) -> None:
    table_model_providers = Table(expand=True)
    table_model_providers.add_column("Name")
    table_model_providers.add_column("Endpoint")
    table_model_providers.add_column("API Key")
    for model_provider in model_providers:
        api_key = model_provider.api_key
        if model_provider.api_key == OPENAI_API_KEY_ENV_VAR_NAME:
            if get_openai_api_key() is not None:
                api_key = mask_api_key(get_openai_api_key())
            else:
                api_key = f"* {OPENAI_API_KEY_ENV_VAR_NAME!r} not set in environment variables * "
        elif model_provider.api_key == NVIDIA_API_KEY_ENV_VAR_NAME:
            if get_nvidia_api_key() is not None:
                api_key = mask_api_key(get_nvidia_api_key())
            else:
                api_key = f"* {NVIDIA_API_KEY_ENV_VAR_NAME!r} not set in environment variables *"
        else:
            api_key = mask_api_key(model_provider.api_key)
        table_model_providers.add_row(model_provider.name, model_provider.endpoint, api_key)
    group = Group(Rule(title="Model Providers"), table_model_providers)
    console.print(group)


def mask_api_key(api_key: str | None) -> str:
    """Mask API keys for display.

    Environment variable names (all uppercase) are kept visible.
    Actual API keys are masked to show only the last 4 characters.

    Args:
        api_key: The API key to mask.

    Returns:
        Masked API key string or "(not set)" if None.
    """
    if not api_key:
        return "(not set)"

    # Keep environment variable names visible
    if api_key.isupper():
        return api_key

    # Mask actual API keys
    return "***" + api_key[-4:] if len(api_key) > 4 else "***"


def convert_to_row_element(elem):
    try:
        elem = Pretty(json.loads(elem))
    except (TypeError, json.JSONDecodeError):
        pass
    if isinstance(elem, (np.integer, np.floating, np.ndarray)):
        elem = str(elem)
    elif isinstance(elem, (list, dict)):
        elem = Pretty(elem)
    return elem


def pad_console_element(elem, padding=(1, 0, 1, 0)):
    return Padding(elem, padding)


def _get_field_type(field: dict) -> str:
    """Extract human-readable type information from a JSON Schema field."""

    # single type
    if "type" in field:
        if field["type"] == "array":
            return " | ".join([f"{f.strip()}[]" for f in _get_field_type(field["items"]).split("|")])
        if field["type"] == "object":
            return "dict"
        return field["type"]

    # union type
    elif "anyOf" in field:
        types = []
        for f in field["anyOf"]:
            if "$ref" in f:
                types.append("enum")
            elif f.get("type") == "array":
                if "items" in f and "$ref" in f["items"]:
                    types.append("enum[]")
                else:
                    types.append(f"{f['items']['type']}[]")
            else:
                types.append(f.get("type", ""))
        return " | ".join(t for t in types if t)

    return ""


def _get_field_constraints(field: dict, schema: dict) -> str:
    """Extract human-readable constraints from a JSON Schema field."""
    constraints = []

    # numeric constraints
    if "minimum" in field:
        constraints.append(f">= {field['minimum']}")
    if "exclusiveMinimum" in field:
        constraints.append(f"> {field['exclusiveMinimum']}")
    if "maximum" in field:
        constraints.append(f"<= {field['maximum']}")
    if "exclusiveMaximum" in field:
        constraints.append(f"< {field['exclusiveMaximum']}")

    # string constraints
    if "minLength" in field:
        constraints.append(f"len > {field['minLength']}")
    if "maxLength" in field:
        constraints.append(f"len < {field['maxLength']}")

    # array constraints
    if "minItems" in field:
        constraints.append(f"len > {field['minItems']}")
    if "maxItems" in field:
        constraints.append(f"len < {field['maxItems']}")

    # enum constraints
    if "enum" in _get_field_type(field) and "$defs" in schema:
        enum_values = []
        for defs in schema["$defs"].values():
            if "enum" in defs:
                enum_values.extend(defs["enum"])
        if len(enum_values) > 0:
            enum_values = OrderedDict.fromkeys(enum_values)
            constraints.append(f"allowed: {', '.join(enum_values.keys())}")

    return ", ".join(constraints)
