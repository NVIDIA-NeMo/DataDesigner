# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html
import io
import json
import logging
import os
import re
from collections import OrderedDict
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.pretty import Pretty
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.base import ConfigBase
from data_designer.config.column_types import DataDesignerColumnType
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.config.sampler_params import SamplerType
from data_designer.config.utils.code_lang import code_lang_to_syntax_lexer
from data_designer.config.utils.constants import (
    DEFAULT_DISPLAY_WIDTH,
    NVIDIA_API_KEY_ENV_VAR_NAME,
    OPENAI_API_KEY_ENV_VAR_NAME,
)
from data_designer.config.utils.errors import DatasetSampleDisplayError
from data_designer.config.utils.image_helpers import (
    extract_base64_from_data_uri,
    is_base64_image,
    is_image_path,
    is_image_url,
    load_image_path_to_base64,
)

if TYPE_CHECKING:
    import pandas as pd

    from data_designer.config.config_builder import DataDesignerConfigBuilder
    from data_designer.config.dataset_metadata import DatasetMetadata

console = Console()
logger = logging.getLogger(__name__)


def _display_image_if_in_notebook(image_data: str, col_name: str) -> bool:
    """Display image with caption in Jupyter notebook if available.

    Args:
        image_data: Base64-encoded image data, data URI, file path, or URL.
        col_name: Name of the column (used for caption).

    Returns:
        True if image was displayed, False otherwise.
    """
    try:
        # Check if we're in a Jupyter environment
        from IPython.display import HTML, display

        get_ipython()  # This will raise NameError if not in IPython/Jupyter

        # Escape column name to prevent HTML injection
        escaped_col_name = html.escape(col_name)

        # URLs: render directly as <img src='url'>
        if is_image_url(image_data):
            escaped_url = html.escape(image_data)
            html_content = f"""
            <div style='display: flex; flex-direction: column; align-items: flex-start; margin-top: 20px; margin-bottom: 20px;'>
                <div style='margin-bottom: 10px;'><strong>🖼️ {escaped_col_name}</strong></div>
                <img src='{escaped_url}'/>
            </div>
            """
            display(HTML(html_content))
            return True

        # File paths: load from disk and convert to base64
        if is_image_path(image_data) and not image_data.startswith("data:image/"):
            loaded_base64 = load_image_path_to_base64(image_data)
            if loaded_base64 is None:
                console.print(
                    f"[yellow]⚠️ Could not load image from path '{image_data}' for column '{col_name}'[/yellow]"
                )
                return False
            base64_data = loaded_base64
        else:
            base64_data = image_data

        # Extract base64 from data URI if present
        img_base64 = extract_base64_from_data_uri(base64_data)

        # Create HTML with caption and image in left-aligned container
        html_content = f"""
        <div style='display: flex; flex-direction: column; align-items: flex-start; margin-top: 20px; margin-bottom: 20px;'>
            <div style='margin-bottom: 10px;'><strong>🖼️ {escaped_col_name}</strong></div>
            <img src='data:image/png;base64,{img_base64}'/>
        </div>
        """
        display(HTML(html_content))
        return True
    except (ImportError, NameError):
        # Not in a notebook environment
        return False
    except Exception as e:
        console.print(f"[yellow]⚠️ Could not display image for column '{col_name}': {e}[/yellow]")
        return False


def get_nvidia_api_key() -> str | None:
    return os.getenv(NVIDIA_API_KEY_ENV_VAR_NAME)


def get_openai_api_key() -> str | None:
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
    dataset_metadata: DatasetMetadata | None

    @cached_property
    def _record_sampler_dataset(self) -> pd.DataFrame:
        if hasattr(self, "dataset") and self.dataset is not None and isinstance(self.dataset, lazy.pd.DataFrame):
            return self.dataset
        elif (
            hasattr(self, "load_dataset")
            and callable(self.load_dataset)
            and (dataset := self.load_dataset()) is not None
            and isinstance(dataset, lazy.pd.DataFrame)
        ):
            return dataset
        else:
            raise DatasetSampleDisplayError("No valid dataset found in results object.")

    def _has_processor_artifacts(self) -> bool:
        return hasattr(self, "processor_artifacts") and self.processor_artifacts is not None

    def display_sample_record(
        self,
        index: int | None = None,
        *,
        syntax_highlighting_theme: str = "dracula",
        background_color: str | None = None,
        processors_to_display: list[str] | None = None,
        hide_seed_columns: bool = False,
        save_path: str | Path | None = None,
        theme: Literal["dark", "light"] = "dark",
        display_width: int = DEFAULT_DISPLAY_WIDTH,
    ) -> None:
        """Display a sample record from the Data Designer dataset preview.

        Args:
            index: Index of the record to display. If None, the next record will be displayed.
                This is useful for running the cell in a notebook multiple times.
            syntax_highlighting_theme: Theme to use for syntax highlighting. See the `Syntax`
                documentation from `rich` for information about available themes.
            background_color: Background color to use for the record. See the `Syntax`
                documentation from `rich` for information about available background colors.
            processors_to_display: List of processors to display the artifacts for. If None, all processors will be displayed.
            hide_seed_columns: If True, seed columns will not be displayed separately.
            save_path: Optional path to save the rendered output as an HTML or SVG file.
            theme: Color theme for saved HTML files (dark or light).
            display_width: Width of the rendered output in characters.
        """
        i = self._display_cycle_index if index is None else index

        num_records = len(self._record_sampler_dataset)
        try:
            record = self._record_sampler_dataset.iloc[i]
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

        seed_column_names = (
            None if hide_seed_columns or self.dataset_metadata is None else self.dataset_metadata.seed_column_names
        )

        display_sample_record(
            record=record,
            processor_data_to_display=processor_data_to_display,
            config_builder=self._config_builder,
            background_color=background_color,
            syntax_highlighting_theme=syntax_highlighting_theme,
            record_index=i,
            seed_column_names=seed_column_names,
            save_path=save_path,
            theme=theme,
            display_width=display_width,
        )
        if index is None:
            self._display_cycle_index = (self._display_cycle_index + 1) % num_records


def create_rich_histogram_table(
    data: dict[str, int | float],
    column_names: tuple[int, int],
    name_style: str = ColorPalette.BLUE.value,
    value_style: str = ColorPalette.TEAL.value,
    title: str | None = None,
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
    record: dict | pd.Series | pd.DataFrame,
    config_builder: DataDesignerConfigBuilder,
    processor_data_to_display: dict[str, list[str] | str] | None = None,
    background_color: str | None = None,
    syntax_highlighting_theme: str = "dracula",
    record_index: int | None = None,
    seed_column_names: list[str] | None = None,
    save_path: str | Path | None = None,
    theme: Literal["dark", "light"] = "dark",
    display_width: int = DEFAULT_DISPLAY_WIDTH,
) -> None:
    if isinstance(record, (dict, lazy.pd.Series)):
        record = lazy.pd.DataFrame([record]).iloc[0]
    elif isinstance(record, lazy.pd.DataFrame):
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

    # Display seed columns if seed_column_names is provided and not empty
    if seed_column_names:
        table = Table(title="Seed Columns", **table_kws)
        table.add_column("Name")
        table.add_column("Value")
        for col_name in seed_column_names:
            if col_name in record.index:
                table.add_row(col_name, convert_to_row_element(record[col_name]))
        render_list.append(pad_console_element(table))

    non_code_columns = (
        config_builder.get_columns_of_type(DataDesignerColumnType.SAMPLER)
        + config_builder.get_columns_of_type(DataDesignerColumnType.EXPRESSION)
        + config_builder.get_columns_of_type(DataDesignerColumnType.LLM_TEXT)
        + config_builder.get_columns_of_type(DataDesignerColumnType.LLM_STRUCTURED)
        + config_builder.get_columns_of_type(DataDesignerColumnType.EMBEDDING)
        + config_builder.get_columns_of_type(DataDesignerColumnType.CUSTOM)
    )
    if len(non_code_columns) > 0:
        table = Table(title="Generated Columns", **table_kws)
        table.add_column("Name")
        table.add_column("Value")
        for col in non_code_columns:
            if not col.drop:
                if col.column_type == DataDesignerColumnType.EMBEDDING:
                    record[col.name]["embeddings"] = [
                        get_truncated_list_as_string(embd) for embd in record[col.name].get("embeddings")
                    ]
                table.add_row(col.name, convert_to_row_element(record[col.name]))
                # Also display side_effect_columns for custom generators
                if col.column_type == DataDesignerColumnType.CUSTOM:
                    for output_col in col.side_effect_columns:
                        if output_col in record:
                            table.add_row(output_col, convert_to_row_element(record[output_col]))
        render_list.append(pad_console_element(table))

    # Collect image generation columns (will be displayed at the end)
    image_columns = config_builder.get_columns_of_type(DataDesignerColumnType.IMAGE)
    images_to_display_later = []
    if len(image_columns) > 0:
        # Check if we're in a notebook to decide display style
        try:
            get_ipython()
            in_notebook = True
        except NameError:
            in_notebook = False

        # Create table for image columns
        table = Table(title="Images", **table_kws)
        table.add_column("Name")
        table.add_column("Preview")

        for col in image_columns:
            if col.drop:
                continue
            image_data = record[col.name]

            # Handle list of images
            if isinstance(image_data, list):
                previews = []
                for idx, img in enumerate(image_data):
                    if is_base64_image(img):
                        previews.append(f"[{idx}] <base64, {len(img)} chars>")
                        if in_notebook:
                            images_to_display_later.append((f"{col.name}[{idx}]", img))
                    elif is_image_url(img):
                        previews.append(f"[{idx}] <URL: {img[:30]}...>")
                        if in_notebook:
                            images_to_display_later.append((f"{col.name}[{idx}]", img))
                    elif is_image_path(img):
                        previews.append(f"[{idx}] <path: {img}>")
                        if in_notebook:
                            images_to_display_later.append((f"{col.name}[{idx}]", img))
                    else:
                        previews.append(f"[{idx}] {str(img)[:30]}")
                preview = "\n".join(previews) if previews else "<empty list>"
            # Handle single image (backwards compatibility)
            elif is_base64_image(image_data):
                preview = f"<base64 encoded, {len(image_data)} chars>"
                if in_notebook:
                    images_to_display_later.append((col.name, image_data))
            elif is_image_url(image_data):
                preview = f"<URL: {image_data[:50]}...>"
                if in_notebook:
                    images_to_display_later.append((col.name, image_data))
            elif is_image_path(image_data):
                preview = f"<path: {image_data}>"
                if in_notebook:
                    images_to_display_later.append((col.name, image_data))
            else:
                preview = str(image_data)[:100] + "..." if len(str(image_data)) > 100 else str(image_data)

            table.add_row(col.name, preview)

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

    if save_path is not None:
        recording_console = Console(record=True, width=display_width, file=io.StringIO())
        recording_console.print(Group(*render_list), markup=False)
        _save_console_output(recording_console, save_path, theme=theme)
    else:
        terminal_width = console.width
        capped_width = min(terminal_width, display_width)
        display_console = Console(width=capped_width)
        display_console.print(Group(*render_list), markup=False)

    # Display images at the bottom with captions (only in notebook)
    if len(images_to_display_later) > 0:
        for col_name, image_data in images_to_display_later:
            _display_image_if_in_notebook(image_data, col_name)


def get_truncated_list_as_string(long_list: list[Any], max_items: int = 2) -> str:
    if max_items <= 0:
        raise ValueError("max_items must be greater than 0")
    if len(long_list) > max_items:
        truncated_part = long_list[:max_items]
        return f"[{', '.join(str(x) for x in truncated_part)}, ...]"
    else:
        return str(long_list)


def display_sampler_table(
    sampler_params: dict[SamplerType, ConfigBase],
    title: str | None = None,
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
    table_model_configs.add_column("Inference Parameters")
    for model_config in model_configs:
        params_display = model_config.inference_parameters.format_for_display()

        table_model_configs.add_row(
            model_config.alias,
            model_config.model,
            model_config.provider,
            params_display,
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


def convert_to_row_element(elem: Any) -> Any:
    try:
        elem = Pretty(json.loads(elem))
    except (TypeError, json.JSONDecodeError):
        pass
    if isinstance(elem, (lazy.np.integer, lazy.np.floating, lazy.np.ndarray)):
        elem = str(elem)
    elif isinstance(elem, (list, dict)):
        elem = Pretty(elem)
    return elem


def pad_console_element(elem: Any, padding: tuple[int, int, int, int] = (1, 0, 1, 0)) -> Padding:
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


_SAMPLE_RECORD_DARK_CSS = """
:root { color-scheme: dark; }
html, body { background: #020a1d !important; color: #dbe8ff !important; }
pre, code { color: inherit !important; }
table, th, td { border-color: rgba(184, 210, 255, 0.5) !important; }
"""


def apply_html_post_processing(html_path: str | Path, *, theme: Literal["dark", "light"] = "dark") -> None:
    """Inject viewport meta tag and optional dark-mode CSS into a Rich-exported HTML file."""
    path = Path(html_path)
    try:
        content = path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError) as exc:
        logger.warning("Could not post-process HTML at %s: %s", path, exc)
        return

    if 'name="viewport"' in content:
        return

    viewport_tag = '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    injection = viewport_tag

    if theme == "dark":
        dark_css = _SAMPLE_RECORD_DARK_CSS.strip()
        injection += f'<style id="data-designer-styles">\n{dark_css}\n</style>\n'

    if re.search(r"</head>", content, flags=re.I):
        content = re.sub(r"</head>", lambda m: injection + m.group(), content, count=1, flags=re.I)
    else:
        content = injection + content
    path.write_text(content, encoding="utf-8")


def _save_console_output(
    recorded_console: Console, save_path: str | Path, *, theme: Literal["dark", "light"] = "dark"
) -> None:
    save_path = str(save_path)
    suffix = Path(save_path).suffix.lower()
    if suffix == ".html":
        recorded_console.save_html(save_path)
        apply_html_post_processing(save_path, theme=theme)
    elif suffix == ".svg":
        recorded_console.save_svg(save_path, title="")
    else:
        raise DatasetSampleDisplayError(
            f"The extension of the save path must be either .html or .svg. You provided {save_path}."
        )
