# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Script to generate Colab-compatible notebooks from notebook source files.

This script processes jupytext percent-format Python files and:
1. Injects Colab-specific setup cells (pip install, API key from secrets)
2. Injects cells before the "Import the essentials" section
3. Saves the result as .ipynb files in docs/colab_notebooks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jupytext
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell

COLAB_SETUP_MARKDOWN = """\
### ⚡ Colab Setup

Run the cells below to install the dependencies and set up the API key. If you don't have an API key, you can generate one from [build.nvidia.com](https://build.nvidia.com).
"""

ADDITIONAL_DEPENDENCIES = {
    "4-providing-images-as-context.py": "pillow>=12.0.0",
}

COLAB_INSTALL_CELL = """\
%%capture
!pip install -U data-designer"""

COLAB_API_KEY_CELL = """\
import getpass
import os

from google.colab import userdata

try:
    os.environ["NVIDIA_API_KEY"] = userdata.get("NVIDIA_API_KEY")
except userdata.SecretNotFoundError:
    os.environ["NVIDIA_API_KEY"] = getpass.getpass("Enter your NVIDIA API key: ")"""


def create_colab_setup_cells(additional_dependencies: str) -> list[NotebookNode]:
    """Create the Colab-specific setup cells to inject before imports."""
    cells = []
    cells += [new_markdown_cell(source=COLAB_SETUP_MARKDOWN)]

    install_cell = COLAB_INSTALL_CELL
    if additional_dependencies:
        install_cell += f" {additional_dependencies}"
    cells += [new_code_cell(source=install_cell)]

    cells += [new_code_cell(source=COLAB_API_KEY_CELL)]
    return cells


def find_import_section_index(cells: list[NotebookNode]) -> int:
    """Find the index of the 'Import the essentials' markdown cell."""
    first_code_cell_index = -1
    for i, cell in enumerate(cells):
        if first_code_cell_index == -1 and cell.get("cell_type") == "code":
            first_code_cell_index = i

        if cell.get("cell_type") == "markdown":
            source = cell.get("source", "")
            if "import" in source.lower() and "essentials" in source.lower():
                return i
    return first_code_cell_index


def process_notebook(notebook: NotebookNode, source_path: Path) -> NotebookNode:
    """Process a notebook to make it Colab-compatible.

    Args:
        notebook: The input notebook

    Returns:
        The processed notebook with Colab setup cells injected
    """
    cells = notebook.cells

    additional_dependencies = ADDITIONAL_DEPENDENCIES.get(source_path.name, "")

    # Find where to insert Colab setup (before "Import the essentials")
    import_idx = find_import_section_index(cells)

    if import_idx == -1:
        # If not found, insert after first cell (title)
        import_idx = 1

    # Insert Colab setup cells before the import section
    colab_cells = create_colab_setup_cells(additional_dependencies)
    processed_cells = cells[:import_idx] + colab_cells + cells[import_idx:]

    notebook.cells = processed_cells
    return notebook


def generate_colab_notebook(source_path: Path, output_dir: Path) -> Path:
    """Generate a Colab-compatible notebook from a source file.

    Args:
        source_path: Path to the jupytext percent-format Python source file
        output_dir: Directory to save the output notebook

    Returns:
        Path to the generated notebook
    """
    # Read the source file using jupytext
    notebook = jupytext.read(source_path)

    # Process the notebook for Colab
    notebook = process_notebook(notebook, source_path)

    # Determine output path
    output_path = output_dir / f"{source_path.stem}.ipynb"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write the notebook
    jupytext.write(notebook, output_path, config={"metadata": {"jupytext": {"cell_metadata_filter": "-id"}}})

    return output_path


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Generate Colab-compatible notebooks from notebook source files.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("docs/notebook_source"),
        help="Directory containing notebook source files (default: docs/notebook_source)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/colab_notebooks"),
        help="Directory to save Colab notebooks (default: docs/colab_notebooks)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Specific files to process (if not specified, process all .py files)",
    )

    args = parser.parse_args()

    # Get list of source files
    if args.files:
        source_files = [args.source_dir / f for f in args.files]
    else:
        source_files = sorted(args.source_dir.glob("*.py"))
        # Filter out files starting with underscore (like _README.md, _pyproject.toml)
        source_files = [f for f in source_files if not f.name.startswith("_")]

    if not source_files:
        print(f"No source files found in {args.source_dir}")
        return

    print(f"📓 Generating Colab notebooks from {len(source_files)} source file(s)...")
    print(f"   Source: {args.source_dir}")
    print(f"   Output: {args.output_dir}")
    print()

    for source_path in source_files:
        if not source_path.exists():
            print(f"⚠️  Skipping {source_path} (file not found)")
            continue

        try:
            output_path = generate_colab_notebook(source_path, args.output_dir)
            print(f"✅ {source_path.name} → {output_path.name}")
        except Exception as e:
            print(f"❌ {source_path.name}: {e}")

    print()
    print(f"✨ Colab notebooks saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
