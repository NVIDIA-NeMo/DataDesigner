#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert Jupyter notebooks (.ipynb) to Fern NotebookViewer JSON+TS format.

Reads notebook JSON and outputs a minimal format with cells array:
  { "cells": [ { "type": "markdown"|"code", "source": "...", "source_html"?: "...",
    "language"?: "python", "outputs"?: [{ "type": "text"|"image", "data": "...", "format"?: "plain"|"html" }] } ] }

Code cells include source_html (Pygments syntax-highlighted HTML) and outputs when available.

The leading "Open In Colab" badge cell that `generate_colab_notebooks.py` injects is skipped:
NotebookViewer renders its own colab banner from the wrapper MDX's `colabUrl` prop, so the raw
HTML anchor would otherwise leak into the page.

Writes both <name>.json (canonical data) and <name>.ts (default-export wrapper that MDX
imports — Fern's bundler doesn't follow JSON imports cleanly).

Usage:
  python ipynb-to-fern-json.py input.ipynb -o output.json
  python ipynb-to-fern-json.py docs/colab_notebooks/1-the-basics.ipynb -o fern/components/notebooks/1-the-basics.json

Run after: make convert-execute-notebooks && make generate-colab-notebooks
  (executed notebooks preserve outputs; generate-colab injects the colab setup cell.)
"""

from __future__ import annotations

import base64
import io
import json
import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt
from PIL import Image
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

# Cap each image's longest edge so notebook .ts payloads stay small enough for
# Fern's SSR bundler (a 22MB module from full-resolution Flux outputs broke
# server-render evaluation). 800px is plenty for in-doc viewing.
MAX_IMAGE_DIMENSION = 800

# CommonMark-compliant markdown renderer with table + strikethrough +
# raw-HTML support. Used to pre-render markdown cell sources to HTML at
# build time so NotebookViewer doesn't have to ship a JS markdown parser.
_MD = MarkdownIt("commonmark", {"html": True, "linkify": False, "breaks": False}).enable("table").enable("strikethrough")

COLAB_BADGE_RE = re.compile(
    r"colab\.research\.google\.com/(?:assets/colab-badge\.svg|github/)",
    re.IGNORECASE,
)


def get_language(metadata: dict) -> str:
    info = metadata.get("kernelspec", {}) or {}
    lang = info.get("language", "python")
    return "python" if lang == "python3" else lang


def highlight_code(source: str, language: str) -> str | None:
    try:
        lexer = get_lexer_by_name(language, stripall=True)
    except ClassNotFound:
        return None
    formatter = HtmlFormatter(noclasses=True, style="friendly", nowrap=True)
    return highlight(source, lexer, formatter)


def _join_source(source: list | str) -> str:
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def is_colab_badge_cell(cell: dict) -> bool:
    """True if the cell is the redundant Colab badge prepended by generate_colab_notebooks.py."""
    if cell.get("cell_type") != "markdown":
        return False
    src = _join_source(cell.get("source", []))
    return bool(COLAB_BADGE_RE.search(src))


def shrink_image_b64(b64: str, max_dim: int = MAX_IMAGE_DIMENSION) -> tuple[str, str]:
    """Decode a base64 PNG, resize so its longest edge is at most max_dim, and
    re-encode as JPEG (q=82) for photographic Flux outputs. Massively cheaper
    than PNG for the .ts payload — full-resolution Flux PNGs blow past Fern's
    SSR module-size threshold (a 22MB combined .ts breaks server-render).

    Returns (b64, mime). On any failure, returns the original PNG bytes."""
    try:
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw))
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        if img.mode in ("RGBA", "LA"):
            # Flatten alpha onto white before JPEG encode
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82, optimize=True, progressive=True)
        return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"
    except Exception:
        return b64, "image/png"


def extract_outputs(outputs: list) -> list[dict]:
    result: list[dict] = []
    for out in outputs:
        out_type = out.get("output_type", "")
        if out_type == "stream":
            text = _join_source(out.get("text", []))
            if text.strip():
                result.append({"type": "text", "data": text.rstrip("\n"), "format": "plain"})
        elif out_type in ("display_data", "execute_result"):
            data = out.get("data", {})
            if "image/png" in data:
                b64 = data["image/png"]
                if isinstance(b64, list):
                    b64 = "".join(b64)
                b64, mime = shrink_image_b64(b64)
                result.append({"type": "image", "data": b64, "mime": mime})
            elif "text/html" in data:
                html = data["text/html"]
                if isinstance(html, list):
                    html = "".join(html)
                if html.strip():
                    result.append({"type": "text", "data": html, "format": "html"})
            elif "text/plain" in data:
                text = data["text/plain"]
                if isinstance(text, list):
                    text = "".join(text)
                if text.strip():
                    result.append({"type": "text", "data": text.rstrip("\n"), "format": "plain"})
    return result


def convert_cell(cell: dict, default_language: str) -> dict:
    cell_type = cell.get("cell_type", "code")
    source = _join_source(cell.get("source", [])).rstrip("\n")
    result: dict = {"type": cell_type, "source": source}
    if cell_type == "code":
        result["language"] = default_language
        source_html = highlight_code(source, default_language)
        if source_html:
            result["source_html"] = source_html
        raw_outputs = cell.get("outputs", [])
        if raw_outputs:
            result["outputs"] = extract_outputs(raw_outputs)
    elif cell_type == "markdown" and source:
        # Pre-render markdown to HTML at build time. NotebookViewer renders
        # this directly, side-stepping the JS-side markdown parser that
        # didn't handle blockquotes, fenced code, tables, or nested lists.
        result["source_html"] = _MD.render(source)
    return result


def convert_notebook(ipynb_path: Path) -> tuple[dict, int]:
    """Convert a .ipynb file to Fern format. Returns (data, n_skipped_colab_cells)."""
    with open(ipynb_path, encoding="utf-8") as f:
        nb = json.load(f)
    metadata = nb.get("metadata", {})
    default_language = get_language(metadata)
    raw_cells = nb.get("cells", [])
    skipped = 0
    cells = []
    for cell in raw_cells:
        if is_colab_badge_cell(cell):
            skipped += 1
            continue
        cells.append(convert_cell(cell, default_language))
    return {"cells": cells}, skipped


def write_ts_export(data: dict, ts_path: Path) -> None:
    """Write a .ts file that exports the notebook data inline (MDX imports the .ts, not the .json)."""
    cells_json = json.dumps(data["cells"], indent=2, ensure_ascii=False)
    ts_path.write_text(
        f"/** Auto-generated by ipynb-to-fern-json.py - do not edit */\n"
        f"export default {{ cells: {cells_json} }};\n",
        encoding="utf-8",
    )


def main() -> int:
    args = sys.argv[1:]
    if not args or "-h" in args or "--help" in args:
        print(__doc__)
        return 0
    input_path = Path(args[0])
    output_path: Path | None = None
    if "-o" in args:
        idx = args.index("-o")
        if idx + 1 < len(args):
            output_path = Path(args[idx + 1])
    if not output_path:
        output_path = input_path.with_suffix(".json")
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        return 1
    data, skipped = convert_notebook(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {output_path}")
    ts_path = output_path.with_suffix(".ts")
    write_ts_export(data, ts_path)
    print(f"Wrote {ts_path}")
    if skipped:
        print(f"  (skipped {skipped} colab-badge cell{'s' if skipped != 1 else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
