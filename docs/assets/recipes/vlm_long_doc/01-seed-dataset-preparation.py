# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "pymupdf",
#     "pandas",
#     "pyarrow",
# ]
# ///
"""Long-Document Understanding Seed Dataset Preparation

This script uses HuggingFace's FinePDFs dataset (HuggingFaceFW/finepdfs) as
an example data source to demonstrate how to prepare seed data for the rest
of the recipes. It downloads the original PDFs, renders each page to a PNG
image, and produces three seed parquet files:

  1. **per-page seed** – one row per page, with a ``png_images_base64``
     column containing a JSON array with a single base64-encoded PNG of
     that page. Suitable for single-page recipes (02 through 06).
  2. **windowed seed** – one row per window of consecutive pages, with a
     ``png_images_base64`` column containing a JSON array of base64-encoded
     PNGs for the pages in that window. The window size adapts to document
     length (2 pages for short documents up to 8 for long ones).
     Suitable for the multi-page windowed recipe (07).
  3. **whole-document seed** – one row per document, with a
     ``png_images_base64`` column containing a JSON array of base64-encoded
     PNGs for all pages. Suitable for the whole-document recipe (08).

Prerequisites:
    - Internet access to download PDFs from their original URLs.

Run:
    # Prepare seeds from 10 English PDFs (default)
    uv run 01-seed-dataset-preparation.py --output-dir ./seed_data

    # Prepare seeds from 50 PDFs
    uv run 01-seed-dataset-preparation.py --output-dir ./seed_data --num-docs 50

    # Use a different language subset
    uv run 01-seed-dataset-preparation.py --output-dir ./seed_data --subset fra_Latn

    # Skip documents that fail to download (default behaviour) and set
    # a custom timeout
    uv run 01-seed-dataset-preparation.py --output-dir ./seed_data --timeout 30

    # For help
    uv run 01-seed-dataset-preparation.py --help
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.request
from argparse import ArgumentParser
from pathlib import Path

import fitz  # pymupdf
import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)

DPI = 144
FINEPDFS_REPO = "HuggingFaceFW/finepdfs"


def download_pdf(url: str, timeout: int = 20) -> bytes | None:
    """Download a PDF from *url*, returning raw bytes or None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:
        log.warning("Failed to download %s: %s", url, exc)
        return None


def render_pages(pdf_bytes: bytes, dpi: int = DPI) -> list[bytes]:
    """Render every page of *pdf_bytes* to PNG, returning a list of raw PNG bytes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[bytes] = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        pages.append(pix.tobytes("png"))
    doc.close()
    return pages


def png_to_base64(png_bytes: bytes) -> str:
    """Encode raw PNG bytes as a base64 string."""
    return base64.b64encode(png_bytes).decode("ascii")


def adaptive_window_size(n_pages: int) -> int:
    """Choose a window size that scales with document length.

    Short documents get small windows (2 pages) so multi-page questions
    remain feasible; longer documents get larger windows (up to 8) to
    cover more context per seed row.
    """
    if n_pages > 10 and n_pages < 20:
        return 3
    elif n_pages > 20 and n_pages < 30:
        return 4
    elif n_pages > 30 and n_pages < 40:
        return 5
    elif n_pages > 40 and n_pages < 50:
        return 6
    elif n_pages > 50 and n_pages < 60:
        return 7
    elif n_pages > 60:
        return 8
    return 2


def main() -> None:
    parser = ArgumentParser(description="Prepare seed parquets from FinePDFs")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output parquet files",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=10,
        help="Number of PDF documents to process (default: 10)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="eng_Latn",
        help="FinePDFs language subset (default: eng_Latn)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP download timeout in seconds (default: 20)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DPI,
        help=f"Render resolution in DPI (default: {DPI})",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Skip documents with more pages than this (default: 50)",
    )
    parser.add_argument(
        "--min-window-pages",
        type=int,
        default=2,
        help="Minimum pages in a window; documents shorter than this are skipped for windowed output (default: 2)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Streaming %d documents from %s (subset=%s)",
        args.num_docs,
        FINEPDFS_REPO,
        args.subset,
    )

    ds = load_dataset(
        FINEPDFS_REPO,
        name=args.subset,
        split="train",
        streaming=True,
    )

    per_page_rows: list[dict] = []
    windowed_rows: list[dict] = []
    whole_doc_rows: list[dict] = []

    docs_processed = 0
    for row in ds:
        if docs_processed >= args.num_docs:
            break

        doc_id = row.get("id", f"doc_{docs_processed:06d}")
        url = row["url"]
        date = row.get("date", "")

        pdf_bytes = download_pdf(url, timeout=args.timeout)
        if pdf_bytes is None:
            continue

        try:
            page_pngs = render_pages(pdf_bytes, dpi=args.dpi)
        except Exception as exc:
            log.warning("Failed to render %s: %s", url, exc)
            continue

        if len(page_pngs) == 0:
            log.warning("No pages rendered for %s, skipping", url)
            continue

        if len(page_pngs) > args.max_pages:
            log.info(
                "Skipping %s (%d pages > --max-pages %d)",
                url,
                len(page_pngs),
                args.max_pages,
            )
            continue

        page_b64s: list[str] = []

        for page_idx, png_bytes in enumerate(page_pngs):
            b64 = png_to_base64(png_bytes)
            page_b64s.append(b64)

            per_page_rows.append(
                {
                    "id": doc_id,
                    "url": url,
                    "date": date,
                    "page_number": page_idx,
                    "total_pages": len(page_pngs),
                    "png_images_base64": json.dumps([b64]),
                }
            )

        whole_doc_rows.append(
            {
                "id": doc_id,
                "url": url,
                "date": date,
                "total_pages": len(page_pngs),
                "png_images_base64": json.dumps(page_b64s),
            }
        )

        n_pages = len(page_b64s)
        win_size = adaptive_window_size(n_pages)
        n_windows = n_pages // win_size
        for i in range(n_windows):
            win_start = i * win_size
            win_end = win_start + win_size
            if win_end - win_start < args.min_window_pages:
                break
            windowed_rows.append(
                {
                    "id": doc_id,
                    "url": url,
                    "date": date,
                    "total_pages": n_pages,
                    "start_page": win_start,
                    "end_page": win_end,
                    "window_size": win_end - win_start,
                    "png_images_base64": json.dumps(page_b64s[win_start:win_end]),
                }
            )

        docs_processed += 1
        log.info(
            "[%d/%d] %s — %d pages",
            docs_processed,
            args.num_docs,
            url,
            len(page_pngs),
        )

    if not per_page_rows:
        log.error("No documents were successfully processed. Exiting.")
        return

    per_page_path = output_dir / "seed_per_page.parquet"
    windowed_path = output_dir / "seed_windowed.parquet"
    whole_doc_path = output_dir / "seed_whole_document.parquet"

    pd.DataFrame(per_page_rows).to_parquet(per_page_path, index=False)
    if windowed_rows:
        pd.DataFrame(windowed_rows).to_parquet(windowed_path, index=False)
    pd.DataFrame(whole_doc_rows).to_parquet(whole_doc_path, index=False)

    log.info("Per-page seed:       %s (%d rows)", per_page_path, len(per_page_rows))
    log.info("Windowed seed:       %s (%d rows)", windowed_path, len(windowed_rows))
    log.info("Whole-document seed: %s (%d rows)", whole_doc_path, len(whole_doc_rows))


if __name__ == "__main__":
    main()
    # Force-exit to avoid hanging on background threads from datasets/fsspec.
    os._exit(0)
