# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File-based review session for the dataset review UI."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ReviewSession:
    """Loads a dataset from a parquet/JSON file and manages review annotations."""

    def __init__(self, data_file: Path) -> None:
        self._data_file = data_file.resolve()
        self._rows: list[dict[str, Any]] = []
        self._columns: list[str] = []
        self._version = 0
        self._annotations: dict[int, dict[str, Any]] = {}
        self._finished = False

        self._load_file()

    def _load_file(self) -> None:
        path = self._data_file
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            table = pq.read_table(str(path))
            df = table.to_pandas()
            # Convert through JSON to ensure all numpy/pyarrow types become native Python
            self._rows = json.loads(df.to_json(orient="records"))
            self._columns = list(df.columns)
        elif path.suffix == ".json":
            data = json.loads(path.read_text())
            if isinstance(data, list):
                self._rows = data
                self._columns = list(data[0].keys()) if data else []
            elif isinstance(data, dict) and "rows" in data:
                self._rows = data["rows"]
                self._columns = data.get("columns", list(self._rows[0].keys()) if self._rows else [])
            else:
                raise ValueError(f"Unrecognized JSON format in {path}")
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}. Use .parquet or .json")

        logger.info(f"Loaded {len(self._rows)} rows, {len(self._columns)} columns from {path.name}")

    def reload(self) -> None:
        """Re-read the data file from disk, clear annotations, increment version."""
        self._load_file()
        self._annotations = {}
        self._finished = False
        self._version += 1
        logger.info(f"Reloaded data (version {self._version})")

    # -- Data access --------------------------------------------------------

    @property
    def file_name(self) -> str:
        return self._data_file.name

    @property
    def file_path(self) -> str:
        return str(self._data_file)

    @property
    def version(self) -> int:
        return self._version

    def get_session_info(self) -> dict[str, Any]:
        visible_cols = [c for c in self._columns if not c.endswith("__trace") and not c.endswith("__reasoning_content")]
        return {
            "file_name": self.file_name,
            "file_path": self.file_path,
            "row_count": len(self._rows),
            "columns": visible_cols,
            "all_columns": self._columns,
            "version": self._version,
            "finished": self._finished,
        }

    def get_rows(self) -> list[dict[str, Any]]:
        return self._rows

    def get_trace(self, row: int, column: str) -> list[dict[str, Any]]:
        if row >= len(self._rows):
            return []
        trace_key = f"{column}__trace"
        trace = self._rows[row].get(trace_key)
        if isinstance(trace, list):
            return trace
        return []

    # -- Annotations --------------------------------------------------------

    def annotate_row(self, row: int, rating: str | None, note: str) -> None:
        """Set the overall row-level annotation."""
        ann = self._annotations.setdefault(row, {"rating": None, "note": "", "columns": {}})
        ann["rating"] = rating
        ann["note"] = note
        self._save_annotations()

    def annotate_column(self, row: int, column: str, rating: str | None, note: str) -> None:
        """Set a per-column annotation within a row."""
        ann = self._annotations.setdefault(row, {"rating": None, "note": "", "columns": {}})
        if rating is None and not note:
            ann["columns"].pop(column, None)
        else:
            ann["columns"][column] = {"rating": rating, "note": note}
        self._save_annotations()

    def get_annotations(self) -> dict[str, Any]:
        return {str(k): v for k, v in self._annotations.items()}

    def get_annotations_summary(self) -> dict[str, int]:
        total = len(self._rows)
        good = sum(1 for a in self._annotations.values() if a.get("rating") == "good")
        bad = sum(1 for a in self._annotations.values() if a.get("rating") == "bad")
        return {"good": good, "bad": bad, "unreviewed": total - good - bad, "total": total}

    def finish_review(self) -> dict[str, Any]:
        """Write the final rich annotations file and mark session complete."""
        self._finished = True
        self._save_annotations()
        summary = self.get_annotations_summary()
        return {"status": "finished", "summary": summary, "file": self._annotations_path()}

    def _annotations_path(self) -> str:
        stem = self._data_file.stem
        return str(self._data_file.parent / f"{stem}_annotations.json")

    def _save_annotations(self) -> None:
        """Write rich annotations JSON with row data context."""
        path = Path(self._annotations_path())
        visible_cols = [c for c in self._columns if not c.endswith("__trace") and not c.endswith("__reasoning_content")]

        annotations_list = []
        for row_idx, ann in sorted(self._annotations.items()):
            row_data = {}
            if row_idx < len(self._rows):
                row_data = {k: self._rows[row_idx].get(k) for k in visible_cols}
            annotations_list.append({
                "row": row_idx,
                "rating": ann.get("rating"),
                "note": ann.get("note", ""),
                "columns": ann.get("columns", {}),
                "data": row_data,
            })

        output = {
            "source_file": str(self._data_file),
            "summary": self.get_annotations_summary(),
            "annotations": annotations_list,
        }

        try:
            path.write_text(json.dumps(output, indent=2, default=str))
        except Exception as e:
            logger.warning(f"Failed to save annotations: {e}")
