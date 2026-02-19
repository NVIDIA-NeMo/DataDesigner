# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from pathlib import Path

from data_designer.cli.utils.sample_records_pager import PAGER_FILENAME, create_sample_records_pager


def test_create_sample_records_pager_creates_browser_file(tmp_path: Path) -> None:
    """Test that create_sample_records_pager writes sample_records_browser.html."""
    create_sample_records_pager(sample_records_dir=tmp_path, num_records=3)

    out = tmp_path / PAGER_FILENAME
    assert out.exists()
    content = out.read_text()
    assert "Sample Records Browser" in content
    assert "record_0.html" in content
    assert "record_1.html" in content
    assert "record_2.html" in content
    assert "record_3.html" not in content


def test_create_sample_records_pager_title_uses_parent_dir_name(tmp_path: Path) -> None:
    """Test that the pager HTML title includes the parent directory name."""
    sample_dir = tmp_path / "sample_records"
    sample_dir.mkdir()
    create_sample_records_pager(sample_records_dir=sample_dir, num_records=1)

    content = (sample_dir / PAGER_FILENAME).read_text()
    assert tmp_path.name in content


def test_create_sample_records_pager_zero_records(tmp_path: Path) -> None:
    """Test that no file is created when num_records=0."""
    create_sample_records_pager(sample_records_dir=tmp_path, num_records=0)

    out = tmp_path / PAGER_FILENAME
    assert not out.exists()


def test_create_sample_records_pager_single_record(tmp_path: Path) -> None:
    """Test correct HTML generation with a single record."""
    create_sample_records_pager(sample_records_dir=tmp_path, num_records=1)

    out = tmp_path / PAGER_FILENAME
    assert out.exists()
    content = out.read_text()
    assert "record_0.html" in content
    assert "record_1.html" not in content
    assert "Sample Records Browser" in content


def test_create_sample_records_pager_xss_safe_title(tmp_path: Path) -> None:
    """Test that HTML-special chars in the parent directory name are escaped in the title."""
    xss_dir = tmp_path / "a&b<c" / "sample_records"
    xss_dir.mkdir(parents=True)
    create_sample_records_pager(sample_records_dir=xss_dir, num_records=1)

    content = (xss_dir / PAGER_FILENAME).read_text()
    assert "a&amp;b&lt;c" in content
    assert "a&b<c" not in content


def test_create_sample_records_pager_valid_json_records(tmp_path: Path) -> None:
    """Test that the embedded JSON records array is valid and has the correct structure."""
    create_sample_records_pager(sample_records_dir=tmp_path, num_records=3)

    content = (tmp_path / PAGER_FILENAME).read_text()
    match = re.search(r"const records = (\[.*?\]);", content, re.DOTALL)
    assert match is not None, "Could not find records JSON in pager HTML"

    records = json.loads(match.group(1))
    assert len(records) == 3
    for i, record in enumerate(records):
        assert record == f"record_{i}.html"


def test_create_sample_records_pager_has_record_counter(tmp_path: Path) -> None:
    """Test that the pager contains a record counter element updated by the show() function."""
    create_sample_records_pager(sample_records_dir=tmp_path, num_records=5)

    content = (tmp_path / PAGER_FILENAME).read_text()
    assert 'id="counter"' in content
    assert "counter.textContent" in content


def test_create_sample_records_pager_dark_theme(tmp_path: Path) -> None:
    """Test that the default dark theme uses dark color-scheme."""
    create_sample_records_pager(sample_records_dir=tmp_path, num_records=1)

    content = (tmp_path / PAGER_FILENAME).read_text()
    assert "color-scheme: dark" in content


def test_create_sample_records_pager_light_theme(tmp_path: Path) -> None:
    """Test that the light theme uses light color-scheme."""
    create_sample_records_pager(sample_records_dir=tmp_path, num_records=1, theme="light")

    content = (tmp_path / PAGER_FILENAME).read_text()
    assert "color-scheme: light" in content
    assert "color-scheme: dark" not in content
