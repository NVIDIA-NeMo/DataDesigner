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


def test_create_sample_records_pager_records_list(tmp_path: Path) -> None:
    """Test that the pager contains correct record paths for iframe."""
    create_sample_records_pager(sample_records_dir=tmp_path, num_records=2)

    text = (tmp_path / PAGER_FILENAME).read_text()
    assert '"path": "record_0.html"' in text
    assert '"path": "record_1.html"' in text


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
        assert "path" in record
        assert record["path"] == f"record_{i}.html"


def test_create_sample_records_pager_has_theme_toggle(tmp_path: Path) -> None:
    """Test that the pager contains a theme toggle button with dynamic sun/moon icon."""
    create_sample_records_pager(sample_records_dir=tmp_path, num_records=1)

    content = (tmp_path / PAGER_FILENAME).read_text()
    assert 'id="theme-toggle"' in content
    assert "themeToggle.textContent" in content
    assert "\u2600\ufe0f" in content
    assert "\U0001f319" in content


def test_create_sample_records_pager_contains_both_themes(tmp_path: Path) -> None:
    """Test that the pager always contains both dark and light theme CSS as JS constants."""
    create_sample_records_pager(sample_records_dir=tmp_path, num_records=1)

    content = (tmp_path / PAGER_FILENAME).read_text()
    assert "DARK_THEME" in content
    assert "LIGHT_THEME" in content
    assert "color-scheme: dark" in content
    assert "color-scheme: light" in content


def test_create_sample_records_pager_dark_initial_theme(tmp_path: Path) -> None:
    """Test that the default theme initializes to dark."""
    create_sample_records_pager(sample_records_dir=tmp_path, num_records=1)

    content = (tmp_path / PAGER_FILENAME).read_text()
    assert '"dark" === "dark"' in content


def test_create_sample_records_pager_light_initial_theme(tmp_path: Path) -> None:
    """Test that theme='light' initializes to light."""
    create_sample_records_pager(sample_records_dir=tmp_path, num_records=1, theme="light")

    content = (tmp_path / PAGER_FILENAME).read_text()
    assert '"light" === "dark"' in content


def test_create_sample_records_pager_has_iframe_theme_logic(tmp_path: Path) -> None:
    """Test that the pager uses postMessage to apply theme to the iframe content."""
    create_sample_records_pager(sample_records_dir=tmp_path, num_records=1)

    content = (tmp_path / PAGER_FILENAME).read_text()
    assert "applyThemeToIframe" in content
    assert "postMessage" in content


def test_create_sample_records_pager_has_record_counter(tmp_path: Path) -> None:
    """Test that the pager contains a record counter element updated by the show() function."""
    create_sample_records_pager(sample_records_dir=tmp_path, num_records=5)

    content = (tmp_path / PAGER_FILENAME).read_text()
    assert 'id="counter"' in content
    assert "counter.textContent" in content


def test_create_sample_records_pager_no_unused_css_vars(tmp_path: Path) -> None:
    """Test that unused CSS variables are not present in the pager HTML."""
    create_sample_records_pager(sample_records_dir=tmp_path, num_records=1)

    content = (tmp_path / PAGER_FILENAME).read_text()
    for var in ["--bg-primary", "--bg-secondary", "--panel-strong", "--chip-text", "--chip-bg"]:
        assert var not in content, f"Unused CSS variable {var} should have been removed"
