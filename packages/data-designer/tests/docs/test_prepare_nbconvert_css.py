# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
from importlib.metadata import PathDistribution
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[4] / "docs" / "scripts" / "prepare_nbconvert_css.py"


@pytest.fixture
def css_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, Path, Path]:
    spec = importlib.util.spec_from_file_location("prepare_nbconvert_css", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    prefix = tmp_path / "venv"
    info = prefix / "lib" / "site-packages" / "nbconvert-7.17.2.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Name: nbconvert\nVersion: 7.17.2\n")
    record = info / "RECORD"
    record.write_text("nbconvert-7.17.2.dist-info/RECORD,,\n")
    installed = PathDistribution(info)
    monkeypatch.setattr(module, "distribution", lambda name: installed)
    monkeypatch.setattr(module.sys, "prefix", str(prefix))

    stylesheets = {name: f"/* {name} */".encode() for name in module.CSS_SHA256}
    payload = io.BytesIO()
    with ZipFile(payload, "w") as wheel:
        for name, content in stylesheets.items():
            wheel.writestr(f"{module.WHEEL_TEMPLATES}/{name}", content)
        wheel.writestr("nbconvert/__init__.py", "raise RuntimeError('must not install wheel code')")
    wheel_bytes = payload.getvalue()
    monkeypatch.setattr(module, "WHEEL_SHA256", hashlib.sha256(wheel_bytes).hexdigest())
    monkeypatch.setattr(
        module, "CSS_SHA256", {name: hashlib.sha256(data).hexdigest() for name, data in stylesheets.items()}
    )
    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: io.BytesIO(wheel_bytes))
    return module, prefix, record


def test_restores_only_css_and_tracks_assets_for_uninstall(css_environment: tuple[ModuleType, Path, Path]) -> None:
    module, prefix, record = css_environment

    module.prepare_nbconvert_css()

    templates = prefix / "share" / "jupyter" / "nbconvert" / "templates"
    assert {str(path.relative_to(templates)) for path in templates.rglob("*.css")} == module.CSS_SHA256.keys()
    assert not (record.parent.parent / "nbconvert").exists()
    with record.open(newline="") as stream:
        records = list(csv.reader(stream))
    assert len(records) == 5
    for name, digest, size in records[1:]:
        restored = record.parent.parent / name
        assert restored.is_file()
        assert digest.startswith("sha256=")
        assert len(restored.read_bytes()) == int(size)


def test_complete_installation_needs_no_network(
    css_environment: tuple[ModuleType, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _, record = css_environment
    module.prepare_nbconvert_css()
    previous = record.read_bytes()

    def offline(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not download existing assets")

    monkeypatch.setattr(module, "urlopen", offline)
    module.prepare_nbconvert_css()

    assert record.read_bytes() == previous


@pytest.mark.parametrize("corruption", ["wheel", "stylesheet", "missing-member"])
def test_rejects_invalid_assets_before_writing(
    css_environment: tuple[ModuleType, Path, Path], monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    module, prefix, record = css_environment
    previous = record.read_bytes()
    if corruption == "wheel":
        monkeypatch.setattr(module, "WHEEL_SHA256", "0" * 64)
    elif corruption == "stylesheet":
        monkeypatch.setitem(module.CSS_SHA256, "classic/static/style.css", "0" * 64)
    else:
        monkeypatch.setitem(module.CSS_SHA256, "classic/static/missing.css", "0" * 64)

    with pytest.raises((ValueError, KeyError), match="checksum mismatch|missing.css"):
        module.prepare_nbconvert_css()

    assert not (prefix / "share").exists()
    assert record.read_bytes() == previous


@pytest.mark.parametrize("corruption", ["missing", "modified"])
def test_repairs_incomplete_installation_without_duplicate_records(
    css_environment: tuple[ModuleType, Path, Path], corruption: str
) -> None:
    module, prefix, record = css_environment
    module.prepare_nbconvert_css()
    stylesheet = prefix / "share" / "jupyter" / "nbconvert" / "templates" / "classic" / "static" / "style.css"
    expected = stylesheet.read_bytes()
    if corruption == "missing":
        stylesheet.unlink()
    else:
        stylesheet.write_bytes(b"modified")

    module.prepare_nbconvert_css()

    assert stylesheet.read_bytes() == expected
    with record.open(newline="") as stream:
        assert len(list(csv.reader(stream))) == 5
