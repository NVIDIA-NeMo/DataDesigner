# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

SCRIPT_PATH = Path(__file__).parents[1] / "serve-local-docs-preview.py"
SPEC = importlib.util.spec_from_file_location("serve_local_docs_preview", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
serve_local_docs_preview = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(serve_local_docs_preview)


@pytest.fixture
def docs_root(tmp_path: Path) -> Path:
    root = tmp_path / "fern"
    (root / "components").mkdir(parents=True)
    (root / "styles").mkdir()
    (root / "versions/latest/pages").mkdir(parents=True)
    (root / "docs.yml").write_text(
        yaml.safe_dump(
            {
                "global-theme": "nvidia",
                "title": "Data Designer",
                "css": "./styles/base.css",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / "components/Authors.tsx").write_text("export const Authors = 'initial';\n", encoding="utf-8")
    (root / "styles/base.css").write_text("body {}\n", encoding="utf-8")
    (root / "styles/local-preview.css").write_text(":root {}\n", encoding="utf-8")
    (root / "versions/latest/pages/index.mdx").write_text("# Initial\n", encoding="utf-8")
    return root


def test_build_preview_root_materializes_local_theme(docs_root: Path, tmp_path: Path) -> None:
    preview_root = tmp_path / "preview"
    preview_root.mkdir()

    serve_local_docs_preview.build_preview_root(docs_root, preview_root)

    config = yaml.safe_load((preview_root / "docs.yml").read_text(encoding="utf-8"))
    assert "global-theme" not in config
    assert config["colors"] == serve_local_docs_preview.LOCAL_THEME_CONFIG["colors"]
    assert config["css"] == ["./styles/base.css", "./styles/local-preview.css"]
    assert (preview_root / "versions/latest/pages/index.mdx").read_text(encoding="utf-8") == "# Initial\n"
    assert (preview_root / "components/Authors").read_text(encoding="utf-8") == ("export const Authors = 'initial';\n")
    assert all(not path.is_symlink() for path in preview_root.rglob("*"))
    assert all(path.resolve().is_relative_to(preview_root.resolve()) for path in preview_root.rglob("*"))


def test_build_preview_root_uses_first_component_extension(docs_root: Path, tmp_path: Path) -> None:
    preview_root = tmp_path / "preview"
    preview_root.mkdir()
    component = docs_root / "components/Authors.ts"
    component.write_text("export const Authors = 'ts';\n", encoding="utf-8")

    state = serve_local_docs_preview.build_preview_root(docs_root, preview_root)

    assert (preview_root / "components/Authors").read_text(encoding="utf-8") == "export const Authors = 'ts';\n"
    component.unlink()
    serve_local_docs_preview.sync_preview_root(docs_root, preview_root, state)
    assert (preview_root / "components/Authors").read_text(encoding="utf-8") == "export const Authors = 'initial';\n"


def test_sync_preview_root_updates_adds_and_removes_files(docs_root: Path, tmp_path: Path) -> None:
    preview_root = tmp_path / "preview"
    preview_root.mkdir()
    state = serve_local_docs_preview.build_preview_root(docs_root, preview_root)
    page = docs_root / "versions/latest/pages/index.mdx"
    replacement = page.with_suffix(".tmp")
    replacement.write_text("# Updated\n", encoding="utf-8")
    replacement.replace(page)
    added_page = docs_root / "versions/latest/pages/added.mdx"
    added_page.write_text("# Added\n", encoding="utf-8")
    (docs_root / "styles/base.css").unlink()

    serve_local_docs_preview.sync_preview_root(docs_root, preview_root, state)

    assert (preview_root / "versions/latest/pages/index.mdx").read_text(encoding="utf-8") == "# Updated\n"
    assert (preview_root / "versions/latest/pages/added.mdx").read_text(encoding="utf-8") == "# Added\n"
    assert not (preview_root / "styles/base.css").exists()


def test_snapshot_source_skips_unreadable_file(docs_root: Path) -> None:
    page = docs_root / "versions/latest/pages/index.mdx"
    original_stat = Path.stat

    def deny_page_stat(self: Path, **kwargs: object) -> object:
        if self == page:
            raise PermissionError
        return original_stat(self, **kwargs)

    with patch.object(Path, "stat", deny_page_stat):
        state = serve_local_docs_preview.snapshot_source(docs_root)

    assert page.relative_to(docs_root) not in state
    assert Path("docs.yml") in state


def test_sync_preview_root_retries_transient_copy_error(docs_root: Path, tmp_path: Path) -> None:
    preview_root = tmp_path / "preview"
    preview_root.mkdir()
    state = serve_local_docs_preview.build_preview_root(docs_root, preview_root)
    page = docs_root / "versions/latest/pages/added.mdx"
    page.write_text("# Added\n", encoding="utf-8")
    relative_path = page.relative_to(docs_root)

    with patch.object(serve_local_docs_preview, "copy_preview_file", side_effect=PermissionError):
        state = serve_local_docs_preview.sync_preview_root(docs_root, preview_root, state)

    assert relative_path not in state
    assert not (preview_root / relative_path).exists()
    state = serve_local_docs_preview.sync_preview_root(docs_root, preview_root, state)
    assert (preview_root / relative_path).read_text(encoding="utf-8") == "# Added\n"


def test_sync_preview_root_removes_file_deleted_during_copy(docs_root: Path, tmp_path: Path) -> None:
    preview_root = tmp_path / "preview"
    preview_root.mkdir()
    state = serve_local_docs_preview.build_preview_root(docs_root, preview_root)
    page = docs_root / "versions/latest/pages/index.mdx"
    page.write_text("# Updated\n", encoding="utf-8")
    relative_path = page.relative_to(docs_root)

    def delete_source(*args: object, **kwargs: object) -> None:
        page.unlink()
        raise FileNotFoundError

    with patch.object(serve_local_docs_preview, "copy_preview_file", side_effect=delete_source):
        state = serve_local_docs_preview.sync_preview_root(docs_root, preview_root, state)

    assert relative_path in state
    state = serve_local_docs_preview.sync_preview_root(docs_root, preview_root, state)
    assert relative_path not in state
    assert not (preview_root / relative_path).exists()


def test_sync_preview_root_regenerates_docs_config(docs_root: Path, tmp_path: Path) -> None:
    preview_root = tmp_path / "preview"
    preview_root.mkdir()
    state = serve_local_docs_preview.build_preview_root(docs_root, preview_root)
    config = yaml.safe_load((docs_root / "docs.yml").read_text(encoding="utf-8"))
    config["title"] = "Updated title"
    config["global-theme"] = "replacement-theme"
    (docs_root / "docs.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    serve_local_docs_preview.sync_preview_root(docs_root, preview_root, state)

    preview_config = yaml.safe_load((preview_root / "docs.yml").read_text(encoding="utf-8"))
    assert preview_config["title"] == "Updated title"
    assert "global-theme" not in preview_config
    assert preview_config["colors"] == serve_local_docs_preview.LOCAL_THEME_CONFIG["colors"]


@pytest.mark.parametrize("invalid_config", ["", "title: ["])
def test_sync_preview_root_retries_invalid_docs_config(
    docs_root: Path,
    tmp_path: Path,
    invalid_config: str,
) -> None:
    preview_root = tmp_path / "preview"
    preview_root.mkdir()
    state = serve_local_docs_preview.build_preview_root(docs_root, preview_root)
    previous_docs_state = state[serve_local_docs_preview.DOCS_CONFIG_PATH]
    docs_config = docs_root / "docs.yml"
    docs_config.write_text(invalid_config, encoding="utf-8")

    state = serve_local_docs_preview.sync_preview_root(docs_root, preview_root, state)

    assert state[serve_local_docs_preview.DOCS_CONFIG_PATH] == previous_docs_state
    assert yaml.safe_load((preview_root / "docs.yml").read_text(encoding="utf-8"))["title"] == "Data Designer"
    docs_config.write_text("title: Updated title\n", encoding="utf-8")
    state = serve_local_docs_preview.sync_preview_root(docs_root, preview_root, state)
    assert state[serve_local_docs_preview.DOCS_CONFIG_PATH] != previous_docs_state
    assert yaml.safe_load((preview_root / "docs.yml").read_text(encoding="utf-8"))["title"] == "Updated title"


def test_sync_preview_root_updates_and_removes_component_alias(docs_root: Path, tmp_path: Path) -> None:
    preview_root = tmp_path / "preview"
    preview_root.mkdir()
    state = serve_local_docs_preview.build_preview_root(docs_root, preview_root)
    component = docs_root / "components/Authors.tsx"
    component.write_text("export const Authors = 'updated';\n", encoding="utf-8")

    state = serve_local_docs_preview.sync_preview_root(docs_root, preview_root, state)

    assert (preview_root / "components/Authors.tsx").read_text(encoding="utf-8") == (
        "export const Authors = 'updated';\n"
    )
    assert (preview_root / "components/Authors").read_text(encoding="utf-8") == ("export const Authors = 'updated';\n")
    component.unlink()

    serve_local_docs_preview.sync_preview_root(docs_root, preview_root, state)

    assert not (preview_root / "components/Authors.tsx").exists()
    assert not (preview_root / "components/Authors").exists()


def test_run_command_synchronizes_until_child_exits(tmp_path: Path) -> None:
    process = MagicMock()
    process.wait.side_effect = [subprocess.TimeoutExpired(["fern"], 0.25), 0]
    initial_state = {Path("docs.yml"): (1, 1, 1)}
    next_state = {Path("docs.yml"): (2, 1, 1)}

    with (
        patch.object(serve_local_docs_preview.subprocess, "Popen", return_value=process),
        patch.object(serve_local_docs_preview, "sync_preview_root", return_value=next_state) as sync,
    ):
        result = serve_local_docs_preview.run_command(["fern"], tmp_path, tmp_path, initial_state)

    assert result == 0
    sync.assert_called_once_with(tmp_path, tmp_path, initial_state)


def test_run_command_stops_child_when_synchronization_fails(tmp_path: Path) -> None:
    process = MagicMock()
    process.wait.side_effect = [subprocess.TimeoutExpired(["fern"], 0.25), 0]

    with (
        patch.object(serve_local_docs_preview.subprocess, "Popen", return_value=process),
        patch.object(serve_local_docs_preview, "sync_preview_root", side_effect=OSError("sync failed")),
        pytest.raises(OSError, match="sync failed"),
    ):
        serve_local_docs_preview.run_command(["fern"], tmp_path, tmp_path, {})

    process.send_signal.assert_called_once_with(serve_local_docs_preview.signal.SIGINT)
