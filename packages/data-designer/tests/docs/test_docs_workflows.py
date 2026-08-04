# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

WORKFLOWS_DIR = Path(__file__).resolve().parents[4] / ".github" / "workflows"


def test_notebook_cache_is_scoped_to_execution_profile() -> None:
    workflow = (WORKFLOWS_DIR / "build-notebooks.yml").read_text()

    assert (
        "DATA_DESIGNER_FLUX_2_PRO_CREATE_NUM_RECORDS: ${{ github.event_name == 'schedule' && '2' || '5' }}" in workflow
    )
    assert "NOTEBOOK_EXECUTION_PROFILE: ${{ github.event_name == 'schedule'" in workflow
    assert "NOTEBOOK_CACHE_CONTEXT=${NOTEBOOK_EXECUTION_PROFILE}:" in workflow
    assert workflow.count("'docs/scripts/build_notebooks_cached.sh'") == 2
    assert "key: notebooks-${{ env.NOTEBOOK_EXECUTION_PROFILE }}-" in workflow
    assert "notebooks-${{ env.NOTEBOOK_EXECUTION_PROFILE }}-\n" in workflow
    assert "gh run list --workflow build-fern-docs.yml --status success" in workflow


def test_notebook_cache_can_be_disabled() -> None:
    workflow = (WORKFLOWS_DIR / "build-notebooks.yml").read_text()

    assert "NOTEBOOK_CACHE_ENABLED: ${{ inputs.use_cache && '1' || '0' }}" in workflow
    assert 'if [ "$NOTEBOOK_CACHE_ENABLED" != "1" ]; then' in workflow
    assert "rm -rf .notebook-cache" in workflow


def test_fern_publish_excludes_cancelled_notebook_builds() -> None:
    workflow = (WORKFLOWS_DIR / "build-fern-docs.yml").read_text()

    assert "(needs.build-notebooks.result == 'success' || needs.build-notebooks.result == 'failure')" in workflow
    assert "if: needs.build-notebooks.result == 'failure'" in workflow


def test_fern_release_resolution_sets_repository() -> None:
    workflow = (WORKFLOWS_DIR / "build-fern-docs.yml").read_text()

    assert 'gh release list --repo "$GITHUB_REPOSITORY"' in workflow


def test_fern_publish_persists_and_restores_notebook_snapshots() -> None:
    workflow = (WORKFLOWS_DIR / "build-fern-docs.yml").read_text()

    assert "Publish source notebook snapshot" in workflow
    assert 'source-fallback "$archive"' in workflow
    assert "Restore prepared notebook snapshot" in workflow
    assert "Run the Build Fern docs workflow successfully once" in workflow
    assert 'gh release download "$release_tag"' in workflow
    assert "Publish executed notebook snapshot" in workflow
    assert 'executed "$archive"' in workflow
    assert "git add fern/notebook-snapshot.json" in workflow


def test_devnotes_publish_does_not_reuse_notebook_artifacts() -> None:
    workflow = (WORKFLOWS_DIR / "publish-fern-devnotes.yml").read_text()

    assert "Reuse notebooks from last successful docs build" not in workflow
    assert "gh run download" not in workflow
    assert "Require published notebook snapshot" in workflow
    assert "Run the Build Fern docs workflow successfully once" in workflow
    assert "Restore published notebook snapshot" in workflow
    assert 'gh release download "$release_tag"' in workflow
    assert "run: make -f ../workflow/Makefile check-fern-published-docs" in workflow
    assert "run: make check-fern-docs\n" not in workflow
