# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from data_designer.plugins import PluginType
from data_designer.plugins.registry import PluginRegistry
from slurm_test_fakes import FakeDependencyInstaller, FakeDependencyResolver


def test_fake_dependency_resolver_scripts_compatible_incompatible_and_missing_cases() -> None:
    compatible = ("fake-data-designer-plugin==1.0.0",)
    incompatible = ("fake-data-designer-plugin==2.0.0",)
    missing = ("missing-plugin==1.0.0",)
    resolver = FakeDependencyResolver(
        (
            (compatible, "resolved-lock"),
            (incompatible, ValueError("incompatible Data Designer version")),
            (missing, FileNotFoundError("dependency artifact is missing")),
        )
    )

    assert resolver.resolve(compatible) == "resolved-lock"
    with pytest.raises(ValueError, match="incompatible"):
        resolver.resolve(incompatible)
    with pytest.raises(FileNotFoundError, match="missing"):
        resolver.resolve(missing)

    assert resolver.calls == [compatible, incompatible, missing]
    resolver.assert_complete()


def test_fake_dependency_installer_scripts_success_and_digest_mismatch(tmp_path: Path) -> None:
    first_target = tmp_path / "compatible"
    second_target = tmp_path / "digest-mismatch"
    installer = FakeDependencyInstaller(
        (
            (("resolved-lock", first_target), ("fake-data-designer-plugin==1.0.0",)),
            (("mismatched-lock", second_target), ValueError("lock digest mismatch")),
        )
    )

    assert installer.install("resolved-lock", first_target) == ("fake-data-designer-plugin==1.0.0",)
    with pytest.raises(ValueError, match="digest mismatch"):
        installer.install("mismatched-lock", second_target)

    assert installer.calls == [("resolved-lock", first_target), ("mismatched-lock", second_target)]
    installer.assert_complete()


def test_dependency_fakes_raise_cancellation_signals(tmp_path: Path) -> None:
    resolver = FakeDependencyResolver((("cancelled", KeyboardInterrupt()),))
    installer = FakeDependencyInstaller(((("lock", tmp_path), SystemExit(2)),))

    with pytest.raises(KeyboardInterrupt):
        resolver.resolve("cancelled")
    with pytest.raises(SystemExit, match="2"):
        installer.install("lock", tmp_path)


def test_dependency_fakes_reject_unexpected_calls(tmp_path: Path) -> None:
    resolver = FakeDependencyResolver((("expected", "lock"),))
    installer = FakeDependencyInstaller(((("lock", tmp_path / "expected"), "installed"),))

    with pytest.raises(AssertionError, match="expected dependency request"):
        resolver.resolve("unexpected")
    with pytest.raises(AssertionError, match="expected dependency installation"):
        installer.install("lock", tmp_path / "unexpected")


def test_fake_plugin_overlay_supports_real_entry_point_discovery(
    fake_plugin_overlay: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(fake_plugin_overlay))
    importlib.invalidate_caches()
    monkeypatch.setattr("data_designer.plugins.registry.PLUGINS_DISABLED", False)
    PluginRegistry.reset()
    try:
        discovered = tuple(
            entry_point
            for entry_point in entry_points(group="data_designer.plugins")
            if entry_point.name == "fake-slurm-column"
        )
        assert len(discovered) == 1
        assert discovered[0].dist is not None
        assert discovered[0].dist.version == "1.0.0"

        plugin = discovered[0].load()
        registry = PluginRegistry()

        assert plugin.name == "fake-slurm-column"
        assert plugin.plugin_type is PluginType.COLUMN_GENERATOR
        assert registry.get_plugin("fake-slurm-column") == plugin
    finally:
        PluginRegistry.reset()
