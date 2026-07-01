# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib

import pytest

import data_designer.engine.models.telemetry as telemetry_module
from data_designer.engine.models.telemetry import DeploymentTypeEnum


class TestDeploymentTypeEnvVar:
    def _reload(self) -> object:
        return importlib.reload(telemetry_module)

    def test_default_is_library(self, monkeypatch):
        monkeypatch.delenv("NEMO_DEPLOYMENT_TYPE", raising=False)
        assert self._reload().DEPLOYMENT_TYPE == DeploymentTypeEnum.LIBRARY

    def test_nvidia_internal(self, monkeypatch):
        monkeypatch.setenv("NEMO_DEPLOYMENT_TYPE", "nvidia-internal")
        assert self._reload().DEPLOYMENT_TYPE == DeploymentTypeEnum.NVIDIA_INTERNAL

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("NEMO_DEPLOYMENT_TYPE", "NVIDIA-INTERNAL")
        assert self._reload().DEPLOYMENT_TYPE == DeploymentTypeEnum.NVIDIA_INTERNAL

    def test_api(self, monkeypatch):
        monkeypatch.setenv("NEMO_DEPLOYMENT_TYPE", "api")
        assert self._reload().DEPLOYMENT_TYPE == DeploymentTypeEnum.API

    def test_invalid_raises(self, monkeypatch):
        monkeypatch.setenv("NEMO_DEPLOYMENT_TYPE", "definitely-not-real")
        with pytest.raises(ValueError, match="Invalid NEMO_DEPLOYMENT_TYPE"):
            self._reload()


class TestTelemetryEnabledEnvVar:
    def _reload(self) -> object:
        return importlib.reload(telemetry_module)

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "YES"])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", value)
        assert self._reload().TELEMETRY_ENABLED is True

    @pytest.mark.parametrize("value", ["false", "False", "FALSE", "0", "no", "NO", "off", "random"])
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", value)
        assert self._reload().TELEMETRY_ENABLED is False

    def test_default_is_enabled(self, monkeypatch):
        monkeypatch.delenv("NEMO_TELEMETRY_ENABLED", raising=False)
        assert self._reload().TELEMETRY_ENABLED is True


class TestTelemetryEndpointEnvVar:
    def _reload(self) -> object:
        return importlib.reload(telemetry_module)

    def test_endpoint_is_lowercased(self, monkeypatch):
        monkeypatch.setenv("NEMO_TELEMETRY_ENDPOINT", "HTTPS://EVENTS.EXAMPLE.COM/v1")
        mod = self._reload()
        assert mod.NEMO_TELEMETRY_ENDPOINT == "https://events.example.com/v1"

    def test_default_endpoint(self, monkeypatch):
        monkeypatch.delenv("NEMO_TELEMETRY_ENDPOINT", raising=False)
        mod = self._reload()
        assert "events.telemetry.data.nvidia.com" in mod.NEMO_TELEMETRY_ENDPOINT


def test_schema_version():
    from data_designer.engine.models.telemetry import TelemetryEvent

    assert TelemetryEvent._schema_version == "1.9"
