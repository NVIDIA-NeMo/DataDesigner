# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys

import pytest

import data_designer.engine.models.clients as model_clients
import data_designer.engine.models.clients.factory as model_client_factory
import data_designer.engine.models.errors as model_errors
import data_designer.engine.models.utils as model_utils


@pytest.mark.parametrize(
    "module_name",
    ["data_designer.engine.models.errors", "data_designer.engine.models.registry"],
    ids=["errors", "registry"],
)
def test_module_imports_first_in_fresh_process(module_name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Importing {module_name} in a fresh process failed:\n{result.stderr}"


def test_formatted_llm_error_message_import_paths_share_one_class() -> None:
    assert model_errors.FormattedLLMErrorMessage is model_utils.FormattedLLMErrorMessage


def test_clients_package_exports_create_model_client() -> None:
    assert "create_model_client" in model_clients.__all__
    assert model_clients.create_model_client is model_client_factory.create_model_client
