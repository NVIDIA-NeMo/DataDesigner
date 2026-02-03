# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile

import yaml


def test_data_designer_config_to_dict(stub_data_designer_config):
    assert isinstance(stub_data_designer_config.to_dict(), dict)


def test_data_designer_config_to_yaml(stub_data_designer_config):
    assert isinstance(stub_data_designer_config.to_yaml(), str)
    with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp_file:
        result = stub_data_designer_config.to_yaml(tmp_file.name)
        assert result is None
        with open(tmp_file.name, "r") as f:
            assert yaml.safe_load(f) == stub_data_designer_config.to_dict()


def test_data_designer_config_to_json(stub_data_designer_config):
    assert isinstance(stub_data_designer_config.to_json(), str)
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
        result = stub_data_designer_config.to_json(tmp_file.name)
        assert result is None
        with open(tmp_file.name, "r") as f:
            assert json.loads(f.read()) == stub_data_designer_config.to_dict()
