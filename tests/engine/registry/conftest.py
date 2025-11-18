# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.config.base import ConfigBase
from data_designer.config.utils.type_helpers import StrEnum
from data_designer.engine.configurable_task import ConfigurableTask


@pytest.fixture
def stub_config_class():
    class MockConfig(ConfigBase):
        value: str = "test"

    return MockConfig


@pytest.fixture
def stub_task_class():
    class MockTask(ConfigurableTask):
        def _validate(self):
            pass

        def _initialize(self):
            pass

    return MockTask


@pytest.fixture
def stub_enum():
    class MockEnum(StrEnum):
        TEST_TASK = "test_task"
        ANOTHER_TASK = "another_task"

    return MockEnum
