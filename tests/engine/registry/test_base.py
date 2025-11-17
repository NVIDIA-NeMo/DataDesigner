# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading

import pytest

from data_designer.config.base import ConfigBase
from data_designer.config.utils.type_helpers import StrEnum
from data_designer.engine.configurable_task import ConfigurableTask
from data_designer.engine.registry.base import TaskRegistry
from data_designer.engine.registry.errors import NotFoundInRegistryError, RegistryItemNotTypeError


@pytest.fixture
def stub_test_enum():
    class TestEnum(StrEnum):
        TASK_A = "task_a"
        TASK_B = "task_b"

    return TestEnum


@pytest.fixture
def stub_test_config_class():
    class TestConfig(ConfigBase):
        value: str = "default"

    return TestConfig


@pytest.fixture
def stub_test_task_class():
    class TestTask(ConfigurableTask):
        def _validate(self):
            pass

        def _initialize(self):
            pass

    return TestTask


@pytest.fixture
def stub_clean_registry():
    """Clean registry state for testing."""
    TaskRegistry._registry.clear()
    TaskRegistry._reverse_registry.clear()
    TaskRegistry._config_registry.clear()
    TaskRegistry._reverse_config_registry.clear()
    TaskRegistry._instance = None
    yield
    # Clean up after test
    TaskRegistry._registry.clear()
    TaskRegistry._reverse_registry.clear()
    TaskRegistry._config_registry.clear()
    TaskRegistry._reverse_config_registry.clear()
    TaskRegistry._instance = None


def test_registry_initialization(stub_clean_registry):
    assert TaskRegistry._registry == {}
    assert TaskRegistry._reverse_registry == {}
    assert TaskRegistry._config_registry == {}
    assert TaskRegistry._reverse_config_registry == {}
    assert TaskRegistry._instance is None


@pytest.mark.parametrize(
    "test_case,should_raise,expected_error",
    [
        ("register_task_success", False, None),
        ("register_task_collision_raises", True, ValueError),
        ("register_task_collision_no_raise", False, None),
    ],
)
def test_register_task_scenarios(
    stub_clean_registry,
    stub_test_enum,
    stub_test_task_class,
    stub_test_config_class,
    test_case,
    should_raise,
    expected_error,
):
    if test_case == "register_task_success":
        TaskRegistry.register(stub_test_enum.TASK_A, stub_test_task_class, stub_test_config_class)

        assert TaskRegistry._registry[stub_test_enum.TASK_A] == stub_test_task_class
        assert TaskRegistry._reverse_registry[stub_test_task_class] == stub_test_enum.TASK_A
        assert TaskRegistry._config_registry[stub_test_enum.TASK_A] == stub_test_config_class
        assert TaskRegistry._reverse_config_registry[stub_test_config_class] == stub_test_enum.TASK_A
    elif test_case == "register_task_collision_raises":
        TaskRegistry.register(stub_test_enum.TASK_A, stub_test_task_class, stub_test_config_class)

        with pytest.raises(expected_error, match="has already been registered!"):
            TaskRegistry.register(
                stub_test_enum.TASK_A, stub_test_task_class, stub_test_config_class, raise_on_collision=True
            )
    elif test_case == "register_task_collision_no_raise":
        TaskRegistry.register(stub_test_enum.TASK_A, stub_test_task_class, stub_test_config_class)
        # Default behavior is raise_on_collision=False, so no need to pass it explicitly
        TaskRegistry.register(stub_test_enum.TASK_A, stub_test_task_class, stub_test_config_class)


@pytest.mark.parametrize(
    "test_case,invalid_task,invalid_config,expected_error",
    [
        ("register_non_type_task_raises", "not_a_class", None, RegistryItemNotTypeError),
        ("register_non_type_config_raises", None, "not_a_class", RegistryItemNotTypeError),
    ],
)
def test_register_validation_errors(
    stub_clean_registry,
    stub_test_enum,
    stub_test_task_class,
    stub_test_config_class,
    test_case,
    invalid_task,
    invalid_config,
    expected_error,
):
    task_class = invalid_task if invalid_task else stub_test_task_class
    config_class = invalid_config if invalid_config else stub_test_config_class

    with pytest.raises(expected_error, match="is not a class!"):
        TaskRegistry.register(stub_test_enum.TASK_A, task_class, config_class)


@pytest.mark.parametrize(
    "test_case,method_name,expected_result,expected_error",
    [
        ("get_task_type_success", "get_task_type", "test_task_class", None),
        ("get_task_type_not_found", "get_task_type", None, NotFoundInRegistryError),
        ("get_config_type_success", "get_config_type", "test_config_class", None),
        ("get_config_type_not_found", "get_config_type", None, NotFoundInRegistryError),
        ("get_registered_name_success", "get_registered_name", "stub_test_enum.TASK_A", None),
        ("get_registered_name_not_found", "get_registered_name", None, NotFoundInRegistryError),
        ("get_for_config_type_success", "get_for_config_type", "test_task_class", None),
        ("get_for_config_type_not_found", "get_for_config_type", None, NotFoundInRegistryError),
    ],
)
def test_get_methods_scenarios(
    stub_clean_registry,
    stub_test_enum,
    stub_test_task_class,
    stub_test_config_class,
    test_case,
    method_name,
    expected_result,
    expected_error,
):
    if "success" in test_case:
        TaskRegistry.register(stub_test_enum.TASK_A, stub_test_task_class, stub_test_config_class)

        if method_name == "get_task_type":
            result = TaskRegistry.get_task_type(stub_test_enum.TASK_A)
            assert result == stub_test_task_class
        elif method_name == "get_config_type":
            result = TaskRegistry.get_config_type(stub_test_enum.TASK_A)
            assert result == stub_test_config_class
        elif method_name == "get_registered_name":
            result = TaskRegistry.get_registered_name(stub_test_task_class)
            assert result == stub_test_enum.TASK_A
        elif method_name == "get_for_config_type":
            result = TaskRegistry.get_for_config_type(stub_test_config_class)
            assert result == stub_test_task_class
    else:
        if method_name == "get_task_type":
            with pytest.raises(expected_error, match="not found in registry"):
                TaskRegistry.get_task_type(stub_test_enum.TASK_A)
        elif method_name == "get_config_type":
            with pytest.raises(expected_error, match="not found in registry"):
                TaskRegistry.get_config_type(stub_test_enum.TASK_A)
        elif method_name == "get_registered_name":
            with pytest.raises(expected_error, match="not found in registry"):
                TaskRegistry.get_registered_name(stub_test_task_class)
        elif method_name == "get_for_config_type":
            with pytest.raises(expected_error, match="not found in registry"):
                TaskRegistry.get_for_config_type(stub_test_config_class)


def test_singleton_behavior(stub_clean_registry):
    instance1 = TaskRegistry()
    instance2 = TaskRegistry()

    assert instance1 is instance2
    assert TaskRegistry._instance is instance1


def test_singleton_thread_safety(stub_clean_registry):
    instances = []

    def create_instance():
        instances.append(TaskRegistry())

    threads = [threading.Thread(target=create_instance) for _ in range(10)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # All instances should be the same
    assert all(instance is instances[0] for instance in instances)
    assert len(set(id(instance) for instance in instances)) == 1


def test_multiple_registrations(stub_clean_registry, stub_test_enum, stub_test_task_class, stub_test_config_class):
    class AnotherTask(ConfigurableTask):
        def _validate(self):
            pass

        def _initialize(self):
            pass

    class AnotherConfig(ConfigBase):
        another_value: str = "another"

    TaskRegistry.register(stub_test_enum.TASK_A, stub_test_task_class, stub_test_config_class)
    TaskRegistry.register(stub_test_enum.TASK_B, AnotherTask, AnotherConfig)

    assert len(TaskRegistry._registry) == 2
    assert len(TaskRegistry._config_registry) == 2
    assert TaskRegistry.get_task_type(stub_test_enum.TASK_A) == stub_test_task_class
    assert TaskRegistry.get_task_type(stub_test_enum.TASK_B) == AnotherTask
