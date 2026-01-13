# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from pathlib import Path

import pytest

from data_designer.engine.errors import SecretResolutionError
from data_designer.engine.secret_resolver import CompositeResolver, EnvironmentResolver, SecretsFileResolver

TEST_SECRETS = {
    "FOO": "foo123",
    "BAR": "bar789",
}


@pytest.fixture
def stub_secrets_file():
    with tempfile.NamedTemporaryFile() as tmpfile:
        with open(tmpfile.name, "w") as f:
            json.dump(TEST_SECRETS, f)

        yield Path(tmpfile.name)


def test_secrets_file_resolution(stub_secrets_file: Path):
    resolver = SecretsFileResolver(stub_secrets_file)
    assert resolver.resolve("FOO") == "foo123"


def test_secrets_file_key_not_found(stub_secrets_file: Path):
    resolver = SecretsFileResolver(stub_secrets_file)

    with pytest.raises(SecretResolutionError):
        resolver.resolve("QUUX")


def test_secrets_file_doesnt_exist():
    # the resolver will instantiate...
    resolver = SecretsFileResolver(Path("/this/will/not/exist.json"))

    # ...but never find anything
    with pytest.raises(SecretResolutionError):
        resolver.resolve("FOO")


def test_malformed_file_fails_immediately():
    with tempfile.NamedTemporaryFile() as tmpfile:
        with open(tmpfile.name, "w") as f:
            f.write("hello world")

        with pytest.raises(json.decoder.JSONDecodeError):
            SecretsFileResolver(Path(tmpfile.name))


def test_env_resolver(monkeypatch):
    resolver = EnvironmentResolver()

    with pytest.raises(SecretResolutionError):
        resolver.resolve("FOO")

    monkeypatch.setenv("FOO", "foo123")
    assert resolver.resolve("FOO") == "foo123"


def test_composite_resolver(monkeypatch, stub_secrets_file: Path):
    resolvers = [EnvironmentResolver(), SecretsFileResolver(stub_secrets_file)]

    resolver = CompositeResolver(resolvers)

    monkeypatch.setenv("FOO", "foo000")

    assert resolver.resolve("FOO") == "foo000"
    assert resolver.resolve("BAR") == "bar789"


def test_composite_resolver_error(stub_secrets_file: Path):
    resolvers = [EnvironmentResolver(), SecretsFileResolver(stub_secrets_file)]

    resolver = CompositeResolver(resolvers)

    with pytest.raises(SecretResolutionError) as excinfo:
        resolver.resolve("QUUX")

    # The composite error message aggregates the individual resolvers' error messages
    assert "Environment variable" in str(excinfo.value)
    assert "secret" in str(excinfo.value)
