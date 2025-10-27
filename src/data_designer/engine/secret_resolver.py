# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from pathlib import Path
from typing import Protocol

from data_designer.engine.errors import SecretResolutionError

logger = logging.getLogger(__name__)


class SecretResolver(Protocol):
    def resolve(self, secret: str) -> str: ...


class SecretsFileResolver(SecretResolver):
    _secrets: dict[str, str]

    def __init__(self, filepath: Path):
        if not filepath.exists():
            self._secrets = {}
        else:
            with open(filepath) as f:
                self._secrets = json.load(f)

    def resolve(self, secret: str) -> str:
        try:
            return self._secrets[secret]
        except KeyError:
            raise SecretResolutionError(f"No secret found with key {secret!r}")


class EnvironmentResolver(SecretResolver):
    def resolve(self, secret: str) -> str:
        try:
            return os.environ[secret]
        except KeyError:
            raise SecretResolutionError(f"No env var found with name {secret!r}")


class PlaintextResolver(SecretResolver):
    def resolve(self, secret: str) -> str:
        return secret


class CompositeResolver(SecretResolver):
    _resolvers: list[SecretResolver]

    def __init__(self, resolvers: list[SecretResolver]):
        if len(resolvers) == 0:
            raise SecretResolutionError("Must provide at least one SecretResolver to CompositeResolver")
        self._resolvers = resolvers

    def resolve(self, secret: str) -> str:
        for resolver in self._resolvers:
            try:
                return resolver.resolve(secret)
            except SecretResolutionError:
                continue

        raise SecretResolutionError(f"No configured resolvers were able to resolve secret {secret!r}")
