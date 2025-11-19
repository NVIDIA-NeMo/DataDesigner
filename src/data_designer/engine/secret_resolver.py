# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
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
            raise SecretResolutionError(f"No secret found in secrets file with key {secret!r}")


class EnvironmentResolver(SecretResolver):
    def resolve(self, secret: str) -> str:
        try:
            return os.environ[secret]
        except KeyError:
            raise SecretResolutionError(
                f"Environment variable with name {secret!r} is required but not set. Please set it in your environment and try again."
            )


class PlaintextResolver(SecretResolver):
    def resolve(self, secret: str) -> str:
        return secret


class CompositeResolver(SecretResolver):
    _resolvers: Sequence[SecretResolver]

    def __init__(self, resolvers: Sequence[SecretResolver]):
        if len(resolvers) == 0:
            raise SecretResolutionError("Must provide at least one SecretResolver to CompositeResolver")
        self._resolvers = resolvers

    @property
    def resolvers(self) -> Sequence[SecretResolver]:
        """Get the sequence of resolvers in this composite resolver.

        Returns:
            Sequence of SecretResolver instances used to resolve secrets.
        """
        return self._resolvers

    def resolve(self, secret: str) -> str:
        errors = []
        for resolver in self._resolvers:
            try:
                return resolver.resolve(secret)
            except SecretResolutionError as err:
                errors.append(str(err))
                continue

        raise SecretResolutionError(
            f"No configured resolvers were able to resolve secret {secret!r}: {', '.join(errors)}"
        )
