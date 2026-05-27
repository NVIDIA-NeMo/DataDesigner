# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SchedulingMetadataKind = Literal["local", "model", "custom_model"]


@dataclass(frozen=True)
class SchedulingMetadata:
    """Static generator-facing scheduling metadata.

    The metadata describes broad resource shape only. It intentionally does
    not expose ready-queue state, task-admission state, request-admission
    pressure, provider cooldowns, or adaptive request limits.
    """

    kind: SchedulingMetadataKind = "local"
    identity: tuple[str, ...] = ("local", "default")
    weight: int = 1
    diagnostics: dict[str, object] = field(default_factory=dict)

    @classmethod
    def local(cls, resource_name: str = "default", *, weight: int = 1) -> SchedulingMetadata:
        return cls(kind="local", identity=("local", resource_name), weight=weight)

    @classmethod
    def model(
        cls,
        provider_name: str,
        model_id: str,
        generation_kind: str,
        *,
        weight: int,
        diagnostics: dict[str, object] | None = None,
    ) -> SchedulingMetadata:
        return cls(
            kind="model",
            identity=("model", provider_name, model_id, generation_kind),
            weight=weight,
            diagnostics=diagnostics or {},
        )

    @classmethod
    def custom_model(
        cls,
        plugin_namespace: str,
        resource_name: str,
        version: str,
        *,
        weight: int = 1,
        diagnostics: dict[str, object] | None = None,
    ) -> SchedulingMetadata:
        return cls(
            kind="custom_model",
            identity=("custom_model", plugin_namespace, resource_name, version),
            weight=weight,
            diagnostics=diagnostics or {},
        )

    def __post_init__(self) -> None:
        if self.kind not in {"local", "model", "custom_model"}:
            raise SchedulingMetadataError(
                code="invalid_kind",
                message=f"Unknown scheduling metadata kind: {self.kind!r}",
                diagnostics={"kind": self.kind},
            )
        if not isinstance(self.identity, tuple) or not self.identity:
            raise SchedulingMetadataError(
                code="invalid_identity",
                message="Scheduling metadata identity must be a non-empty tuple of non-empty strings.",
                diagnostics={"identity": self.identity},
            )
        if any(not isinstance(part, str) or not part for part in self.identity):
            raise SchedulingMetadataError(
                code="invalid_identity",
                message="Scheduling metadata identity must contain only non-empty strings.",
                diagnostics={"identity": self.identity},
            )
        expected_identity_lengths = {"local": 2, "model": 4, "custom_model": 4}
        if self.identity[0] != self.kind or len(self.identity) != expected_identity_lengths[self.kind]:
            raise SchedulingMetadataError(
                code="invalid_identity",
                message=f"Scheduling metadata identity for kind {self.kind!r} has an invalid shape.",
                diagnostics={
                    "kind": self.kind,
                    "identity": self.identity,
                    "expected_prefix": self.kind,
                    "expected_length": expected_identity_lengths[self.kind],
                },
            )
        if isinstance(self.weight, bool) or not isinstance(self.weight, int) or self.weight <= 0:
            raise SchedulingMetadataError(
                code="invalid_weight",
                message="Scheduling metadata weight must be a positive integer.",
                diagnostics={"weight": self.weight},
            )


class SchedulingMetadataError(ValueError):
    """Typed scheduling metadata resolution error."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        fallback: SchedulingMetadata | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.fallback = fallback
        self.diagnostics = diagnostics or {}
