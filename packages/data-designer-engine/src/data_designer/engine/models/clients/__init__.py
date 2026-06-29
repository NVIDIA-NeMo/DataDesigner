# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
from data_designer.engine.models.clients.base import ModelClient
from data_designer.engine.models.clients.errors import (
    ProviderError,
    ProviderErrorKind,
    map_http_error_to_provider_error,
    map_http_status_to_provider_error_kind,
)
from data_designer.engine.models.clients.model_request_executor import ModelRequestExecutor
from data_designer.engine.models.clients.retry import RetryConfig
from data_designer.engine.models.clients.types import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    HttpResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImagePayload,
    ToolCall,
    Usage,
)

if TYPE_CHECKING:
    from data_designer.engine.models.clients.factory import create_model_client  # noqa: F401

__all__ = [
    "AssistantMessage",
    "ChatCompletionChoice",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "HttpResponse",
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "ImagePayload",
    "ModelClient",
    "ModelRequestExecutor",
    "OpenAICompatibleClient",
    "ProviderError",
    "ProviderErrorKind",
    "RetryConfig",
    "ToolCall",
    "Usage",
    "create_model_client",
    "map_http_error_to_provider_error",
    "map_http_status_to_provider_error_kind",
]


def __getattr__(name: str) -> object:
    if name == "create_model_client":
        module = importlib.import_module("data_designer.engine.models.clients.factory")
        attr = getattr(module, name)
        globals()[name] = attr
        return attr

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__
