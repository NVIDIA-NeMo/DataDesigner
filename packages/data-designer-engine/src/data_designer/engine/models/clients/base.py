# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Protocol

from data_designer.engine.models.clients.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
)


class ChatCompletionClient(Protocol):
    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse: ...

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse: ...


class EmbeddingClient(Protocol):
    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse: ...

    async def aembeddings(self, request: EmbeddingRequest) -> EmbeddingResponse: ...


class ImageGenerationClient(Protocol):
    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse: ...

    async def agenerate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse: ...


class ModelClient(ChatCompletionClient, EmbeddingClient, ImageGenerationClient, Protocol):
    """Unified protocol for model provider adapters.

    The ``supports_*`` methods indicate whether this **adapter implementation**
    is capable of handling a given modality (i.e. it has the code paths and
    route mappings to make the call).  They do **not** reflect whether a
    specific model or endpoint actually supports that modality — that is a
    configuration concern handled by ``ModelConfig.generation_type``.
    """

    provider_name: str

    def supports_chat_completion(self) -> bool: ...

    def supports_embeddings(self) -> bool: ...

    def supports_image_generation(self) -> bool: ...

    def close(self) -> None: ...

    async def aclose(self) -> None: ...
