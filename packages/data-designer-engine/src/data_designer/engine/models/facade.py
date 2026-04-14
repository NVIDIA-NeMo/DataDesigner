# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from data_designer.config.models import GenerationType, ModelConfig, ModelProvider
from data_designer.config.utils.constants import (
    ATTRIBUTION_TITLE,
    OPENROUTER_ATTRIBUTION_HEADERS,
    OPENROUTER_PROVIDER_NAME,
)
from data_designer.config.utils.image_helpers import is_image_diffusion_model
from data_designer.engine.mcp.errors import MCPConfigurationError
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.clients.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    Usage,
)
from data_designer.engine.models.errors import (
    GenerationValidationFailureError,
    ImageGenerationError,
    acatch_llm_exceptions,
    catch_llm_exceptions,
    get_exception_primary_cause,
)
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.telemetry import TELEMETRY_ENABLED
from data_designer.engine.models.usage import ImageUsageStats, ModelUsageStats, RequestUsageStats, TokenUsageStats
from data_designer.engine.models.utils import ChatMessage, prompt_to_messages

if TYPE_CHECKING:
    from data_designer.engine.mcp.facade import MCPFacade
    from data_designer.engine.mcp.registry import MCPRegistry
    from data_designer.engine.models.clients.base import ModelClient


def _identity(x: Any) -> Any:
    """Identity function for default parser. Module-level for pickling compatibility."""
    return x


logger = logging.getLogger(__name__)


def _classify_generation_failure_kind(exc: ParserException) -> str:
    detail = " ".join(str(get_exception_primary_cause(exc)).split()).lower()
    if "response_schema" in detail or "model_validate" in detail:
        return "schema_validation"
    if "validation error" in detail or "doesn't match requested" in detail:
        return "schema_validation"
    return "parse_error"


def _build_generation_validation_error(summary: str, exc: ParserException) -> GenerationValidationFailureError:
    return GenerationValidationFailureError(
        summary,
        detail=str(get_exception_primary_cause(exc)),
        failure_kind=_classify_generation_failure_kind(exc),
    )


# Known keyword arguments extracted into request fields for each modality.
# Note: `extra_body` and `extra_headers` appear in every set but receive special
# treatment in `consolidate_kwargs` (merged with provider-level overrides) and in
# `TransportKwargs` (extra_body is either flattened into the request body or
# preserved as a nested dict depending on the adapter; extra_headers are
# forwarded as HTTP headers).  They are NOT regular model parameters.
_COMPLETION_REQUEST_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "seed",
        "response_format",
        "frequency_penalty",
        "presence_penalty",
        "timeout",
        "tools",
        "extra_body",
        "extra_headers",
    }
)

_EMBEDDING_REQUEST_FIELDS = frozenset(
    {
        "encoding_format",
        "dimensions",
        "timeout",
        "extra_body",
        "extra_headers",
    }
)

_IMAGE_GENERATION_REQUEST_FIELDS = frozenset(
    {
        "timeout",
        "extra_body",
        "extra_headers",
    }
)


class ModelFacade:
    def __init__(
        self,
        model_config: ModelConfig,
        model_provider_registry: ModelProviderRegistry,
        *,
        client: ModelClient,
        mcp_registry: MCPRegistry | None = None,
    ) -> None:
        self._model_config = model_config
        self._model_provider_registry = model_provider_registry
        self._client = client
        self._mcp_registry = mcp_registry
        self._usage_stats = ModelUsageStats()

    @property
    def model_name(self) -> str:
        return self._model_config.model

    @property
    def model_provider(self) -> ModelProvider:
        return self._model_provider_registry.get_provider(self._model_config.provider)

    @property
    def model_generation_type(self) -> GenerationType:
        return self._model_config.generation_type

    @property
    def model_provider_name(self) -> str:
        return self.model_provider.name

    @property
    def model_alias(self) -> str:
        return self._model_config.alias

    @property
    def max_parallel_requests(self) -> int:
        return self._model_config.inference_parameters.max_parallel_requests

    @property
    def usage_stats(self) -> ModelUsageStats:
        return self._usage_stats

    def consolidate_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        # Remove purpose from kwargs to avoid passing it to the model
        kwargs.pop("purpose", None)
        kwargs = {**self._model_config.inference_parameters.generate_kwargs, **kwargs}
        if self.model_provider.extra_body:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **self.model_provider.extra_body}
        if self.model_provider.extra_headers:
            kwargs["extra_headers"] = {**(kwargs.get("extra_headers") or {}), **self.model_provider.extra_headers}
        # Inject framework-level attribution header when telemetry is enabled.
        # Applied last so that user-supplied or provider-level headers take precedence.
        if TELEMETRY_ENABLED:
            headers = kwargs.get("extra_headers") or {}
            if "X-Title" not in headers:
                kwargs["extra_headers"] = {"X-Title": ATTRIBUTION_TITLE, **headers}
            # Inject OpenRouter-specific attribution headers when the provider is
            # OpenRouter.  This ensures attribution works even when existing users
            # have ``extra_headers: null`` in their provider config.  Provider- or
            # user-supplied values take precedence (only missing keys are filled).
            if self.model_provider.name == OPENROUTER_PROVIDER_NAME:
                headers = kwargs.get("extra_headers") or {}
                merged = {**OPENROUTER_ATTRIBUTION_HEADERS, **headers}
                kwargs["extra_headers"] = merged
        return kwargs

    # --- completion / acompletion ---

    def completion(
        self, messages: list[ChatMessage], skip_usage_tracking: bool = False, **kwargs: Any
    ) -> ChatCompletionResponse:
        message_payloads = [message.to_dict() for message in messages]
        logger.debug(
            f"Prompting model {self.model_name!r}...",
            extra={"model": self.model_name, "messages": message_payloads},
        )
        response = None
        kwargs = self.consolidate_kwargs(**kwargs)
        try:
            request = self._build_chat_completion_request(message_payloads, kwargs)
            response = self._client.completion(request)
            logger.debug(
                f"Received completion from model {self.model_name!r}",
                extra={
                    "model": self.model_name,
                    "response": response,
                    "text": response.message.content,
                    "usage": self._usage_stats.model_dump(),
                },
            )
            return response
        finally:
            if not skip_usage_tracking:
                self._track_usage(
                    response.usage if response is not None else None,
                    is_request_successful=response is not None,
                )

    async def acompletion(
        self, messages: list[ChatMessage], skip_usage_tracking: bool = False, **kwargs: Any
    ) -> ChatCompletionResponse:
        message_payloads = [message.to_dict() for message in messages]
        logger.debug(
            f"Prompting model {self.model_name!r}...",
            extra={"model": self.model_name, "messages": message_payloads},
        )
        response = None
        kwargs = self.consolidate_kwargs(**kwargs)
        try:
            request = self._build_chat_completion_request(message_payloads, kwargs)
            response = await self._client.acompletion(request)
            logger.debug(
                f"Received completion from model {self.model_name!r}",
                extra={
                    "model": self.model_name,
                    "response": response,
                    "text": response.message.content,
                    "usage": self._usage_stats.model_dump(),
                },
            )
            return response
        finally:
            if not skip_usage_tracking:
                self._track_usage(
                    response.usage if response is not None else None,
                    is_request_successful=response is not None,
                )

    # --- generate / agenerate ---

    @catch_llm_exceptions
    def generate(
        self,
        prompt: str,
        *,
        parser: Callable[[str], Any] = _identity,
        system_prompt: str | None = None,
        multi_modal_context: list[dict[str, Any]] | None = None,
        tool_alias: str | None = None,
        max_correction_steps: int = 0,
        max_conversation_restarts: int = 0,
        skip_usage_tracking: bool = False,
        purpose: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, list[ChatMessage]]:
        """Generate a parsed output with correction steps.

        This generation call will attempt to generate an output which is
        valid according to the specified parser, where "valid" implies
        that the parser can process the LLM response without raising
        an exception.

        `ParserExceptions` are routed back
        to the LLM as new rounds in the conversation, where the LLM is provided its
        earlier response along with the "user" role responding with the exception string
        (not traceback). This will continue for the number of rounds specified by
        `max_correction_steps`.

        Args:
            prompt (str): Task prompt.
            system_prompt (str, optional): Optional system instructions. If not specified,
                no system message is provided and the model should use its default system
                prompt.
            parser (func(str) -> Any): A function applied to the LLM response which processes
                an LLM response into some output object. Default: identity function.
            tool_alias (str | None): Optional tool configuration alias. When provided,
                the model may call permitted tools from the configured MCP providers.
                The alias must reference a ToolConfig registered in the MCPRegistry.
            max_correction_steps (int): Maximum number of correction rounds permitted
                within a single conversation. Note, many rounds can lead to increasing
                context size without necessarily improving performance -- small language
                models can enter repeated cycles which will not be solved with more steps.
                Default: `0` (no correction).
            max_conversation_restarts (int): Maximum number of full conversation restarts permitted
                if generation fails.  Default: `0` (no restarts).
            skip_usage_tracking (bool): Whether to skip usage tracking. Default: `False`.
            purpose (str): The purpose of the model usage to show as context in the error message.
                It is expected to be used by the @catch_llm_exceptions decorator.
            **kwargs: Additional arguments to pass to the model.

        Returns:
            A tuple containing:
                - The parsed output object from the parser.
                - The full trace of ChatMessage entries in the conversation, including any tool calls,
                  corrections, and reasoning traces. Callers can decide whether to store this.

        Raises:
            GenerationValidationFailureError: If the maximum number of retries or
                correction steps are met and the last response failures on
                generation validation.
            MCPConfigurationError: If tool_alias is specified but no MCPRegistry is configured.
        """
        output_obj = None
        tool_schemas = None
        tool_call_turns = 0
        total_tool_calls = 0
        curr_num_correction_steps = 0
        curr_num_restarts = 0

        mcp_facade = self._get_mcp_facade(tool_alias)

        # Checkpoint for restarts - updated after tool calls so we don't repeat them
        restart_checkpoint = prompt_to_messages(
            user_prompt=prompt, system_prompt=system_prompt, multi_modal_context=multi_modal_context
        )
        checkpoint_tool_call_turns = 0
        messages: list[ChatMessage] = deepcopy(restart_checkpoint)

        if mcp_facade is not None:
            tool_schemas = mcp_facade.get_tool_schemas()

        while True:
            completion_kwargs = dict(kwargs)
            if tool_schemas is not None:
                completion_kwargs["tools"] = tool_schemas

            completion_response = self.completion(
                messages,
                skip_usage_tracking=skip_usage_tracking,
                **completion_kwargs,
            )

            # Process any tool calls in the response (handles parallel tool calling)
            if mcp_facade is not None and mcp_facade.has_tool_calls(completion_response):
                tool_call_turns += 1
                total_tool_calls += mcp_facade.get_tool_call_count(completion_response)

                if tool_call_turns > mcp_facade.max_tool_call_turns:
                    # Gracefully refuse tool calls when budget is exhausted
                    messages.extend(mcp_facade.refuse_completion_response(completion_response))
                else:
                    messages.extend(mcp_facade.process_completion_response(completion_response))

                # Update checkpoint so restarts don't repeat tool calls
                restart_checkpoint = deepcopy(messages)
                checkpoint_tool_call_turns = tool_call_turns

                continue  # Back to top

            # No tool calls remaining to process
            response = (completion_response.message.content or "").strip()
            reasoning_trace = completion_response.message.reasoning_content
            messages.append(ChatMessage.as_assistant(content=response, reasoning_content=reasoning_trace or None))
            curr_num_correction_steps += 1

            try:
                output_obj = parser(response)  # type: ignore - if not a string will cause a ParserException below
                break
            except ParserException as exc:
                if max_correction_steps == 0 and max_conversation_restarts == 0:
                    raise _build_generation_validation_error(
                        "Unsuccessful generation attempt. No retries were attempted.",
                        exc,
                    ) from exc

                if curr_num_correction_steps <= max_correction_steps:
                    # Add user message with error for correction
                    messages.append(ChatMessage.as_user(content=str(get_exception_primary_cause(exc))))

                elif curr_num_restarts < max_conversation_restarts:
                    curr_num_correction_steps = 0
                    curr_num_restarts += 1
                    messages = deepcopy(restart_checkpoint)
                    tool_call_turns = checkpoint_tool_call_turns

                else:
                    raise _build_generation_validation_error(
                        (
                            f"Unsuccessful generation despite {max_correction_steps} correction steps "
                            f"and {max_conversation_restarts} conversation restarts."
                        ),
                        exc,
                    ) from exc

        if not skip_usage_tracking and mcp_facade is not None:
            self._usage_stats.tool_usage.extend(
                tool_calls=total_tool_calls,
                tool_call_turns=tool_call_turns,
            )

        return output_obj, messages

    @acatch_llm_exceptions
    async def agenerate(
        self,
        prompt: str,
        *,
        parser: Callable[[str], Any] = _identity,
        system_prompt: str | None = None,
        multi_modal_context: list[dict[str, Any]] | None = None,
        tool_alias: str | None = None,
        max_correction_steps: int = 0,
        max_conversation_restarts: int = 0,
        skip_usage_tracking: bool = False,
        purpose: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, list[ChatMessage]]:
        output_obj = None
        tool_schemas = None
        tool_call_turns = 0
        total_tool_calls = 0
        curr_num_correction_steps = 0
        curr_num_restarts = 0

        mcp_facade = self._get_mcp_facade(tool_alias)

        restart_checkpoint = prompt_to_messages(
            user_prompt=prompt, system_prompt=system_prompt, multi_modal_context=multi_modal_context
        )
        checkpoint_tool_call_turns = 0
        messages: list[ChatMessage] = deepcopy(restart_checkpoint)

        if mcp_facade is not None:
            tool_schemas = await asyncio.to_thread(mcp_facade.get_tool_schemas)

        while True:
            completion_kwargs = dict(kwargs)
            if tool_schemas is not None:
                completion_kwargs["tools"] = tool_schemas

            completion_response = await self.acompletion(
                messages,
                skip_usage_tracking=skip_usage_tracking,
                **completion_kwargs,
            )

            if mcp_facade is not None and mcp_facade.has_tool_calls(completion_response):
                tool_call_turns += 1
                total_tool_calls += mcp_facade.get_tool_call_count(completion_response)

                if tool_call_turns > mcp_facade.max_tool_call_turns:
                    messages.extend(mcp_facade.refuse_completion_response(completion_response))
                else:
                    messages.extend(
                        await asyncio.to_thread(mcp_facade.process_completion_response, completion_response)
                    )

                restart_checkpoint = deepcopy(messages)
                checkpoint_tool_call_turns = tool_call_turns

                continue

            response = (completion_response.message.content or "").strip()
            reasoning_trace = completion_response.message.reasoning_content
            messages.append(ChatMessage.as_assistant(content=response, reasoning_content=reasoning_trace or None))
            curr_num_correction_steps += 1

            try:
                output_obj = parser(response)
                break
            except ParserException as exc:
                if max_correction_steps == 0 and max_conversation_restarts == 0:
                    raise _build_generation_validation_error(
                        "Unsuccessful generation attempt. No retries were attempted.",
                        exc,
                    ) from exc

                if curr_num_correction_steps <= max_correction_steps:
                    messages.append(ChatMessage.as_user(content=str(get_exception_primary_cause(exc))))

                elif curr_num_restarts < max_conversation_restarts:
                    curr_num_correction_steps = 0
                    curr_num_restarts += 1
                    messages = deepcopy(restart_checkpoint)
                    tool_call_turns = checkpoint_tool_call_turns

                else:
                    raise _build_generation_validation_error(
                        (
                            f"Unsuccessful generation despite {max_correction_steps} correction steps "
                            f"and {max_conversation_restarts} conversation restarts."
                        ),
                        exc,
                    ) from exc

        if not skip_usage_tracking and mcp_facade is not None:
            self._usage_stats.tool_usage.extend(
                tool_calls=total_tool_calls,
                tool_call_turns=tool_call_turns,
            )

        return output_obj, messages

    # --- generate_text_embeddings / agenerate_text_embeddings ---

    @catch_llm_exceptions
    def generate_text_embeddings(
        self, input_texts: list[str], skip_usage_tracking: bool = False, **kwargs: Any
    ) -> list[list[float]]:
        logger.debug(
            f"Generating embeddings with model {self.model_name!r}...",
            extra={
                "model": self.model_name,
                "input_count": len(input_texts),
            },
        )
        kwargs = self.consolidate_kwargs(**kwargs)
        response: EmbeddingResponse | None = None
        try:
            request = self._build_embedding_request(input_texts, kwargs)
            response = self._client.embeddings(request)
            logger.debug(
                f"Received embeddings from model {self.model_name!r}",
                extra={
                    "model": self.model_name,
                    "embedding_count": len(response.vectors),
                    "usage": self._usage_stats.model_dump(),
                },
            )
            if len(response.vectors) == len(input_texts):
                return response.vectors
            raise ValueError(f"Expected {len(input_texts)} embeddings, but received {len(response.vectors)}")
        finally:
            if not skip_usage_tracking:
                self._track_usage(
                    response.usage if response is not None else None,
                    is_request_successful=response is not None,
                )

    @acatch_llm_exceptions
    async def agenerate_text_embeddings(
        self, input_texts: list[str], skip_usage_tracking: bool = False, **kwargs: Any
    ) -> list[list[float]]:
        logger.debug(
            f"Generating embeddings with model {self.model_name!r}...",
            extra={
                "model": self.model_name,
                "input_count": len(input_texts),
            },
        )
        kwargs = self.consolidate_kwargs(**kwargs)
        response: EmbeddingResponse | None = None
        try:
            request = self._build_embedding_request(input_texts, kwargs)
            response = await self._client.aembeddings(request)
            logger.debug(
                f"Received embeddings from model {self.model_name!r}",
                extra={
                    "model": self.model_name,
                    "embedding_count": len(response.vectors),
                    "usage": self._usage_stats.model_dump(),
                },
            )
            if len(response.vectors) == len(input_texts):
                return response.vectors
            raise ValueError(f"Expected {len(input_texts)} embeddings, but received {len(response.vectors)}")
        finally:
            if not skip_usage_tracking:
                self._track_usage(
                    response.usage if response is not None else None,
                    is_request_successful=response is not None,
                )

    # --- generate_image / agenerate_image ---

    @catch_llm_exceptions
    def generate_image(
        self,
        prompt: str,
        multi_modal_context: list[dict[str, Any]] | None = None,
        skip_usage_tracking: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        """Generate image(s) and return base64-encoded data.

        Automatically detects the appropriate API based on model name:
        - Diffusion models (DALL-E, Stable Diffusion, Imagen, etc.) -> image_generation API
        - All other models -> chat/completions API (default)

        Both paths return base64-encoded image data. If the API returns multiple images,
        all are returned in the list.

        Args:
            prompt: The prompt for image generation
            multi_modal_context: Optional list of image contexts for multi-modal generation.
                Only used with autoregressive models via chat completions API.
            skip_usage_tracking: Whether to skip usage tracking
            **kwargs: Additional arguments to pass to the model

        Returns:
            List of base64-encoded image strings (without data URI prefix)

        Raises:
            ImageGenerationError: If image generation fails or returns invalid data
        """
        logger.debug(
            f"Generating image with model {self.model_name!r}...",
            extra={"model": self.model_name, "prompt": prompt},
        )

        kwargs = self.consolidate_kwargs(**kwargs)
        response: ImageGenerationResponse | None = None
        got_usable_images = False
        try:
            request = self._build_image_generation_request(prompt, multi_modal_context, kwargs)
            response = self._client.generate_image(request)

            images = [img.b64_data for img in response.images]

            if not images:
                raise ImageGenerationError("No image data found in image generation response")

            got_usable_images = True
            if not skip_usage_tracking:
                self._usage_stats.extend(image_usage=ImageUsageStats(total_images=len(images)))

            return images
        finally:
            if not skip_usage_tracking:
                self._track_usage(
                    response.usage if response is not None else None,
                    is_request_successful=got_usable_images,
                )

    @acatch_llm_exceptions
    async def agenerate_image(
        self,
        prompt: str,
        multi_modal_context: list[dict[str, Any]] | None = None,
        skip_usage_tracking: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        """Async version of generate_image. Generate image(s) and return base64-encoded data.

        Automatically detects the appropriate API based on model name:
        - Diffusion models (DALL-E, Stable Diffusion, Imagen, etc.) -> image_generation API
        - All other models -> chat/completions API (default)

        Both paths return base64-encoded image data. If the API returns multiple images,
        all are returned in the list.

        Args:
            prompt: The prompt for image generation
            multi_modal_context: Optional list of image contexts for multi-modal generation.
                Only used with autoregressive models via chat completions API.
            skip_usage_tracking: Whether to skip usage tracking
            **kwargs: Additional arguments to pass to the model

        Returns:
            List of base64-encoded image strings (without data URI prefix)

        Raises:
            ImageGenerationError: If image generation fails or returns invalid data
        """
        logger.debug(
            f"Generating image with model {self.model_name!r}...",
            extra={"model": self.model_name, "prompt": prompt},
        )

        kwargs = self.consolidate_kwargs(**kwargs)
        response: ImageGenerationResponse | None = None
        got_usable_images = False
        try:
            request = self._build_image_generation_request(prompt, multi_modal_context, kwargs)
            response = await self._client.agenerate_image(request)

            images = [img.b64_data for img in response.images]

            if not images:
                raise ImageGenerationError("No image data found in image generation response")

            got_usable_images = True
            if not skip_usage_tracking:
                self._usage_stats.extend(image_usage=ImageUsageStats(total_images=len(images)))

            return images
        finally:
            if not skip_usage_tracking:
                self._track_usage(
                    response.usage if response is not None else None,
                    is_request_successful=got_usable_images,
                )

    # --- close / aclose ---

    def close(self) -> None:
        """Release resources held by the underlying client."""
        self._client.close()

    async def aclose(self) -> None:
        """Async release resources held by the underlying client."""
        await self._client.aclose()

    # --- private helpers ---

    def _get_mcp_facade(self, tool_alias: str | None) -> MCPFacade | None:
        if tool_alias is None:
            return None
        if self._mcp_registry is None:
            raise MCPConfigurationError(f"Tool alias {tool_alias!r} specified but no MCPRegistry configured.")

        try:
            return self._mcp_registry.get_mcp(tool_alias=tool_alias)
        except ValueError as exc:
            raise MCPConfigurationError(f"Tool alias {tool_alias!r} is not registered.") from exc

    def _build_chat_completion_request(
        self, messages: list[dict[str, Any]], kwargs: dict[str, Any]
    ) -> ChatCompletionRequest:
        """Build a ChatCompletionRequest from message payloads and consolidated kwargs."""
        request_fields: dict[str, Any] = {"model": self.model_name, "messages": messages}
        metadata: dict[str, Any] = {}

        for key, value in kwargs.items():
            if key in _COMPLETION_REQUEST_FIELDS:
                request_fields[key] = value
            else:
                metadata[key] = value

        if metadata:
            logger.debug(
                "Unknown kwargs %s dropped (not forwarded as model parameters). "
                "Use 'extra_body' to pass non-standard parameters to the model.",
                metadata.keys(),
            )

        return ChatCompletionRequest(**request_fields)

    def _build_embedding_request(self, input_texts: list[str], kwargs: dict[str, Any]) -> EmbeddingRequest:
        """Build an EmbeddingRequest from input texts and consolidated kwargs."""
        unknown = kwargs.keys() - _EMBEDDING_REQUEST_FIELDS
        if unknown:
            logger.debug(
                "Unknown kwargs %s dropped from embedding request. "
                "Use 'extra_body' to pass non-standard parameters to the model.",
                unknown,
            )
        return EmbeddingRequest(
            model=self.model_name,
            inputs=input_texts,
            encoding_format=kwargs.get("encoding_format"),
            dimensions=kwargs.get("dimensions"),
            timeout=kwargs.get("timeout"),
            extra_body=kwargs.get("extra_body"),
            extra_headers=kwargs.get("extra_headers"),
        )

    def _build_image_generation_request(
        self,
        prompt: str,
        multi_modal_context: list[dict[str, Any]] | None,
        kwargs: dict[str, Any],
    ) -> ImageGenerationRequest:
        """Build an ImageGenerationRequest, choosing chat-completion vs diffusion path."""
        unknown = kwargs.keys() - _IMAGE_GENERATION_REQUEST_FIELDS
        if unknown:
            logger.debug(
                "Unknown kwargs %s dropped from image generation request. "
                "Use 'extra_body' to pass non-standard parameters to the model.",
                unknown,
            )
        is_diffusion = is_image_diffusion_model(self.model_name)

        if is_diffusion:
            return ImageGenerationRequest(
                model=self.model_name,
                prompt=prompt,
                timeout=kwargs.get("timeout"),
                extra_body=kwargs.get("extra_body"),
                extra_headers=kwargs.get("extra_headers"),
            )

        chat_messages = [
            m.to_dict() for m in prompt_to_messages(user_prompt=prompt, multi_modal_context=multi_modal_context)
        ]
        return ImageGenerationRequest(
            model=self.model_name,
            prompt=prompt,
            messages=chat_messages,
            timeout=kwargs.get("timeout"),
            extra_body=kwargs.get("extra_body"),
            extra_headers=kwargs.get("extra_headers"),
        )

    def _track_usage(self, usage: Usage | None, *, is_request_successful: bool) -> None:
        """Unified usage tracking from canonical Usage type."""
        if not is_request_successful:
            self._usage_stats.extend(request_usage=RequestUsageStats(successful_requests=0, failed_requests=1))
            return

        token_usage = None
        if usage is not None and usage.input_tokens is not None:
            token_usage = TokenUsageStats(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens or 0,
            )

        self._usage_stats.extend(
            token_usage=token_usage,
            request_usage=RequestUsageStats(successful_requests=1, failed_requests=0),
        )
