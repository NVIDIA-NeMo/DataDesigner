# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from data_designer.config.models import GenerationType, ModelConfig, ModelProvider
from data_designer.engine.mcp.errors import MCPConfigurationError
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.errors import (
    GenerationValidationFailureError,
    catch_llm_exceptions,
    get_exception_primary_cause,
)
from data_designer.engine.models.litellm_overrides import CustomRouter, LiteLLMRouterDefaultKwargs
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.usage import ModelUsageStats, RequestUsageStats, TokenUsageStats
from data_designer.engine.models.utils import ChatMessage, prompt_to_messages
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.lazy_heavy_imports import litellm

if TYPE_CHECKING:
    import litellm

    from data_designer.engine.mcp.facade import MCPFacade
    from data_designer.engine.mcp.registry import MCPRegistry

logger = logging.getLogger(__name__)


class ModelFacade:
    def __init__(
        self,
        model_config: ModelConfig,
        secret_resolver: SecretResolver,
        model_provider_registry: ModelProviderRegistry,
        *,
        mcp_registry: MCPRegistry | None = None,
    ) -> None:
        self._model_config = model_config
        self._secret_resolver = secret_resolver
        self._model_provider_registry = model_provider_registry
        self._mcp_registry = mcp_registry
        self._litellm_deployment = self._get_litellm_deployment(model_config)
        self._router = CustomRouter([self._litellm_deployment], **LiteLLMRouterDefaultKwargs().model_dump())
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
    def usage_stats(self) -> ModelUsageStats:
        return self._usage_stats

    def completion(
        self, messages: list[ChatMessage], skip_usage_tracking: bool = False, **kwargs
    ) -> litellm.ModelResponse:
        message_payloads = [message.to_dict() for message in messages]
        logger.debug(
            f"Prompting model {self.model_name!r}...",
            extra={"model": self.model_name, "messages": message_payloads},
        )
        response = None
        kwargs = self.consolidate_kwargs(**kwargs)
        try:
            response = self._router.completion(model=self.model_name, messages=message_payloads, **kwargs)
            logger.debug(
                f"Received completion from model {self.model_name!r}",
                extra={
                    "model": self.model_name,
                    "response": response,
                    "text": response.choices[0].message.content,
                    "usage": self._usage_stats.model_dump(),
                },
            )
            return response
        except Exception as e:
            raise e
        finally:
            if not skip_usage_tracking and response is not None:
                self._track_usage(response)

    def consolidate_kwargs(self, **kwargs) -> dict[str, Any]:
        # Remove purpose from kwargs to avoid passing it to the model
        kwargs.pop("purpose", None)
        kwargs = {**self._model_config.inference_parameters.generate_kwargs, **kwargs}
        if self.model_provider.extra_body:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **self.model_provider.extra_body}
        if self.model_provider.extra_headers:
            kwargs["extra_headers"] = self.model_provider.extra_headers
        return kwargs

    def _get_mcp_facade(self, tool_alias: str | None) -> MCPFacade | None:
        if tool_alias is None:
            return None
        if self._mcp_registry is None:
            raise MCPConfigurationError(f"Tool alias {tool_alias!r} specified but no MCPRegistry configured.")

        try:
            return self._mcp_registry.get_mcp(tool_alias=tool_alias)
        except ValueError as exc:
            raise MCPConfigurationError(f"Tool alias {tool_alias!r} is not registered.") from exc

    @catch_llm_exceptions
    def generate_text_embeddings(
        self, input_texts: list[str], skip_usage_tracking: bool = False, **kwargs
    ) -> list[list[float]]:
        logger.debug(
            f"Generating embeddings with model {self.model_name!r}...",
            extra={
                "model": self.model_name,
                "input_count": len(input_texts),
            },
        )
        kwargs = self.consolidate_kwargs(**kwargs)
        response = None
        try:
            response = self._router.embedding(model=self.model_name, input=input_texts, **kwargs)
            logger.debug(
                f"Received embeddings from model {self.model_name!r}",
                extra={
                    "model": self.model_name,
                    "embedding_count": len(response.data) if response.data else 0,
                    "usage": self._usage_stats.model_dump(),
                },
            )
            if response.data and len(response.data) == len(input_texts):
                return [data["embedding"] for data in response.data]
            else:
                raise ValueError(f"Expected {len(input_texts)} embeddings, but received {len(response.data)}")
        except Exception as e:
            raise e
        finally:
            if not skip_usage_tracking and response is not None:
                self._track_usage_from_embedding(response)

    @catch_llm_exceptions
    def generate_image(self, prompt: str, skip_usage_tracking: bool = False, **kwargs) -> str:
        """Generate image using either autoregressive or diffusion models.

        Automatically detects the appropriate API based on the model's generation_type:
        - CHAT_COMPLETION_IMAGE → chat/completions API (GPT-5, gpt-image-*, Gemini)
        - DIFFUSION_IMAGE → image_generation API (DALL-E, Imagen, Stable Diffusion)

        Args:
            prompt: The prompt for image generation.
            skip_usage_tracking: Whether to skip usage tracking. Default: False.
            **kwargs: Additional arguments to pass to the model.

        Returns:
            Base64-encoded image data (without data URI prefix for autoregressive models).
            For diffusion models: URL string or base64 data depending on output_format.
        """
        from data_designer.config.models import GenerationType

        logger.debug(
            f"Generating image with model {self.model_name!r}...",
            extra={"model": self.model_name, "prompt": prompt},
        )

        # Determine which API to use based on generation_type
        gen_type = self.model_generation_type

        if gen_type == GenerationType.DIFFUSION_IMAGE:
            return self._generate_image_diffusion(prompt, skip_usage_tracking, **kwargs)
        else:
            # Default to chat-completion (CHAT_COMPLETION_IMAGE or backward compatibility)
            return self._generate_image_chat_completion(prompt, skip_usage_tracking, **kwargs)

    def _generate_image_chat_completion(self, prompt: str, skip_usage_tracking: bool = False, **kwargs) -> str:
        """Generate image using autoregressive model via chat completions API."""
        kwargs = self.consolidate_kwargs(**kwargs)

        # Build messages for image generation
        messages = [ChatMessage.as_user(content=prompt)]

        response = None
        try:
            response = self.completion(
                messages=messages,
                skip_usage_tracking=skip_usage_tracking,
                **kwargs,
            )
            logger.debug(
                f"Received image from autoregressive model {self.model_name!r}",
                extra={"model": self.model_name, "response": response},
            )

            # Check if response has images attribute (some models return images here)
            if hasattr(response.choices[0].message, "images") and response.choices[0].message.images:
                # Extract base64 from first image
                first_image = response.choices[0].message.images[0]
                if isinstance(first_image, dict) and "image_url" in first_image:
                    image_url = first_image["image_url"]
                    if isinstance(image_url, dict) and "url" in image_url:
                        # Extract base64 data from data URL
                        url = image_url["url"]
                        if url.startswith("data:image/"):
                            # Remove data URI prefix to get pure base64
                            return url.split(",", 1)[1] if "," in url else url
                        return url
                    elif isinstance(image_url, str):
                        if image_url.startswith("data:image/"):
                            return image_url.split(",", 1)[1] if "," in image_url else image_url
                        return image_url
                return str(first_image)

            # If no images attribute, check content for base64 or image data
            content = response.choices[0].message.content or ""
            if content.startswith("data:image/"):
                # Remove data URI prefix
                return content.split(",", 1)[1] if "," in content else content

            # Return content as-is (might be base64 or other format)
            return content

        except Exception as e:
            raise e

    def _generate_image_diffusion(self, prompt: str, skip_usage_tracking: bool = False, **kwargs) -> str:
        """Generate image using diffusion model via image_generation API."""
        from data_designer.config.models import ModalityDataType

        kwargs = self.consolidate_kwargs(**kwargs)
        response = None
        try:
            response = self._router.image_generation(prompt=prompt, model=self.model_name, **kwargs)
            logger.debug(
                f"Received image from diffusion model {self.model_name!r}",
                extra={"model": self.model_name, "response": response},
            )

            # Return URL or base64 based on output_format
            output_format = getattr(self._model_config.inference_parameters, "output_format", ModalityDataType.BASE64)
            if output_format == ModalityDataType.URL:
                return response.data[0].url
            else:
                return response.data[0].b64_json

        except Exception as e:
            raise e
        finally:
            if not skip_usage_tracking and response is not None:
                self._track_usage_from_image_diffusion(response)

    @catch_llm_exceptions
    def generate(
        self,
        prompt: str,
        *,
        parser: Callable[[str], Any],
        system_prompt: str | None = None,
        multi_modal_context: list[dict[str, Any]] | None = None,
        tool_alias: str | None = None,
        max_correction_steps: int = 0,
        max_conversation_restarts: int = 0,
        skip_usage_tracking: bool = False,
        purpose: str | None = None,
        **kwargs,
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
                an LLM response into some output object.
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
            response = completion_response.choices[0].message.content or ""
            reasoning_trace = getattr(completion_response.choices[0].message, "reasoning_content", None)
            messages.append(ChatMessage.as_assistant(content=response, reasoning_content=reasoning_trace or None))
            curr_num_correction_steps += 1

            try:
                output_obj = parser(response)  # type: ignore - if not a string will cause a ParserException below
                break
            except ParserException as exc:
                if max_correction_steps == 0 and max_conversation_restarts == 0:
                    raise GenerationValidationFailureError(
                        "Unsuccessful generation attempt. No retries were attempted."
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
                    raise GenerationValidationFailureError(
                        f"Unsuccessful generation despite {max_correction_steps} correction steps "
                        f"and {max_conversation_restarts} conversation restarts."
                    ) from exc

        return output_obj, messages

    def _get_litellm_deployment(self, model_config: ModelConfig) -> litellm.DeploymentTypedDict:
        provider = self._model_provider_registry.get_provider(model_config.provider)
        api_key = None
        if provider.api_key:
            api_key = self._secret_resolver.resolve(provider.api_key)
        api_key = api_key or "not-used-but-required"

        litellm_params = litellm.LiteLLM_Params(
            model=f"{provider.provider_type}/{model_config.model}",
            api_base=provider.endpoint,
            api_key=api_key,
        )
        return {
            "model_name": model_config.model,
            "litellm_params": litellm_params.model_dump(),
        }

    def _track_usage(self, response: litellm.types.utils.ModelResponse | None) -> None:
        if response is None:
            self._usage_stats.extend(request_usage=RequestUsageStats(successful_requests=0, failed_requests=1))
            return
        if (
            response.usage is not None
            and response.usage.prompt_tokens is not None
            and response.usage.completion_tokens is not None
        ):
            self._usage_stats.extend(
                token_usage=TokenUsageStats(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                ),
                request_usage=RequestUsageStats(successful_requests=1, failed_requests=0),
            )

    def _track_usage_from_embedding(self, response: litellm.types.utils.EmbeddingResponse | None) -> None:
        if response is None:
            self._usage_stats.extend(request_usage=RequestUsageStats(successful_requests=0, failed_requests=1))
            return
        if response.usage is not None and response.usage.prompt_tokens is not None:
            self._usage_stats.extend(
                token_usage=TokenUsageStats(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=0,
                ),
                request_usage=RequestUsageStats(successful_requests=1, failed_requests=0),
            )

    def _track_usage_from_response(self, response: litellm.types.utils.ResponseResponse | None) -> None:
        """Track usage from Responses API response."""
        if response is None:
            self._usage_stats.extend(request_usage=RequestUsageStats(successful_requests=0, failed_requests=1))
            return
        if response.usage is not None:
            input_tokens = getattr(response.usage, "input_tokens", 0) or 0
            output_tokens = getattr(response.usage, "output_tokens", 0) or 0
            self._usage_stats.extend(
                token_usage=TokenUsageStats(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
                request_usage=RequestUsageStats(successful_requests=1, failed_requests=0),
            )

    def _track_usage_from_image_diffusion(self, response: litellm.types.utils.ImageResponse | None) -> None:
        """Track usage from image_generation API response."""
        if response is None:
            self._usage_stats.extend(request_usage=RequestUsageStats(successful_requests=0, failed_requests=1))
            return
        if response.usage is not None and isinstance(response.usage, litellm.types.utils.ImageUsage):
            self._usage_stats.extend(
                token_usage=TokenUsageStats(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                ),
                request_usage=RequestUsageStats(successful_requests=1, failed_requests=0),
            )
