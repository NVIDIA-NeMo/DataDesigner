# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import logging
from typing import Any

from litellm.types.router import DeploymentTypedDict, LiteLLM_Params
from litellm.types.utils import EmbeddingResponse, ImageResponse, ImageUsage, ModelResponse

from data_designer.config.models import GenerationType, ModelConfig, ModelProvider
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.errors import (
    GenerationValidationFailureError,
    catch_llm_exceptions,
    get_exception_primary_cause,
)
from data_designer.engine.models.litellm_overrides import CustomRouter, LiteLLMRouterDefaultKwargs
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.usage import ModelUsageStats, RequestUsageStats, TokenUsageStats
from data_designer.engine.models.utils import prompt_to_messages, str_to_message
from data_designer.engine.secret_resolver import SecretResolver

logger = logging.getLogger(__name__)


class ModelFacade:
    def __init__(
        self,
        model_config: ModelConfig,
        secret_resolver: SecretResolver,
        model_provider_registry: ModelProviderRegistry,
    ):
        self._model_config = model_config
        self._secret_resolver = secret_resolver
        self._model_provider_registry = model_provider_registry
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

    def completion(self, messages: list[dict[str, str]], skip_usage_tracking: bool = False, **kwargs) -> ModelResponse:
        logger.debug(
            f"Prompting model {self.model_name!r}...",
            extra={"model": self.model_name, "messages": messages},
        )
        response = None
        kwargs = self.consolidate_kwargs(**kwargs)
        try:
            response = self._router.completion(model=self.model_name, messages=messages, **kwargs)
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
        return kwargs

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
    def generate_image(self, prompt: str, skip_usage_tracking: bool = False, **kwargs) -> ImageResponse:
        logger.debug(
            f"Generating image with model {self.model_name!r}...",
            extra={"model": self.model_name, "prompt": prompt},
        )
        kwargs = self.consolidate_kwargs(**kwargs)
        response = None
        try:
            response = self._router.image_generation(prompt=prompt, model=self.model_name, **kwargs)
            logger.debug(
                f"Received image from model {self.model_name!r}",
                extra={"model": self.model_name, "response": response},
            )
            return response
        except Exception as e:
            raise e
        finally:
            if not skip_usage_tracking and response is not None:
                self._track_usage_from_image(response)

    @catch_llm_exceptions
    def generate(
        self,
        prompt: str,
        *,
        parser: Callable[[str], Any],
        system_prompt: str | None = None,
        multi_modal_context: list[dict[str, Any]] | None = None,
        max_correction_steps: int = 0,
        max_conversation_restarts: int = 0,
        skip_usage_tracking: bool = False,
        purpose: str | None = None,
        **kwargs,
    ) -> tuple[Any, str | None]:
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

        Raises:
            GenerationValidationFailureError: If the maximum number of retries or
                correction steps are met and the last response failures on
                generation validation.
        """
        output_obj = None
        curr_num_correction_steps = 0
        curr_num_restarts = 0
        curr_generation_attempt = 0
        max_generation_attempts = (max_correction_steps + 1) * (max_conversation_restarts + 1)

        starting_messages = prompt_to_messages(
            user_prompt=prompt, system_prompt=system_prompt, multi_modal_context=multi_modal_context
        )
        messages = deepcopy(starting_messages)

        while True:
            curr_generation_attempt += 1
            logger.debug(
                f"Starting generation attempt {curr_generation_attempt} of {max_generation_attempts} attempts."
            )

            completion_response = self.completion(messages, skip_usage_tracking=skip_usage_tracking, **kwargs)
            response = completion_response.choices[0].message.content or ""
            reasoning_trace = getattr(completion_response.choices[0].message, "reasoning_content", None)

            if reasoning_trace:
                ## There are generally some extra newlines with how these get parsed.
                response = response.strip()
                reasoning_trace = reasoning_trace.strip()

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
                    ## Add turns to loop-back errors for correction
                    messages += [
                        str_to_message(content=response, role="assistant"),
                        str_to_message(content=str(get_exception_primary_cause(exc)), role="user"),
                    ]
                elif curr_num_restarts < max_conversation_restarts:
                    curr_num_correction_steps = 0
                    curr_num_restarts += 1
                    messages = deepcopy(starting_messages)
                else:
                    raise GenerationValidationFailureError(
                        f"Unsuccessful generation attempt despite {max_generation_attempts} attempts."
                    ) from exc
        return output_obj, reasoning_trace

    def _get_litellm_deployment(self, model_config: ModelConfig) -> DeploymentTypedDict:
        provider = self._model_provider_registry.get_provider(model_config.provider)
        api_key = None
        if provider.api_key:
            api_key = self._secret_resolver.resolve(provider.api_key)
        api_key = api_key or "not-used-but-required"

        litellm_params = LiteLLM_Params(
            model=f"{provider.provider_type}/{model_config.model}",
            api_base=provider.endpoint,
            api_key=api_key,
        )
        return {
            "model_name": model_config.model,
            "litellm_params": litellm_params.model_dump(),
        }

    def _track_usage(self, response: ModelResponse | None) -> None:
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
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                ),
                request_usage=RequestUsageStats(successful_requests=1, failed_requests=0),
            )

    def _track_usage_from_embedding(self, response: EmbeddingResponse | None) -> None:
        if response is None:
            self._usage_stats.extend(request_usage=RequestUsageStats(successful_requests=0, failed_requests=1))
            return
        if response.usage is not None and response.usage.prompt_tokens is not None:
            self._usage_stats.extend(
                token_usage=TokenUsageStats(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=0,
                ),
                request_usage=RequestUsageStats(successful_requests=1, failed_requests=0),
            )

    def _track_usage_from_image(self, response: ImageResponse | None) -> None:
        if response is None:
            self._usage_stats.extend(request_usage=RequestUsageStats(successful_requests=0, failed_requests=1))
            return
        if response.usage is not None and isinstance(response.usage, ImageUsage):
            self._usage_stats.extend(
                token_usage=TokenUsageStats(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                ),
                request_usage=RequestUsageStats(successful_requests=1, failed_requests=0),
            )
