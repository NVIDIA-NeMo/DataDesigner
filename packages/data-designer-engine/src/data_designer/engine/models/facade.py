# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from data_designer.config.mcp import MCPToolConfig
from data_designer.config.models import GenerationType, ModelConfig, ModelProvider
from data_designer.engine.mcp.errors import MCPConfigurationError, MCPToolError
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
from data_designer.lazy_heavy_imports import litellm

if TYPE_CHECKING:
    import litellm
    from data_designer.engine.mcp.manager import MCPClientManager

logger = logging.getLogger(__name__)


class ModelFacade:
    def __init__(
        self,
        model_config: ModelConfig,
        secret_resolver: SecretResolver,
        model_provider_registry: ModelProviderRegistry,
        *,
        mcp_client_manager: MCPClientManager | None = None,
    ):
        self._model_config = model_config
        self._secret_resolver = secret_resolver
        self._model_provider_registry = model_provider_registry
        self._mcp_client_manager = mcp_client_manager
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
        self, messages: list[dict[str, str]], skip_usage_tracking: bool = False, **kwargs
    ) -> litellm.ModelResponse:
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
        if self.model_provider.extra_headers:
            kwargs["extra_headers"] = self.model_provider.extra_headers
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
    def generate(
        self,
        prompt: str,
        *,
        parser: Callable[[str], Any],
        system_prompt: str | None = None,
        multi_modal_context: list[dict[str, Any]] | None = None,
        tool_config: MCPToolConfig | None = None,
        include_full_traces: bool = False,
        max_correction_steps: int = 0,
        max_conversation_restarts: int = 0,
        skip_usage_tracking: bool = False,
        purpose: str | None = None,
        **kwargs,
    ) -> tuple[Any, str | None, list[dict[str, Any]] | None]:
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
            tool_config (MCPToolConfig | None): Optional MCP tool configuration. When provided,
                the model may call permitted tools from the configured MCP server.
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
        tool_schemas = None
        tool_calls_used = 0
        curr_num_correction_steps = 0
        curr_num_restarts = 0
        curr_generation_attempt = 0
        max_generation_attempts = (max_correction_steps + 1) * (max_conversation_restarts + 1)

        starting_messages = prompt_to_messages(
            user_prompt=prompt, system_prompt=system_prompt, multi_modal_context=multi_modal_context
        )
        messages = deepcopy(starting_messages)
        trace_messages: list[dict[str, Any]] | None = deepcopy(starting_messages) if include_full_traces else None

        if tool_config is not None:
            tool_schemas = self._get_tool_schemas(tool_config)

        while True:
            curr_generation_attempt += 1
            logger.debug(
                f"Starting generation attempt {curr_generation_attempt} of {max_generation_attempts} attempts."
            )

            completion_kwargs = dict(kwargs)
            if tool_schemas is not None:
                completion_kwargs["tools"] = tool_schemas
            completion_response = self.completion(
                messages,
                skip_usage_tracking=skip_usage_tracking,
                **completion_kwargs,
            )
            response = completion_response.choices[0].message.content or ""
            reasoning_trace = getattr(completion_response.choices[0].message, "reasoning_content", None)

            if reasoning_trace:
                ## There are generally some extra newlines with how these get parsed.
                response = response.strip()
                reasoning_trace = reasoning_trace.strip()

            tool_calls = self._extract_tool_calls(completion_response.choices[0].message)
            if tool_config is not None and len(tool_calls) > 0:
                tool_calls_used += len(tool_calls)
                if tool_calls_used > tool_config.max_tool_calls:
                    raise MCPToolError(
                        f"Exceeded maximum MCP tool calls ({tool_config.max_tool_calls}) for server "
                        f"{tool_config.server_name!r}."
                    )
                assistant_tool_message = self._build_assistant_tool_message(response, tool_calls)
                tool_messages = self._execute_tool_calls(tool_config, tool_calls)

                messages.append(assistant_tool_message)
                messages.extend(tool_messages)

                if trace_messages is not None:
                    assistant_trace_message = dict(assistant_tool_message)
                    if reasoning_trace:
                        assistant_trace_message["reasoning_content"] = reasoning_trace
                    trace_messages.append(assistant_trace_message)
                    trace_messages.extend(tool_messages)
                continue

            curr_num_correction_steps += 1

            try:
                output_obj = parser(response)  # type: ignore - if not a string will cause a ParserException below
                if trace_messages is not None:
                    assistant_trace_message: dict[str, Any] = {"role": "assistant", "content": response}
                    if reasoning_trace:
                        assistant_trace_message["reasoning_content"] = reasoning_trace
                    trace_messages.append(assistant_trace_message)
                break
            except ParserException as exc:
                if max_correction_steps == 0 and max_conversation_restarts == 0:
                    raise GenerationValidationFailureError(
                        "Unsuccessful generation attempt. No retries were attempted."
                    ) from exc
                if curr_num_correction_steps <= max_correction_steps:
                    ## Add turns to loop-back errors for correction
                    assistant_message = str_to_message(content=response, role="assistant")
                    user_message = str_to_message(content=str(get_exception_primary_cause(exc)), role="user")
                    messages += [assistant_message, user_message]
                    if trace_messages is not None:
                        assistant_trace_message = dict(assistant_message)
                        if reasoning_trace:
                            assistant_trace_message["reasoning_content"] = reasoning_trace
                        trace_messages += [assistant_trace_message, user_message]
                elif curr_num_restarts < max_conversation_restarts:
                    curr_num_correction_steps = 0
                    curr_num_restarts += 1
                    messages = deepcopy(starting_messages)
                    if trace_messages is not None:
                        trace_messages = deepcopy(starting_messages)
                else:
                    raise GenerationValidationFailureError(
                        f"Unsuccessful generation attempt despite {max_generation_attempts} attempts."
                    ) from exc
        return output_obj, reasoning_trace, trace_messages

    def _get_tool_schemas(self, tool_config: MCPToolConfig) -> list[dict[str, Any]]:
        if self._mcp_client_manager is None:
            raise MCPConfigurationError("MCP tool configuration was provided but no MCP servers were configured.")
        return self._mcp_client_manager.get_tool_schemas(tool_config)

    def _extract_tool_calls(self, message: Any) -> list[dict[str, Any]]:
        raw_tool_calls = getattr(message, "tool_calls", None)
        if raw_tool_calls is None and isinstance(message, dict):
            raw_tool_calls = message.get("tool_calls")
        if not raw_tool_calls:
            return []
        tool_calls: list[dict[str, Any]] = []
        for raw_tool_call in raw_tool_calls:
            tool_calls.append(self._normalize_tool_call(raw_tool_call))
        return tool_calls

    def _normalize_tool_call(self, raw_tool_call: Any) -> dict[str, Any]:
        if isinstance(raw_tool_call, dict):
            tool_call_id = raw_tool_call.get("id")
            function = raw_tool_call.get("function") or {}
            name = function.get("name") or raw_tool_call.get("name")
            arguments = function.get("arguments") or raw_tool_call.get("arguments")
        else:
            tool_call_id = getattr(raw_tool_call, "id", None)
            function = getattr(raw_tool_call, "function", None)
            name = getattr(function, "name", None) if function is not None else getattr(raw_tool_call, "name", None)
            arguments = (
                getattr(function, "arguments", None)
                if function is not None
                else getattr(raw_tool_call, "arguments", None)
            )

        if not name:
            raise MCPToolError("MCP tool call is missing a tool name.")

        arguments_payload: dict[str, Any]
        arguments_json: str
        if arguments is None or arguments == "":
            arguments_payload = {}
            arguments_json = "{}"
        elif isinstance(arguments, str):
            try:
                arguments_payload = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise MCPToolError(f"Invalid tool arguments for '{name}': {arguments}") from exc
            arguments_json = arguments
        elif isinstance(arguments, dict):
            arguments_payload = arguments
            arguments_json = json.dumps(arguments_payload)
        else:
            raise MCPToolError(f"Unsupported tool arguments type for '{name}': {type(arguments)!r}")

        return {
            "id": tool_call_id or uuid.uuid4().hex,
            "name": name,
            "arguments": arguments_payload,
            "arguments_json": arguments_json,
        }

    def _build_assistant_tool_message(self, response: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": response or "",
            "tool_calls": [
                {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {"name": tool_call["name"], "arguments": tool_call["arguments_json"]},
                }
                for tool_call in tool_calls
            ],
        }

    def _execute_tool_calls(self, tool_config: MCPToolConfig, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._mcp_client_manager is None:
            raise MCPConfigurationError("MCP tool configuration was provided but no MCP servers were configured.")

        allowed_tools = set(tool_config.tool_names) if tool_config.tool_names else None
        tool_messages: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            if allowed_tools is not None and tool_name not in allowed_tools:
                raise MCPToolError(f"Tool {tool_name!r} is not permitted for server {tool_config.server_name!r}.")
            result = self._mcp_client_manager.call_tool(tool_config.server_name, tool_name, tool_call["arguments"])
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result.content,
                }
            )
        return tool_messages

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
