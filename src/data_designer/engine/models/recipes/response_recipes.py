# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable

from pydantic import BaseModel

from data_designer.config.utils.code_lang import CodeLang
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.parsers.parser import LLMResponseParser
from data_designer.engine.models.parsers.postprocessors import (
    StructuredDataBlock,
    deserialize_json_code,
    merge_text_blocks,
)
from data_designer.engine.models.parsers.types import CodeBlock
from data_designer.engine.models.recipes.base import (
    ResponseRecipe,
)
from data_designer.engine.processing.gsonschema.validators import JSONSchemaValidationError, validate


class TextResponseRecipe(ResponseRecipe[str]):
    """Default text-parser.

    This parser is meant to cover the "pass-through" case of natural language LLM responses.
    """

    @property
    def example_template(self) -> str:
        return "{example}"

    def serialize_output(self, output: str) -> str:
        return output

    def deserialize_output(self, serialized_output: str) -> str:
        return serialized_output

    def _build_parser_fn(self) -> Callable[[str], str]:
        parser = LLMResponseParser(
            postprocessors=[
                merge_text_blocks,
            ]
        )

        return lambda x: parser.parse(x).response


class StructuredResponseRecipe(ResponseRecipe[dict]):
    """Recipe for structured responses.

    This recipe is intended to cover the generic case of
    prompting-based requests for structured data outputs,
    and the structure in question is determined by a
    provided JSON Schema.

    The LLM's response us validated against the provided
    JSON Schema, however the object returned is python
    dictionary obtained from deserializing the LLM's
    JSON response.
    """

    json_schema: dict
    pruning: bool
    no_extra_properties: bool

    def __init__(
        self,
        json_schema: dict,
        pruning: bool = True,
        no_extra_properties: bool = True,
        **kwargs,
    ):
        """Initialize StructuredResponseRecipe.

        Args:
            json_schema (dict): A target JSON schema that the LLM
                should adhere to when making its response.
            pruning (bool): If `True`, then any extra fields in the returned
                JSON object will be removed. Otherwise, they are retained,
                which could raise validation errors. Default=True
            no_extra_properties (bool) If `True`, then validation will fail
                if extra properties are encountered in the returned JSON response.
                Default=True.
        """
        super().__init__(**kwargs)
        self.json_schema = json_schema
        self.pruning = pruning
        self.no_extra_properties = no_extra_properties

    @property
    def task_instructions(self) -> str:
        return (
            "* Your response must be in JSON format.\n"
            "* Your JSON response must be returned within a Markdown, ```json code fence.\n"
            "* The JSON format is given as a JSON Schema description within <response_schema> markup tags.\n\n"
            f"<response_schema>\n{self.schema}\n</response_schema>"
        )

    @property
    def example_template(self) -> str:
        return "```json\n{example}\n```"

    def generate_response_example(self, example: dict) -> str:
        return self.example_template.format(example=json.dumps(example))

    @property
    def schema(self) -> str:
        return json.dumps(self.json_schema)

    def serialize_output(self, output: dict) -> str:
        return json.dumps(output, ensure_ascii=False)

    def deserialize_output(self, serialized_output: str) -> dict:
        return json.loads(serialized_output)

    @property
    def _validate_args(self):
        return {
            "schema": self.json_schema,
            "pruning": self.pruning,
            "no_extra_properties": self.no_extra_properties,
        }

    def _build_parser_fn(self) -> Callable[[str], dict]:
        parser = LLMResponseParser(
            postprocessors=[
                merge_text_blocks,
                deserialize_json_code,
            ]
        )

        def parse_fn(response: str) -> dict:
            try:
                obj = parser.parse(response).filter([StructuredDataBlock]).parsed.pop().obj
                return validate(obj, **self._validate_args)
            except IndexError:
                raise ParserException(
                    "No parsable JSON structure within ```json markdown fence.",
                    source=response,
                ) from None
            except JSONSchemaValidationError as exc:
                raise ParserException(
                    "Response doesn't match requested <response_schema>\n" + str(exc),
                    source=response,
                ) from None

        return parse_fn


class PydanticResponseRecipe(ResponseRecipe[BaseModel]):
    """Recipe for Pydantic responses.

    This recipe covers the case that we have a Pydantic
    data type (BaseModel) already specified in the runtime
    making LLM calls, and we want to obtain an object of
    that same data type as the output from the parser.

    This recipe operates in a very similar fashion to
    `StructuredResponseRecipe` except that it is initialized
    from a Pydantic `BaseModel` and does the extra step of
    validating against that `BaseModel` using
    `BaseModel.model_validate` for its return.
    """

    data_type: type[BaseModel]

    def __init__(self, data_type: type[BaseModel], **kwargs):
        """Initialize a PydanticResponseRecipe.

        Args:
            data_type (type(BaseModel)): The target Pydantic BaseModel
                subclass that the LLM should adhere to in its response,
                and defines the output type of the parser.
        """
        super().__init__(**kwargs)
        self.data_type = data_type

    @property
    def schema(self) -> str:
        return json.dumps(self.data_type.model_json_schema())

    @property
    def task_instructions(self) -> str:
        return (
            "* Your response must be in JSON format.\n"
            "* Your JSON response must be returned within a Markdown, ```json code fence.\n"
            "* The JSON format is given as a JSON Schema description within <response_schema> markup tags.\n\n"
            f"<response_schema>\n{self.schema}\n</response_schema>"
        )

    @property
    def example_template(self) -> str:
        return "```json\n{example}\n```"

    def generate_response_example(self, example: BaseModel) -> str:
        return self.example_template.format(example=example.model_dump_json())

    def serialize_output(self, output: BaseModel) -> str:
        return output.model_dump_json()

    def deserialize_output(self, serialized_output: str) -> BaseModel:
        return self.data_type.model_validate_json(serialized_output)

    def _build_parser_fn(self) -> Callable[[str], BaseModel]:
        parser = LLMResponseParser(
            postprocessors=[
                merge_text_blocks,
                deserialize_json_code,
            ]
        )

        def parse_fn(response: str) -> BaseModel:
            try:
                obj = parser.parse(response).filter([StructuredDataBlock]).parsed.pop().obj
                return self.data_type.model_validate(obj)
            except IndexError:
                raise ParserException(
                    "No parsable JSON structure within ```json markdown fence.",
                    source=response,
                ) from None
            except Exception as exc:
                raise ParserException(
                    "Response doesn't match requested <response_schema>\n" + str(exc),
                    source=response,
                ) from None

        return parse_fn


class CodeResponseRecipe(ResponseRecipe[str]):
    """Obtain a code snippet from an LLM."""

    def __init__(self, syntax: str | CodeLang, **kwargs):
        """Initialize a CodeResponseRecipe.

        Args:
            syntax (str | CodeLang): The code syntax that the
                LLM should adhere to, e.g. `"python"`, `"sql"`, etc.
        """
        super().__init__(**kwargs)
        self.syntax = CodeLang.parse_lang(syntax)

    @property
    def task_instructions(self) -> str:
        return (
            f"* Your response must be code written in {self.syntax}.\n"
            "* You will follow accepted and common syntax and best-practices.\n"
            f"* Your response will be given in markdown code fences specifying the correct language.\n"
            "* Only respond with a SINGLE code block."
        )

    @property
    def example_template(self) -> str:
        return f"```{self.syntax}\n{{example}}\n```\n"

    def serialize_output(self, output: str) -> str:
        return output

    def deserialize_output(self, serialized_output: str) -> str:
        return serialized_output

    def _build_parser_fn(self) -> Callable[[str], str]:
        parser = LLMResponseParser(
            postprocessors=[
                merge_text_blocks,
            ]
        )

        def parse_fn(response: str) -> str:
            try:
                code_block = parser.parse(response).filter([CodeBlock]).parsed.pop()
                # For the type checker -- should always pass
                assert isinstance(code_block, CodeBlock)
            except IndexError:
                raise ParserException(
                    "No parsable code response.",
                    source=response,
                ) from None

            # Only report this as a parser error if there was a mismatch.
            if code_block.code_lang and code_block.code_lang != self.syntax:
                raise ParserException(
                    f"Responded with code not matching the requested syntax ({self.syntax}).",
                    source=response,
                )

            return code_block.code.strip()

        return parse_fn
