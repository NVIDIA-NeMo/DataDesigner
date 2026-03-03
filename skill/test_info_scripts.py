#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for the data designer info scripts.

Run from the skill/ directory:
    uv run test_info_scripts.py

Tests cover:
  - Unit tests for pydantic_info_utils helper functions
  - Integration tests for all four info scripts (column, sampler, validator, processor)
  - CLI behavior: help, list, single entry, all entries, invalid input, exit codes
  - Nested BaseModel expansion (Score, ImageContext)
  - Enum value expansion (SamplerType, TraceType, CodeLang, etc.)
  - Discriminated union non-expansion (params fields)
  - Cycle/depth protection in nested model printing
"""

import io
import re
import subprocess
import sys
from contextlib import redirect_stdout
from enum import Enum
from pathlib import Path
from typing import Annotated

# Resolve the scripts directory relative to this test file
SCRIPTS_DIR = Path(__file__).resolve().parent / "data-designer" / "scripts"

# Add scripts dir to sys.path so we can import the helpers module
sys.path.insert(0, str(SCRIPTS_DIR))

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0
_errors: list[str] = []


def check(condition: bool, name: str) -> None:
    global _passed, _failed
    if condition:
        _passed += 1
    else:
        _failed += 1
        _errors.append(name)
        print(f"  FAIL: {name}")


def run_script(script: str, *args: str, expect_fail: bool = False) -> tuple[str, str, int]:
    """Run a script in skills/data-designer/scripts/ via uv and return (stdout, stderr, returncode)."""
    script_path = str(SCRIPTS_DIR / script)
    result = subprocess.run(
        ["uv", "run", script_path, *args],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if not expect_fail and result.returncode != 0:
        print(f"  WARNING: {script} {' '.join(args)} exited with code {result.returncode}")
        if result.stderr:
            print(f"    stderr: {result.stderr[:200]}")
    return result.stdout, result.stderr, result.returncode


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


# ===========================================================================
# PART 1: Unit tests for pydantic_info_utils helpers
# ===========================================================================

section("Unit tests: type introspection helpers")

from helpers.pydantic_info_utils import (
    _extract_enum_class,
    _is_basemodel_subclass,
    _is_enum_subclass,
    extract_nested_basemodel,
    format_type,
    get_brief_description,
    get_field_info,
)

# ---- Test models & enums for unit tests ----


class MyEnum(str, Enum):
    A = "a"
    B = "b"
    C = "c"


class AnotherEnum(Enum):
    X = 1
    Y = 2


class Nested(BaseModel):
    x: int = 0
    y: str = "hello"


class Nested2(BaseModel):
    z: float = 1.0


class Outer(BaseModel):
    """Outer model for testing."""

    plain: str = "foo"
    nested: Nested = Field(default_factory=Nested)
    nested_list: list[Nested] = Field(default_factory=list)
    nested_optional: Nested | None = None
    nested_list_optional: list[Nested] | None = None
    nested_dict: dict[str, Nested] = Field(default_factory=dict)
    my_enum: MyEnum = MyEnum.A
    enum_optional: MyEnum | None = None
    annotated_enum: Annotated[MyEnum, "some metadata"] = MyEnum.A


class SelfRef(BaseModel):
    """Model that references itself indirectly."""

    name: str = ""
    child: "SelfRef | None" = None


SelfRef.model_rebuild()


# Model with a discriminated-union-like field (multiple BaseModel subclasses)
class DiscriminatedOuter(BaseModel):
    choice: Nested | Nested2 = Field(default_factory=Nested)


# ---- _is_basemodel_subclass ----
print("\n_is_basemodel_subclass:")
check(_is_basemodel_subclass(Nested) is True, "concrete subclass -> True")
check(_is_basemodel_subclass(BaseModel) is False, "BaseModel itself -> False")
check(_is_basemodel_subclass(str) is False, "str -> False")
check(_is_basemodel_subclass(int) is False, "int -> False")
check(_is_basemodel_subclass(MyEnum) is False, "Enum -> False")
check(_is_basemodel_subclass(None) is False, "None -> False")
check(_is_basemodel_subclass(list) is False, "list -> False")

# ---- _is_enum_subclass ----
print("\n_is_enum_subclass:")
check(_is_enum_subclass(MyEnum) is True, "str Enum subclass -> True")
check(_is_enum_subclass(AnotherEnum) is True, "int Enum subclass -> True")
check(_is_enum_subclass(Enum) is False, "Enum itself -> False")
check(_is_enum_subclass(str) is False, "str -> False")
check(_is_enum_subclass(BaseModel) is False, "BaseModel -> False")
check(_is_enum_subclass(None) is False, "None -> False")

# ---- _extract_enum_class ----
print("\n_extract_enum_class:")
check(_extract_enum_class(MyEnum) is MyEnum, "direct enum class")
check(_extract_enum_class(MyEnum | None) is MyEnum, "enum | None")
check(_extract_enum_class(MyEnum | None) is MyEnum, "Optional[enum]")
check(_extract_enum_class(Annotated[MyEnum, "meta"]) is MyEnum, "Annotated[enum, ...]")
check(_extract_enum_class(str) is None, "str -> None")
check(_extract_enum_class(int) is None, "int -> None")
check(_extract_enum_class(None) is None, "None -> None")
check(_extract_enum_class(Nested) is None, "BaseModel -> None")
check(_extract_enum_class(list[MyEnum]) is None, "list[enum] -> None (not unwrapped)")

# ---- extract_nested_basemodel ----
print("\nextract_nested_basemodel:")
check(extract_nested_basemodel(Nested) is Nested, "direct BaseModel subclass")
check(extract_nested_basemodel(list[Nested]) is Nested, "list[Model]")
check(extract_nested_basemodel(Nested | None) is Nested, "Model | None")
check(extract_nested_basemodel(list[Nested] | None) is Nested, "list[Model] | None")
check(extract_nested_basemodel(dict[str, Nested]) is Nested, "dict[str, Model]")
check(
    extract_nested_basemodel(Annotated[Nested, "meta"]) is Nested,
    "Annotated[Model, ...]",
)
check(
    extract_nested_basemodel(Annotated[list[Nested], "meta"]) is Nested,
    "Annotated[list[Model], ...]",
)

# Should return None for these:
check(extract_nested_basemodel(str) is None, "str -> None")
check(extract_nested_basemodel(int) is None, "int -> None")
check(extract_nested_basemodel(None) is None, "None -> None")
check(extract_nested_basemodel(BaseModel) is None, "BaseModel itself -> None")
check(extract_nested_basemodel(list[str]) is None, "list[str] -> None")
check(extract_nested_basemodel(dict[str, int]) is None, "dict[str, int] -> None")
check(extract_nested_basemodel(MyEnum) is None, "Enum -> None")
# Discriminated union: 2+ BaseModel subclasses -> None
check(
    extract_nested_basemodel(Nested | Nested2) is None,
    "Model | Model2 (discriminated) -> None",
)
check(
    extract_nested_basemodel(Nested | Nested2 | None) is None,
    "Model | Model2 | None -> None",
)

# ---- format_type ----
print("\nformat_type:")
check(format_type(str) == "str", "str")
check(format_type(int) == "int", "int")
check("None" in format_type(str | None), "str | None contains 'None'")
check("list" in format_type(list[str]).lower(), "list[str]")

# ---- get_brief_description ----
print("\nget_brief_description:")
check(get_brief_description(Outer) == "Outer model for testing.", "docstring extraction")
check(
    get_brief_description(type("NoDoc", (), {})) == "No description available.",
    "no docstring",
)

# ---- get_field_info: tuple structure ----
print("\nget_field_info tuple structure:")
fields = get_field_info(Outer, {})
check(len(fields) > 0, "returns non-empty list")
check(all(len(f) == 5 for f in fields), "all tuples have 5 elements")

# Verify nested detection in field tuples
field_dict = {f[0]: f for f in fields}

# plain str field: no nested, no enum
f = field_dict["plain"]
check(f[3] is None, "plain str -> nested_cls is None")
check(f[4] is None, "plain str -> enum_cls is None")

# nested field: should detect Nested
f = field_dict["nested"]
check(f[3] is Nested, "nested field -> nested_cls is Nested")
check(f[4] is None, "nested field -> enum_cls is None")

# nested_list: should detect Nested
f = field_dict["nested_list"]
check(f[3] is Nested, "nested_list -> nested_cls is Nested")

# nested_optional: should detect Nested
f = field_dict["nested_optional"]
check(f[3] is Nested, "nested_optional -> nested_cls is Nested")

# nested_list_optional: should detect Nested
f = field_dict["nested_list_optional"]
check(f[3] is Nested, "nested_list_optional -> nested_cls is Nested")

# nested_dict: should detect Nested
f = field_dict["nested_dict"]
check(f[3] is Nested, "nested_dict -> nested_cls is Nested")

# my_enum: should detect MyEnum
f = field_dict["my_enum"]
check(f[3] is None, "enum field -> nested_cls is None")
check(f[4] is MyEnum, "my_enum -> enum_cls is MyEnum")

# enum_optional: should detect MyEnum
f = field_dict["enum_optional"]
check(f[4] is MyEnum, "enum_optional -> enum_cls is MyEnum")

# annotated_enum
f = field_dict["annotated_enum"]
check(f[4] is MyEnum, "annotated_enum -> enum_cls is MyEnum")


# ===========================================================================
# PART 2: Unit tests for _print_fields and print_yaml_entry
# ===========================================================================

section("Unit tests: output formatting")

from helpers.pydantic_info_utils import _print_fields, print_yaml_entry

# ---- _print_fields: basic output ----
print("\n_print_fields basic output:")
fields = get_field_info(Nested, {})
buf = io.StringIO()
with redirect_stdout(buf):
    _print_fields(fields, {})
output = buf.getvalue()
check("x:" in output, "nested field 'x' printed")
check("y:" in output, "nested field 'y' printed")
check("type: int" in output, "type: int printed")
check("type: str" in output, "type: str printed")

# ---- _print_fields: enum expansion ----
print("\n_print_fields enum expansion:")
fields = get_field_info(Outer, {})
enum_fields = [f for f in fields if f[0] == "my_enum"]
buf = io.StringIO()
with redirect_stdout(buf):
    _print_fields(enum_fields, {})
output = buf.getvalue()
check("values:" in output, "values: line present for enum field")
check("A" in output and "B" in output and "C" in output, "all enum member names present")

# ---- _print_fields: nested model expansion ----
print("\n_print_fields nested model expansion:")
fields = get_field_info(Outer, {})
nested_fields = [f for f in fields if f[0] == "nested"]
buf = io.StringIO()
with redirect_stdout(buf):
    _print_fields(nested_fields, {})
output = buf.getvalue()
check("schema (Nested):" in output, "schema (Nested): header present")
check("        x:" in output, "nested field x printed at deeper indent")
check("        y:" in output, "nested field y printed at deeper indent")

# ---- _print_fields: cycle protection ----
print("\n_print_fields cycle protection:")
fields = get_field_info(SelfRef, {})
buf = io.StringIO()
with redirect_stdout(buf):
    _print_fields(fields, {})
output = buf.getvalue()
# Should expand SelfRef once but not infinitely recurse
count = output.count("schema (SelfRef):")
check(count == 1, f"SelfRef expanded exactly once (got {count})")

# ---- _print_fields: discriminated union not expanded ----
print("\n_print_fields discriminated union not expanded:")
fields = get_field_info(DiscriminatedOuter, {})
choice_field = [f for f in fields if f[0] == "choice"]
buf = io.StringIO()
with redirect_stdout(buf):
    _print_fields(choice_field, {})
output = buf.getvalue()
check("schema" not in output, "discriminated union field not expanded")

# ---- _print_fields: depth protection ----
print("\n_print_fields depth limit:")


class Level3(BaseModel):
    val: int = 0


class Level2(BaseModel):
    child: Level3 = Field(default_factory=Level3)


class Level1(BaseModel):
    child: Level2 = Field(default_factory=Level2)


class Level0(BaseModel):
    child: Level1 = Field(default_factory=Level1)


fields = get_field_info(Level0, {})
buf = io.StringIO()
with redirect_stdout(buf):
    _print_fields(fields, {}, max_depth=2)
output = buf.getvalue()
check("schema (Level1):" in output, "depth 0->1: Level1 expanded")
check("schema (Level2):" in output, "depth 1->2: Level2 expanded")
check("schema (Level3):" not in output, "depth 2->3: Level3 NOT expanded (max_depth=2)")

# ---- print_yaml_entry ----
print("\nprint_yaml_entry:")
buf = io.StringIO()
with redirect_stdout(buf):
    print_yaml_entry("test_type", "my_value", Outer, {"plain": "A plain field"})
output = buf.getvalue()
check(output.startswith("Outer:"), "starts with class name")
check("  test_type: my_value" in output, "type key/value printed")
check("  description: Outer model for testing." in output, "description printed")
check("  fields:" in output, "fields header printed")
check("    plain:" in output, "field plain printed at indent 4")
check("      description: A plain field" in output, "default description used")
check("schema (Nested):" in output, "nested model expanded in yaml entry")
check("values: [A, B, C]" in output, "enum values expanded in yaml entry")

# ---- print_yaml_entry with uppercase_value ----
print("\nprint_yaml_entry uppercase_value:")
buf = io.StringIO()
with redirect_stdout(buf):
    print_yaml_entry("test_type", "my_value", Nested, {}, uppercase_value=True)
output = buf.getvalue()
check("  test_type: MY_VALUE" in output, "uppercase value printed")


# ===========================================================================
# PART 3: Integration tests — get_column_info.py
# ===========================================================================

section("Integration: get_column_info.py")

SCRIPT = "get_column_info.py"

# ---- help ----
print("\nhelp output:")
out, err, rc = run_script(SCRIPT, "--help")
check(rc == 0, "help exits 0")
check("Usage:" in out, "help contains Usage:")
check("column_type" in out, "help mentions column_type")
check("config_class" in out, "help mentions config_class")
check("Examples:" in out, "help contains Examples section")
check("Available column_types" in out, "help lists available types")

# No-arg should also show help
out2, _, rc2 = run_script(SCRIPT)
check(rc2 == 0, "no-arg exits 0")
check("Usage:" in out2, "no-arg shows help")

# ---- list ----
print("\nlist output:")
out, err, rc = run_script(SCRIPT, "--list")
check(rc == 0, "list exits 0")
check("column_type" in out, "list header has column_type")
check("config_class" in out, "list header has config_class")
expected_types = [
    "custom",
    "embedding",
    "expression",
    "llm-code",
    "llm-judge",
    "llm-structured",
    "llm-text",
    "sampler",
    "seed-dataset",
    "validation",
]
for t in expected_types:
    check(t in out, f"list contains '{t}'")
check("LLMJudgeColumnConfig" in out, "list contains LLMJudgeColumnConfig class name")
check("SamplerColumnConfig" in out, "list contains SamplerColumnConfig class name")

# -l alias
out_l, _, _ = run_script(SCRIPT, "-l")
check(out_l == out, "-l produces same output as --list")

# ---- single entry: llm-judge (nested model expansion) ----
print("\nsingle entry: llm-judge:")
out, err, rc = run_script(SCRIPT, "llm-judge")
check(rc == 0, "llm-judge exits 0")
check(out.startswith("LLMJudgeColumnConfig:"), "starts with class name")
check("  column_type: llm-judge" in out, "column_type value")
check("  description:" in out, "has description")
check("  fields:" in out, "has fields header")
# Score nested expansion
check("    scores:" in out, "scores field present")
check("      type: list[Score]" in out, "scores type is list[Score]")
check("      schema (Score):" in out, "Score schema expanded")
check("        name:" in out, "Score.name field at indent 8")
check("        description:" in out, "Score.description field at indent 8")
check("        options:" in out, "Score.options field at indent 8")
check("          type: dict[int | str, str]" in out, "Score.options type at indent 10")
# ImageContext nested expansion (inherited from LLMTextColumnConfig)
check("      schema (ImageContext):" in out, "ImageContext schema expanded")
check("        modality:" in out, "ImageContext.modality field")
check("        data_type:" in out, "ImageContext.data_type field")
# Enum expansion
check("      values: [NONE, LAST_MESSAGE, ALL_MESSAGES]" in out, "TraceType enum values")

# ---- single entry: llm-text (ImageContext expansion) ----
print("\nsingle entry: llm-text:")
out, err, rc = run_script(SCRIPT, "llm-text")
check(rc == 0, "llm-text exits 0")
check(out.startswith("LLMTextColumnConfig:"), "starts with class name")
check("      schema (ImageContext):" in out, "ImageContext expanded")
check("        modality:" in out, "ImageContext.modality at indent 8")
check("          values: [IMAGE]" in out, "Modality enum values for modality field")
check("          values: [URL, BASE64]" in out, "ModalityDataType enum values")
check("          values: [PNG, JPG, JPEG, GIF, WEBP]" in out, "ImageFormat enum values")
# Verify scores is NOT present in llm-text (it's only in llm-judge)
check("scores:" not in out, "scores not in llm-text")

# ---- single entry: sampler (enum expansion, no discriminated union expansion) ----
print("\nsingle entry: sampler:")
out, err, rc = run_script(SCRIPT, "sampler")
check(rc == 0, "sampler exits 0")
check(out.startswith("SamplerColumnConfig:"), "starts with class name")
check("    sampler_type:" in out, "sampler_type field present")
check("      type: SamplerType" in out, "sampler_type type is SamplerType")
# Enum values for SamplerType
check("      values: [BERNOULLI, " in out, "SamplerType enum values start with BERNOULLI")
check("CATEGORY" in out, "CATEGORY in enum values")
check("UNIFORM" in out, "UNIFORM in enum values")
check("UUID]" in out, "UUID at end of enum values")
# params should NOT be expanded (it's a discriminated union of 14 BaseModel subclasses)
check(
    "schema (CategorySamplerParams):" not in out,
    "params not expanded to CategorySamplerParams",
)
check(
    "schema (UniformSamplerParams):" not in out,
    "params not expanded to UniformSamplerParams",
)

# ---- single entry: llm-code (CodeLang enum expansion) ----
print("\nsingle entry: llm-code:")
out, err, rc = run_script(SCRIPT, "llm-code")
check(rc == 0, "llm-code exits 0")
check("    code_lang:" in out, "code_lang field present")
check("PYTHON" in out, "PYTHON in CodeLang values")
check("JAVASCRIPT" in out, "JAVASCRIPT in CodeLang values")

# ---- single entry: validation (ValidatorType enum) ----
print("\nsingle entry: validation:")
out, err, rc = run_script(SCRIPT, "validation")
check(rc == 0, "validation exits 0")
check("    validator_type:" in out, "validator_type field present")
check("      values: [CODE, LOCAL_CALLABLE, REMOTE]" in out, "ValidatorType enum values")

# ---- single entry: custom (GenerationStrategy enum) ----
print("\nsingle entry: custom:")
out, err, rc = run_script(SCRIPT, "custom")
check(rc == 0, "custom exits 0")
check("      values: [CELL_BY_CELL, FULL_COLUMN]" in out, "GenerationStrategy enum values")

# ---- all ----
print("\nall output:")
out, err, rc = run_script(SCRIPT, "all")
check(rc == 0, "all exits 0")
check("# Data Designer Column Types Reference" in out, "header title present")
check("# 10 types discovered" in out, "type count in header")
for t in expected_types:
    # Each type should appear as "  column_type: <type>"
    check(f"  column_type: {t}" in out, f"all output contains column_type: {t}")

# ---- invalid type ----
print("\ninvalid type:")
out, err, rc = run_script(SCRIPT, "nonexistent", expect_fail=True)
check(rc == 1, "invalid type exits 1")
check("Error:" in err, "error message on stderr")
check("nonexistent" in err, "mentions the invalid type")
check("Available types:" in err, "lists available types")

# ---- case sensitivity (column_info is case-sensitive) ----
print("\ncase sensitivity:")
out, err, rc = run_script(SCRIPT, "LLM-TEXT", expect_fail=True)
check(rc == 1, "uppercase LLM-TEXT fails (case-sensitive)")

# ---- too many args ----
print("\ntoo many args:")
out, err, rc = run_script(SCRIPT, "llm-text", "extra", expect_fail=True)
check(rc == 1, "too many args exits 1")
check("Usage:" in out, "shows help on too many args")


# ===========================================================================
# PART 4: Integration tests — get_sampler_info.py
# ===========================================================================

section("Integration: get_sampler_info.py")

SCRIPT = "get_sampler_info.py"

# ---- help ----
print("\nhelp output:")
out, err, rc = run_script(SCRIPT, "--help")
check(rc == 0, "sampler help exits 0")
check("sampler_type" in out, "help mentions sampler_type")
check("params_class" in out, "help mentions params_class")

# ---- list ----
print("\nlist output:")
out, err, rc = run_script(SCRIPT, "--list")
check(rc == 0, "sampler list exits 0")
expected_samplers = [
    "bernoulli",
    "bernoulli_mixture",
    "binomial",
    "category",
    "datetime",
    "gaussian",
    "person",
    "person_from_faker",
    "poisson",
    "scipy",
    "subcategory",
    "timedelta",
    "uniform",
    "uuid",
]
for s in expected_samplers:
    check(s in out, f"sampler list contains '{s}'")
check("CategorySamplerParams" in out, "list has CategorySamplerParams class")

# ---- single entry: category ----
print("\nsingle entry: category:")
out, err, rc = run_script(SCRIPT, "category")
check(rc == 0, "category exits 0")
check(out.startswith("CategorySamplerParams:"), "starts with class name")
# sampler_type should be UPPERCASE
check("  sampler_type: CATEGORY" in out, "sampler_type displayed as CATEGORY (uppercase)")
check("    values:" in out, "values field present")
check("    weights:" in out, "weights field present")

# ---- case insensitivity (sampler_info is case-insensitive) ----
print("\ncase insensitivity:")
out_lower, _, rc1 = run_script(SCRIPT, "category")
out_upper, _, rc2 = run_script(SCRIPT, "CATEGORY")
check(rc1 == 0 and rc2 == 0, "both cases succeed")
check(out_lower == out_upper, "case-insensitive: same output for category/CATEGORY")

out_mixed, _, rc3 = run_script(SCRIPT, "Category")
check(rc3 == 0, "mixed case succeeds")
check(out_mixed == out_lower, "mixed case same output")

# ---- all ----
print("\nall output:")
out, err, rc = run_script(SCRIPT, "all")
check(rc == 0, "sampler all exits 0")
check("# Data Designer Sampler Types Reference" in out, "header title")
check(f"# {len(expected_samplers)} types discovered" in out, "type count in header")
for s in expected_samplers:
    check(f"  sampler_type: {s.upper()}" in out, f"all has sampler_type: {s.upper()}")

# ---- invalid type ----
print("\ninvalid type:")
_, err, rc = run_script(SCRIPT, "nonexistent", expect_fail=True)
check(rc == 1, "invalid sampler exits 1")
check("Error:" in err, "error on stderr")


# ===========================================================================
# PART 5: Integration tests — get_validator_info.py
# ===========================================================================

section("Integration: get_validator_info.py")

SCRIPT = "get_validator_info.py"

# ---- help ----
print("\nhelp output:")
out, err, rc = run_script(SCRIPT, "--help")
check(rc == 0, "validator help exits 0")
check("validator_type" in out, "help mentions validator_type")

# ---- list ----
print("\nlist output:")
out, err, rc = run_script(SCRIPT, "--list")
check(rc == 0, "validator list exits 0")
expected_validators = ["code", "local_callable", "remote"]
for v in expected_validators:
    check(v in out, f"validator list contains '{v}'")

# ---- single entry: code (CodeLang enum expansion) ----
print("\nsingle entry: code:")
out, err, rc = run_script(SCRIPT, "code")
check(rc == 0, "code validator exits 0")
check(out.startswith("CodeValidatorParams:"), "starts with class name")
check("  validator_type: CODE" in out, "validator_type displayed as CODE (uppercase)")
check("    code_lang:" in out, "code_lang field present")
check("      values:" in out, "CodeLang enum values present")
check("PYTHON" in out, "PYTHON in CodeLang values")
check("SQL_POSTGRES" in out, "SQL_POSTGRES in CodeLang values")

# ---- case insensitivity ----
print("\ncase insensitivity:")
out1, _, rc1 = run_script(SCRIPT, "code")
out2, _, rc2 = run_script(SCRIPT, "CODE")
check(rc1 == 0 and rc2 == 0, "both cases succeed")
check(out1 == out2, "case-insensitive: same output")

# ---- all ----
print("\nall output:")
out, err, rc = run_script(SCRIPT, "all")
check(rc == 0, "validator all exits 0")
check("# Data Designer Validator Types Reference" in out, "header title")
check(f"# {len(expected_validators)} types discovered" in out, "type count")

# ---- invalid ----
print("\ninvalid type:")
_, err, rc = run_script(SCRIPT, "nonexistent", expect_fail=True)
check(rc == 1, "invalid validator exits 1")


# ===========================================================================
# PART 6: Integration tests — get_processor_info.py
# ===========================================================================

section("Integration: get_processor_info.py")

SCRIPT = "get_processor_info.py"

# ---- help ----
print("\nhelp output:")
out, err, rc = run_script(SCRIPT, "--help")
check(rc == 0, "processor help exits 0")
check("processor_type" in out, "help mentions processor_type")

# ---- list ----
print("\nlist output:")
out, err, rc = run_script(SCRIPT, "--list")
check(rc == 0, "processor list exits 0")
expected_processors = ["drop_columns", "schema_transform"]
for p in expected_processors:
    check(p in out, f"processor list contains '{p}'")

# ---- single entry: drop_columns ----
print("\nsingle entry: drop_columns:")
out, err, rc = run_script(SCRIPT, "drop_columns")
check(rc == 0, "drop_columns exits 0")
check(out.startswith("DropColumnsProcessorConfig:"), "starts with class name")
# processor_info uses uppercase_value=False
check(
    "  processor_type: drop_columns" in out,
    "processor_type is lowercase (not uppercased)",
)
check("    column_names:" in out, "column_names field present")
# BuildStage enum expansion
check("      values:" in out, "BuildStage enum values present")

# ---- single entry: schema_transform ----
print("\nsingle entry: schema_transform:")
out, err, rc = run_script(SCRIPT, "schema_transform")
check(rc == 0, "schema_transform exits 0")
check(out.startswith("SchemaTransformProcessorConfig:"), "starts with class name")
check("    template:" in out, "template field present")

# ---- case insensitivity ----
print("\ncase insensitivity:")
out1, _, rc1 = run_script(SCRIPT, "drop_columns")
out2, _, rc2 = run_script(SCRIPT, "DROP_COLUMNS")
check(rc1 == 0 and rc2 == 0, "both cases succeed")
check(out1 == out2, "case-insensitive: same output")

# ---- all ----
print("\nall output:")
out, err, rc = run_script(SCRIPT, "all")
check(rc == 0, "processor all exits 0")
check("# Data Designer Processor Types Reference" in out, "header title")

# ---- invalid ----
print("\ninvalid type:")
_, err, rc = run_script(SCRIPT, "nonexistent", expect_fail=True)
check(rc == 1, "invalid processor exits 1")


# ===========================================================================
# PART 7: Cross-cutting tests — YAML structure validation
# ===========================================================================

section("Cross-cutting: YAML structure validation")


def validate_yaml_structure(output: str, script_name: str) -> None:
    """Validate common YAML structure in output from any info script."""
    lines = output.rstrip().split("\n")

    # First line should be ClassName:
    check(
        re.match(r"^\w+:", lines[0]) is not None,
        f"{script_name}: first line is ClassName:",
    )

    # Should have fields: header
    check(
        any(line == "  fields:" for line in lines),
        f"{script_name}: has 'fields:' at indent 2",
    )

    # Every field name should be at indent 4 under fields:
    in_fields = False
    field_names = []
    for line in lines:
        if line == "  fields:":
            in_fields = True
            continue
        if in_fields and re.match(r"^    \w[\w_]*:$", line):
            field_names.append(line.strip().rstrip(":"))

    check(len(field_names) > 0, f"{script_name}: has at least one field")

    # Every field should have a type: line
    for i, line in enumerate(lines):
        if re.match(r"^    \w[\w_]*:$", line) and in_fields:
            # Next non-empty line should contain "type:"
            if i + 1 < len(lines):
                check(
                    "type:" in lines[i + 1],
                    f"{script_name}: field '{line.strip().rstrip(':')}' has type line",
                )

    # No schema line should appear without a corresponding nested model
    for line in lines:
        if "schema (" in line:
            match = re.search(r"schema \((\w+)\):", line)
            check(
                match is not None,
                f"{script_name}: schema line has valid format",
            )

    # No values: line should be empty brackets
    for line in lines:
        if "values:" in line:
            check(
                "values: []" not in line,
                f"{script_name}: values: is not empty",
            )


print("\nValidating YAML structure for each script + type:")
for script, types_to_check in [
    (
        "get_column_info.py",
        ["llm-text", "llm-judge", "sampler", "expression", "validation", "custom"],
    ),
    ("get_sampler_info.py", ["category", "uniform", "person"]),
    ("get_validator_info.py", ["code", "remote"]),
    ("get_processor_info.py", ["drop_columns", "schema_transform"]),
]:
    for t in types_to_check:
        out, _, rc = run_script(script, t)
        if rc == 0:
            validate_yaml_structure(out, f"{script} {t}")


# ===========================================================================
# PART 8: Nested expansion consistency tests
# ===========================================================================

section("Nested expansion consistency")

# ImageContext should be expanded identically across all LLM column types
print("\nImageContext expansion consistency:")
llm_types_with_imagecontext = ["llm-text", "llm-code", "llm-judge", "llm-structured"]
imagecontext_blocks = {}
for t in llm_types_with_imagecontext:
    out, _, rc = run_script("get_column_info.py", t)
    if rc == 0 and "schema (ImageContext):" in out:
        # Extract the ImageContext block
        start = out.index("schema (ImageContext):")
        # Find the next field at indent 4 (sibling) or end of output
        rest = out[start:]
        lines = rest.split("\n")
        block_lines = [lines[0]]
        for line in lines[1:]:
            # Stop at a line that's at indent <= 6 (sibling or parent level)
            if line and not line.startswith("        "):
                break
            block_lines.append(line)
        imagecontext_blocks[t] = "\n".join(block_lines)

if len(imagecontext_blocks) >= 2:
    reference = list(imagecontext_blocks.values())[0]
    for t, block in imagecontext_blocks.items():
        check(block == reference, f"ImageContext block identical in {t}")
else:
    check(False, "ImageContext found in at least 2 LLM types")


# ===========================================================================
# PART 9: Indentation consistency tests
# ===========================================================================

section("Indentation consistency")


def check_indentation(output: str, label: str) -> None:
    """Verify indentation is always a multiple of 2 spaces."""
    for i, line in enumerate(output.split("\n"), 1):
        if not line.strip():
            continue
        leading = len(line) - len(line.lstrip())
        check(
            leading % 2 == 0,
            f"{label} line {i}: indent {leading} is multiple of 2",
        )


print("\nIndentation check for representative outputs:")
for script, arg in [
    ("get_column_info.py", "llm-judge"),
    ("get_sampler_info.py", "category"),
    ("get_validator_info.py", "code"),
    ("get_processor_info.py", "drop_columns"),
]:
    out, _, _ = run_script(script, arg)
    check_indentation(out, f"{script} {arg}")


# ===========================================================================
# Summary
# ===========================================================================

section("RESULTS")
total = _passed + _failed
print(f"\n  Total:  {total}")
print(f"  Passed: {_passed}")
print(f"  Failed: {_failed}")

if _errors:
    print("\n  Failed tests:")
    for e in _errors:
        print(f"    - {e}")

print()
sys.exit(0 if _failed == 0 else 1)
