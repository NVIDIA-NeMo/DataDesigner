# Validators

Validators are quality assurance mechanisms in Data Designer that check generated content against rules and return structured pass/fail results. They enable automated verification of data for correctness, code quality, and adherence to specifications.

!!! note "Quality Gates for Generated Data"
    Validators act as **quality gates** in your generation pipeline. Use them to filter invalid records, score code quality, verify format compliance, or integrate with external validation services.

## Overview

Validation columns execute validation logic against target columns and produce structured results indicating:

- **`is_valid`**: Boolean pass/fail status
- **Additional metadata**: Error messages, scores, severity levels, and custom fields

Validators currently support three execution strategies:

1. **Code validation**: Lint and check Python or SQL code using industry-standard tools
2. **Local callable validation**: Execute custom Python functions for flexible validation logic
3. **Remote validation**: Send data to HTTP endpoints for external validation services

## Validator Types

### 🐍 Python Code Validator

The Python code validator runs generated Python code through [Ruff](https://github.com/astral-sh/ruff), a fast Python linter that checks for syntax errors, undefined variables, and code quality issues.

**Configuration:**

```python
from data_designer.essentials import CodeLang, CodeValidatorParams

validator_params = CodeValidatorParams(code_lang=CodeLang.PYTHON)
```

**Validation Output:**

Each validated record returns:

- **`is_valid`**: `True` if no fatal or error-level issues found
- **`python_linter_score`**: Quality score from 0-10 (based on pylint formula)
- **`python_linter_severity`**: Highest severity level found (`"none"`, `"convention"`, `"refactor"`, `"warning"`, `"error"`, `"fatal"`)
- **`python_linter_messages`**: List of linter messages with line numbers, columns, and descriptions

**Severity Levels:**

- **Fatal**: Syntax errors preventing code execution
- **Error**: Undefined names, invalid syntax
- **Warning**: Code smells and potential issues
- **Refactor**: Simplification opportunities
- **Convention**: Style guide violations

A record is marked valid if it has no messages or only messages at warning/convention/refactor levels.

**Example Validation Result:**

```python
{
    "is_valid": False,
    "python_linter_score": 0,
    "python_linter_severity": "error",
    "python_linter_messages": [
        {
            "type": "error",
            "symbol": "F821",
            "line": 1,
            "column": 7,
            "message": "Undefined name `it`"
        }
    ]
}
```

### 🗄️ SQL Code Validator

The SQL code validator uses [SQLFluff](https://github.com/sqlfluff/sqlfluff), a dialect-aware SQL linter that checks query syntax and structure.

**Configuration:**

```python
from data_designer.essentials import CodeLang, CodeValidatorParams

validator_params = CodeValidatorParams(code_lang=CodeLang.SQL_POSTGRES)
```

!!! tip "Multiple Dialects"
    The SQL code validator supports multiple dialects: `SQL_POSTGRES`, `SQL_ANSI`, `SQL_MYSQL`, `SQL_SQLITE`, `SQL_TSQL` and `SQL_BIGQUERY`.

**Validation Output:**

Each validated record returns:

- **`is_valid`**: `True` if no parsing errors found
- **`error_messages`**: Concatenated error descriptions (empty string if valid)

The validator focuses on parsing errors (PRS codes) that indicate malformed SQL. It also checks for common pitfalls like `DECIMAL` definitions without scale parameters.

**Example Validation Result:**

```python
# Valid SQL
{
    "is_valid": True,
    "error_messages": ""
}

# Invalid SQL
{
    "is_valid": False,
    "error_messages": "PRS: Line 1, Position 1: Found unparsable section: 'NOT SQL'"
}
```

### 🔧 Local Callable Validator

The local callable validator executes custom Python functions for flexible validation logic.

**Configuration:**

```python
import pandas as pd

from data_designer.essentials import LocalCallableValidatorParams

def my_validation_function(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that values are positive.

    Args:
        df: DataFrame with target columns

    Returns:
        DataFrame with is_valid column and optional metadata
    """
    result = pd.DataFrame()
    result["is_valid"] = df["price"] > 0
    result["error_message"] = result["is_valid"].apply(
        lambda valid: "" if valid else "Price must be positive"
    )
    return result

validator_params = LocalCallableValidatorParams(
    validation_function=my_validation_function,
    output_schema={  # Optional: enforce output schema
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "is_valid": {"type": ["boolean", "null"]},
                        "error_message": {"type": "string"}
                    },
                    "required": ["is_valid"]
                }
            }
        }
    }
)
```

**Function Requirements:**

- **Input**: DataFrame with target columns
- **Output**: DataFrame with `is_valid` column (boolean or null)
- **Extra fields**: Any additional columns become validation metadata

The `output_schema` parameter is optional but recommended—it validates the function's output against a JSON schema, catching unexpected return formats.

### 🌐 Remote Validator

The remote validator sends data to HTTP endpoints for validation-as-a-service. This is useful for when you have validation software that needs to run on external compute and you can expose it through a service. Some examples are:

- External linting services
- Security scanners
- Domain-specific validators
- Proprietary validation systems

!!! note "Authentication"
    Currently, the remote validator is only able to perform unauthenticated API calls. When implementing your own service, you can rely on network isolation for security. If you need to reach a service that requires authentication, you should implement a local proxy.

**Configuration:**

```python
from data_designer.essentials import RemoteValidatorParams

validator_params = RemoteValidatorParams(
    endpoint_url="https://api.example.com/validate",
    timeout=30.0,  # Request timeout in seconds
    max_retries=3,  # Retry attempts on failure
    retry_backoff=2.0,  # Exponential backoff factor
    max_parallel_requests=4,  # Concurrent request limit
    output_schema={  # Optional: enforce response schema
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "is_valid": {"type": ["boolean", "null"]},
                        "confidence": {"type": "string"}
                    }
                }
            }
        }
    }
)
```

**Request Format:**

The validator sends POST requests with this structure:

```json
{
    "data": [
        {"column1": "value1", "column2": "value2"},
        {"column1": "value3", "column2": "value4"}
    ]
}
```

**Expected Response Format:**

The endpoint must return:

```json
{
    "data": [
        {
            "is_valid": true,
            "custom_field": "any additional metadata"
        },
        {
            "is_valid": false,
            "custom_field": "more metadata"
        }
    ]
}
```

**Retry Behavior:**

The validator automatically retries on:

- Network errors
- HTTP status codes: 429 (rate limit), 500, 502, 503, 504

Failed requests use exponential backoff: `delay = retry_backoff^attempt`.

**Parallelization:**

Set `max_parallel_requests` to control concurrency. Higher values improve throughput but increase server load. The validator batches requests according to the `batch_size` parameter in the validation column configuration.

## Using Validators in Columns

Add validation columns to your configuration using the builder's `add_column` method:

```python
from data_designer.essentials import (
    CodeValidatorParams,
    CodeLang,
    DataDesignerConfigBuilder,
    LLMCodeColumnConfig,
    ValidationColumnConfig,
)

builder = DataDesignerConfigBuilder()

# Generate Python code
builder.add_column(
    LLMCodeColumnConfig(
        name="sorting_algorithm",
        prompt="Write a Python function to sort a list using bubble sort.",
        code_lang="python",
        model_alias="my-model"
    )
)

# Validate the generated code
builder.add_column(
    ValidationColumnConfig(
        name="code_validation",
        target_columns=["sorting_algorithm"],
        validator_type="code",
        validator_params=CodeValidatorParams(code_lang=CodeLang.PYTHON),
        batch_size=10,
        drop=False,
    )
)
```

The `target_columns` parameter specifies which columns to validate. All target columns are passed to the validator together (except for code validators, which process each column separately).

### Configuration Parameters

See more about parameters used to instantiate `ValidationColumnConfig` in the [code reference](../../code_reference/column_configs/#data_designer.config.column_configs.ValidationColumnConfig).

### Batch Size Considerations

Larger batch sizes improve efficiency but consume more memory:

- **Code validators**: 5-20 records (file I/O overhead)
- **Local callable**: 10-50 records (depends on function complexity)
- **Remote validators**: 1-10 records (network latency, server capacity)

Adjust based on:

- Validator computational cost
- Available memory
- Network bandwidth (for remote validators)
- Server rate limits

If the validation logic uses information from other samples, only samples in the batch will be considered.

### Multiple Column Validation

Validate multiple columns simultaneously:

```python
from data_designer.essentials import RemoteValidatorParams, ValidationColumnConfig

builder.add_column(
    ValidationColumnConfig(
        name="multi_column_validation",
        target_columns=["column_a", "column_b", "column_c"],
        validator_type="remote",
        validator_params=RemoteValidatorParams(
            endpoint_url="https://api.example.com/validate"
        )
    )
)
```

**Note**: Code validators always process each target column separately, even when multiple columns are specified. Local callable and remote validators receive all target columns together.

## See Also

- [Validator Parameters Reference](../code_reference/validator_params.md): Configuration object schemas

