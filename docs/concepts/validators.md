# Validators

Validators are quality assurance mechanisms in Data Designer that check generated content against rules and return structured pass/fail results. They enable automated verification of data correctness, code quality, and adherence to specifications.

!!! note "Quality Gates for Generated Data"
    Validators act as **quality gates** in your generation pipeline. Use them to filter invalid records, score code quality, verify format compliance, or integrate with external validation services.

## Overview

Validation columns execute validation logic against target columns and produce structured results indicating:

- **`is_valid`**: Boolean pass/fail status
- **Additional metadata**: Error messages, scores, severity levels, and custom fields

Validators support three execution strategies:

1. **Code validation**: Lint and check Python or SQL code using industry-standard tools
2. **Local callable validation**: Execute custom Python functions for flexible validation logic
3. **Remote validation**: Send data to HTTP endpoints for external validation services

## Validator Types

### 🐍 Code Validator (Python)

The Python code validator runs generated Python code through **Ruff**, a fast Python linter that checks for syntax errors, undefined variables, and code quality issues.

**Configuration:**

```python
from data_designer.config.validator_params import CodeValidatorParams
from data_designer.config.utils.code_lang import CodeLang

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

### 🗄️ Code Validator (SQL)

The SQL code validator uses **SQLFluff**, a dialect-aware SQL linter that checks query syntax and structure.

**Configuration:**

```python
from data_designer.config.validator_params import CodeValidatorParams
from data_designer.config.utils.code_lang import CodeLang

# Supports multiple SQL dialects
validator_params = CodeValidatorParams(code_lang=CodeLang.SQL_POSTGRES)
# Or: SQL_ANSI, SQL_MYSQL, SQL_SQLITE, SQL_TSQL, SQL_BIGQUERY
```

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

The local callable validator executes custom Python functions for flexible validation logic. This validator is **only available when running Data Designer locally**—it cannot be serialized to YAML or used in remote execution environments.

**Configuration:**

```python
from data_designer.config.validator_params import LocalCallableValidatorParams
import pandas as pd

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

The remote validator sends data to HTTP endpoints for validation-as-a-service. Use this for:

- External linting services
- Security scanners
- Domain-specific validators
- Proprietary validation systems

**Configuration:**

```python
from data_designer.config.validator_params import RemoteValidatorParams

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
from data_designer.essentials import DataDesignerConfigBuilder
from data_designer.config.validator_params import CodeValidatorParams
from data_designer.config.utils.code_lang import CodeLang

builder = DataDesignerConfigBuilder()

# Generate Python code
builder.add_column(
    name="sorting_algorithm",
    column_type="llm-code",
    prompt="Write a Python function to sort a list using bubble sort.",
    code_lang="python",
    model_alias="my-model"
)

# Validate the generated code
builder.add_column(
    name="code_validation",
    column_type="validation",
    target_columns=["sorting_algorithm"],
    validator_type="code",
    validator_params=CodeValidatorParams(code_lang=CodeLang.PYTHON),
    batch_size=10  # Process 10 records per batch
)
```

The `target_columns` parameter specifies which columns to validate. All target columns are passed to the validator together (except for code validators, which process each column separately).

### Multiple Column Validation

Validate multiple columns simultaneously:

```python
builder.add_column(
    name="multi_column_validation",
    column_type="validation",
    target_columns=["column_a", "column_b", "column_c"],
    validator_type="remote",
    validator_params=RemoteValidatorParams(
        endpoint_url="https://api.example.com/validate"
    )
)
```

**Note**: Code validators always process each target column separately, even when multiple columns are specified. Other validators receive all target columns together.

### SQL Validation Example

```python
# Generate SQL queries
builder.add_column(
    name="analytics_query",
    column_type="llm-code",
    prompt="Write a SQL query to calculate {{ metric }} grouped by {{ dimension }}.",
    code_lang="sql:postgres",
    model_alias="my-model"
)

# Validate SQL syntax
builder.add_column(
    name="query_validation",
    column_type="validation",
    target_columns=["analytics_query"],
    validator_type="code",
    validator_params=CodeValidatorParams(code_lang=CodeLang.SQL_POSTGRES),
    batch_size=5
)
```

### Filtering with Validation Results

Use validation results to filter your dataset:

```python
# After generation
result = designer.run()
df = result.data

# Keep only valid code
valid_code = df[df["code_validation"].apply(lambda x: x["is_valid"])]

# Filter by quality score
high_quality = df[df["code_validation"].apply(lambda x: x.get("python_linter_score", 0) >= 8.0)]

# Check severity levels
no_errors = df[df["code_validation"].apply(
    lambda x: x.get("python_linter_severity") not in ["error", "fatal"]
)]
```

## Validation Column Configuration

Validation columns are configured using `ValidationColumnConfig`:

```python
from data_designer.config.column_configs import ValidationColumnConfig
from data_designer.config.validator_params import CodeValidatorParams
from data_designer.config.utils.code_lang import CodeLang

config = ValidationColumnConfig(
    name="my_validation",
    target_columns=["code_column"],
    validator_type="code",
    validator_params=CodeValidatorParams(code_lang=CodeLang.PYTHON),
    batch_size=10,
    drop=False  # Include validation results in output
)
```

### Configuration Parameters

- **`name`**: Column name for validation results
- **`target_columns`**: List of columns to validate (must exist before validation runs)
- **`validator_type`**: Validator strategy (`"code"`, `"local_callable"`, or `"remote"`)
- **`validator_params`**: Type-specific configuration object
- **`batch_size`**: Records per validation batch (default: 10)
- **`drop`**: Whether to exclude validation results from final output (default: `False`)

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

## Common Patterns

### Quality Scoring Pipeline

```python
# Generate code
builder.add_column(
    name="implementation",
    column_type="llm-code",
    prompt="Implement {{ task }}",
    code_lang="python",
    model_alias="code-model"
)

# Validate code quality
builder.add_column(
    name="quality_check",
    column_type="validation",
    target_columns=["implementation"],
    validator_type="code",
    validator_params=CodeValidatorParams(code_lang=CodeLang.PYTHON)
)

# Score with LLM judge
builder.add_column(
    name="human_readability",
    column_type="llm-judge",
    prompt="Rate the readability of this code:\n{{ implementation }}",
    scores=[
        Score(
            name="readability",
            description="Code readability",
            options={"1": "Poor", "2": "Fair", "3": "Good", "4": "Excellent"}
        )
    ],
    model_alias="judge-model"
)
```

### Multi-Stage Validation

```python
# Stage 1: Syntax validation
builder.add_column(
    name="syntax_check",
    column_type="validation",
    target_columns=["query"],
    validator_type="code",
    validator_params=CodeValidatorParams(code_lang=CodeLang.SQL_POSTGRES)
)

# Stage 2: Semantic validation (remote service)
builder.add_column(
    name="semantic_check",
    column_type="validation",
    target_columns=["query"],
    validator_type="remote",
    validator_params=RemoteValidatorParams(
        endpoint_url="https://api.example.com/validate-sql-semantics"
    )
)
```

### Development vs. Production

Use `drop=True` to include validation during development but exclude from production:

```python
builder.add_column(
    name="debug_validation",
    column_type="validation",
    target_columns=["generated_content"],
    validator_type="code",
    validator_params=CodeValidatorParams(code_lang=CodeLang.PYTHON),
    drop=True  # Generate but don't include in final output
)
```

## Best Practices

### 1. Validate Early

Place validation columns immediately after generation to catch issues early:

```python
builder.add_column(name="code", ...)
builder.add_column(name="code_validation", target_columns=["code"], ...)  # Immediate validation
builder.add_column(name="documentation", prompt="Document {{ code }}", ...)  # Use validated code
```

### 2. Use Appropriate Batch Sizes

- Start with small batches (5-10) for expensive validators
- Increase batch size if validation is fast and memory permits
- Monitor memory usage and adjust accordingly

### 3. Handle Validation in Analysis

Always check validation results before using generated data:

```python
result = designer.run()
df = result.data

# Log validation statistics
validation_col = df["my_validation"]
total = len(validation_col)
valid = validation_col.apply(lambda x: x["is_valid"]).sum()
print(f"Validation: {valid}/{total} passed ({100*valid/total:.1f}%)")

# Filter to valid records only
clean_data = df[validation_col.apply(lambda x: x["is_valid"])]
```

### 4. Combine Multiple Validators

Use different validators for different quality aspects:

```python
builder.add_column(name="syntax_check", validator_type="code", ...)  # Syntax correctness
builder.add_column(name="security_check", validator_type="remote", ...)  # Security scan
builder.add_column(name="quality_score", column_type="llm-judge", ...)  # Human-like quality
```

### 5. Schema Validation for Custom Validators

Always provide `output_schema` for local callable and remote validators to catch unexpected output formats:

```python
validator_params = LocalCallableValidatorParams(
    validation_function=my_function,
    output_schema={...}  # Define expected structure
)
```

## Advanced Usage

### Custom Validation Logic (Local Callable)

```python
import pandas as pd
from data_designer.config.validator_params import LocalCallableValidatorParams

def validate_email_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Check if emails use allowed domains."""
    allowed_domains = ["@company.com", "@company.org"]

    result = pd.DataFrame()
    result["is_valid"] = df["email"].apply(
        lambda email: any(domain in email for domain in allowed_domains)
    )
    result["domain"] = df["email"].apply(lambda email: email.split("@")[1] if "@" in email else "")
    result["error"] = result["is_valid"].apply(
        lambda valid: "" if valid else f"Domain must be one of {allowed_domains}"
    )
    return result

builder.add_column(
    name="email_validation",
    column_type="validation",
    target_columns=["email"],
    validator_type="local_callable",
    validator_params=LocalCallableValidatorParams(
        validation_function=validate_email_domain
    )
)
```

### Remote Validator with Authentication

For endpoints requiring authentication, configure your HTTP client in the remote service:

```python
# Note: Current implementation doesn't support auth headers directly
# You may need to implement a proxy service that adds authentication
validator_params = RemoteValidatorParams(
    endpoint_url="https://api.example.com/validate",
    max_parallel_requests=2,  # Respect rate limits
    timeout=60.0  # Longer timeout for complex validation
)
```

## Validation Output Structure

All validators return results conforming to this structure:

```python
{
    "is_valid": bool | None,  # Pass/fail status (None if validation couldn't run)
    # ... additional validator-specific fields
}
```

The `is_valid` field is required; all other fields are validator-specific metadata.

### Accessing Validation Results

```python
result = designer.run()
df = result.data

# Access validation column
validation_results = df["my_validation"]

# Extract specific fields
is_valid = validation_results.apply(lambda x: x["is_valid"])
scores = validation_results.apply(lambda x: x.get("python_linter_score", None))

# Convert to separate columns for easier analysis
df["is_valid"] = is_valid
df["linter_score"] = scores
```

## Limitations

### Code Validators

- **Python**: Validates syntax and basic semantics; doesn't execute code or check runtime correctness
- **SQL**: Validates query structure; doesn't check against actual database schemas

### Local Callable Validators

- **Not serializable**: Cannot be saved to YAML or used in distributed environments
- **No sandboxing**: Validation functions run in the same process; be cautious with untrusted code

### Remote Validators

- **Network dependency**: Requires stable internet connection and responsive endpoints
- **Latency**: Slower than local validators due to network overhead
- **Authentication**: Limited support for complex authentication schemes

## See Also

- [Columns Concept](columns.md): Understanding column types and dependencies
- [Column Configuration Reference](../code_reference/column_configs.md): Detailed API documentation
- [Validator Parameters Reference](../code_reference/column_configs.md): Configuration object schemas

