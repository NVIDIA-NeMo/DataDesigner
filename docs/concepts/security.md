# Security

Data Designer can run in two very different trust models:

- **Trusted / monolithic**: The same user or team writes the config and runs the engine.
- **Untrusted / shared execution**: One user submits a config and a different process, service, or team executes it.

That distinction matters for features that evaluate user-supplied configuration at runtime, such as Jinja template rendering. In a trusted local workflow, broader template flexibility may be acceptable. In a shared-service deployment, user-supplied Jinja becomes part of the engine's remote code execution surface. A template sandbox escape would execute inside the process running Data Designer.

See [Deployment Options](deployment-options.md) for the architectures where that trust boundary changes.

## Jinja Rendering Modes

Data Designer exposes the renderer choice through `RunConfig`:

```python
import data_designer.config as dd

run_config = dd.RunConfig(
    jinja_rendering_engine=dd.JinjaRenderingEngine.SECURE,
)
```

`SECURE` is the default. Opt into `NATIVE` only when you are comfortable treating the config author and the engine operator as the same trust domain.

| Mode | What it uses | Best fit |
|------|---------------|----------|
| `SECURE` | Data Designer's hardened renderer built on top of Jinja2's sandbox | Shared services, microservices, internal platforms, or any deployment where config submission is separated from execution |
| `NATIVE` | Jinja2's built-in sandbox with Data Designer's variable whitelist | Local library usage and other trusted, monolithic workflows that want broader Jinja behavior |

!!! warning "Treat untrusted Jinja as a security boundary"
    If many users can submit configs to one engine, or if configs are accepted over an API and executed elsewhere, keep `JinjaRenderingEngine.SECURE`. In that model, Jinja templates are no longer just prompt-formatting helpers. They are untrusted user programs being evaluated by your engine.

## Compatibility Matrix

`NATIVE` is not an unrestricted Python template engine. The matrix below shows what each mode permits, restricts, or adds on top of Jinja2's standard sandbox behavior.

| Capability | `NATIVE` | `SECURE` |
|------|------|----------|
| Jinja2 `ImmutableSandboxedEnvironment` baseline | Yes | Yes |
| References to explicitly provided dataset variables only | Yes | Yes |
| Standard Jinja built-in filter set | Yes | Subset only |
| Data Designer `jsonpath` filter | Yes | Yes |
| `import`, `macro`, `set`, `extends`, `block` support | Yes | No |
| Nested or recursive `for` loops | Yes | No |
| Unbounded AST complexity | Yes | No |
| Template context sanitized to JSON-compatible types before render | No | Yes |
| Empty, oversized, or built-in-like rendered output is permitted | Yes | No |

## What `SECURE` Adds on Top of Standard Jinja Sandbox

The `SECURE` renderer uses a hardened environment implemented in the [renderer source file on GitHub](https://github.com/NVIDIA-NeMo/DataDesigner/blob/v0.5.6/packages/data-designer-engine/src/data_designer/engine/processing/ginja/environment.py). Compared with the standard Jinja sandbox, it adds several additional controls.

### Record Sanitization Before Render

Before rendering, `SECURE` forces template context through a JSON-compatible serialization step. That means remote templates operate on plain data, not arbitrary Python objects.

```python
# Intended shape for remote template context
record = {
    "user": {
        "name": "alice",
        "roles": ["admin", "reviewer"],
    }
}
```

```python
# Not the kind of server-side object SECURE wants to expose directly
record = {
    "user": SomePythonObject(...),
}
```

In a remote execution setting, that matters because rich Python objects can expose attributes, methods, or descriptors that were never meant to be reachable from user-authored templates. The relevant history here is mostly Jinja sandbox-escape CVEs rather than Python interpreter CVEs: Jinja's [sandbox security considerations](https://jinja.palletsprojects.com/en/stable/sandbox/) explicitly note that the sandbox is not a complete security boundary, and fixes for untrusted-template execution have included [`str.format` sandbox escape (CVE-2016-10745)](https://nvd.nist.gov/vuln/detail/CVE-2016-10745), [`str.format_map` sandbox escape (CVE-2019-10906)](https://github.com/advisories/GHSA-462w-v97r-4m45), [indirect `str.format` reference escape (CVE-2024-56326)](https://nvd.nist.gov/vuln/detail/CVE-2024-56326), and [`|attr`-based access to `format` (CVE-2025-27516)](https://nvd.nist.gov/vuln/detail/CVE-2025-27516). The broader `instance -> __class__ -> mro -> modules -> os` style chain is better understood as server-side template injection and sandbox-escape technique literature; PortSwigger's [server-side template injection research](https://portswigger.net/research/server-side-template-injection) is a useful reference for that model.

### Filter Allowlist

`SECURE` does not expose the full Jinja filter surface. It keeps a small approved subset plus the Data Designer `jsonpath` filter.

```jinja
{{ payload | jsonpath("$.customer.name") }}
```

```jinja
{{ items | join(", ") }}
```

The first example is supported. The second is broader Jinja behavior that `NATIVE` permits but `SECURE` intentionally rejects. In a shared engine, narrowing the filter surface reduces the number of operations that user templates can compose into server-side execution.

### Template Features Removed

`SECURE` rejects `import`, `macro`, `set`, `extends`, and `block`.

```jinja
{% macro render_name(name) %}{{ name }}{% endmacro %}
{{ render_name(customer_name) }}
```

```jinja
{% set temp = user_id %}
{{ temp }}
```

Those features are useful in trusted authoring environments, but they also make user templates more expressive and stateful. In a remote execution model, `SECURE` intentionally narrows the language so templates stay closer to data interpolation than to a reusable programming layer.

### Loop Restrictions

`SECURE` rejects recursive loops and nested `for` loops.

```jinja
{% for row in rows %}
  {% for item in row %}
    {{ item }}
  {% endfor %}
{% endfor %}
```

Nested and recursive loops are especially risky in shared execution because they can amplify compute cost and output size in ways that are hard to reason about from the outside.

### AST Complexity Limits

`SECURE` statically analyzes the parsed Jinja AST and rejects templates that exceed the current limits of 600 nodes or depth 10.

```jinja
{% if a %}
  {% if b %}
    {% if c %}
      {{ value }}
    {% endif %}
  {% endif %}
{% endif %}
```

This is not about any one feature being unsafe by itself. It is about limiting how much control flow and composition untrusted templates can pack into a single server-side render operation.

### `self` References Blocked

`SECURE` rejects references to `self`.

```jinja
{{ self }}
```

The point is to avoid exposing template internals back to the submitter. In a remote setting, even accidental access to those internals is unnecessary surface area.

### Rendered Output Guards

`SECURE` validates rendered output after template execution. It rejects empty output, very large output, and strings that look like Python built-in or function representations.

```jinja
{{ "" }}
```

```text
<built-in method ...>
<function ...>
```

These checks matter because not all bad outcomes come from parse-time behavior. Some templates are syntactically valid but still produce output that is clearly broken, oversized, or revealing internal implementation details.

### Sanitized User-Facing Errors

At the engine boundary, `SECURE` normalizes most template failures into a generic invalid-template message.

```text
User provided prompt generation template is invalid.
```

That matters in remote execution because exception details can leak information about server-side implementation, supported objects, or internal execution paths that untrusted users do not need to see.

These controls exist because the standard sandbox is a good baseline, but shared-service deployments need a narrower and more defensive execution model.

## Why This Matters in Multi-User Deployments

The security posture changes as soon as config submission and execution are separated.

Examples:

- A centralized Data Designer service accepts configs from many users.
- An internal platform lets users upload or edit configs that are executed by a background worker.
- A REST API accepts Jinja-containing configs and runs them on server-side infrastructure.

In those environments, templates are no longer just local convenience syntax. They are untrusted input being evaluated by infrastructure the submitter does not control. In practice, that makes Jinja rendering a remote code execution concern, which is why `SECURE` exists and why it remains the default.

If you are deciding between local library usage and a shared service model, read [Deployment Options](deployment-options.md). The library patterns are often still "trusted" deployments. The shared microservice pattern is not.

## When To Use `NATIVE`

Use `NATIVE` when all of the following are true:

- The person submitting the config is also the person running the engine, or they are in the same trusted operational boundary.
- You want broader standard Jinja behavior than `SECURE` allows.
- You understand that this is a flexibility tradeoff, not the safer default.

For example, this is often reasonable in a notebook, local script, or other single-user library workflow.

## Related Reading

- [Deployment Options](deployment-options.md)
- [Run Config Reference](../code_reference/run_config.md)
