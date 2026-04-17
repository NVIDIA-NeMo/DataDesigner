# Security

Data Designer can run in two very different trust models:

- **Trusted / monolithic**: The same user or team writes the config and runs the engine.
- **Untrusted / shared execution**: One user submits a config and a different process, service, or team executes it.

That distinction matters for Jinja template rendering. In a trusted local workflow, broader template flexibility may be acceptable. In a shared-service deployment, user-supplied Jinja becomes part of the engine's remote code execution surface. A template sandbox escape would execute inside the process running Data Designer.

See [Deployment Options](deployment-options.md) for the architectures where that trust boundary changes.

!!! warning "Treat untrusted Jinja as a security boundary"
    If many users can submit configs to one engine, or if configs are accepted over an API and executed elsewhere, keep `JinjaRenderingEngine.SECURE`. In that model, Jinja templates are no longer just prompt-formatting helpers. They are untrusted user programs being evaluated by your engine.

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

## What Both Modes Already Do

`NATIVE` is not an unrestricted Python template engine. Both modes still provide some baseline containment:

- Both use Jinja2's `ImmutableSandboxedEnvironment`.
- Both only allow references to explicitly provided dataset variables.
- Both still reject sandboxed operations that Jinja2 itself disallows.

The difference is that `SECURE` adds another layer of Data Designer-specific restrictions on top of that baseline.

## What `SECURE` Adds on Top of Standard Jinja Sandbox

The `SECURE` renderer uses a hardened environment implemented in `packages/data-designer-engine/src/data_designer/engine/processing/ginja/environment.py`. Compared with the standard Jinja sandbox, it adds several additional controls:

- **Record sanitization before render**: Template context is serialized and deserialized into basic JSON-compatible types before rendering. This reduces the chance that unexpected Python objects expose attributes or callables to the template.
- **Filter allowlist**: Only a limited set of filters is available. This includes a small set of built-in Jinja filters plus Data Designer's custom `jsonpath` filter.
- **Unsupported template features removed**: `SECURE` rejects `import`, `macro`, `set`, `extends`, and `block`.
- **Loop restrictions**: Recursive loops and nested `for` loops are rejected.
- **AST complexity limits**: Templates are statically analyzed and rejected if they exceed the current complexity thresholds of 600 AST nodes or depth 10.
- **`self` references blocked**: Templates cannot reference `self`, which reduces access to template internals.
- **Rendered output guards**: Empty output is rejected, very large output is rejected, and rendered strings that look like Python built-in or function representations are rejected.
- **Sanitized user-facing errors**: At the engine boundary, most template errors are normalized to a generic invalid-template message instead of surfacing internal exception details.

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
