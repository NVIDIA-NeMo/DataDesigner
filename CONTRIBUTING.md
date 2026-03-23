# 🎨✨ Contributing to NeMo Data Designer 🎨✨

The skills and workflows in this repository are for **developing** DataDesigner. If you're looking to **use** DataDesigner to build datasets, see the [product documentation](https://nvidia-nemo.github.io/DataDesigner/) instead.

---

This project uses agent-assisted development. Contributors are expected to use agents for investigation, planning, and implementation. The repository includes [skills and guidance](.agents/) that make agents effective contributors.

Agents accelerate work; humans stay accountable. People make design decisions and own quality — agents help get there faster.

## How to Contribute

1. **Open an issue** using the appropriate [issue template](https://github.com/NVIDIA-NeMo/DataDesigner/issues/new/choose).
2. **Include investigation output.** If you used an agent, paste its diagnostics. If you didn't, include the troubleshooting you tried.
3. **For non-trivial changes, submit a plan** in the issue for review before building. Plans should describe the approach, trade-offs considered, and affected subsystems.
4. **Once approved, implement** using agent-assisted development. See [DEVELOPMENT.md](DEVELOPMENT.md) for local setup and workflow.

## Before You Open an Issue

- Clone the repo and point your agent at it
- Have the agent search docs and existing issues (the `search-docs` and `search-github` skills can help)
- If the agent can't resolve it, include the diagnostics in your issue
- If you didn't use an agent, include the troubleshooting you already tried

## When to Open an Issue

- Real bugs — reproduced or agent-confirmed
- Feature proposals with design context
- Problems that `search-docs` / `search-github` couldn't resolve

## When NOT to Open an Issue

- Questions about how things work — an agent can answer these from the codebase
- Configuration problems — an agent can diagnose these
- "How do I..." requests — try the [product documentation](https://nvidia-nemo.github.io/DataDesigner/) first

---

## Development Skills

The repository includes skills for common development tasks. These are located in [`.agents/skills/`](.agents/skills/) and are automatically discovered by agent harnesses.

| Category      | Skills                             | Purpose                                |
| ------------- | ---------------------------------- | -------------------------------------- |
| Investigation | `search-docs`, `search-github`     | Find information, check for duplicates |
| Development   | `commit`, `create-pr`, `update-pr` | Standard development cycle             |
| Review        | `review-code`                      | Multi-pass code review                 |

---

## Pull Requests

- PRs should link to the issue they address (`Fixes #NNN` or `Closes #NNN`)
- Use the `create-pr` skill for well-formatted PR descriptions, or follow the PR template
- Ensure all checks pass before requesting review:
  ```bash
  make check-all-fix
  make test
  ```

### Pull Request Review Process

- Maintainers will review your PR and may request changes
- Address feedback by pushing additional commits to your branch
- Reply to the feedback comment with a link to the commit that addresses it
- Once approved, a maintainer will merge your PR

---

## Commit Messages

- Use imperative mood ("add feature" not "added feature")
- Keep the subject line under 50 characters (hard limit: 72)
- Reference issue numbers when applicable (`Fixes #123`)

---

## License Headers

All code files must include the NVIDIA copyright header:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

Use `make update-license-headers` to add headers automatically.

## Signing Off on Your Work (DCO)

When contributing, you must agree that you have authored 100% of the content, that you have the necessary rights to the content, and that the content you contribute may be provided under the project license. All contributors are asked to sign the [Developer Certificate of Origin (DCO)](DCO) when submitting their first pull request. The process is automated by a bot that will comment on the pull request.

## Code of Conduct

Data Designer follows the Contributor Covenant Code of Conduct. Please read our complete [Code of Conduct](CODE_OF_CONDUCT.md) for full details.

---

## Reference

- [AGENTS.md](AGENTS.md) — architecture, layering, design principles
- [STYLEGUIDE.md](STYLEGUIDE.md) — code style, naming, imports, type annotations
- [DEVELOPMENT.md](DEVELOPMENT.md) — local setup, testing, day-to-day workflow
