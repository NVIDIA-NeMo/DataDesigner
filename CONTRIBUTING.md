# 🎨✨ Contributing to NeMo Data Designer 🎨✨

Thank you for your interest in contributing to Data Designer!

We welcome contributions from the community and sincerely appreciate your efforts to improve the project. Whether you're fixing a typo, reporting a bug, proposing a new feature, or implementing a major enhancement, your work helps make Data Designer better for everyone 🎉.

This guide will help you get started with the contribution process.

## Table of Contents

- [Getting Started](#getting-started)
- [Ways to Contribute](#ways-to-contribute)
- [Feature Requests](#feature-requests)
- [Development Guide](#development-guide)
- [Code Quality Standards](#code-quality-standards)
- [Submitting Changes](#submitting-changes)
- [Code of Conduct](#code-of-conduct)
- [Signing off on your work](#signing-off-on-your-work)


## Getting Started
👋 Welcome to the Data Designer community! We're excited to have you here.

Whether you're new to the project or ready to dive in, the resources below will help you get oriented and productive quickly:

1. **[README.md](README.md)** – best place to start to learn the basics of the project

2. **[AGENTS.md](AGENTS.md)** – code structure and development guidelines for humans and agents

3. **[Documentation](docs/)** – detailed documentation on Data Designer's capabilities and usage

## Ways to Contribute

There are many ways to contribute to Data Designer:

### 🐛 Bug Fixes

Found a bug? Before reporting, please
1. Verify you're using the latest version: `uv pip install --upgrade data-designer`
2. Search for duplicates in the [issue tracker](https://github.com/NVIDIA-NeMo/DataDesigner/issues)

When [creating a bug report](https://github.com/NVIDIA-NeMo/DataDesigner/issues/new), please include:
- Data Designer version: `python -c "import data_designer; print(data_designer.__version__)"`
- Python version and operating system
- Minimal reproducible example
- Expected vs. actual behavior
- Full error messages and stack traces

If you are interested in fixing the bug yourself, that's AWESOME! Please follow the [development guide](#development-guide) to get started.

### ✨ Feature Implementation
Want to add new functionality? Great! Please review [our development approach](#feature-requests) and open a feature request to discuss the idea and get feedback before investing significant time on the implementation.

### 📖 Documentation Improvements
Documentation is crucial for user adoption. Contributions that clarify usage, add examples, or fix typos are highly valued.

### 💡 Examples and Tutorials
Share your use cases! Example notebooks and tutorials help others understand how to leverage Data Designer effectively.

### 🧪 Test Coverage
Help us improve test coverage by adding tests for untested code paths or edge cases.

## Feature Requests

Data Designer is designed to be as flexible and extensible as possible, and we welcome your ideas for pushing its capabilities even further! To keep the core library maintainable, while also supporting innovation, we take an incremental approach: we explore what's already possible, extend through plugins when needed, and integrate the most broadly useful features into the core library:

### How We Grow Data Designer

1. 🧗 **Explore what's possible**: Can your use case be achieved with current features? We've designed Data Designer to be composable – sometimes creative combinations of existing tools can accomplish what you need. Check out our examples or open an issue if you'd like help exploring this!

2. 🔌 **Extend through plugins**: If existing features aren't quite enough, consider implementing your idea as a plugin that extends the core library. Plugins let you experiment and share innovations while keeping the core focused.

3. ⚙️ **Integrate into the core**: If your feature or plugin proves broadly useful and aligns with Data Designer's goals, we'd love to integrate it into the core! We're happy to discuss whether it's a good fit and how to move forward together.

This approach helps us grow thoughtfully while keeping Data Designer reliable for everyone.

### Submitting a Feature Request

Open a [new issue](https://github.com/NVIDIA-NeMo/DataDesigner/issues/new) with:

- **Clear title**: Concise description of the feature
- **Use case**: Explain what problem this solves and why it's important
- **Proposed solution**: Describe how you envision the feature working
- **Alternatives considered**: Other approaches you've thought about
- **Examples**: Code examples or mockups of how users would interact with the feature
- **Willingness to implement**: Are you interested in implementing this yourself?

## Development Guide

Data Designer uses [`uv`](https://github.com/astral-sh/uv) for dependency management. If you don't have uv installed, follow their [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

### Initial Setup

0. **Create or find an issue**

   Before starting work, ensure there's an issue tracking your contribution:
   - For bug fixes: Search [existing issues](https://github.com/NVIDIA-NeMo/DataDesigner/issues) or [create a new one](https://github.com/NVIDIA-NeMo/DataDesigner/issues/new)
   - For new features: Open a [feature request](#feature-requests) to discuss the approach first
   - Comment on the issue to let maintainers know you're working on it

1. **Fork and clone the repository**

   Start by [forking the Data Designer repository](https://github.com/NVIDIA-NeMo/DataDesigner/fork), then clone your fork and add the upstream remote:

   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/DataDesigner.git

   cd DataDesigner

   git remote add upstream https://github.com/NVIDIA-NeMo/DataDesigner.git
   ```

2. **Install dependencies**

   ```bash
   # Install project with dev dependencies
   make install-dev

   # Or, if you use Jupyter / IPython for development
   make install-dev-notebooks
   ```

   This creates a virtual environment in `.venv`. Activate it with:

   ```bash
   source .venv/bin/activate
   ```

3. **Verify your setup**

   ```bash
   make test && make check-all
   ```

   If no errors are reported, you're ready to develop 🚀

### Making Changes

1. **Create a feature branch**

   ```bash
   git checkout main
   git pull upstream main
   git checkout -b <username>/<type-of-change>/<issue-number>-<short-description>
   ```

   Example types of change:
   - `feat` for new features
   - `fix` for bug fixes
   - `docs` for documentation updates
   - `test` for testing changes
   - `refactor` for code refactoring
   - `chore` for chore tasks
   - `style` for style changes
   - `perf` for performance improvements

   Example branch name:
   - `johnnygreco/feat/123-add-xyz-generator` for a new feature by @johnnygreco, addressing issue #123

2. **Develop your changes**

   Please follow the patterns and conventions outlined in [AGENTS.md](AGENTS.md).

3. **Test and validate**

   ```bash
   make check-all-fix  # Format code and fix linting issues
   make test           # Run all tests
   make coverage       # Check test coverage (must be >90%)
   ```

   **Writing tests**: Place tests in [tests/](tests/) mirroring the source structure. Use fixtures from [tests/conftest.py](tests/conftest.py), mock external services with `unittest.mock` or `pytest-httpx`, and test both success and failure cases. See [AGENTS.md](AGENTS.md) for patterns and examples.

4. **Commit your work**

   Write clear, descriptive commit messages, optionally including a brief summary (50 characters or less) and reference issue numbers when applicable (e.g., "Fixes #123").

   ```bash
   git commit -m "Add XYZ generator for synthetic data" -m "Fixes #123"
   ```

5. **Stay up to date**

   Regularly sync your branch with upstream changes:

   ```bash
   git fetch upstream
   git merge upstream/main
   ```

## Submitting Changes

### Before Submitting

Ensure your changes meet the following criteria:

- Code follows patterns and conventions outlined in [AGENTS.md](AGENTS.md)
- All tests pass (`make test`)
- Code is formatted and linted (`make check-all-fix`)
- New functionality includes tests
- Documentation is updated (README, docstrings, examples)
- License headers are present on all new files
- Commit messages are clear and descriptive

### Creating a Pull Request

1. **Push your changes** to your fork:

   ```bash
   git push origin <username>/<type-of-change>/<issue-number>-<short-description>
   ```

2. **Open a pull request** on GitHub from your fork to the main repository

3. **Respond to review feedback** promptly and update your PR as needed

### Pull Request Review Process

- Maintainers will review your PR and may request changes
- Address feedback by pushing additional commits to your branch
- Once approved, a maintainer will merge your PR
- Your contribution will be included in the next release!

## Code of Conduct

Data Designer follows the Contributor Covenant Code of Conduct. We are committed to providing a welcoming and inclusive environment for all contributors.

**Please read our complete [Code of Conduct](CODE_OF_CONDUCT.md)** for full details on our standards and expectations.

### Copyright

All code files that are added to this repository must include the appropriate NVIDIA copyright header:

```python
# SPDX-FileCopyrightText: Copyright (c) {YEAR} NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

Use `make update-license-headers` to add headers automatically.

## Getting Help

Need help with your contribution?

- **Documentation**: Check the [documentation](docs/) and [AGENTS.md](AGENTS.md) for additional information
- **Issues**: Browse [existing issues](https://github.com/NVIDIA-NeMo/DataDesigner/issues) for similar questions
- **Contact**: Reach out to the core maintainers at [data-designer@nvidia.com](mailto:data-designer@nvidia.com)


## Signing off on your work

When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project license. All contributors are asked to sign the Data Designer [Developer Certificate of Origin (DCO)](DCO) when submitting their first pull request. The process is automated by a bot that will comment on the pull request. Our DCO is the same as the Linux Foundation requires its contributors to sign.

---

Thank you for contributing to NeMo Data Designer! Your efforts help make synthetic data generation more accessible and powerful for everyone. 🎨✨
