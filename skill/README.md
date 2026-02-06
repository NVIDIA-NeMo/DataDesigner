# Data Designer Skill for Claude Code

A [Claude Code skill](https://docs.anthropic.com/en/docs/claude-code/skills) that teaches Claude how to generate synthetic datasets using [NVIDIA NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner).

When activated, Claude can design and build complete data generation pipelines — choosing the right column types, writing prompts, wiring up dependencies, and iterating on previews — all from a natural language description of the dataset you want.

## What's in the skill

```
.claude/skills/data-designer/
├── SKILL.md                 # Core skill definition and workflow guide
├── references/
│   ├── api_reference.md     # Complete API documentation
│   └── advanced_patterns.md # Custom columns, MCP tools, multimodal, etc.
├── examples/                # 5 runnable pattern-reference scripts
├── scripts/                 # Discovery tools for API introspection
└── hooks/                   # Session startup check, ruff lint, ty type-check
```

## Prerequisites

- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — used for environment management and required by the skill's session hooks
- **[Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview)** — the CLI that runs the skill
- **Python 3.10+** — any version from 3.10 to 3.13 works (`uv` will install it for you)
- **An LLM provider API key** — e.g., an [NVIDIA API key](https://build.nvidia.com/) (`NVIDIA_API_KEY`)

## Quick start

### 1. Set up a project and download the skill

```bash
mkdir my-project && cd my-project
mkdir -p .claude/skills
```

Download the `skill/data-designer` folder into `.claude/skills/data-designer`:

```bash
# with curl
curl -L https://github.com/NVIDIA-NeMo/DataDesigner/archive/refs/heads/main.tar.gz \
  | tar xz --strip-components=2 -C .claude/skills "DataDesigner-main/skill/data-designer"

# or with wget
wget -qO- https://github.com/NVIDIA-NeMo/DataDesigner/archive/refs/heads/main.tar.gz \
  | tar xz --strip-components=2 -C .claude/skills "DataDesigner-main/skill/data-designer"
```

### 2. Create a Python environment and install Data Designer

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install --pre data-designer
```

> **Note:** The `--pre` flag installs the latest pre-release.

### 3. Set up your default model providers and models

Use the Data Designer CLI to configure your LLM provider(s) and model(s) interactively:

```bash
# Configure a provider (endpoint, API key, etc.)
data-designer config providers

# Configure model(s) that use the provider
data-designer config models

# Verify your configuration
data-designer config list
```

The CLI walks you through each setting with an interactive prompt. You only need to do this once — configurations are saved to `~/.data-designer/`.

### 4. Launch Claude Code

```bash
claude
```
