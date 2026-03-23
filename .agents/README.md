# .agents/

This is the tool-agnostic home for shared agent infrastructure used in **developing** DataDesigner.

## Structure

```
.agents/
├── skills/       # Development skills (commit, create-pr, review-code, etc.)
├── agents/       # Sub-agent persona definitions (docs-searcher, github-searcher)
└── README.md     # This file
```

## Compatibility

Tool-specific directories symlink back here so each harness resolves skills from the same source:

- `.claude/skills` → `.agents/skills`
- `.claude/agents` → `.agents/agents`

## Scope

All skills and agents in this directory are for **contributors developing DataDesigner** — not for end users building datasets.

The usage skill for building datasets with DataDesigner lives separately at [`skills/data-designer/`](../skills/data-designer/). For product documentation, see the [docs site](https://nvidia-nemo.github.io/DataDesigner/).
