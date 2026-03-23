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

All skills and agents in this directory are for **contributors developing DataDesigner** — not for end users building datasets. If you're looking for usage tooling, see the [product documentation](https://nvidia-nemo.github.io/DataDesigner/).
