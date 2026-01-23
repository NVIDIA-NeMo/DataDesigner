---
name: search-docs
description: Search local documentation in the docs/ folder for content related to a topic
argument-hint: <search-topic>
---

# Documentation Search

Use the `docs-searcher` subagent to search local documentation for content related to: **$ARGUMENTS**

Call the Task tool with:
- `subagent_type: "docs-searcher"`
- `mode: "bypassPermissions"`
- `prompt`: the search topic

Report the results back to the user exactly as returned by the agent.
