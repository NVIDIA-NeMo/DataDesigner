---
name: search-docs
description: Search local Fern documentation for content related to a topic
argument-hint: <search-topic>
metadata:
    internal: true
---

# Documentation Search

Use the `docs-searcher` subagent to search local Fern documentation for content related to: **$ARGUMENTS**

Call the Task tool with:
- `subagent_type: "docs-searcher"`
- `mode: "bypassPermissions"`
- `prompt`: the search topic

Report the results back to the user exactly as returned by the agent.
