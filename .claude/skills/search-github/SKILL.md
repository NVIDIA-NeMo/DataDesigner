---
name: search-github
description: Search GitHub issues, discussions, and PRs for content related to a topic
argument-hint: <search-topic>
---

# GitHub Search

Use the `github-searcher` subagent to search GitHub for content related to: **$ARGUMENTS**

Call the Task tool with:
- `subagent_type: "github-searcher"`
- `mode: "bypassPermissions"`
- `prompt`: the search topic

Report the results back to the user exactly as returned by the agent.
