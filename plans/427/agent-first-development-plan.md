---

## date: 2026-03-17
authors:
  - nmulepati

# Plan: Agent-Assisted Development Principles

## Problem

DataDesigner was built entirely by humans, and the codebase reflects that with strong architecture, comprehensive tests, and thoughtful design. We are now increasingly moving toward an agent-assisted planning and development workflow. The project already has meaningful agent-oriented infrastructure: seven skills, an agent introspection CLI, and supporting tooling. But a new contributor reading `README.md`, `CONTRIBUTING.md`, or `AGENTS.md` would not immediately discover that these workflows exist. The repository supports agent-assisted work, yet the top-level documentation still presents the project mostly as a conventional human-only codebase.

## Inspiration

This proposal draws strong inspiration from [NVIDIA/OpenShell](https://github.com/NVIDIA/OpenShell), which makes agent workflows and contributor guidance highly visible from the repository root. The goal is to bring those ideas into DataDesigner in a way that makes the project more agent-friendly while fitting its role as a public open-source Python library for synthetic data generation. Because DataDesigner serves a different audience, the resulting workflow should remain lighter-weight and more flexible than OpenShell's.

## Current State


| Asset             | Status                                                                        | Agent-First Signal                                                                                      |
| ----------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| README.md         | Product-focused, usage-first                                                  | Zero mention of agent-first development                                                                 |
| CONTRIBUTING.md   | Standard OSS contributing guide                                               | Zero mention of skills or agent workflows                                                               |
| AGENTS.md         | ~500 lines, mixes architecture, code style, and engineering workflow guidance | No skills inventory, no workflow chains                                                                 |
| CLAUDE.md         | 1 line: `@AGENTS.md`                                                          | Minimal                                                                                                 |
| Issue templates   | 4 templates (bug, feature, dev task, config)                                  | No agent investigation fields                                                                           |
| PR template       | Doesn't exist                                                                 | --                                                                                                      |
| CODEOWNERS        | Catch-all `* @NVIDIA-NeMo/data_designer_reviewers`                            | No agent infra ownership                                                                                |
| `.claude/skills/` | Current skill location                                                        | Invisible from top-level docs                                                                           |
| `.agents/skills/` | Doesn't exist yet                                                             | Planned future location, but not present or documented today                                            |
| `architecture/`   | Doesn't exist                                                                 | --                                                                                                      |
| STYLEGUIDE.md     | Doesn't exist (inlined in AGENTS.md)                                          | --                                                                                                      |
| DEVELOPMENT.md    | Doesn't exist                                                                 | Setup, testing, and day-to-day workflow guidance are scattered across `AGENTS.md` and `CONTRIBUTING.md` |


## Principles

1. **Agents accelerate work; humans stay accountable.** Agents can speed up planning, implementation, and review, but people still make design decisions and own quality.
2. **Design intent should remain explicit.** The project should communicate that systems are deliberately engineered, with agents supporting the work rather than replacing architectural judgment.
3. **Encourage agent investigation without blocking real users.** Issue templates should normalize agent-assisted investigation, but contributors who cannot or did not use an agent still need a clear path to report bugs and propose features.
4. **Skills are part of the contributor surface area.** The future `.agents/skills/` directory should be treated as a maintained interface between the project and contributor agents.
5. **Top-level docs should advertise the workflow.** `README.md`, `CONTRIBUTING.md`, and `AGENTS.md` should make agent-assisted paths obvious to new contributors.

---

## Phase 1: Skill & Agent Infrastructure

This phase lands first so the repository has a stable, tool-agnostic home for shared agent assets before the documentation starts pointing contributors at it.

### 1a. Consolidate `.agents/` and `.claude/`

**Goal:** `.agents/` becomes the primary tool-agnostic location for shared agent infrastructure. `.claude/` remains a compatibility layer for Claude Code-specific runtime state.

**Current layout before consolidation:**

```
.claude/
  skills/           # 7 skills — Claude Code-specific location
  agents/            # 2 sub-agent definitions (docs-searcher, github-searcher)
  settings.json
  settings.local.json
```

**Target layout:**

```
.agents/
  skills/            # 7 skills (moved from .claude/skills/)
  agents/            # Sub-agent persona definitions (moved from .claude/agents/)
    docs-searcher.md
    github-searcher.md

.claude/
  skills             # Symlink → ../.agents/skills
  agents             # Symlink → ../.agents/agents (or keep Claude-specific if frontmatter differs)
  agent-memory/      # Stays here (Claude Code-specific, not portable)
  settings.json
  settings.local.json

.codex/
  skills             # Symlink → ../.agents/skills
```

**Changes:**

1. Create `.agents/skills/` and move `.claude/skills/` into it
2. Move `.claude/agents/*.md` to `.agents/agents/*.md`
3. Create symlinks from `.claude/` and `.codex/` if both harnesses resolve them correctly; otherwise keep mirrored directories and add a drift-check task
4. Add `.claude/README.md` explaining the structure

### 1b. Skill Cross-Reference Cleanup

- Verify all skill files reference `.agents/skills/` not `.claude/skills/`
- Verify sub-agent references point to `.agents/agents/`
- Ensure all cross-skill references use consistent naming

---

## Phase 2: Foundation Documents

This phase updates the contributor-facing docs after the agent-infrastructure paths and terminology are settled.

### 2a. README.md

**Current state:** Product-focused. Zero signal of agent-first development.

**Target state:** Retains the product pitch while making it obvious that DataDesigner supports agent-assisted development. A new developer should quickly understand that the repo contains workflows and guidance their agent can use.


| Section                           | Action                                                                                                                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hero / intro paragraph            | Add a line signaling agent-first development alongside the product pitch                                                                                                        |
| New: "Get Oriented with an Agent" | After quickstart. Show contributors how to clone the repo, point an agent at it, and use the repo guidance to answer questions quickly                                          |
| New: "How Agent Workflows Fit In" | After the product sections. Explain how agent-assisted workflows support development here, and link to `AGENTS.md` for the authoritative skills inventory and workflow guidance |
| Contributing                      | Expand from current brief mention to set expectations: agent-first contributions, link to CONTRIBUTING.md                                                                       |


**Key language to establish:**

- "DataDesigner supports agent-assisted planning, implementation, and review."
- "Before opening an issue, consider asking your coding agent to inspect the repository first."

### 2b. CONTRIBUTING.md

**Current state:** Standard OSS contributing guide. Welcoming tone, good fork→develop→PR workflow, but zero agent awareness.

**Target state:** Agent-assisted contribution workflow is the recommended path. Human-only paths are fully supported but agent workflows are encouraged and well-documented.


| Section                          | Action                                                                                                                                                                                                                                                                                          |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| New: Opening philosophy          | 2-3 sentences explaining that this project supports agent-assisted development and includes repo guidance and skills that make agents more effective contributors.                                                                                                                              |
| New: "Before You Open an Issue"  | Recommended path. Checklist: (1) Clone the repo, (2) Point your agent at it, (3) Load relevant skills, (4) Have your agent diagnose/investigate. If the agent can't solve it, include the diagnostics. If you couldn't use an agent, say why and include the troubleshooting you already tried. |
| New: "Contributor Skill Map"     | Short category summary plus a link to `AGENTS.md`, which remains the authoritative skill inventory                                                                                                                                                                                              |
| New: "Common Agent Workflows"    | Document typical paths such as investigation → development, and future spike → build                                                                                                                                                                                                            |
| Getting Started                  | Keep as-is (fork, clone, install)                                                                                                                                                                                                                                                               |
| Development Guide                | Keep the contributor-facing summary, but link to `DEVELOPMENT.md` for detailed setup, testing, and day-to-day engineering workflow                                                                                                                                                              |
| Pull Requests                    | Update to reference the new PR template and the `create-pr` skill                                                                                                                                                                                                                               |
| New: "When to Open an Issue"     | Clear guidance: real bugs your agent confirmed or you reproduced yourself with enough detail, feature proposals with design context, problems the `search-docs`/`search-github` skills couldn't resolve                                                                                         |
| New: "When NOT to Open an Issue" | Questions about how things work (agent can answer), configuration problems (agent can diagnose), "how do I..." requests (agent has skills for this)                                                                                                                                             |
| Commit Messages / DCO            | Keep as-is                                                                                                                                                                                                                                                                                      |


**Skill categories to summarize (keep `AGENTS.md` as the authoritative inventory):**


| Category        | Skills                             | Purpose                                        |
| --------------- | ---------------------------------- | ---------------------------------------------- |
| Getting Started | `search-docs`, `search-github`     | Find information, check for duplicates         |
| Data Generation | `new-sdg`                          | Design synthetic data generators interactively |
| Development     | `commit`, `create-pr`, `update-pr` | Standard development cycle                     |
| Review          | `review-code`                      | Multi-pass code review                         |


### 2c. AGENTS.md

**Current state:** ~500 lines. Mixes project overview, architecture, code style, development workflow, and testing guidance into one file. No skills inventory, no workflow chains, no project identity statement.

**Target state:** The main onboarding document for agents working on this codebase. It should provide enough architectural and workflow context to make an agent effective, while moving code-authoring reference material to `STYLEGUIDE.md` and detailed development/testing workflow guidance to `DEVELOPMENT.md`.


| Section                                               | Action                                                                                                                                                                                             |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Opening                                               | Keep the CONTRIBUTING.md reference. Add: "This file is the main onboarding surface for agents contributing to DataDesigner."                                                                       |
| New: "Project Identity"                               | 3-4 sentences: DataDesigner is a synthetic-data project built by humans that now supports agent-assisted planning and development, while keeping human ownership over design decisions             |
| New: "Skills"                                         | Authoritative skill inventory. Note where skills live after consolidation, how harnesses discover them, and keep the full inventory here rather than duplicating it across README and CONTRIBUTING |
| New: "Suggested Workflows"                            | Document the common skill sequences and when to use them                                                                                                                                           |
| Architecture                                          | Keep existing 3-layer overview, key files, registries. Add brief component map.                                                                                                                    |
| New: "Issue and PR Conventions"                       | Reference the templates. When creating issues, use the template format. When creating PRs, use the PR template. Skills should produce output conforming to these templates.                        |
| Development Workflow                                  | Move detailed setup and day-to-day engineering commands to `DEVELOPMENT.md`; keep only a short summary and link                                                                                    |
| Working Guidelines                                    | Split the section: code-authoring guidance (license headers, `__future__` imports, comments) moves to `STYLEGUIDE.md`, while operational safety guidance stays in `AGENTS.md`                      |
| Testing                                               | Move detailed test commands and "when to run what" guidance to `DEVELOPMENT.md`; keep only short expectations and link                                                                             |
| New: "Security"                                       | Don't commit secrets, don't run destructive operations without confirmation, scope changes to the issue at hand                                                                                    |
| Pre-commit                                            | Move command details to `DEVELOPMENT.md`; keep only a brief mention in `AGENTS.md` if needed                                                                                                       |
| Column/Model Configuration                            | Keep as-is (brief summaries)                                                                                                                                                                       |
| Registry System                                       | Keep as-is                                                                                                                                                                                         |
| **REMOVE** Code Style sections                        | Move to STYLEGUIDE.md (see 2d)                                                                                                                                                                     |
| **MOVE** detailed workflow/testing reference sections | Move to `DEVELOPMENT.md` (see 2e)                                                                                                                                                                  |


### 2d. STYLEGUIDE.md (new file)

Extract from AGENTS.md. Contains all code style reference material:

- General formatting (line length, quotes, indentation, target version)
- Type annotations (full section with all examples)
- Import style (absolute imports, lazy loading, TYPE_CHECKING — full section)
- Naming conventions (PEP 8 rules, verb-first function names)
- Code organization (public/private ordering, class method order, section comments)
- Design principles (DRY, KISS, YAGNI, SOLID — with examples)
- Common pitfalls (all 5 with code examples)
- Code-authoring guidance from Working Guidelines: license headers, `from __future__ import annotations`, and comment expectations
- Active linter rules (ruff rule reference)

**CLAUDE.md remains:**

```
@AGENTS.md
```

`AGENTS.md` should link to `STYLEGUIDE.md` and `DEVELOPMENT.md` when deeper reference material is needed.

**Why:** AGENTS.md is loaded into every agent conversation. Code style is reference material — needed when writing code, not when triaging issues or creating spikes. Splitting reduces context cost only if `STYLEGUIDE.md` is not loaded unconditionally.

### 2e. DEVELOPMENT.md (new file)

Collect from `AGENTS.md` and `CONTRIBUTING.md`. Contains development and testing reference material:

- Local setup and install commands (`uv`, `make install-dev`, notebook/dev variants)
- Day-to-day engineering workflow (branching, syncing with upstream, validation commands)
- Testing commands and guidance on when to run which test suites
- Pre-commit usage and expected local checks before opening a PR
- Practical contributor workflow details that are too operational for `AGENTS.md` and too detailed for `CONTRIBUTING.md`

`AGENTS.md` and `CONTRIBUTING.md` should link to `DEVELOPMENT.md` rather than duplicating these details.

**Why:** Development workflow and testing guidance are operational reference material, not project identity or code style. Moving them out keeps `AGENTS.md` focused on onboarding and architecture, keeps `STYLEGUIDE.md` focused on how code should look, and keeps `CONTRIBUTING.md` concise.

### 2f. Create `architecture/` Directory (Skeleton)

Create stub files for each major subsystem. Each stub lists section headings but doesn't contain full content yet. Docs are populated incrementally as features are built.

```
architecture/
├── overview.md                  # System architecture, package relationships, data flow diagram
├── config.md                    # Config layer: builder, column types, unions, plugin system
├── engine.md                    # Engine layer: compilation, generators, DAG execution, batching
├── models.md                    # Model facade, client adapters, retry/throttle, usage tracking
├── mcp.md                       # MCP: I/O service, session pooling, coalescing, tool execution
├── dataset-builders.md          # Column-wise builder, async scheduler, DAG, concurrency
├── sampling.md                  # Person/entity sampling, locale system, data sources
├── cli.md                       # CLI architecture: commands → controllers → services → repos
├── agent-introspection.md       # Agent CLI commands, type discovery, family specs
└── plugins.md                   # Plugin system: entry points, registry, discovery, validation
```

Each stub follows this template:

```markdown
# <Subsystem Name>

> Stub — to be populated. See source code at `<package path>`.

## Overview
<!-- What this subsystem does and why it exists -->

## Key Components
<!-- Main classes, modules, entry points -->

## Data Flow
<!-- How data moves through this subsystem -->

## Design Decisions
<!-- Why things are the way they are -->

## Cross-References
<!-- Links to related architecture docs -->
```

**Why:** Agents producing plans can reference architecture docs to understand subsystems. Stubs establish the structure; content grows organically as features are built.

---

## Phase 3: GitHub Machinery

### 3a. Issue Templates

Update existing `.github/ISSUE_TEMPLATE/` templates:

**bug-report.yml** updates:

- Add **Agent Diagnostic / Prior Investigation** (textarea, recommended): "If you used an agent, paste the output from its investigation. If you couldn't or didn't, briefly say why and include the troubleshooting you already tried."
- Add **Checklist** (required):
  - I reproduced this issue or provided a minimal example
  - I searched the docs/issues myself, or had my agent do so
  - If I used an agent, I included its diagnostics above
- Keep existing fields (priority, description, reproduction, expected behavior)

**feature-request.yml** updates:

- Keep existing fields (problem, solution, alternatives)
- Add **Agent Investigation** (textarea, optional): "If your agent explored the codebase to assess feasibility (e.g., using the `search-docs` skill), paste its findings here."
- Add **Checklist**:
  - I've reviewed existing issues and the documentation
  - This is a design proposal, not a "please build this" request

**development-task.yml** updates:

- Clarify that it is for tracked internal work, refactors, and infra changes rather than end-user support requests
- Add **Investigation / Context** (textarea, optional): relevant issue links, notes, or architecture context
- Add **Agent Plan / Findings** (textarea, optional): what the agent found, proposed, or could not resolve

**config.yml** updates:

- Keep the Discussions link, but update the copy to: "Have a question? Try pointing your agent at the repo first. It has skills for searching docs, finding issues, and more. See CONTRIBUTING.md for the workflow and AGENTS.md for the full skill inventory."

### 3b. PR Template

Create `.github/PULL_REQUEST_TEMPLATE.md`:

```markdown
## Summary
<!-- 1-3 sentences: what this PR does and why -->

## Related Issue
<!-- Link to the issue this addresses: Fixes #NNN or Closes #NNN -->

## Changes
<!-- Bullet list of key changes -->

## Testing
<!-- What testing was done? -->
- [ ] `make test` passes
- [ ] Unit tests added/updated
- [ ] E2E tests added/updated (if applicable)

## Checklist
- [ ] Follows commit message conventions
- [ ] Commits are signed off (DCO)
- [ ] Architecture docs updated (if applicable)
```

Intentionally lean. The `create-pr` and `review-code` skills already produce well-structured descriptions; the template provides guardrails without fighting the skills.

### 3c. CODEOWNERS

Update `.github/CODEOWNERS` to explicitly call out agent-infrastructure ownership paths:

```
# Broad ownership — core team reviews everything
* @NVIDIA-NeMo/data_designer_reviewers

# Agent infrastructure — explicit path callouts for visibility
.agents/ @NVIDIA-NeMo/data_designer_reviewers
AGENTS.md @NVIDIA-NeMo/data_designer_reviewers
STYLEGUIDE.md @NVIDIA-NeMo/data_designer_reviewers
```

### 3d. Label Taxonomy

Create labels for workflow state:


| Label                | Purpose                                                                           | Used by                                     |
| -------------------- | --------------------------------------------------------------------------------- | ------------------------------------------- |
| `agent-ready`        | Human-approved, agent can build                                                   | `build-from-issue` (future)                 |
| `review-ready`       | Agent has posted a plan, needs human review                                       | `build-from-issue`, `create-spike` (future) |
| `in-progress`        | Agent is actively building                                                        | `build-from-issue` (future)                 |
| `pr-opened`          | Implementation complete, PR submitted                                             | `build-from-issue` (future)                 |
| `spike`              | Needs deeper investigation                                                        | `create-spike` (future)                     |
| `needs-more-context` | Issue is missing useful reproduction or investigation context and needs follow-up | Triage automation (future)                  |
| `good-first-issue`   | Suitable for new contributors (with agents)                                       | Manual                                      |


### 3e. Update Skills to Conform to Templates

Skills that create GitHub artifacts should produce output matching the new templates:
**`create-pr`:**
- Produce PR descriptions matching the PR template structure (Summary / Related Issue / Changes / Testing / Checklist)
- Include the testing checklist populated based on what was actually run

`**review-code`:**

- When reviewing PRs created from templates, check that the template sections are properly filled

---

## Phase 4: Future Work (Requires Separate Planning)

### 4a. New Skills


| Skill                    | Purpose                                       | Depends on                            |
| ------------------------ | --------------------------------------------- | ------------------------------------- |
| `build-from-issue`       | Stateful plan → review → build → PR pipeline  | Labels, `principal-engineer-reviewer` |
| `create-spike`           | Investigate problem, create structured issue  | `principal-engineer-reviewer`         |
| `debug-sdg`              | Debug failing data generation pipelines       | Architecture docs                     |
| `generate-column-config` | Generate column configs from natural language | Agent introspection API               |
| `watch-github-actions`   | Monitor CI workflow runs                      | None                                  |
| `sync-agent-infra`       | Detect drift across agent files               | Skills inventory                      |


### 4b. Sub-Agent Personas


| Agent                         | Purpose                                           | Scope                         |
| ----------------------------- | ------------------------------------------------- | ----------------------------- |
| `principal-engineer-reviewer` | Analyze code, generate plans, review architecture | Read-only                     |
| `arch-doc-writer`             | Update `architecture/` docs after features land   | Write to `architecture/` only |


### 4c. Issue Triage Workflow

Create `.github/workflows/issue-triage.yml`:

- **Trigger:** `issues.opened`
- **Logic:** Check if the issue was created using the bug report template and the investigation/context fields are empty or contain only placeholder text
- **Action:** Add the `needs-more-context` label and post a comment asking for more reproduction or investigation detail; recommend agent output as a fast path, but do not require it
- This is a simple deterministic check — not an LLM-powered triage bot

---

## Delivery Strategy

Land this work as a sequence of incremental PRs rather than a single large rollout:

1. **Phase 1 PR(s)** — agent infrastructure consolidation and path cleanup.
2. **Phase 2 PR(s)** — documentation restructuring (`AGENTS.md`, `STYLEGUIDE.md`, `DEVELOPMENT.md`, `CONTRIBUTING.md`, `README.md`, and optional `architecture/` skeleton).
3. **Phase 3 PR(s)** — GitHub machinery such as templates, labels, and skill output conformance.
4. **Phase 4** — do not start implementation directly from this plan. Treat it as follow-on work that requires another planning pass, design review, and then its own incremental PRs.

The default PR boundary should be the phase boundary. If a phase is still too large, split it into a small sequence of focused PRs, but keep the phases ordered and avoid mixing deliverables from different phases in one PR.

---

## Execution Order


| Step | Deliverable                                                | Dependencies                           | Parallelizable      |
| ---- | ---------------------------------------------------------- | -------------------------------------- | ------------------- |
| 1    | Skill consolidation (`.agents/`, `.claude/` cleanup)       | None                                   | --                  |
| 2    | AGENTS.md restructure + STYLEGUIDE.md/DEVELOPMENT.md split | Step 1 (canonical paths settled)       | --                  |
| 3    | CONTRIBUTING.md overhaul                                   | AGENTS.md (references it)              | --                  |
| 4    | README.md updates                                          | CONTRIBUTING.md (references it)        | --                  |
| 5    | Issue templates                                            | CONTRIBUTING.md (templates link to it) | Yes (with 6-9)      |
| 6    | PR template                                                | None                                   | Yes (with 5, 7-9)   |
| 7    | CODEOWNERS update                                          | None                                   | Yes (with 5-6, 8-9) |
| 8    | Label creation (via `gh label create`)                     | None                                   | Yes (with 5-7, 9)   |
| 9    | `architecture/` skeleton                                   | None                                   | Yes (with 5-8)      |
| 10   | Skill template conformance updates                         | Issue/PR templates (steps 5-6)         | --                  |


Step 1 lands first so the docs can reference canonical paths truthfully. Steps 2-4 are sequential. Steps 5-9 are independent and can be parallelized. Step 10 depends on earlier steps.

In practice, Steps 1-4 map cleanly to Phase 1 and Phase 2 PRs, while Steps 5-10 map to one or more Phase 3 PRs. Phase 4 should be planned separately before any implementation PRs are opened.

---

## Out of Scope

- **New skills** — the 7 existing skills are sufficient; this plan surfaces them
- **LLM-powered issue triage** — deliberate choice to keep triage deterministic
- **Vouch system** — defer until external contributor volume warrants it
- **CI/CD changes** — existing workflows are solid
- **Full architecture docs** — create skeleton only, populate incrementally
- **Dependabot / Renovate** — dependency management automation (separate concern)

## Open Questions

1. **Reference doc naming** — `STYLEGUIDE.md` vs `CODE_STYLE.md` vs `STYLE.md`, and `DEVELOPMENT.md` vs `DEVELOPMENT_GUIDE.md`?
2. **README tone** — How prominent should the agent-first messaging be? A line in the hero paragraph, or a dedicated section?
3. **CODEOWNERS granularity** — Keep the explicit path callouts only, or introduce a distinct owner group for agent infra later?
4. `**development-task.yml` audience** — Keep it public for all contributors, or narrow it to maintainers/internal work?
5. **Symlink compatibility** — Do Claude/Codex harnesses handle symlinked skill directories reliably enough, or should we prefer mirrored directories plus drift checks?
