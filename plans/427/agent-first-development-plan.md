---
date: 2026-03-17
authors:
  - nmulepati
---

# Plan: Agent-Assisted Development Principles

## Problem

DataDesigner was built entirely by humans, and the codebase reflects that with strong architecture, comprehensive tests, and thoughtful design. We are now increasingly moving toward an agent-assisted planning and development workflow. The project already has meaningful agent-oriented infrastructure: seven skills, an agent introspection CLI, and supporting tooling. But a new contributor reading `README.md`, `CONTRIBUTING.md`, or `AGENTS.md` would not immediately discover that these workflows exist. The repository supports agent-assisted work, yet the top-level documentation still presents the project mostly as a conventional human-only codebase.

This plan distinguishes two surfaces for agent tooling:

- **Usage tooling** — skills and workflows that help end users build synthetic datasets with DataDesigner (e.g., the forthcoming official "build a dataset" skill, which runs outside the repo).
- **Development tooling** — skills and workflows that help contributors plan, implement, test, and review changes to DataDesigner itself (e.g., `search-docs`, `review-code`, `create-pr`).

The implementation work in this plan focuses on the **development** surface, but the plan must also ensure agents can clearly distinguish between the two. An agent working inside the repo should understand it is contributing to DataDesigner, not using it to build datasets. An agent helping a user build a dataset should not be confused by development-oriented skills and guidance.

## Inspiration

This proposal draws strong inspiration from [NVIDIA/OpenShell](https://github.com/NVIDIA/OpenShell), which makes agent workflows and contributor guidance highly visible from the repository root. The goal is to bring those ideas into DataDesigner in a way that makes the project more agent-friendly while fitting its role as a public open-source Python library for synthetic data generation. Because DataDesigner serves a different audience, the resulting workflow should remain lighter-weight and more flexible than OpenShell's.

## Current State


| Asset             | Status                                                                        | Agent-First Signal                                                                                      |
| ----------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| README.md         | Product-focused, usage-first                                                  | Zero mention of agent-first development                                                                 |
| CONTRIBUTING.md   | Standard OSS contributing guide                                               | Zero mention of skills or agent workflows                                                               |
| AGENTS.md         | ~500 lines, mixes architecture, code style, and engineering workflow guidance | Bloated; mixes reference material with architectural invariants                                         |
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
4. **Development and usage are separate surfaces.** An agent working inside the repo is a contributor; an agent helping a user build a dataset is a consumer. The repo's agent infrastructure (skills, `AGENTS.md`, `CONTRIBUTING.md`) serves the development surface. Usage tooling lives outside the repo. Both docs and skill metadata should make the boundary unambiguous so agents don't confuse the two contexts.
5. **README is for users; CONTRIBUTING.md is for contributors.** Agent-assisted development messaging belongs in `CONTRIBUTING.md` and `AGENTS.md`, not prominently in `README.md`.

---

## Phase 0: AGENTS.md

AGENTS.md is injected into every agent prompt. It must land first because every subsequent phase references the architectural invariants it establishes. It should also be the most stable file in the repo — if it changes often, something is wrong.

**Current state:** ~500 lines. Mixes project overview, architecture, code style, development workflow, and testing guidance into one file.

**Target state:** ~50 lines. Only high-level design decisions that are consequential for development. Content that changes when features ship (key files, registries, column types) or that duplicates tooling enforcement (code style, linter rules) does not belong here.

**Target sections:**

| Section                  | Content                                                                                                                                                                     |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Identity                 | 3-4 sentences: what DataDesigner is, the "declare, don't orchestrate" contract, and the implication for every change. Must state that this file is for agents *developing* DataDesigner. If you are an agent helping a user *build a dataset*, see the product documentation and tutorials instead — not this file. |
| The Layering Is Structural | Three packages (config → engine → interface), what each owns, and the PEP 420 namespace package detail                                                                   |
| Core Concepts            | One-liner definitions: columns, samplers, seed datasets, processors, models, plugins                                                                                        |
| Core Design Principles   | Declarative config vs. imperative engine, registries connect types to behavior, errors normalize at boundaries                                                               |
| Structural Invariants    | Import direction, fast imports, no relative imports, typed code, follow established patterns, no untested code paths                                                         |
| Development              | Four `make` targets only: `check-all-fix`, `test`, `update-license-headers`, `perf-import`                                                                                  |

**What moves out:**

- Code style, type annotations, import patterns, lazy loading, `TYPE_CHECKING`, naming conventions, common pitfalls → `STYLEGUIDE.md`
- Development workflow, testing commands, pre-commit, setup → `DEVELOPMENT.md`
- Key files list, column/model configuration details, registry system → removed (agents discover these via code search; they change too often to maintain in a static doc)

**What is NOT added:**

- Skills inventory — agent harnesses have built-in skill discovery and loading; duplicating the inventory in AGENTS.md creates a maintenance burden with no benefit
- Suggested workflows / skill sequences — these belong in skill files themselves or in `CONTRIBUTING.md`
- Issue and PR conventions — these belong in the templates and in `CONTRIBUTING.md`

**CLAUDE.md remains:**

```
@AGENTS.md
```

This preserves Claude Code's native include mechanism and the ability to compose multiple files in the future (e.g., `@AGENTS.md` + `@STYLEGUIDE.md`).

---

## Phase 1: Skill & Agent Infrastructure

This phase lands after AGENTS.md so the repository has a stable, tool-agnostic home for shared agent assets before the remaining documentation starts pointing contributors at it.

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
- Each skill's description or frontmatter should identify it as a **development** skill (e.g., "for contributors developing DataDesigner") so that agent harnesses with skill discovery can distinguish repo skills from usage skills

---

## Phase 2: Foundation Documents

This phase updates the contributor-facing docs after the agent-infrastructure paths and terminology are settled.

### 2a. README.md

**Current state:** Product-focused, usage-first. Designed for humans. Zero signal of agent-assisted development.

**Target state:** Remains a product-focused document for users. Agent-assisted development gets a brief mention — a sentence or two near the development installation section — not a prominent hero section. The README is often the only documentation a DataDesigner *user* reads; it should not be dominated by contributor workflow details. The README should also help agents distinguish the two surfaces: users who want to *build datasets* should be directed toward usage documentation and the official usage skill (once available), not toward the in-repo development skills.

| Section                | Action                                                                                                                                                |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Quickstart / usage     | Keep product-focused. When usage tooling (e.g., the official "build a dataset" skill) ships, link to it here so user-facing agents find it naturally. |
| Development install    | Add 1-2 sentences noting that the repo supports agent-assisted development and linking to `CONTRIBUTING.md` for the contributor workflow.              |
| Contributing           | Brief mention linking to `CONTRIBUTING.md`. No dedicated agent workflow sections in README itself.                                                     |

**Key language:** Keep it minimal. Something like: "This repository supports agent-assisted development — see [CONTRIBUTING.md](CONTRIBUTING.md) for the recommended workflow."

### 2b. CONTRIBUTING.md

**Current state:** Standard OSS contributing guide. Welcoming tone, good fork→develop→PR workflow, but zero agent awareness.

**Target state:** A complete overhaul that reflects how contributions actually happen now. The traditional fork→develop→PR workflow is being replaced by agent-assisted planning and development. CONTRIBUTING.md should guide contributors toward submitting plans via issues, using agents for investigation and implementation, and treating PRs as the output of an agent-assisted workflow rather than a purely manual one.

| Section                          | Action                                                                                                                                                                                                                                                                                       |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Opening philosophy               | 2-3 sentences: this project uses agent-assisted development. Contributors are expected to use agents for investigation, planning, and implementation. The repo includes skills and guidance that make agents effective.                                                                        |
| "How to Contribute"              | Primary path: (1) Open an issue using the appropriate template, (2) Include agent investigation output, (3) For non-trivial changes, submit a plan in the issue for review before building, (4) Once approved, use agent-assisted development to implement. Link to `DEVELOPMENT.md` for setup. |
| "Before You Open an Issue"       | Checklist: clone the repo, point your agent at it, have it search docs/issues. If the agent can't resolve it, include the diagnostics. If you didn't use an agent, include the troubleshooting you tried.                                                                                    |
| "When to Open an Issue"          | Real bugs (reproduced or agent-confirmed), feature proposals with design context, problems that `search-docs`/`search-github` couldn't resolve.                                                                                                                                              |
| "When NOT to Open an Issue"      | Questions about how things work (agent can answer), configuration problems (agent can diagnose), "how do I..." requests.                                                                                                                                                                     |
| Pull Requests                    | Reference the PR template and the `create-pr` skill. PRs should link to the issue they address.                                                                                                                                                                                              |
| Commit Messages / DCO            | Keep as-is.                                                                                                                                                                                                                                                                                  |


CONTRIBUTING.md should open by clarifying the boundary: "The skills and workflows in this repository are for *developing* DataDesigner. If you're looking to *use* DataDesigner to build datasets, see the product documentation and the official usage skill (once available)."

**Repo skill categories (development only):**


| Category      | Skills                             | Purpose                                |
| ------------- | ---------------------------------- | -------------------------------------- |
| Investigation | `search-docs`, `search-github`     | Find information, check for duplicates |
| Development   | `commit`, `create-pr`, `update-pr` | Standard development cycle             |
| Review        | `review-code`                      | Multi-pass code review                 |


### 2c. STYLEGUIDE.md (new file)

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

**Why:** AGENTS.md is loaded into every agent conversation. Code style is reference material — needed when writing code, not when triaging issues or creating spikes. Splitting reduces context cost only if `STYLEGUIDE.md` is not loaded unconditionally.

### 2d. DEVELOPMENT.md (new file)

Collect from `AGENTS.md` and `CONTRIBUTING.md`. Contains development and testing reference material:

- Local setup and install commands (`uv`, `make install-dev`, notebook/dev variants)
- Day-to-day engineering workflow (branching, syncing with upstream, validation commands)
- Testing commands and guidance on when to run which test suites
- Pre-commit usage and expected local checks before opening a PR
- Practical contributor workflow details that are too operational for `AGENTS.md` and too detailed for `CONTRIBUTING.md`

`AGENTS.md` and `CONTRIBUTING.md` should link to `DEVELOPMENT.md` rather than duplicating these details.

**Why:** Development workflow and testing guidance are operational reference material, not project identity or code style. Moving them out keeps `AGENTS.md` focused on onboarding and architecture, keeps `STYLEGUIDE.md` focused on how code should look, and keeps `CONTRIBUTING.md` concise.

### 2e. Create `architecture/` Directory (Skeleton)

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

- Keep the Discussions link, but update the copy to: "Have a question? Try pointing your agent at the repo first — it can search docs, find issues, and more. See CONTRIBUTING.md for the recommended workflow."

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

Keep the existing single-group ownership for now. All paths — including agent infrastructure — are owned by `@NVIDIA-NeMo/data_designer_reviewers`. Introduce a distinct agent infra owner group only if the need becomes clear later.

```
* @NVIDIA-NeMo/data_designer_reviewers
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

**`review-code`:**

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

0. **Phase 0 PR** — `AGENTS.md` restructure (~50 lines). Lands first because it is injected into every agent prompt and every subsequent phase references it.
1. **Phase 1 PR(s)** — agent infrastructure consolidation and path cleanup.
2. **Phase 2 PR(s)** — remaining documentation (`STYLEGUIDE.md`, `DEVELOPMENT.md`, `CONTRIBUTING.md`, `README.md`, and optional `architecture/` skeleton).
3. **Phase 3 PR(s)** — GitHub machinery such as templates, labels, and skill output conformance.
4. **Phase 4** — do not start implementation directly from this plan. Treat it as follow-on work that requires another planning pass, design review, and then its own incremental PRs.

The default PR boundary should be the phase boundary. If a phase is still too large, split it into a small sequence of focused PRs, but keep the phases ordered and avoid mixing deliverables from different phases in one PR.

---

## Execution Order


| Step | Deliverable                                                | Dependencies                           | Parallelizable       |
| ---- | ---------------------------------------------------------- | -------------------------------------- | -------------------- |
| 0    | AGENTS.md restructure (~50 lines)                          | None                                   | --                   |
| 1    | Skill consolidation (`.agents/`, `.claude/` cleanup)       | Step 0 (AGENTS.md settled)             | --                   |
| 2    | STYLEGUIDE.md + DEVELOPMENT.md (extracted from old AGENTS.md) | Step 0                              | --                   |
| 3    | CONTRIBUTING.md overhaul                                   | Step 0 (references it)                 | --                   |
| 4    | README.md updates                                          | CONTRIBUTING.md (references it)        | --                   |
| 5    | Issue templates                                            | CONTRIBUTING.md (templates link to it) | Yes (with 6-9)       |
| 6    | PR template                                                | None                                   | Yes (with 5, 7-9)    |
| 7    | CODEOWNERS update                                          | None                                   | Yes (with 5-6, 8-9)  |
| 8    | Label creation (via `gh label create`)                     | None                                   | Yes (with 5-7, 9)    |
| 9    | `architecture/` skeleton                                   | None                                   | Yes (with 5-8)       |
| 10   | Skill template conformance updates                         | Issue/PR templates (steps 5-6)         | --                   |


Step 0 lands first because AGENTS.md is injected into every prompt and establishes the architectural vocabulary. Steps 1-4 are sequential. Steps 5-9 are independent and can be parallelized. Step 10 depends on earlier steps.

In practice, Step 0 is Phase 0, Steps 1-4 map to Phase 1 and Phase 2 PRs, and Steps 5-10 map to one or more Phase 3 PRs. Phase 4 should be planned separately before any implementation PRs are opened.

---

## Out of Scope

- **New skills in Phases 0–3** — the 7 existing development skills are sufficient for the work in this plan; Phase 4 captures future skill additions that require a separate planning pass
- **LLM-powered issue triage** — deliberate choice to keep triage deterministic
- **Vouch system** — defer until external contributor volume warrants it
- **CI/CD changes** — existing workflows are solid
- **Full architecture docs** — create skeleton only, populate incrementally
- **Dependabot / Renovate** — dependency management automation (separate concern)

## Resolved Questions

1. **Reference doc naming** — `STYLEGUIDE.md` and `DEVELOPMENT.md` (no suffix).
2. **README tone** — Minimal. A sentence or two near the development installation section. The README is for users, not contributors.
3. **CODEOWNERS granularity** — Keep it simple: all data designer maintainers for now. No separate agent infra group until the need is clear.
4. **Symlink compatibility** — Prefer symlinks. They've been used in popular repos and are cleaner than mirrored directories. If a harness doesn't resolve them, fall back to mirrored directories with a drift-check task.

## Open Questions

1. **`development-task.yml` audience** — Keep it public for all contributors, or narrow it to maintainers/internal work?
2. **Development vs. dataset-building agents** — How should the repo handle an agent that is both developing DataDesigner and using it to build datasets in the same session (e.g., running a tutorial notebook to verify a change)? Should there be an explicit context-switching mechanism, or is the AGENTS.md redirect plus skill metadata sufficient?
