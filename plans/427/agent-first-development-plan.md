---
date: 2026-03-17
authors:
  - nmulepati
---

# Plan: Agent-First Development Ethos

## Problem

DataDesigner was built entirely by humans — and the codebase reflects that with strong architecture, comprehensive tests, and thoughtful design. But we are increasingly moving to an agent-assisted planning and development workflow. The project already has 7 agent skills, an agent introspection CLI, deep MCP integration, and a plugin system. None of this is visible from the three entry documents (README, CONTRIBUTING, AGENTS.md). The infrastructure exists but the front door doesn't signal it.

## Current State

| Asset | Status | Agent-First Signal |
|-------|--------|--------------------|
| README.md | Product-focused, usage-first | Zero mention of agent-first development |
| CONTRIBUTING.md | Standard OSS contributing guide | Zero mention of skills or agent workflows |
| AGENTS.md | ~500 lines, mixes code style with architecture | No skills inventory, no workflow chains |
| CLAUDE.md | 1 line: `@AGENTS.md` | Minimal |
| Issue templates | 4 templates (bug, feature, dev task, config) | No agent investigation fields |
| PR template | Doesn't exist | -- |
| CODEOWNERS | Catch-all `* @NVIDIA-NeMo/data_designer_reviewers` | No agent infra ownership |
| `.claude/skills/` | 7 skills (~25KB) | Claude Code-locked, invisible from top-level docs |
| `.agents/skills/` | Doesn't exist (yet) | -- |
| `architecture/` | Doesn't exist | -- |
| STYLEGUIDE.md | Doesn't exist (inlined in AGENTS.md) | -- |

## Principles

1. **Agent-assisted, not agent-only.** We are a human-built project adopting agent workflows. Agents accelerate planning, development, and review — humans make design decisions and own quality.
2. **Designed, not vibed.** Humans architect systems; agents help implement. The distinction matters and should be visible. This is not vibe coding.
3. **Firm gates, clear paths.** Issue templates encourage agent investigation. But the paths to success are well-lit for both agent-assisted and human-only contributors.
4. **Skills are the API.** The `.agents/skills/` directory is the contract between the project and contributor agents. Treat it as a first-class interface.
5. **Front door tells the story.** README, CONTRIBUTING, and AGENTS.md should signal that agent-assisted workflows are available and encouraged. A new contributor's immediate next step is "clone and point your agent at the repo."

---

## Phase 1: Foundation Documents

Three files define the front door. All three need to tell the same story.

### 1a. README.md

**Current state:** Product-focused. Zero signal of agent-first development.

**Target state:** Retains the product pitch but frames DataDesigner as an agent-friendly project. A new developer's immediate next step is "clone and point your agent at the repo."

| Section | Action |
|---------|--------|
| Hero / intro paragraph | Add a line signaling agent-first development alongside the product pitch |
| New: "Explore with Your Agent" | After quickstart. Clone the repo, point your agent at it, let it load the skills and answer your questions |
| New: "Built With Agents" | After the product sections. Surfaces `.agents/skills/` infrastructure, the workflow chains. This is the "how we work" section |
| Contributing | Expand from current brief mention to set expectations: agent-first contributions, link to CONTRIBUTING.md |

**Key language to establish:**
- "DataDesigner supports agent-assisted development. We provide skills that help agents plan, build, and review code."
- "Before opening an issue, try pointing your agent at the repo. It has skills to help."

### 1b. CONTRIBUTING.md

**Current state:** Standard OSS contributing guide. Welcoming tone, good fork→develop→PR workflow, but zero agent awareness.

**Target state:** Agent-assisted contribution workflow is the recommended path. Human-only paths are fully supported but agent workflows are encouraged and well-documented.

| Section | Action |
|---------|--------|
| New: Opening philosophy | 2-3 sentences. This project supports agent-assisted development. Your agent is a powerful collaborator — we provide skills to help it help you. |
| New: "Before You Open an Issue" | **The gate.** Checklist: (1) Clone the repo, (2) Point your agent at it, (3) Load relevant skills, (4) Have your agent diagnose/investigate. If the agent can't solve it, open an issue with the diagnostics attached. |
| New: "Agent Skills for Contributors" | Table of all skills grouped by workflow category |
| New: "Workflow Chains" | Document the natural pipelines: investigation → development, and future spike → build |
| Getting Started | Keep as-is (fork, clone, install) |
| Development Guide | Keep as-is |
| Pull Requests | Update to reference the new PR template and the `create-pr` skill |
| New: "When to Open an Issue" | Clear guidance: real bugs your agent confirmed, feature proposals with design context, problems the `search-docs`/`search-github` skills couldn't resolve |
| New: "When NOT to Open an Issue" | Questions about how things work (agent can answer), configuration problems (agent can diagnose), "how do I..." requests (agent has skills for this) |
| Commit Messages / DCO | Keep as-is |

**Skill groupings for the table:**

| Category | Skills | Purpose |
|----------|--------|---------|
| Getting Started | `search-docs`, `search-github` | Find information, check for duplicates |
| Data Generation | `new-sdg` | Design synthetic data generators interactively |
| Development | `commit`, `create-pr`, `update-pr` | Standard development cycle |
| Review | `review-code` | Multi-pass code review |

### 1c. AGENTS.md

**Current state:** ~500 lines. Mixes project overview, architecture, code style, type annotations, linting rules, design principles, and testing patterns into one file. No skills inventory, no workflow chains, no project identity statement.

**Target state:** The comprehensive entrypoint for any agent working on this codebase. Every agent reads this on load. It should give an agent everything it needs to be effective — without the code style reference (which moves to STYLEGUIDE.md).

| Section | Action |
|---------|--------|
| Opening | Keep the CONTRIBUTING.md reference. Add: "This file is the primary instruction surface for agents contributing to DataDesigner." |
| New: "Project Identity" | 3-4 sentences: agent-assisted development, designed not vibed, the product generates synthetic datasets and increasingly uses agents for planning and development |
| New: "Skills" | Note that skills live in `.agents/skills/`. Agent harnesses can discover and load them natively. |
| New: "Workflow Chains" | Document the natural skill pipelines |
| Architecture | Keep existing 3-layer overview, key files, registries. Add brief component map. |
| New: "Issue and PR Conventions" | Reference the templates. When creating issues, use the template format. When creating PRs, use the PR template. Skills should produce output conforming to these templates. |
| Development Workflow | Keep as-is (`uv`, `make`, test commands) |
| Working Guidelines | Keep as-is (license headers, `__future__` imports, comments) |
| Testing | Keep as-is, add brief guidance on when to run which tests |
| New: "Security" | Don't commit secrets, don't run destructive operations without confirmation, scope changes to the issue at hand |
| Pre-commit | Keep as-is |
| Column/Model Configuration | Keep as-is (brief summaries) |
| Registry System | Keep as-is |
| **REMOVE** Code Style sections | Move to STYLEGUIDE.md (see 1d) |

### 1d. STYLEGUIDE.md (new file)

Extract from AGENTS.md. Contains all code style reference material:

- General formatting (line length, quotes, indentation, target version)
- Type annotations (full section with all examples)
- Import style (absolute imports, lazy loading, TYPE_CHECKING — full section)
- Naming conventions (PEP 8 rules, verb-first function names)
- Code organization (public/private ordering, class method order, section comments)
- Design principles (DRY, KISS, YAGNI, SOLID — with examples)
- Common pitfalls (all 5 with code examples)
- Active linter rules (ruff rule reference)

**CLAUDE.md updated to:**
```
@AGENTS.md
@STYLEGUIDE.md
```

**Why:** AGENTS.md is loaded into every agent conversation. Code style is reference material — needed when writing code, not when triaging issues or creating spikes. Splitting reduces context cost and makes each file single-purpose.

### 1e. Create `architecture/` Directory (Skeleton)

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

## Phase 2: GitHub Machinery

### 2a. Issue Templates

Update existing `.github/ISSUE_TEMPLATE/` templates:

**bug-report.yml** updates:
- Add **Agent Diagnostic** (textarea, required): "Paste the output from your agent's investigation of this bug. What skills did it use? What did it find? If you haven't had your agent investigate, please do that first — see CONTRIBUTING.md."
- Add **Checklist** (required):
  - [ ] I pointed my agent at the repo and had it investigate this issue
  - [ ] I loaded relevant skills (e.g., `search-docs`, `search-github`)
  - [ ] My agent could not resolve this — the diagnostics above explain why
- Keep existing fields (priority, description, reproduction, expected behavior)

**feature-request.yml** updates:
- Keep existing fields (problem, solution, alternatives)
- Add **Agent Investigation** (textarea, optional): "If your agent explored the codebase to assess feasibility (e.g., using the `search-docs` skill), paste its findings here."
- Add **Checklist**:
  - [ ] I've reviewed existing issues and the documentation
  - [ ] This is a design proposal, not a "please build this" request

**config.yml** updates:
- Update contact link to: "Have a question? Point your agent at the repo. It has skills for searching docs, finding issues, and more. See CONTRIBUTING.md for the full list."

### 2b. PR Template

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

### 2c. CODEOWNERS

Update `.github/CODEOWNERS` to add agent infrastructure ownership:

```
# Broad ownership — core team reviews everything
* @NVIDIA-NeMo/data_designer_reviewers

# Agent infrastructure — tighter review
.agents/ @NVIDIA-NeMo/data_designer_reviewers
AGENTS.md @NVIDIA-NeMo/data_designer_reviewers
STYLEGUIDE.md @NVIDIA-NeMo/data_designer_reviewers
```

### 2d. Label Taxonomy

Create labels for workflow state:

| Label | Purpose | Used by |
|-------|---------|---------|
| `agent-ready` | Human-approved, agent can build | `build-from-issue` (future) |
| `review-ready` | Agent has posted a plan, needs human review | `build-from-issue`, `create-spike` (future) |
| `in-progress` | Agent is actively building | `build-from-issue` (future) |
| `pr-opened` | Implementation complete, PR submitted | `build-from-issue` (future) |
| `spike` | Needs deeper investigation | `create-spike` (future) |
| `needs-agent-triage` | Opened without agent diagnostics — redirect | Triage automation (future) |
| `good-first-issue` | Suitable for new contributors (with agents) | Manual |

---

## Phase 3: Skill & Agent Infrastructure

### 3a. Consolidate `.agents/` and `.claude/`

**Goal:** `.agents/` is the canonical home for all agent infrastructure. `.claude/` becomes a thin shim for Claude Code-specific runtime state.

**Current layout:**
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
1. Move `.claude/skills/` to `.agents/skills/` (done in prototype)
2. Move `.claude/agents/*.md` to `.agents/agents/*.md`
3. Create symlinks from `.claude/` and `.codex/`
4. Add `.claude/README.md` explaining the structure

### 3b. Update Skills to Conform to Templates

Skills that create GitHub artifacts should produce output matching the new templates:

**`create-pr`:**
- Produce PR descriptions matching the PR template structure (Summary / Related Issue / Changes / Testing / Checklist)
- Include the testing checklist populated based on what was actually run

**`review-code`:**
- When reviewing PRs created from templates, check that the template sections are properly filled

### 3c. Skill Cross-Reference Cleanup

- Verify all skill files reference `.agents/skills/` not `.claude/skills/`
- Verify sub-agent references point to `.agents/agents/`
- Ensure all cross-skill references use consistent naming

---

## Phase 4: Future Work (Separate PRs)

### 4a. New Skills

| Skill | Purpose | Depends on |
|-------|---------|------------|
| `build-from-issue` | Stateful plan → review → build → PR pipeline | Labels, `principal-engineer-reviewer` |
| `create-spike` | Investigate problem, create structured issue | `principal-engineer-reviewer` |
| `debug-sdg` | Debug failing data generation pipelines | Architecture docs |
| `generate-column-config` | Generate column configs from natural language | Agent introspection API |
| `watch-github-actions` | Monitor CI workflow runs | None |
| `sync-agent-infra` | Detect drift across agent files | Skills inventory |

### 4b. Sub-Agent Personas

| Agent | Purpose | Scope |
|-------|---------|-------|
| `principal-engineer-reviewer` | Analyze code, generate plans, review architecture | Read-only |
| `arch-doc-writer` | Update `architecture/` docs after features land | Write to `architecture/` only |

### 4c. Issue Triage Workflow

Create `.github/workflows/issue-triage.yml`:
- **Trigger:** `issues.opened`
- **Logic:** Check if the issue was created using the bug report template and the "Agent Diagnostic" field is empty or contains only placeholder text
- **Action:** Add the `needs-agent-triage` label and post a comment redirecting to CONTRIBUTING.md
- This is a simple deterministic check — not an LLM-powered triage bot

---

## Execution Order

| Step | Deliverable | Dependencies | Parallelizable |
|------|-------------|--------------|----------------|
| 1 | AGENTS.md restructure + STYLEGUIDE.md split | None | -- |
| 2 | CONTRIBUTING.md overhaul | AGENTS.md (references it) | -- |
| 3 | README.md updates | CONTRIBUTING.md (references it) | -- |
| 4 | Issue templates | CONTRIBUTING.md (templates link to it) | Yes (with 5-8) |
| 5 | PR template | None | Yes (with 4, 6-8) |
| 6 | CODEOWNERS update | None | Yes (with 4-5, 7-8) |
| 7 | Label creation (via `gh label create`) | None | Yes (with 4-6, 8) |
| 8 | Skill consolidation (`.agents/`, `.claude/` cleanup) | None | Yes (with 4-7) |
| 9 | `architecture/` skeleton | None | Yes (with 4-8) |
| 10 | Skill template conformance updates | Issue/PR templates (steps 4-5) | -- |

Steps 1-3 are sequential. Steps 4-9 are independent and can be parallelized. Step 10 depends on earlier steps.

---

## Out of Scope

- **New skills** — the 7 existing skills are sufficient; this plan surfaces them
- **LLM-powered issue triage** — deliberate choice to keep triage deterministic
- **Vouch system** — defer until external contributor volume warrants it
- **CI/CD changes** — existing workflows are solid
- **Full architecture docs** — create skeleton only, populate incrementally
- **Dependabot / Renovate** — dependency management automation (separate concern)

## Open Questions

1. **STYLEGUIDE.md naming** — `STYLEGUIDE.md` vs `CODE_STYLE.md` vs `STYLE.md`?
2. **Issue template strictness** — Should agent diagnostic be required (gate) or optional (encouraged) on bug reports? OpenShell requires it. DataDesigner could start with required and relax if it creates too much friction.
3. **README tone** — How prominent should the agent-first messaging be? A line in the hero paragraph, or a dedicated section?
4. **CODEOWNERS granularity** — Keep catch-all or add file-specific ownership for docs/CI?
5. **Phase 2 timing** — Land foundation docs (Phase 1) first and iterate, or ship Phases 1-2 together?
