---
name: review-code
description: Perform a thorough code review of the current branch or a GitHub PR by number.
argument-hint: [pr-number] [special instructions]
disable-model-invocation: true
metadata:
    internal: true
---

# Review Code Changes

Perform a comprehensive code review of either the current branch or a specific GitHub pull request.

## Arguments

`$ARGUMENTS` determines the review mode:

**PR mode** — first argument is a number:
- `366` — review PR #366
- `366 focus on the API changes` — review PR #366 with a focus area

**Branch mode** — no number, or only instructions:
- *(empty)* — review current branch against `main`
- `compare against develop` — review against a different base
- `focus on the API changes` — review current branch with a focus area

Additional instructions work in both modes:
- `be strict about type annotations`
- `skip style nits`

## Step 1: Gather Changes

### If PR mode (argument starts with a number)

Run these commands in parallel using `gh`:

1. **PR details**: `gh pr view <number> --json title,body,author,baseRefName,headRefName,state,additions,deletions,changedFiles,commits,url`
2. **PR diff**: `gh pr diff <number>`
3. **PR files**: `gh pr diff <number> --name-only`
4. **PR commits**: `gh pr view <number> --json commits --jq '.commits[].messageHeadline'`
5. **Existing inline review comments**: `gh api repos/{owner}/{repo}/pulls/<number>/comments --paginate --jq '.[].body'`
5b. **Existing PR-level reviews** (top-level review bodies from "Review changes"): `gh api repos/{owner}/{repo}/pulls/<number>/reviews --paginate --jq '.[].body'`
6. **Repo info**: `gh repo view --json nameWithOwner -q '.nameWithOwner'`

Then get the PR branch locally for full file access. Prefer a **worktree** so your current branch and uncommitted work are untouched:

```bash
git fetch origin pull/<number>/head:pr-<number> --force
git worktree add /tmp/review-<number> pr-<number>
# Cleanup when done: git worktree remove /tmp/review-<number> && git branch -D pr-<number>
```

If worktrees aren't suitable, you can use `gh pr checkout <number>` (this switches your current branch — only if you have no uncommitted work). Run the rest of the review from `/tmp/review-<number>`.

If checkout isn't possible (e.g., external fork), use `gh api` to fetch file contents:

```bash
gh api repos/{owner}/{repo}/contents/{path}?ref={head-branch} --jq '.content' | base64 --decode
```

**Important checks:**
- If the PR number doesn't exist, inform the user
- If the PR is merged or closed, note the state but proceed (useful for post-merge audits)
- If the PR is a draft, note it — review may be on incomplete work
- For very large diffs (>3000 lines), fetch and read changed files individually instead of relying solely on the diff

### If Branch mode (no number)

First, fetch the base branch to ensure the remote ref is current:

0. **Fetch base**: `git fetch origin <base>`

Then run these commands in parallel:

1. **Current branch**: `git branch --show-current`
2. **Commits on branch**: `git log origin/<base>..HEAD --oneline`
3. **File changes summary**: `git diff --stat origin/<base>..HEAD`
4. **Full diff**: `git diff origin/<base>..HEAD`
5. **Uncommitted changes**: `git status --porcelain`
6. **Merge base**: `git merge-base origin/<base> HEAD`

Where `<base>` is `main` unless overridden in arguments.

**Important checks:**
- If no commits ahead of base, inform the user there's nothing to review
- If uncommitted changes exist, note them but review committed changes only
- For very large diffs (>3000 lines), read changed files individually instead of relying solely on the diff

## Step 2: Load Project Guidelines

Read the following files at the repository root to load the project's standards and conventions:

- **`AGENTS.md`** — architecture, layering, core design principles, structural invariants
- **`STYLEGUIDE.md`** — code style rules (formatting, naming, imports, type annotations), design principles (DRY, KISS, YAGNI, SOLID), common pitfalls, lazy loading and `TYPE_CHECKING` patterns
- **`DEVELOPMENT.md`** — testing patterns and expectations

Use these guidelines as the baseline for the entire review. Project-specific rules take precedence over general best practices.

## Step 3: Understand the Scope

Before diving into details, build a mental model:

1. **Read the PR description** (PR mode) or commit messages to understand the stated intent
2. **Read each commit message** to understand the progression of changes
3. **Group changed files** by module/package to identify which areas are affected
4. **Identify the primary goal** (feature, refactor, bugfix, etc.)
5. **Note cross-cutting concerns** (e.g., a rename that touches many files vs. substantive logic changes)
6. **Check existing feedback** (PR mode): inspect both inline comments (Step 1, item 5) and PR-level review bodies (Step 1, item 5b) so you don't duplicate feedback already given
7. **Check PR template conformance** (PR mode): verify the PR description includes the required template sections (📋 Summary, 🔗 Related Issue, 🔄 Changes, 🧪 Testing, ✅ Checklist). Flag missing or empty sections as a warning in the review. The template lives at `.github/PULL_REQUEST_TEMPLATE.md`

## Step 4: Review Each Changed File (Multi-Pass)

Perform **at least 2-3 passes** over the changed files. Each pass has a different focus — this catches issues that a single read-through would miss.

**Scope rule: Only flag issues introduced or modified by this changeset.** Read the full file for context, but do not report pre-existing patterns, style issues, or design choices that were already present before this branch/PR. If existing code was merely moved without modification, don't flag it. The goal is to review what the author changed, not audit the entire file.

### Pass 1: Correctness & Logic

Read each changed file in full (not just the diff), but evaluate only the **new or modified code**:

- Logic errors, off-by-one, wrong operator, inverted condition
- Missing edge case handling (None, empty collections, boundary values)
- Truthy/falsy checks on values where 0, empty string, or None is valid (e.g. `if index:` when index can be 0)
- Defensive `getattr(obj, attr, fallback)` or `.get()` on Pydantic models where the field always exists with a default
- Silent behavior changes for existing users that aren't called out in the PR description
- Race conditions or concurrency issues
- Resource leaks (unclosed files, connections, missing cleanup)
- Incorrect error handling (swallowed exceptions, wrong exception type)
- Input validation at boundaries (user input, API responses, file I/O)
- Graceful degradation on failure

### Pass 2: Design, Architecture & API

Re-read the changed files with a focus on **structure and design of the new/modified code**:

- Does the change fit the existing architecture and patterns?
- Are new abstractions at the right level? (too abstract / too concrete)
- Single responsibility — does each new function/class do one thing?
- Are new dependencies flowing in the right direction?
- Could this introduce circular imports or unnecessary coupling?
- Are new or modified public signatures clear and minimal?
- Are return types precise (not overly broad like `Any`)?
- Could the new API be misused easily? Is it hard to use incorrectly?
- Are breaking changes to existing interfaces intentional and documented?
- Unnecessary wrapper functions or dead code left behind after refactors
- Scalability: in-memory operations that could OOM on large datasets
- Raw exceptions leaking instead of being normalized to project error types (see AGENTS.md / interface errors)
- Obvious inefficiencies introduced by this change (N+1 queries, repeated computation, unnecessary copies)
- Appropriate data structures for the access pattern

### Pass 3: Standards, Testing & Polish

Final pass focused on **project conventions and test quality for new/modified code only**:

**Testing:**
- Are new code paths covered by tests?
- Do new tests verify behavior, not implementation details? (Flag tests that only verify plumbing — e.g. "mock was called" — without exercising actual behavior.)
- Duplicate test setup across tests that should use fixtures or `@pytest.mark.parametrize`
- Prefer flat test functions over test classes unless grouping is meaningful
- Are edge cases tested?
- Are mocks/stubs used appropriately (at boundaries, not deep internals)?
- Do new test names clearly describe what they verify?

**Project Standards (from AGENTS.md and STYLEGUIDE.md) — apply to new/modified code only:**

Verify the items below on lines introduced or changed by this branch. Refer to `AGENTS.md` and `STYLEGUIDE.md` loaded in Step 2 for details and examples.

- License headers: if present, they should be correct (wrong year or format → suggest `make update-license-headers`; don't treat as critical if CI enforces this)
- `from __future__ import annotations` in new files
- Type annotations on new/modified functions, methods, and class attributes
- Modern type syntax (`list[str]`, `str | None` — not `List[str]`, `Optional[str]`)
- Absolute imports only (no relative imports)
- Lazy loading for heavy third-party imports via `lazy_heavy_imports` + `TYPE_CHECKING`
- Naming: snake_case functions starting with a verb, PascalCase classes, UPPER_SNAKE_CASE constants
- No vacuous comments — comments only for non-obvious intent
- Public before private ordering in new classes
- Design principles: DRY (extract on third occurrence), KISS (flat over clever), YAGNI (no speculative abstractions)
- Common pitfalls: no mutable default arguments, no unused imports, simplify where possible

## Step 5: Run Linter

Run the linter on all changed files (requires local checkout). Use the venv directly to avoid sandbox permission issues in some environments (e.g. Claude Code):

```bash
.venv/bin/ruff check <changed-files>
.venv/bin/ruff format --check <changed-files>
```

> **Note**: This runs ruff only on the changed files for speed. For a full project-wide check, use `make check-all` or `uv run ruff check` (and `ruff format --check`) without file arguments.

If the branch isn't checked out locally (e.g., external fork in PR mode), skip this step and note it in the review.

## Step 6: Produce the Review

Output a structured review using the format below. Use the **PR template** or **Branch template** for the overview depending on the mode.

Write the review to a **temporary markdown file outside the repository** (e.g. `/tmp/review-<pr-or-branch>.md`) so other agents or tools can consume it without polluting `git status`. Do not commit this file; treat it as ephemeral.

---

### Overview (PR mode)

| | |
|---|---|
| **PR** | [#\<number\> \<title\>](\<url\>) |
| **Author** | `<author>` |
| **Base** | `<base-branch>` |
| **Files changed** | `<count>` |
| **Insertions/Deletions** | `+<ins> / -<del>` |

### Overview (Branch mode)

| | |
|---|---|
| **Branch** | `<branch-name>` |
| **Commits** | `<count>` commits ahead of `<base>` |
| **Files changed** | `<count>` |
| **Insertions/Deletions** | `+<ins> / -<del>` |

### (Both modes)

**Summary**: 1-2 sentence description of what the changes accomplish. In PR mode, note whether the implementation matches the stated intent in the PR description.

### Findings

Group findings by severity. Format each finding as a heading + bullet list — do NOT use numbered lists:

```
**`path/to/file.py:42` — Short title**
- **What**: Concise description of the issue.
- **Why**: Why it matters.
- **Suggestion**: Concrete fix or improvement (with code snippet when helpful).
```

Separate each finding with a blank line. Use bold file-and-title as a heading line, then bullet points for What/Why/Suggestion. Never use numbered lists (`1.`, `2.`) for findings or their sub-items — they render poorly in terminals.

#### Critical — Must fix before merge
> Issues that would cause bugs, data loss, security vulnerabilities, or broken functionality.

#### Warnings — Strongly recommend fixing
> Design issues, missing error handling, test gaps, or violations of project standards that could cause problems later.

#### Suggestions — Consider improving
> Style improvements, minor simplifications, or optional enhancements that would improve code quality.

### PR Template Conformance (PR mode only)

Check that the PR description follows the repository template (`.github/PULL_REQUEST_TEMPLATE.md`):

- **📋 Summary** — present and describes the "why", not just the "what"
- **🔗 Related Issue** — links to an issue (`Fixes #NNN` or `Closes #NNN`), or explicitly states N/A
- **🔄 Changes** — bullet list of key changes
- **🧪 Testing** — checklist items checked or marked N/A with explanation
- **✅ Checklist** — all items addressed

Flag missing or empty sections as a warning. This check is skipped in branch mode.

### What Looks Good

Call out 2-3 things done well (good abstractions, thorough tests, clean refactoring, etc.). Positive feedback is part of a good review.

### Verdict

One of:
- **Ship it** — No critical issues, ready to merge
- **Ship it (with nits)** — Minor suggestions but nothing blocking
- **Needs changes** — Has issues that should be addressed before merge
- **Needs discussion** — Architectural or design questions that need team input

### Next steps (optional)

After the summary, you may suggest follow-ups when useful:

- Deep dive into a specific file or finding
- Check a specific concern in more detail
- Install dev deps and add smoke tests (e.g. `uv sync --all-extras`, then run tests or suggest minimal smoke tests)

---

## Review Principles

- **Only flag what's new**: Report issues introduced by this changeset — not pre-existing patterns or style in untouched code, unless explicitly asked by the user
- **Be specific**: "This could return None on line 42 when `items` is empty" not "handle edge cases better"
- **Suggest, don't just criticize**: Always pair a problem with a concrete suggestion
- **Distinguish severity honestly**: Don't inflate nits to warnings; don't downplay real issues
- **Consider intent**: Review what the author was trying to do, not what you would have done differently
- **Batch related issues**: If the same pattern appears in multiple places, note it once and list all locations
- **Read the full file**: Diff-only reviews miss context — always read the surrounding code, but only flag new issues
- **Don't repeat existing feedback**: In PR mode, check both inline comments and PR-level review bodies and skip issues already raised

**Do not flag (focus on what CI won't catch):**

- Issues that are supposed to be caught by CI (linter, typechecker, formatter) — mention "run `make check-all`" if relevant, but don't list every style nit
- Pre-existing issues on unmodified lines
- Pedantic nits that don't affect correctness or maintainability
- Intentional functionality or API changes that are clearly documented

## Edge Cases

- **No changes**: Inform user there's nothing to review
- **PR not found**: Inform user the PR number doesn't exist
- **Merged/closed PR**: Note the state, proceed with review anyway
- **Draft PR**: Note it's a draft; review may be on incomplete work
- **External fork**: Can't checkout locally — use `gh api` to fetch file contents and skip the linter step
- **Huge changeset** (>50 files): Summarize by module first, then review the most critical files in detail; ask user if they want the full file-by-file review
- **Only renames/moves**: Note that changes are structural and focus on verifying nothing broke
- **Only test changes**: Focus review on test quality, coverage, and correctness of assertions
- **Only config/docs changes**: Adjust review to focus on accuracy and completeness rather than code quality
