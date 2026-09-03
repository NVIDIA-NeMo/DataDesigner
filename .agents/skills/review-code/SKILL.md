---
name: review-code
description: Perform a thorough code review of the current branch or a GitHub PR by number.
argument-hint: "[pr-number] [special instructions]"
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

**Documentation sources (load when the changeset touches matching areas):**

- **`architecture/*.md`** — subsystem maps aligned with `packages/` (e.g. `engine/mcp/` ↔ `architecture/mcp.md`). Use to verify the PR does not leave recorded architecture false relative to new behavior.
- **`fern/versions/latest/pages/`** — published user-facing documentation. Cross-check when public API, CLI behavior, or config surface changes would affect what readers are told.

Use these guidelines as the baseline for the entire review. Project-specific rules take precedence over general best practices.

## Step 3: Understand the Scope

Before diving into details, build a mental model:

1. **Read the PR description** (PR mode) or commit messages to understand the stated intent
2. **Read each commit message** to understand the progression of changes
3. **Group changed files** by module/package to identify which areas are affected
4. **Identify the primary goal** (feature, refactor, bugfix, etc.)
5. **Note cross-cutting concerns** (e.g., a rename that touches many files vs. substantive logic changes)
6. **Check existing feedback** (PR mode): inspect both inline comments (Step 1, item 5) and PR-level review bodies (Step 1, item 5b) so you don't duplicate feedback already given
7. **Classify the contract impact**: note whether the change introduces or modifies supported public symbols or exports, config or model schemas, serialized formats, extension/plugin boundaries, or builder APIs. If none apply, skip items 8-10 and continue to Step 3.5.
8. **Inventory the changed contract**: list each changed contract element—including symbols, schema fields, serialized representations, extension entry points, and builder operations—and record its exposure, stability commitment, and expected callers or consumers
9. **Map invariant ownership**: identify where each new rule is parsed, normalized, validated, resolved, and compiled so duplicated checks and unclear responsibility are visible before the detailed review
10. **Find the closest existing analogue**: compare the complete neighboring API surface—not just individual names—for builder vocabulary, type precision, export strategy, errors, and contract clarity

## Step 3.5: Structural Impact (if available)

Check for a pre-computed structural impact analysis at
`/tmp/structural-impact-<pr-or-branch>.md`. This file is produced by
`graphify` AST extraction and contains:

- **Risk level** (LOW/MEDIUM/HIGH) based on god nodes touched, import
  violations, and cluster spread
- **Core abstractions modified** - the most-connected entities in the
  codebase (high blast radius if changed)
- **Import direction violations** - cross-package edges that violate the
  layering rule (interface -> engine -> config)
- **High-connectivity changes** - entities with many dependents
- **Cross-package dependencies** - edges crossing package boundaries

If the file exists, read it and use it to calibrate your review:

- **HIGH risk**: apply extra scrutiny in Pass 2 (Design & Architecture).
  Verify backward compatibility for god nodes. Check that cross-package
  changes don't break existing callers.
- **Import violations**: flag real dependency direction issues (not just
  inferred edges) as at least Warnings. Use Critical when the violation
  breaks packaging or establishes a foundational public boundary in the
  wrong layer.
- **LOW risk**: the structural analysis confirms a localized change. You
  can focus more on correctness (Pass 1), but still perform the mandatory
  public-contract checks below when applicable.

If the file does not exist (e.g. local branch review), skip this step.

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
- Single responsibility — does each new function, class, and module do one coherent job?
- If one module owns distinct phases such as authored config, resolution, compilation, and validation, would splitting those responsibilities make ownership and future changes clearer?
- Are new dependencies flowing in the right direction?
- Could this introduce circular imports or unnecessary coupling?
- Does every symbol live in the module that owns it? Treat a leading-underscore helper imported by sibling modules as a signal to inspect ownership and package-export intent. Package-private sharing can be legitimate when the defining module owns the helper and it remains intentionally unexported.
- Are helpers scoped to their actual consumers? Logic used only to validate or construct one class usually belongs on that class instead of in module-level generic machinery.
- Are re-exports intentional and sourced from the defining module rather than incidental import chains?
- Is there one canonical public entry point per operation, or do class methods plus module-level wrappers expose redundant APIs?
- Does each business invariant have one authoritative owner, with complementary boundary checks allowed when representation or trust level changes and without independently redefining the rule?
- Are new or modified public signatures clear and minimal?
- Do generic helpers and `TypeVar`s eliminate enough real duplication to justify their cognitive cost, and are their names consistent with those already used in the changed module?
- Are return types precise (not overly broad like `Any`)?
- Could the new API be misused easily? Is it hard to use incorrectly?
- Are breaking changes to existing interfaces intentional and documented?
- Dead code left behind after refactors
- Scalability: in-memory operations that could OOM on large datasets
- Raw exceptions leaking instead of being normalized to project error types (see AGENTS.md / interface errors)
- Obvious inefficiencies introduced by this change (N+1 queries, repeated computation, unnecessary copies)
- Appropriate data structures for the access pattern

**Public interface inventory (mandatory when Step 3 identifies contract impact):**

Review every item in the Step 3 inventory individually, using the invariant map and neighboring analogue built there. Do not infer that a coherent-looking module implies a coherent public API.

- Does each name identify the value's actual role or lifecycle when callers need that distinction?
- Does each parameter or field name match the concrete value it contains rather than an earlier or later representation?
- Are public Python inputs expressed with concrete config/model classes where those classes already define the contract? Broad `Mapping[str, object]`, `dict[str, Any]`, or `**kwargs` inputs add normalization branches and weaken discoverability unless accepting arbitrary mappings is an explicit requirement.
- Does the API vocabulary match neighboring builders and established cardinality? For example, repeatable collection operations should follow the existing `add_*` convention, while `with_*` should retain its established singleton/configuration meaning.
- Is every exported symbol meant to become a compatibility commitment? Keep implementation bases, compiler internals, and shared utilities private unless external callers need them.
- Are exceptions named and located at the right subsystem boundary? Avoid generic names that become ambiguous when imported, and prefer the project's canonical error hierarchy over parallel local taxonomies.
- Can callers determine parameter semantics, accepted representations, return values, lifecycle transitions, and raised project errors from the public contract itself? Flag ambiguous or incomplete API design separately from project-standard docstring coverage.

**Boundary and representation checks:**

- Treat plugin-owned or opaque payloads as opaque. Core code may validate a stable envelope it owns, but should not infer extension semantics from familiar-looking keys or current built-in payload shapes.
- Test the boundary with an unfamiliar future extension value. If valid unknown fields or plugin payloads are rejected, stripped, or reinterpreted, the abstraction is not actually extensible.
- At exception boundaries, inspect the public error and how actionable detail is preserved. Use a safe chained cause when it will not expose sensitive provider data; otherwise put sanitized detail on the public error and suppress the raw cause with `raise ... from None` according to boundary policy.

**Documentation alignment (same pass — scoped, not a full docs audit):**

When **code** under `packages/` changes behavior, structure, or public contracts in a way that a maintainer would reasonably describe in `architecture/` or Fern docs:

1. Identify the closest **`architecture/<topic>.md`** (and any obvious `fern/versions/latest/pages/` pages) for that subsystem.
2. If the PR **also edits** those docs, sanity-check that the edits match the code.
3. If the PR **does not** edit docs but the change **contradicts** what `architecture/` or Fern docs currently assert, flag it (**Warnings** if contributors rely on that text; **Suggestions** if impact is narrow). Suggest updating the same PR or an explicit follow-up issue.
4. **Skip** this check for pure refactors with no observable behavior change, typo-only PRs, or changes already limited to documentation.

The local **`search-docs`** skill can help locate Fern docs pages by topic when the right file is not obvious.

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
- For new public contracts, do tests exercise direct supported construction as well as builder/factory paths, lifecycle transitions, malformed boundary input, and an unfamiliar extension/plugin case where applicable?
- For normalized exceptions, do tests assert the stable public error and verify that actionable detail is preserved through either a safe chained cause or sanitized public-error detail without leaking provider secrets?

**Project Standards (from AGENTS.md and STYLEGUIDE.md) — apply to new/modified code only:**

Verify the items below on lines introduced or changed by this branch. Refer to `AGENTS.md` and `STYLEGUIDE.md` loaded in Step 2 for details and examples.

- License headers: if present, they should be correct (wrong year or format → suggest `make update-license-headers`; don't treat as critical if CI enforces this)
- `from __future__ import annotations` in new files
- Type annotations on new/modified functions, methods, and class attributes
- Modern type syntax (`list[str]`, `str | None` — not `List[str]`, `Optional[str]`)
- Absolute imports only (no relative imports)
- Lazy loading for heavy third-party imports via `lazy_heavy_imports` + `TYPE_CHECKING`
- Naming: snake_case functions starting with a verb, PascalCase classes, UPPER_SNAKE_CASE constants
- Public API classes and functions have Google-style docstrings; private helpers need them only when their logic is non-obvious
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

For new or substantially expanded production modules, also use complexity rules as an exploratory signal:

```bash
.venv/bin/ruff check --select C901,PLR0912,PLR0913,PLR0915 <changed-production-files>
```

Do not report a metric violation by itself. Use it to identify functions or modules that need a closer cohesion, responsibility, and maintainability review, then report the concrete design problem if one exists.
Expect broad signals in config- or builder-heavy modules; pursue a violation only when it coincides with a concrete cohesion or ownership concern.

If the branch isn't checked out locally (e.g., external fork in PR mode), skip this step and note it in the review.

## Tone

Write as a supportive teammate, not a gatekeeper. The goal is to help the author ship great code, not to prove you found problems.

- **Be cordial and collaborative.** Use "we" language and frame genuine alternatives as questions or suggestions ("Could we …?", "What do you think about …?", "Nice approach — one thought: …"). State confirmed merge blockers directly.
- **Assume good intent.** If something looks off, ask before assuming it's wrong — the author may have context you don't.
- **Lead with what's good.** Acknowledge effort and smart decisions before raising concerns.
- **Keep it conversational.** Avoid stiff, formal phrasing. Write the way you'd talk to a colleague at a whiteboard.
- **Be direct, not blunt.** Clearly state what needs to change and why, but without harsh or commanding language ("This must be fixed" → "This could bite us in production — worth addressing before merge").

## Step 6: Produce the Review

Write the review as **GitHub-flavored Markdown** ready to post as a PR comment. Save it to a temporary file outside the repository (e.g. `/tmp/review-<pr-or-branch>.md`) so it doesn't pollute `git status`. Do not commit this file; treat it as ephemeral.

Use the template below exactly — omit a severity section if it has no findings, include Open Questions only with **Needs discussion**, and omit Residual Risk when empty, but keep all other sections.

---

Open with a brief, genuine thank-you to the author (e.g. "Thanks for putting this together, @author!" or "Nice work on this one, @author — here are my thoughts."). Keep it to one sentence; don't over-do it. Do NOT add a top-level title like "## Code Review" — the comment speaks for itself.

### Summary

1-2 sentence description of what the changes accomplish. In PR mode, note whether the implementation matches the stated intent in the PR description.

### Findings

Group findings by severity. Omit any severity section that has no findings. Format each finding as a heading + bullet list — do NOT use numbered lists:

```
**`path/to/file.py:42` — Short title**
- **What**: Concise description of the issue.
- **Why**: Why it matters.
- **Suggestion**: Concrete fix or improvement (with code snippet when helpful).
```

Separate each finding with a blank line. Use bold file-and-title as a heading line, then bullet points for What/Why/Suggestion. Never use numbered lists (`1.`, `2.`) for findings or their sub-items.

#### Critical — Let's fix these before merge
> Issues that would cause bugs, data loss, security vulnerabilities, or broken functionality; also foundational flaws in a new or materially changed public, serialized, or extension contract that should block merge because correcting them after release would require a breaking compatibility change.

#### Warnings — Worth addressing
> Significant design issues that are not foundational compatibility blockers, missing error handling, test gaps, or violations of project standards that could cause problems later.

#### Suggestions — Take it or leave it
> Style improvements, minor simplifications, or optional enhancements that would improve code quality.

### What Looks Good

Call out 2-3 things done well (good abstractions, thorough tests, clean refactoring, etc.). Be genuine — positive feedback is part of a good review and helps the author know what to keep doing.

### Open Questions (Needs discussion only)

State the material question, why it affects compatibility or merge readiness, and the team decision needed.

### Residual Risk (optional)

Note validation gaps or lower-impact uncertainty that do not change merge readiness. If a risk is concrete enough to require action before merge, report it as a finding instead.

### Verdict

Choose the verdict that matches the **highest severity confirmed finding** in the review. If there are no confirmed Critical or Warning findings but a design question material to compatibility or merge readiness cannot be resolved without team input, use **Needs discussion**.

- **Ship it** — No findings. Ready to merge as-is.
- **Ship it (with nits)** — Only Suggestions (see above — style improvements, simplifications, or optional enhancements). Nothing blocking.
- **Needs changes** — Any Critical or Warning findings. List the items that must be addressed before merge.
- **Needs discussion** — No confirmed Critical or Warning findings, but an architectural or design question material to compatibility or merge readiness needs team input before a decision can be made.

A confirmed foundational contract flaw is Critical and requires **Needs changes**. Reserve **Needs discussion** for material uncertainty that requires team input. Lower-impact uncertainty belongs in **Residual Risk** and does not preclude **Ship it** or **Ship it (with nits)**.

### Signature (PR mode only)

When the review will be posted as a PR comment, end with a signature line so readers can distinguish agent-generated reviews from human ones:

```
---
*This review was generated by an AI assistant.*
```

In branch mode (local only), omit the signature.

---

## Step 7: Post the Review (PR mode only)

In PR mode, display the review to the user and note the temp file path (`/tmp/review-<number>.md`). Then ask if they'd like you to post it as a PR comment. Only run the command after the user confirms:

```bash
gh pr comment <number> --body-file /tmp/review-<number>.md
```

In branch mode, skip this step — display the review to the user and note the temp file path.

---

## Review Principles

- **Only flag what's new**: Report issues introduced by this changeset — not pre-existing patterns or style in untouched code, unless explicitly asked by the user
- **Be specific**: "This could return None on line 42 when `items` is empty" not "handle edge cases better"
- **Suggest, don't just criticize**: Always pair a problem with a concrete suggestion
- **Distinguish severity honestly**: Don't inflate cosmetic naming or internal refactors into blockers. Do treat foundational public-contract design as Critical when merging would create a hard-to-reverse compatibility commitment.
- **Consider intent**: Review what the author was trying to do, not what you would have done differently
- **Batch related issues**: If the same pattern appears in multiple places, note it once and list all locations
- **Read the full file**: Diff-only reviews miss context — always read the surrounding code, but only flag new issues
- **Don't repeat existing feedback**: In PR mode, check both inline comments and PR-level review bodies and skip issues already raised

**Do not flag (focus on what CI won't catch):**

- Issues that are supposed to be caught by CI (linter, typechecker, formatter) — mention "run `make check-all`" if relevant, but don't list every style nit
- Pre-existing issues on unmodified lines
- Pedantic nits that don't affect correctness or maintainability
- Missing docstrings on private helpers when their purpose is clear, or prose-only comment/docstring nits that do not affect correctness or understanding; public API docstrings remain required by `STYLEGUIDE.md`
- Intentional functionality or API changes that are clearly documented, unless the new design introduces a concrete correctness, extensibility, or compatibility risk, or a maintainability risk that would require a breaking change to correct

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
