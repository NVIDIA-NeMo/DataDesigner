# Plans

Development planning artifacts. Each subdirectory holds the design documents for one body of
work — the approach, the trade-offs weighed, the affected subsystems, the delivery sequence, and,
once the work lands, what it actually shipped. [`CONTRIBUTING.md`](../CONTRIBUTING.md) asks for
one of these before building anything non-trivial.

These are **not user documentation and are never published**. [`fern/docs.yml`](../fern/docs.yml)
declares a single version source and all of it lives under `fern/versions/latest/pages/`; nothing
in `plans/` is reachable from the docs site. Read a plan as a record of one change, not as a
description of what the code does today.

## What belongs here

A plan belongs in `plans/` when it records the design and shipped outcome of a specific body of
work. Adjacent cases go elsewhere:

- How the system works today goes in [`architecture/`](../architecture/), which is maintained as
  the code changes.
- User-facing product documentation goes in [`fern/`](../fern/).
- Temporary notes go in `.scratch/`, which is gitignored and never committed.

## Naming

New plans go in `plans/<issue-number>/`, as [`CONTRIBUTING.md`](../CONTRIBUTING.md) specifies,
with no zero padding — `plans/790/`, not `plans/0790/`. One directory per plan, so supporting media
sits beside the document it belongs to.

Some existing directories are named after a workstream (for example `workflow-chaining/`) or a
pull request number rather than an issue. Do not use either form for a new plan.

## Document shape

Observed across the existing plans, not a schema to conform to:

- Optional YAML frontmatter: `date`, `authors`, and sometimes `status` or `issue`. Documents
  that set `status:` use it loosely (`draft`, `proposal`, `in-progress`) — read it as an
  author's note, not as a lifecycle the repository enforces.
- A `# Plan: <title>` heading. Most primary documents use it.
- A body that runs Summary or Problem → Motivation → Goals → Non-goals → Design.
  [`790/engine-native-record-selection.md`](790/engine-native-record-selection.md) and
  [`518/pr-hygiene-plan.md`](518/pr-hygiene-plan.md) are good references.
- `path:line` citations when pointing at code, so a reader can check the claim.
- kebab-case filenames.

A plan that grows past a single document gets an index — see
[`645/README.md`](645/README.md), where a `README.md` fronts the sibling documents and links each
by audience.

## Assets

Diagrams and images go beside the document, or in an `assets/` subdirectory
([`396/assets/`](396/assets/)). Both are in use.

For generated diagrams, commit the source beside the rendered images. [`645/`](645/) is an
example: its README makes the PlantUML file authoritative and asks that a change to it regenerate
the PNGs in the same diff.

## For agents

Keep plans factual. Link the issues and pull requests the plan relates to, cite code by `path:line`
rather than paraphrasing it, and name the open questions instead of resolving them by assumption.
Do not write user-facing prose here.

A plan's lifecycle:

1. Draft the plan and get it reviewed before implementation starts.
2. Keep it aligned with the work as it is built. A plan left describing an abandoned approach is
   worse than no plan.
3. Reconcile it with the delivered behavior before the implementing work is considered complete.
4. After that it stays as the historical record of that change, tied to its issue or pull request.
   A later change to the same behavior gets its own plan rather than an edit to this one.

Nothing refreshes plans automatically — the docs auto-fix job in
`.github/workflows/agentic-ci-daily.yml` does not include `plans/` in the paths it may touch. Even
an up-to-date plan does not tell you how the system works now; the code, `architecture/`, and
`fern/` do.

### How a plan PR is reviewed

Per [`.agents/recipes/pr-review/recipe.md`](../.agents/recipes/pr-review/recipe.md), a pull request
that only touches `plans/` is reviewed on four things:

1. **Completeness** — gaps, missing phases.
2. **Feasibility** — can the proposed approach actually be built.
3. **Alignment** — consistent with [`AGENTS.md`](../AGENTS.md) and the existing
   [`architecture/`](../architecture/) documents.
4. **Open questions** — are the unknowns identified rather than glossed over.

Linting and code-style checks are skipped.
