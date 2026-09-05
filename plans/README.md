# Plans

Development planning artifacts. Each subdirectory holds the point-in-time design documents
written before non-trivial work started — the approach, the trade-offs weighed, the affected
subsystems, and the delivery sequence. [`CONTRIBUTING.md`](../CONTRIBUTING.md) asks for one of
these before building anything non-trivial.

These are **not user documentation and are never published**. [`fern/docs.yml`](../fern/docs.yml)
declares a single version source and all of it lives under `fern/versions/latest/pages/`; nothing
in `plans/` is reachable from the docs site. Read a plan as a record of what the author intended
at the time, not as a description of what the code does today.

## Where a document belongs

| Directory | Holds | Describes |
| --- | --- | --- |
| `plans/` | Design documents written before the work | Work not yet built |
| [`architecture/`](../architecture/) | [`overview.md`](../architecture/overview.md) plus nine subsystem documents | Shipped code |
| `fern/versions/latest/pages/**` | Published product documentation | What users are told |
| [`docs/`](../docs/) | Support files consumed by the Fern build (see [`docs/README.md`](../docs/README.md)) | Not prose |

If a plan's content has shipped and readers need it to understand the running system, it belongs
in `architecture/`, not here.

## Naming

Use `plans/<issue-number>/`, no zero padding — `plans/790/`, not `plans/0790/`. One directory per
plan, so supporting media sits beside the document it belongs to.

`plans/<workstream-name>/` is equally current for work with no single tracking issue.
`workflow-chaining/`, `check-models/`, and `remote-filesystem-seeds/` are all named this way and
all postdate the numbered convention.

Two directories — `299/` and `788/` — key off the implementing **pull request** number, because
the plan landed in the same change as the code. Prefer the issue number for new plans.

## Document shape

Observed across the existing plans, not a schema to conform to:

- Optional YAML frontmatter: `date`, `authors`, and sometimes `status` or `issue`. The handful of
  documents that set `status:` use it loosely (`draft`, `proposal`, `in-progress`) — read it as an
  author's note, not as a lifecycle the repository enforces.
- A `# Plan: <title>` heading. Most primary documents use it.
- A body that runs Summary or Problem → Motivation → Goals → Non-goals → Design.
  [`790/engine-native-record-selection.md`](790/engine-native-record-selection.md) and
  [`518/pr-hygiene-plan.md`](518/pr-hygiene-plan.md) are good references.
- `path:line` citations when pointing at code, so a reader can check the claim.
- kebab-case filenames. `392/refactor_managed_personas_plan.md` is the one snake_case holdout.

A plan that grows past a single document gets an index — see
[`645/README.md`](645/README.md), where a `README.md` fronts the sibling documents and links each
by audience.

## Assets

Diagrams and images go beside the document, or in an `assets/` subdirectory
([`396/assets/`](396/assets/)). Both are in use.

For generated diagrams the source file is authoritative. `645/` states the rule for its PlantUML
diagrams and it applies generally: a change to the source must regenerate the images in the same
diff, or say explicitly why rendering was unavailable.

## For agents

Keep plans factual. Link the issues and pull requests the plan relates to, cite code by `path:line`
rather than paraphrasing it, and name the open questions instead of resolving them by assumption.
Do not write user-facing prose here.

Plans are point-in-time and nothing refreshes them automatically —
`.github/workflows/agentic-ci-daily.yml` excludes `plans/` from the paths its docs auto-fix job is
allowed to touch. When work changes shape, update the plan or supersede it in the same pull request
that changes the work. A plan left describing an approach that was abandoned is worse than no plan.

### How a plan PR is reviewed

Per [`.agents/recipes/pr-review/recipe.md`](../.agents/recipes/pr-review/recipe.md), a pull request
that only touches `plans/` is reviewed on four things:

1. **Completeness** — gaps, missing phases.
2. **Feasibility** — can the proposed approach actually be built.
3. **Alignment** — consistent with [`AGENTS.md`](../AGENTS.md) and the existing
   [`architecture/`](../architecture/) documents.
4. **Open questions** — are the unknowns identified rather than glossed over.

Linting and code-style checks are skipped.
