# RFC: Migration from MkDocs to Fern Docs

**Status:** Draft  
**Author:** [Author Name]  
**Owner:** [Owner Name]  
**Created:** 2026-01-14  
**Last Updated:** 2026-01-14
**Target Completion:** [YYYY-MM-DD]

---

## Summary

This RFC proposes migrating the NeMo Data Designer documentation from MkDocs Material to [Fern Docs](https://buildwithfern.com/learn/docs/getting-started/overview). The migration will be performed incrementally by creating a new `docs-fern/` directory, preserving all existing content while adapting to Fern's component system.

## Motivation

This migration is **mandated** as part of NVIDIA's documentation platform standardization initiative.

**Additional benefits:**

- **Modern documentation platform**: Fern offers AI-native features including Ask Fern and auto-generated MCP servers
- **Enhanced API documentation**: Better support for API reference documentation from OpenAPI specs
- **Improved developer experience**: Rich component library with interactive elements
- **Self-hosting options**: Flexible deployment for enterprise requirements

## Scope

### In Scope

- 1:1 content migration (no content changes)
- Component mapping from MkDocs Material to Fern equivalents
- Navigation structure preservation
- Code reference documentation migration

### Out of Scope

- Content rewrites or restructuring
- New features or sections
- Removal of existing documentation

---

## Current Documentation Inventory

### File Structure

```
docs/
├── index.md                      # Home page
├── installation.md               # Installation guide
├── quick-start.md                # Quick start tutorial
├── CONTRIBUTING.md               # Contribution guide
├── concepts/
│   ├── columns.md
│   ├── validators.md
│   ├── processors.md
│   ├── person_sampling.md
│   └── models/
│       ├── default-model-settings.md
│       ├── custom-model-settings.md
│       ├── configure-model-settings-with-the-cli.md
│       ├── model-providers.md
│       ├── model-configs.md
│       └── inference-parameters.md
├── recipes/
│   ├── cards.md
│   ├── code_generation/
│   │   ├── text_to_python.md
│   │   └── text_to_sql.md
│   └── qa_and_chat/
│       ├── product_info_qa.md
│       └── multi_turn_chat.md
├── plugins/
│   ├── overview.md
│   ├── example.md
│   └── available.md
├── code_reference/               # Auto-generated API docs
│   ├── models.md
│   ├── column_configs.md
│   ├── config_builder.md
│   ├── data_designer_config.md
│   ├── sampler_params.md
│   ├── validator_params.md
│   ├── processors.md
│   └── analysis.md
├── colab_notebooks/              # Jupyter notebooks
│   ├── 1-the-basics.ipynb
│   ├── 2-structured-outputs-and-jinja-expressions.ipynb
│   ├── 3-seeding-with-a-dataset.ipynb
│   └── 4-providing-images-as-context.ipynb
├── assets/
│   └── recipes/                  # Downloadable code files
├── css/                          # Custom styles
├── js/                           # Custom scripts
└── overrides/                    # MkDocs template overrides
```

### Current Navigation Structure

```yaml
nav:
  - Getting Started:
      - Welcome: index.md
      - Installation: installation.md
      - Quick Start: quick-start.md
      - Contributing: CONTRIBUTING.md
  - Concepts:
      - Models: (6 sub-pages)
      - Columns: concepts/columns.md
      - Validators: concepts/validators.md
      - Processors: concepts/processors.md
      - Person Sampling: concepts/person_sampling.md
  - Tutorials:
      - Overview + 4 Jupyter notebooks
  - Recipes:
      - Recipe Cards + 4 recipes
  - Plugins:
      - 3 pages
  - Code Reference:
      - 8 auto-generated API docs
```

---

## Component Mapping

### MkDocs → Fern Component Equivalents

Reference: [Fern Components Overview](https://buildwithfern.com/learn/docs/writing-content/components/overview)

| MkDocs Feature | Current Syntax | Fern Equivalent | Notes |
|----------------|----------------|-----------------|-------|
| **Admonitions** | `!!! note "Title"` | `<Note>`, `<Tip>`, `<Warning>`, `<Info>` | See [Callouts](#1-admonitions--callouts) |
| **Tabbed Content** | `=== "Tab 1"` | `<Tabs>` + `<Tab>` | See [Tabs](#2-tabbed-content) |
| **Code Blocks** | ` ```python ` | ` ```python ` | Direct compatibility |
| **Code Snippets** | `--8<-- "path"` | `<CodeBlock>` with `src` | File embedding |
| **Grid Cards** | `<div class="grid cards">` | `<Cards>` + `<Card>` | See [Cards](#3-grid-cards) |
| **Icons** | `:material-xxx:` | Fern icons or inline SVG | Limited support |
| **Download Links** | `{ .md-button download=}` | Standard markdown links | Simplified |
| **API Docs** | `::: module.path` | Manual or OpenAPI import | See [API Reference](#4-api-reference) |
| **Jupyter Notebooks** | `.ipynb` files | Convert to MDX or embed | See [Notebooks](#5-jupyter-notebooks) |
| **Versioning** | Mike plugin | Fern versioning | Built-in support |

---

## Detailed Component Migrations

### 1. Admonitions → Callouts

**Current MkDocs syntax:**

```markdown
!!! note "The Declarative Approach"
    Columns are **declarative specifications**. You describe *what* you want...

!!! tip "Conditional Sampling"
    Samplers support **conditional parameters**...

!!! question "New to Data Designer?"
    Recipes provide working code...

!!! warning "Important"
    This action cannot be undone.
```

**Fern equivalent:**

```mdx
<Note title="The Declarative Approach">
Columns are **declarative specifications**. You describe *what* you want...
</Note>

<Tip title="Conditional Sampling">
Samplers support **conditional parameters**...
</Tip>

<Info title="New to Data Designer?">
Recipes provide working code...
</Info>

<Warning title="Important">
This action cannot be undone.
</Warning>
```

**Migration mapping:**

| MkDocs Admonition | Fern Callout |
|-------------------|--------------|
| `!!! note` | `<Note>` |
| `!!! tip` | `<Tip>` |
| `!!! info` | `<Info>` |
| `!!! warning` | `<Warning>` |
| `!!! question` | `<Info>` |
| `!!! danger` | `<Warning>` |

### 2. Tabbed Content

**Current MkDocs syntax (installation.md):**

```markdown
=== "pip"

    ```bash
    pip install data-designer
    ```

=== "uv"

    ```bash
    uv add data-designer
    ```
```

**Fern equivalent:**

```mdx
<Tabs>
  <Tab title="pip">
    ```bash
    pip install data-designer
    ```
  </Tab>
  <Tab title="uv">
    ```bash
    uv add data-designer
    ```
  </Tab>
</Tabs>
```

### 3. Grid Cards

**Current MkDocs syntax (recipes/cards.md):**

```markdown
<div class="grid cards" markdown>

-   :material-snake:{ .lg .middle } **Text to Python**

    Generate a dataset of natural language instructions...

    ---

    **Demonstrates:**
    - Python code generation
    - Python code validation

    ---

    [:material-book-open-page-variant: View Recipe](code_generation/text_to_python.md){ .md-button }

</div>
```

**Fern equivalent:**

```mdx
<Cards>
  <Card
    title="Text to Python"
    icon="code"
    href="/recipes/code-generation/text-to-python"
  >
    Generate a dataset of natural language instructions...
    
    **Demonstrates:**
    - Python code generation
    - Python code validation
  </Card>
</Cards>
```

### 4. API Reference (mkdocstrings)

**Current MkDocs syntax (code_reference/models.md):**

```markdown
# Models

The `models` module defines configuration objects...

::: data_designer.config.models
```

**Fern options:**

**Option A: Manual Documentation**
Convert auto-generated docs to manually written MDX with code examples.

**Option B: OpenAPI Integration**
If the API has an OpenAPI spec, use Fern's native API reference generation.

**Option C: TypeDoc/PyDoc Integration**
Use Fern's SDK documentation features if available.

**Recommendation:** Start with Option A (manual) for initial migration, evaluate automation options post-migration.

### 5. Jupyter Notebooks

**Current approach:** `mkdocs-jupyter` plugin renders `.ipynb` files directly.

**Fern options:**

**Option A: Convert to MDX**
Convert notebooks to MDX files with code blocks and output screenshots.

**Option B: Embed as iframes**
Host notebooks on Colab/GitHub and embed links.

**Option C: Use Fern's code playground**
If available, use interactive code features.

**Recommendation:** Convert to MDX with static code blocks and link to Colab for interactive experience (preserves current Colab badge functionality).

### 6. Code Snippets (pymdownx.snippets)

**Current MkDocs syntax:**

```markdown
```python
--8<-- "assets/recipes/code_generation/text_to_python.py"
```
```

**Fern equivalent:**

```mdx
<CodeBlock src="assets/recipes/code_generation/text_to_python.py" />
```

Or inline the code directly if file embedding isn't supported.

---

## Proposed Directory Structure

```
docs-fern/
├── fern.config.json              # Fern configuration
├── docs.yml                      # Navigation and settings
├── pages/
│   ├── index.mdx                 # Home page
│   ├── installation.mdx
│   ├── quick-start.mdx
│   ├── contributing.mdx
│   ├── concepts/
│   │   ├── columns.mdx
│   │   ├── validators.mdx
│   │   ├── processors.mdx
│   │   ├── person-sampling.mdx
│   │   └── models/
│   │       ├── default-model-settings.mdx
│   │       ├── custom-model-settings.mdx
│   │       ├── configure-with-cli.mdx
│   │       ├── model-providers.mdx
│   │       ├── model-configs.mdx
│   │       └── inference-parameters.mdx
│   ├── tutorials/
│   │   ├── overview.mdx
│   │   ├── the-basics.mdx
│   │   ├── structured-outputs.mdx
│   │   ├── seeding-with-dataset.mdx
│   │   └── images-as-context.mdx
│   ├── recipes/
│   │   ├── index.mdx             # Recipe cards
│   │   ├── code-generation/
│   │   │   ├── text-to-python.mdx
│   │   │   └── text-to-sql.mdx
│   │   └── qa-and-chat/
│   │       ├── product-info-qa.mdx
│   │       └── multi-turn-chat.mdx
│   ├── plugins/
│   │   ├── overview.mdx
│   │   ├── example.mdx
│   │   └── available.mdx
│   └── api-reference/
│       ├── models.mdx
│       ├── column-configs.mdx
│       ├── config-builder.mdx
│       ├── data-designer-config.mdx
│       ├── sampler-params.mdx
│       ├── validator-params.mdx
│       ├── processors.mdx
│       └── analysis.mdx
├── assets/
│   ├── favicon.png
│   └── recipes/                  # Downloadable code files
│       ├── code_generation/
│       └── qa_and_chat/
└── styles/
    └── custom.css                # Custom styling (if needed)
```

---

## URL Redirect Mapping

To preserve existing bookmarks and SEO, all old URLs must redirect to their new locations.

### Redirect Rules

| Old MkDocs URL | New Fern URL |
|----------------|--------------|
| `/` | `/docs` |
| `/installation/` | `/docs/installation` |
| `/quick-start/` | `/docs/quick-start` |
| `/CONTRIBUTING/` | `/docs/contributing` |
| `/concepts/columns/` | `/docs/concepts/columns` |
| `/concepts/validators/` | `/docs/concepts/validators` |
| `/concepts/processors/` | `/docs/concepts/processors` |
| `/concepts/person_sampling/` | `/docs/concepts/person-sampling` |
| `/concepts/models/default-model-settings/` | `/docs/concepts/models/default-model-settings` |
| `/concepts/models/custom-model-settings/` | `/docs/concepts/models/custom-model-settings` |
| `/concepts/models/configure-model-settings-with-the-cli/` | `/docs/concepts/models/configure-with-cli` |
| `/concepts/models/model-providers/` | `/docs/concepts/models/model-providers` |
| `/concepts/models/model-configs/` | `/docs/concepts/models/model-configs` |
| `/concepts/models/inference-parameters/` | `/docs/concepts/models/inference-parameters` |
| `/tutorials/` | `/docs/tutorials/overview` |
| `/recipes/cards/` | `/docs/recipes` |
| `/recipes/code_generation/text_to_python/` | `/docs/recipes/code-generation/text-to-python` |
| `/recipes/code_generation/text_to_sql/` | `/docs/recipes/code-generation/text-to-sql` |
| `/recipes/qa_and_chat/product_info_qa/` | `/docs/recipes/qa-and-chat/product-info-qa` |
| `/recipes/qa_and_chat/multi_turn_chat/` | `/docs/recipes/qa-and-chat/multi-turn-chat` |
| `/plugins/overview/` | `/docs/plugins/overview` |
| `/plugins/example/` | `/docs/plugins/example` |
| `/plugins/available/` | `/docs/plugins/available` |
| `/code_reference/*` | `/api/*` |

### Implementation

**Option A: Fern redirects configuration** (if supported)

```yaml
# In docs.yml
redirects:
  - from: /concepts/person_sampling
    to: /docs/concepts/person-sampling
  # ... additional redirects
```

**Option B: Hosting platform redirects**

For Netlify (`_redirects` file):
```
/concepts/person_sampling/*  /docs/concepts/person-sampling/:splat  301
/code_reference/*            /api/:splat                            301
```

For nginx:
```nginx
rewrite ^/concepts/person_sampling(.*)$ /docs/concepts/person-sampling$1 permanent;
rewrite ^/code_reference/(.*)$ /api/$1 permanent;
```

---

## Configuration Files

### fern.config.json

```json
{
  "organization": "nvidia-nemo",
  "version": "1.0.0"
}
```

### docs.yml

```yaml
instances:
  - url: https://datadesigner.docs.nvidia.com

title: NeMo Data Designer

tabs:
  docs:
    display-name: Documentation
    slug: docs
  api:
    display-name: API Reference
    slug: api

navigation:
  - tab: docs
    layout:
      - section: Getting Started
        contents:
          - page: Welcome
            path: pages/index.mdx
          - page: Installation
            path: pages/installation.mdx
          - page: Quick Start
            path: pages/quick-start.mdx
          - page: Contributing
            path: pages/contributing.mdx
      - section: Concepts
        contents:
          - section: Models
            contents:
              - page: Default Model Settings
                path: pages/concepts/models/default-model-settings.mdx
              - page: Custom Model Settings
                path: pages/concepts/models/custom-model-settings.mdx
              - page: Configure with CLI
                path: pages/concepts/models/configure-with-cli.mdx
              - page: Model Providers
                path: pages/concepts/models/model-providers.mdx
              - page: Model Configs
                path: pages/concepts/models/model-configs.mdx
              - page: Inference Parameters
                path: pages/concepts/models/inference-parameters.mdx
          - page: Columns
            path: pages/concepts/columns.mdx
          - page: Validators
            path: pages/concepts/validators.mdx
          - page: Processors
            path: pages/concepts/processors.mdx
          - page: Person Sampling
            path: pages/concepts/person-sampling.mdx
      - section: Tutorials
        contents:
          - page: Overview
            path: pages/tutorials/overview.mdx
          - page: The Basics
            path: pages/tutorials/the-basics.mdx
          - page: Structured Outputs
            path: pages/tutorials/structured-outputs.mdx
          - page: Seeding with a Dataset
            path: pages/tutorials/seeding-with-dataset.mdx
          - page: Images as Context
            path: pages/tutorials/images-as-context.mdx
      - section: Recipes
        contents:
          - page: Recipe Cards
            path: pages/recipes/index.mdx
          - section: Code Generation
            contents:
              - page: Text to Python
                path: pages/recipes/code-generation/text-to-python.mdx
              - page: Text to SQL
                path: pages/recipes/code-generation/text-to-sql.mdx
          - section: QA and Chat
            contents:
              - page: Product Info QA
                path: pages/recipes/qa-and-chat/product-info-qa.mdx
              - page: Multi-Turn Chat
                path: pages/recipes/qa-and-chat/multi-turn-chat.mdx
      - section: Plugins
        contents:
          - page: Overview
            path: pages/plugins/overview.mdx
          - page: Example Plugin
            path: pages/plugins/example.mdx
          - page: Available Plugins
            path: pages/plugins/available.mdx
  - tab: api
    layout:
      - section: API Reference
        contents:
          - page: Models
            path: pages/api-reference/models.mdx
          - page: Column Configs
            path: pages/api-reference/column-configs.mdx
          - page: Config Builder
            path: pages/api-reference/config-builder.mdx
          - page: Data Designer Config
            path: pages/api-reference/data-designer-config.mdx
          - page: Sampler Params
            path: pages/api-reference/sampler-params.mdx
          - page: Validator Params
            path: pages/api-reference/validator-params.mdx
          - page: Processors
            path: pages/api-reference/processors.mdx
          - page: Analysis
            path: pages/api-reference/analysis.mdx

colors:
  accent-primary:
    dark: "#76B900"
    light: "#76B900"
  background:
    dark: "#1a1a1a"
    light: "#ffffff"

logo:
  dark: assets/favicon.png
  light: assets/favicon.png

favicon: assets/favicon.png

navbar-links:
  - type: github
    value: https://github.com/NVIDIA-NeMo/DataDesigner
```

---

## Migration Plan

### Phase 1: Setup (1 day)

1. Create `docs-fern/` directory structure
2. Initialize Fern configuration files
3. Set up local development environment
4. Verify Fern CLI works (`fern check`, `fern generate`)

### Phase 2: Core Pages Migration (2-3 days)

1. Migrate Getting Started section
   - `index.md` → `index.mdx`
   - `installation.md` → `installation.mdx`
   - `quick-start.md` → `quick-start.mdx`
   - `CONTRIBUTING.md` → `contributing.mdx`

2. Migrate Concepts section (6 model pages + 4 concept pages)

3. Migrate Plugins section (3 pages)

### Phase 3: Complex Content Migration (3-4 days)

1. Convert Jupyter notebooks to MDX
   - Extract code cells as code blocks
   - Convert markdown cells directly
   - Add Colab badges/links

2. Migrate Recipes section
   - Convert grid cards to Fern Cards
   - Migrate recipe content pages
   - Handle code snippet embedding

### Phase 4: API Reference Migration (2-3 days)

1. Extract API documentation from mkdocstrings output
2. Manually format as MDX pages
3. Add code examples and cross-references

### Phase 5: Styling and Polish (1-2 days)

1. Apply NVIDIA branding (green accent color)
2. Configure navigation and tabs
3. Add favicon and logos
4. Test responsive design

### Phase 6: Testing and Validation (1-2 days)

1. Review all pages for rendering issues
2. Verify all links work
3. Test navigation flow
4. Compare against original docs for completeness

---

## CI/CD Pipeline Changes

### Current MkDocs Pipeline

```yaml
# Current workflow (to be replaced)
- name: Build docs
  run: mkdocs build

- name: Deploy docs
  run: mkdocs gh-deploy
```

### New Fern Pipeline

```yaml
# .github/workflows/docs.yml
name: Documentation

on:
  push:
    branches: [main]
    paths:
      - 'docs-fern/**'
  pull_request:
    paths:
      - 'docs-fern/**'

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install Fern CLI
        run: npm install -g fern-api

      - name: Validate Fern config
        run: fern check
        working-directory: docs-fern

      - name: Generate docs (PR preview)
        if: github.event_name == 'pull_request'
        run: fern generate --docs --preview
        working-directory: docs-fern
        env:
          FERN_TOKEN: ${{ secrets.FERN_TOKEN }}

      - name: Deploy docs (production)
        if: github.ref == 'refs/heads/main'
        run: fern generate --docs
        working-directory: docs-fern
        env:
          FERN_TOKEN: ${{ secrets.FERN_TOKEN }}
```

### Required Secrets

| Secret | Description |
|--------|-------------|
| `FERN_TOKEN` | API token from Fern dashboard for deployments |

### Local Development

```bash
# Install Fern CLI
npm install -g fern-api

# Navigate to docs directory
cd docs-fern

# Validate configuration
fern check

# Local preview (starts dev server)
fern docs dev

# Generate static output
fern generate --docs
```

---

## Deprecation Timeline

### Week 1-2: Parallel Operation

- `docs-fern/` is the primary documentation source
- `docs/` remains for reference and rollback capability
- Both directories exist in repository
- MkDocs config (`mkdocs.yml`) remains but is not used in CI

### Week 3: Soft Deprecation

- Remove MkDocs from CI/CD pipeline
- Add deprecation notice to `docs/README.md`:
  ```markdown
  > ⚠️ **DEPRECATED**: This directory is no longer maintained.
  > Documentation has moved to `docs-fern/`.
  > This directory will be removed on [DATE].
  ```
- Update `CONTRIBUTING.md` to reference new docs location

### Week 4: Hard Deprecation

- Delete `docs/` directory
- Delete `mkdocs.yml`
- Remove MkDocs dependencies from `pyproject.toml`:
  - `mkdocs`
  - `mkdocs-material`
  - `mkdocs-jupyter`
  - `mkdocstrings`
  - `mkdocstrings-python`
- Update `.gitignore` to remove MkDocs artifacts (`site/`)
- Archive final MkDocs state in git tag: `mkdocs-final`

### Post-Migration Cleanup

- Remove custom CSS (`docs/css/`)
- Remove custom JS (`docs/js/`)
- Remove template overrides (`docs/overrides/`)
- Update README.md documentation links

---

## Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation | Owner |
|------|--------|------------|------------|-------|
| API reference quality loss | High | Medium | Document Python APIs manually with curated examples; add to PR checklist | [Owner] |
| Notebook interactivity loss | Medium | Low | Link to Colab badges at top of each tutorial; keep `.ipynb` files hosted | [Owner] |
| Icon support gaps | Low | High | Replace `:material-xxx:` with emoji or text labels; document in style guide | [Owner] |
| Custom CSS incompatibility | Low | Medium | Use Fern's built-in components; minimal custom CSS only if essential | [Owner] |
| Build/deploy workflow breaks | Medium | Medium | Test CI/CD in separate branch before merging; keep MkDocs as fallback for 2 weeks | [Owner] |
| SEO ranking drop | Medium | Medium | Implement all redirects before deprecating old URLs; submit sitemap to search engines | [Owner] |
| Broken links post-migration | Medium | High | Run automated link checker before go-live; fix all broken links | [Owner] |

---

## Common Pitfalls & Troubleshooting

### Pitfall 1: Nested Admonitions

MkDocs supports nested admonitions; Fern callouts do not nest well.

**Problem:**
```markdown
!!! note
    Some text
    !!! warning
        Nested warning
```

**Solution:** Flatten to sequential callouts:
```mdx
<Note>
Some text
</Note>

<Warning>
Nested warning (now separate)
</Warning>
```

### Pitfall 2: Code Blocks Inside Tabs

Indentation is critical. Fern expects proper nesting.

**Problem (incorrect indentation):**
```mdx
<Tabs>
<Tab title="Python">
```python
code
```
</Tab>
</Tabs>
```

**Solution (correct indentation):**
```mdx
<Tabs>
  <Tab title="Python">
    ```python
    code
    ```
  </Tab>
</Tabs>
```

### Pitfall 3: MkDocs-Specific Syntax

These MkDocs features have no direct Fern equivalent:

| MkDocs Syntax | Action |
|---------------|--------|
| `{ .md-button }` | Remove, use standard links |
| `{ .annotate }` | Remove, use inline notes |
| `[TOC]` | Remove, Fern auto-generates TOC |
| `--8<-- "file"` | Inline the code or use `<CodeBlock>` |
| `::: module.path` | Convert to manual documentation |

### Pitfall 4: Image Paths

MkDocs resolves images relative to the markdown file; Fern resolves from project root.

**MkDocs:**
```markdown
![Alt](../assets/image.png)
```

**Fern:**
```mdx
![Alt](/assets/image.png)
```

### Pitfall 5: Front Matter

Fern uses YAML front matter for page metadata. Add to each file:

```mdx
---
title: Page Title
description: Optional description for SEO
---
```

### Troubleshooting Commands

```bash
# Validate all Fern configuration
fern check

# See detailed errors
fern check --log-level debug

# Preview locally before deploying
fern docs dev

# Check for broken internal links
grep -r '](/[^)]*\.mdx)' docs-fern/pages/ | grep -v '^#'
```

---

## Rollback Plan

If critical issues are discovered post-migration, follow this rollback procedure:

### Trigger Conditions

Initiate rollback if any of these occur within 2 weeks of go-live:

- [ ] >10% of pages have rendering issues
- [ ] Search functionality broken
- [ ] CI/CD pipeline repeatedly failing
- [ ] Critical content missing or incorrect
- [ ] Stakeholder requests rollback

### Rollback Steps

**Step 1: Restore MkDocs CI/CD (15 minutes)**

```yaml
# Revert .github/workflows/docs.yml to MkDocs version
git revert <fern-migration-commit>
git push origin main
```

**Step 2: Restore DNS/Hosting (if changed)**

Point documentation URL back to MkDocs deployment location.

**Step 3: Communicate**

Notify team:
> Documentation rollback initiated due to [REASON].
> MkDocs docs restored at [URL].
> Fern migration paused pending [ISSUE] resolution.

**Step 4: Preserve Fern Work**

```bash
# Don't delete - branch and preserve
git checkout -b fern-migration-paused
git push origin fern-migration-paused
```

**Step 5: Post-Mortem**

Document:
- What triggered the rollback
- Root cause analysis
- Required fixes before retry
- Updated timeline

### Rollback Window

- **Weeks 1-2**: Full rollback capability (MkDocs still in repo)
- **Week 3+**: Rollback requires restoring from `mkdocs-final` tag
- **Week 4+**: Rollback requires significant effort (MkDocs deleted)

---

## Pre-Flight Checklist

Before starting migration, ensure:

- [ ] Fern account created and `FERN_TOKEN` obtained
- [ ] Hosting decision finalized (Section: Decisions #4)
- [ ] Timeline approved and dates filled in (Section: Decisions #5)
- [ ] Owner assigned in RFC header
- [ ] Team notified of upcoming changes
- [ ] Current docs snapshot archived (`git tag mkdocs-snapshot-pre-migration`)

---

## Conversion Checklist

### File-by-File Migration Tracker

Use this checklist during Phase 2-4 to track progress:

#### Getting Started
- [ ] `index.md` → `pages/index.mdx`
- [ ] `installation.md` → `pages/installation.mdx`
- [ ] `quick-start.md` → `pages/quick-start.mdx`
- [ ] `CONTRIBUTING.md` → `pages/contributing.mdx`

#### Concepts - Models
- [ ] `concepts/models/default-model-settings.md` → `pages/concepts/models/default-model-settings.mdx`
- [ ] `concepts/models/custom-model-settings.md` → `pages/concepts/models/custom-model-settings.mdx`
- [ ] `concepts/models/configure-model-settings-with-the-cli.md` → `pages/concepts/models/configure-with-cli.mdx`
- [ ] `concepts/models/model-providers.md` → `pages/concepts/models/model-providers.mdx`
- [ ] `concepts/models/model-configs.md` → `pages/concepts/models/model-configs.mdx`
- [ ] `concepts/models/inference-parameters.md` → `pages/concepts/models/inference-parameters.mdx`

#### Concepts - Other
- [ ] `concepts/columns.md` → `pages/concepts/columns.mdx`
- [ ] `concepts/validators.md` → `pages/concepts/validators.mdx`
- [ ] `concepts/processors.md` → `pages/concepts/processors.mdx`
- [ ] `concepts/person_sampling.md` → `pages/concepts/person-sampling.mdx`

#### Tutorials (Notebook Conversion)
- [ ] `colab_notebooks/1-the-basics.ipynb` → `pages/tutorials/the-basics.mdx`
- [ ] `colab_notebooks/2-structured-outputs-and-jinja-expressions.ipynb` → `pages/tutorials/structured-outputs.mdx`
- [ ] `colab_notebooks/3-seeding-with-a-dataset.ipynb` → `pages/tutorials/seeding-with-dataset.mdx`
- [ ] `colab_notebooks/4-providing-images-as-context.ipynb` → `pages/tutorials/images-as-context.mdx`
- [ ] Create `pages/tutorials/overview.mdx` (new index page)

#### Recipes
- [ ] `recipes/cards.md` → `pages/recipes/index.mdx`
- [ ] `recipes/code_generation/text_to_python.md` → `pages/recipes/code-generation/text-to-python.mdx`
- [ ] `recipes/code_generation/text_to_sql.md` → `pages/recipes/code-generation/text-to-sql.mdx`
- [ ] `recipes/qa_and_chat/product_info_qa.md` → `pages/recipes/qa-and-chat/product-info-qa.mdx`
- [ ] `recipes/qa_and_chat/multi_turn_chat.md` → `pages/recipes/qa-and-chat/multi-turn-chat.mdx`

#### Plugins
- [ ] `plugins/overview.md` → `pages/plugins/overview.mdx`
- [ ] `plugins/example.md` → `pages/plugins/example.mdx`
- [ ] `plugins/available.md` → `pages/plugins/available.mdx`

#### API Reference
- [ ] `code_reference/models.md` → `pages/api-reference/models.mdx`
- [ ] `code_reference/column_configs.md` → `pages/api-reference/column-configs.mdx`
- [ ] `code_reference/config_builder.md` → `pages/api-reference/config-builder.mdx`
- [ ] `code_reference/data_designer_config.md` → `pages/api-reference/data-designer-config.mdx`
- [ ] `code_reference/sampler_params.md` → `pages/api-reference/sampler-params.mdx`
- [ ] `code_reference/validator_params.md` → `pages/api-reference/validator-params.mdx`
- [ ] `code_reference/processors.md` → `pages/api-reference/processors.mdx`
- [ ] `code_reference/analysis.md` → `pages/api-reference/analysis.mdx`

#### Assets
- [ ] Copy `assets/palette-favicon.png` → `assets/favicon.png`
- [ ] Copy `assets/recipes/` → `assets/recipes/`

---

## Success Criteria

- [ ] All existing documentation pages migrated (32 pages total)
- [ ] Navigation structure preserved
- [ ] All code examples render correctly
- [ ] All internal links functional (automated check)
- [ ] All external links functional (automated check)
- [ ] NVIDIA branding applied (green accent: #76B900)
- [ ] Local development workflow documented
- [ ] CI/CD pipeline deployed and tested
- [ ] URL redirects configured and tested
- [ ] PR preview deployments working
- [ ] Page load time < 3 seconds

---

## Decisions

The following decisions have been made to ensure smooth execution:

### 1. API Reference Approach

**Decision:** Manual documentation with code examples (Option A)

**Rationale:**
- Fastest path to migration completion
- Allows curated examples rather than raw docstring dumps
- Fern's Python SDK autodoc is not mature enough for our needs

**Maintenance commitment:**
- API reference pages will be updated alongside code changes
- Add to PR checklist: "Update API docs if public interfaces changed"
- Revisit automation options in Q2 2026

### 2. Notebook Handling

**Decision:** Convert to MDX with Colab links

**Implementation:**
- Extract code cells as fenced code blocks
- Convert markdown cells directly to MDX
- Preserve Colab badge at top of each tutorial
- Link to hosted `.ipynb` files for interactive experience

**Example header for converted notebooks:**
```mdx
---
title: The Basics
---

<Info title="Interactive Version">
Run this tutorial interactively in [Google Colab](https://colab.research.google.com/github/NVIDIA-NeMo/DataDesigner/blob/main/docs/colab_notebooks/1-the-basics.ipynb).
</Info>
```

### 3. Versioning

**Decision:** Single version initially, evaluate multi-version post-launch

**Rationale:**
- Current MkDocs setup is single-version
- No immediate need for versioned docs
- Fern supports versioning when needed

### 4. Hosting

**Decision:** [Fern-hosted | Self-hosted] _(fill in)_

**If Fern-hosted:**
- URL: `https://datadesigner.docs.buildwithfern.com` or custom domain
- Zero infrastructure management
- Built-in CDN and SSL

**If self-hosted:**
- Deploy to existing NVIDIA infrastructure
- Use `fern generate --docs` to produce static output
- Configure redirects on hosting platform

### 5. Timeline

**Decision:** [X weeks] from RFC approval

| Milestone | Target Date |
|-----------|-------------|
| Phase 1 (Setup) complete | [DATE] |
| Phase 2-3 (Content migration) complete | [DATE] |
| Phase 4 (API reference) complete | [DATE] |
| Phase 5-6 (Polish & testing) complete | [DATE] |
| Go-live | [DATE] |
| Old docs deprecated | [DATE + 2 weeks] |

---

## Helper Scripts

The following scripts can assist with automated conversion:

### 1. Admonition Converter

```python
#!/usr/bin/env python3
"""Convert MkDocs admonitions to Fern callouts."""
import re
import sys

ADMONITION_MAP = {
    "note": "Note",
    "tip": "Tip",
    "info": "Info",
    "warning": "Warning",
    "danger": "Warning",
    "question": "Info",
    "example": "Info",
    "abstract": "Note",
    "success": "Tip",
    "failure": "Warning",
    "bug": "Warning",
}

def convert_admonitions(content: str) -> str:
    """Convert !!! admonitions to <Callout> components."""
    pattern = r'!!! (\w+)(?: "([^"]*)")?\n((?:    .*\n?)*)'

    def replace(match: re.Match) -> str:
        admon_type = match.group(1).lower()
        title = match.group(2) or ""
        body = match.group(3)
        # Remove 4-space indent from body
        body = re.sub(r'^    ', '', body, flags=re.MULTILINE).strip()
        fern_type = ADMONITION_MAP.get(admon_type, "Note")
        if title:
            return f'<{fern_type} title="{title}">\n{body}\n</{fern_type}>\n'
        return f'<{fern_type}>\n{body}\n</{fern_type}>\n'

    return re.sub(pattern, replace, content)

if __name__ == "__main__":
    content = sys.stdin.read()
    print(convert_admonitions(content))
```

**Usage:**
```bash
cat docs/concepts/columns.md | python scripts/convert_admonitions.py > docs-fern/pages/concepts/columns.mdx
```

### 2. Tabs Converter

```python
#!/usr/bin/env python3
"""Convert MkDocs tabs to Fern Tabs components."""
import re
import sys

def convert_tabs(content: str) -> str:
    """Convert === tabs to <Tabs> components."""
    # Match tab groups
    pattern = r'((?:=== "([^"]+)"\n((?:    .*\n?)*)\n?)+)'

    def replace_group(match: re.Match) -> str:
        group = match.group(0)
        tabs = re.findall(r'=== "([^"]+)"\n((?:    .*\n?)*)', group)
        result = ["<Tabs>"]
        for title, body in tabs:
            body = re.sub(r'^    ', '', body, flags=re.MULTILINE).strip()
            result.append(f'  <Tab title="{title}">')
            result.append(f'    {body}')
            result.append('  </Tab>')
        result.append("</Tabs>")
        return '\n'.join(result) + '\n'

    return re.sub(pattern, replace_group, content)

if __name__ == "__main__":
    content = sys.stdin.read()
    print(convert_tabs(content))
```

### 3. Notebook to MDX Converter

```python
#!/usr/bin/env python3
"""Convert Jupyter notebook to MDX."""
import json
import sys
from pathlib import Path

def notebook_to_mdx(notebook_path: str, colab_url: str) -> str:
    """Convert a Jupyter notebook to MDX format."""
    with open(notebook_path) as f:
        nb = json.load(f)

    lines = [
        "---",
        f"title: {Path(notebook_path).stem.replace('-', ' ').title()}",
        "---",
        "",
        '<Info title="Interactive Version">',
        f"Run this tutorial interactively in [Google Colab]({colab_url}).",
        "</Info>",
        "",
    ]

    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))

        if cell_type == "markdown":
            lines.append(source)
            lines.append("")
        elif cell_type == "code":
            lines.append("```python")
            lines.append(source)
            lines.append("```")
            lines.append("")

    return "\n".join(lines)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: notebook_to_mdx.py <notebook.ipynb> <colab_url>")
        sys.exit(1)
    print(notebook_to_mdx(sys.argv[1], sys.argv[2]))
```

**Usage:**
```bash
python scripts/notebook_to_mdx.py \
  docs/colab_notebooks/1-the-basics.ipynb \
  "https://colab.research.google.com/github/NVIDIA-NeMo/DataDesigner/blob/main/docs/colab_notebooks/1-the-basics.ipynb" \
  > docs-fern/pages/tutorials/the-basics.mdx
```

### 4. Link Checker

```bash
#!/bin/bash
# Check all links in Fern docs
cd docs-fern

# Internal links
grep -roh '\[.*\]([^)]*\.mdx)' pages/ | sort | uniq

# External links
grep -roh 'https://[^)]*' pages/ | sort | uniq | while read url; do
  if ! curl -s --head "$url" | head -1 | grep -q "200\|301\|302"; then
    echo "BROKEN: $url"
  fi
done
```

### 5. Batch Conversion Script

```bash
#!/bin/bash
# batch_convert.sh - Run all conversions

set -e

SCRIPTS_DIR="scripts"
DOCS_DIR="docs"
FERN_DIR="docs-fern/pages"

# Create directory structure
mkdir -p "$FERN_DIR"/{concepts/models,tutorials,recipes/{code-generation,qa-and-chat},plugins,api-reference}

# Convert simple pages (admonitions + tabs)
for file in index installation quick-start CONTRIBUTING; do
  src="$DOCS_DIR/$file.md"
  if [ -f "$src" ]; then
    dst="$FERN_DIR/${file,,}.mdx"
    cat "$src" | python "$SCRIPTS_DIR/convert_admonitions.py" | python "$SCRIPTS_DIR/convert_tabs.py" > "$dst"
    echo "Converted: $src -> $dst"
  fi
done

echo "Batch conversion complete. Manual review required."
```

---

## References

- [Fern Docs Getting Started](https://buildwithfern.com/learn/docs/getting-started/overview)
- [Fern Components Overview](https://buildwithfern.com/learn/docs/writing-content/components/overview)
- [Fern Configuration](https://buildwithfern.com/learn/docs/configuration/site-level-settings)
- [Current MkDocs Configuration](../mkdocs.yml)

---

## Appendix: Sample Migration

### Before (MkDocs - columns.md excerpt)

```markdown
# Columns

Columns are the fundamental building blocks in Data Designer.

!!! note "The Declarative Approach"
    Columns are **declarative specifications**. You describe *what* you want...

## Column Types

### 🎲 Sampler Columns

Sampler columns generate data using numerical sampling...

!!! tip "Conditional Sampling"
    Samplers support **conditional parameters**...
```

### After (Fern - columns.mdx excerpt)

```mdx
# Columns

Columns are the fundamental building blocks in Data Designer.

<Note title="The Declarative Approach">
Columns are **declarative specifications**. You describe *what* you want...
</Note>

## Column Types

### 🎲 Sampler Columns

Sampler columns generate data using numerical sampling...

<Tip title="Conditional Sampling">
Samplers support **conditional parameters**...
</Tip>
```
