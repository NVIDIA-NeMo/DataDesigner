# Fern Documentation Cheat Sheet

This folder contains the Fern Docs configuration for NeMo Data Designer.

## 📦 Installation

```bash
# Install Fern CLI globally
npm install -g fern-api

# Or use npx (no install needed)
npx fern-api --version
```

## 🔍 Local Preview

**Before first run (for NotebookViewer pages):**
```bash
make generate-colab-notebooks   # Creates docs/colab_notebooks/*.ipynb
make generate-fern-notebooks    # Creates fern/assets/notebooks/*.ts
```

```bash
# From the fern/ directory
cd fern/
fern docs dev

# Or from project root
fern docs dev --project ./fern
```

The docs will be available at `http://localhost:3000`.

See [DOCS-VS-FERN.md](DOCS-VS-FERN.md) for docs/ vs fern/ comparison and migration notes.

## 📁 Folder Structure

```
fern/
├── docs.yml              # Global config (title, colors, versions)
├── fern.config.json      # Fern CLI config (org name)
├── versions/
│   ├── v0.3.3.yml        # Navigation for v0.3.3
│   └── v0.4.0.yml        # Navigation for v0.4.0
├── v0.3.3/
│   └── pages/            # MDX content for v0.3.3
├── v0.4.0/
│   └── pages/            # MDX content for v0.4.0
└── assets/               # Shared images, favicons
```

## 🔄 Bumping the Version

When releasing a new version (e.g., v0.5.0):

### 1. Copy the previous version's content
```bash
cp -r fern/v0.4.0 fern/v0.5.0
```

### 2. Create the navigation file
```bash
cp fern/versions/v0.4.0.yml fern/versions/v0.5.0.yml
```

### 3. Update paths in `versions/v0.5.0.yml`
Change all `../v0.4.0/pages/` → `../v0.5.0/pages/`

### 4. Add the new version to `docs.yml`
```yaml
versions:
  - display-name: v0.5.0
    path: versions/v0.5.0.yml
    slug: v0.5.0
  - display-name: v0.4.0
    path: versions/v0.4.0.yml
    slug: v0.4.0
  # ... older versions
```

### 5. Make your content changes
Edit files in `fern/v0.5.0/pages/`

## ✏️ Editing Content

### Adding a new page

1. Create the MDX file in the appropriate version folder:
   ```bash
   touch fern/v0.3.3/pages/concepts/new-feature.mdx
   ```

2. Add frontmatter:
   ```mdx
   ---
   title: New Feature
   description: Description for SEO.
   ---

   Content starts here...
   ```

3. Add to navigation in `versions/v0.3.3.yml`:
   ```yaml
   - page: New Feature
     path: ../v0.3.3/pages/concepts/new-feature.mdx
   ```

### MDX Components

```mdx
# Callouts
<Note>Informational note</Note>
<Tip>Helpful tip</Tip>
<Warning>Warning message</Warning>
<Info>Info callout</Info>

# Tabs
<Tabs>
  <Tab title="Python">
    ```python
    print("hello")
    ```
  </Tab>
  <Tab title="JavaScript">
    ```javascript
    console.log("hello")
    ```
  </Tab>
</Tabs>

# Cards
<Cards>
  <Card title="Title" href="/path">
    Description
  </Card>
</Cards>
```

## 🚀 Deploying

```bash
# Generate static docs (for CI/CD)
fern generate --docs

# Deploy to Fern hosting
fern docs deploy
```

## 🔗 Useful Links

- [Fern Docs](https://buildwithfern.com/learn/docs)
- [MDX Components Reference](https://buildwithfern.com/learn/docs/components)
- [Versioning Guide](https://buildwithfern.com/learn/docs/configuration/versions)
- [Navigation Configuration](https://buildwithfern.com/learn/docs/configuration/navigation)

## 📓 NotebookViewer Component

A custom React component renders Jupyter notebook content in Fern docs. Use it to display tutorials from `.ipynb` files with a Colab badge.

### Workflow

1. **Generate Colab notebooks** (source of truth):
   ```bash
   make generate-colab-notebooks
   ```

2. **Convert to Fern format**:
   ```bash
   make generate-fern-notebooks
   ```

3. **Use in MDX** (import from `@/components/notebooks/` – inside the components tree so Fern can resolve it):
   ```mdx
   import { NotebookViewer } from "@/components/NotebookViewer";
   import notebook from "@/components/notebooks/1-the-basics";

   <NotebookViewer
     notebook={notebook}
     colabUrl="https://colab.research.google.com/github/NVIDIA-NeMo/DataDesigner/blob/main/docs/colab_notebooks/1-the-basics.ipynb"
   />
   ```

### Files

- `fern/components/NotebookViewer.tsx` – React component
- `fern/components/notebooks/*.ts` – Notebook data (generated, in components tree for import resolution)
- `fern/scripts/ipynb-to-fern-json.py` – Converts `.ipynb` → `.ts` + `.json`
- `fern/styles/notebook-viewer.css` – Component styles

### Requirements

- Fern Pro or Enterprise plan (custom React components)

## ⚠️ Common Issues

### "EISDIR: illegal operation on a directory"
- Check that all `path:` values point to `.mdx` files, not directories

### Page not showing
- Verify the page is listed in the version's navigation file
- Check the path is correct (relative to the versions/ folder)

### Version selector not appearing
- Ensure `versions:` is defined in `docs.yml`
- Each version needs a valid `.yml` file in `versions/`
