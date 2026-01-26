---
name: create-pr
description: Create a GitHub PR with a well-formatted description including summary, categorized changes, and attention areas
argument-hint: [special instructions]
---

# Create Pull Request

Create a well-formatted GitHub pull request for the current branch.

## Arguments

`$ARGUMENTS` can be used for special instructions, such as:
- Specifying a base branch: "use base branch: develop"
- Guiding the summary: "emphasize the performance improvements in the summary"
- Adding context: "this is part of the auth refactor epic"
- Any other guidance for PR creation

Default base branch: `main` (unless specified in arguments)

## Step 1: Gather Information

Run these commands in parallel to understand the changes:

1. **Current branch**: `git branch --show-current`
2. **Uncommitted changes**: `git status --porcelain`
3. **Commits on branch**: `git log origin/main..HEAD --oneline`
4. **File changes summary**: `git diff --stat origin/main..HEAD`
5. **Full diff**: `git diff origin/main..HEAD`
6. **Recent commit style**: `git log -5 --oneline` (to match PR title convention)

**Important checks:**
- If uncommitted changes exist, warn the user and ask if they want to commit first
- If no commits ahead of main, inform the user there's nothing to PR
- If branch isn't pushed, you'll push it in Step 4

## Step 2: Analyze and Categorize Changes

### By Change Type (from commits and diff)
- ✨ **Added**: New files, features, capabilities
- 🔧 **Changed**: Modified existing functionality
- 🗑️ **Removed**: Deleted files or features
- 🐛 **Fixed**: Bug fixes
- 📚 **Docs**: Documentation updates
- 🧪 **Tests**: Test additions/modifications

### Identify Attention Areas 🔍
Flag for special reviewer attention:
- Files with significant changes (>100 lines)
- Changes to base classes, interfaces, or public API
- New dependencies (`pyproject.toml`, `requirements.txt`)
- Configuration schema changes
- Security-related changes

## Step 3: Generate PR Title

Use conventional commit format matching the repo style:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `refactor:` for refactoring
- `chore:` for maintenance
- `test:` for test changes

If commits have mixed types, use the primary/most significant type.

## Step 4: Create the PR

1. **Push branch** (if needed):
   ```bash
   git push -u origin <branch-name>
   ```

2. **Create PR** using this template:

```markdown
## 📋 Summary

[1-2 sentence overview of what this PR accomplishes]

## 🔄 Changes

### ✨ Added
- [New features/files - link to key files when helpful]

### 🔧 Changed
- [Modified functionality - reference commits for specific changes]

### 🗑️ Removed
- [Deleted items]

### 🐛 Fixed
- [Bug fixes - if applicable]

## 🔍 Attention Areas

> ⚠️ **Reviewers:** Please pay special attention to the following:

- `path/to/critical/file.py` - [Why this needs attention]

---
🤖 *Generated with AI*
```

3. **Execute**:
   ```bash
   gh pr create --title "<title>" --body "$(cat <<'EOF'
   <body>
   EOF
   )"
   ```

4. **Return the PR URL** to the user.

## Section Guidelines

- **Summary**: Always include - be concise and focus on the "why"
- **Changes**: Group by type, omit empty sections
- **Attention Areas**: Only include if there are genuinely important items; omit for simple PRs
- **Links**: Include links to code and commits where helpful for reviewers:
  - Link to specific files: `[filename](path/to/file.py)` - GitHub auto-links files in the repo
  - Link to specific lines: `[description](path/to/file.py#L42-L50)` for key code sections
  - Reference commits: `abc1234` - GitHub auto-links commit SHAs
  - For multi-commit PRs, reference individual commits when describing specific changes

## Edge Cases

- **No changes**: Inform user there's nothing to create a PR for
- **Uncommitted work**: Warn and ask before proceeding
- **Large PRs** (>20 files): Summarize by directory/module
- **Single commit**: PR title can match commit message
