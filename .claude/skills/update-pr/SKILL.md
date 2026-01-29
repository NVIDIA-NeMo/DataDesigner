---
name: update-pr
description: Update an existing GitHub PR description to reflect current changes after incorporating feedback
argument-hint: [special instructions]
disable-model-invocation: true
---

# Update Pull Request

Update the description of an existing GitHub pull request for the current branch.

## Arguments

`$ARGUMENTS` can be used for:
- `--title` flag: Include to also update the PR title
- Special instructions (same as create-pr):
  - Guiding the summary: "emphasize the performance improvements"
  - Adding context: "this PR now also includes the auth fixes"
  - Any other guidance for the updated description

## Step 1: Find the Existing PR

1. **Get current branch and PR info**:
   ```bash
   gh pr view --json number,title,body,url,headRefName,state
   ```

**Important checks:**
- If no PR exists for the current branch, **fail** and inform the user to use `/create-pr` instead
- If the PR is merged or closed, **fail** and inform the user the PR is no longer open

## Step 2: Gather Current State

Run these commands in parallel to understand the full state:

1. **Uncommitted changes**: `git status --porcelain`
2. **All commits on branch**: `git log origin/main..HEAD --oneline`
3. **File changes summary**: `git diff --stat origin/main..HEAD`
4. **Full diff**: `git diff origin/main..HEAD`
5. **Repo info**: `gh repo view --json nameWithOwner -q '.nameWithOwner'` (for constructing file URLs)

**Important checks:**
- If uncommitted changes exist, warn the user and ask if they want to commit first

## Step 3: Analyze and Categorize Changes

### By Change Type (from commits and diff)
- ✨ **Added**: New files, features, capabilities
- 🔧 **Changed**: Modified existing functionality
- 🗑️ **Removed**: Deleted files or features
- 🐛 **Fixed**: Bug fixes
- 📚 **Docs**: Documentation updates
- 🧪 **Tests**: Test additions/modifications

### Identify Attention Areas
Flag for special reviewer attention:
- Files with significant changes (>100 lines)
- Changes to base classes, interfaces, or public API
- New dependencies (`pyproject.toml`, `requirements.txt`)
- Configuration schema changes
- Security-related changes

## Step 4: Update the PR

1. **Push any new commits** (if needed):
   ```bash
   git push
   ```

2. **Generate updated description** using the same template as create-pr (see below)

3. **Update the PR body**:
   ```bash
   gh pr edit <number> --body "$(cat <<'EOF'
   <body>
   EOF
   )"
   ```

4. **Optionally update title** (if `--title` flag was provided):
   ```bash
   gh pr edit <number> --title "<new-title>"
   ```

5. **Return the PR URL** to the user.

## PR Description Template

Use the same template as create-pr, with an updated footer:

```markdown
## Summary

[1-2 sentence overview of what this PR accomplishes]

## Changes

### Added
- [New features/files - link to key files when helpful]

### Changed
- [Modified functionality - reference commits for specific changes]

### Removed
- [Deleted items]

### Fixed
- [Bug fixes - if applicable]

## Attention Areas

> Reviewers: Please pay special attention to the following:

- [`path/to/critical/file.py`](https://github.com/<owner>/<repo>/blob/<branch>/path/to/critical/file.py) - [Why this needs attention]

---
*Description updated with AI*
```

## Section Guidelines

- **Summary**: Always include - be concise and focus on the "why"
- **Changes**: Group by type, omit empty sections
- **Attention Areas**: Only include if there are genuinely important items; omit for simple PRs
- **Links**: Include links to code and commits where helpful for reviewers:
  - **File links require full URLs** - relative paths don't work in PR descriptions
  - Link to a file: `[filename](https://github.com/<owner>/<repo>/blob/<branch>/path/to/file.py)`
  - Link to specific lines: `[description](https://github.com/<owner>/<repo>/blob/<branch>/path/to/file.py#L42-L50)`
  - Use the branch name in the URL so links point to the PR's version of files
  - Reference commits: `abc1234` - GitHub auto-links short commit SHAs in PR descriptions

## Edge Cases

- **No PR exists for current branch**: Fail and inform user to use `/create-pr` instead
- **PR is merged/closed**: Fail and inform user the PR is no longer open
- **Uncommitted work**: Warn and ask before proceeding
- **No new changes since PR creation**: Still allow update - user may want to rewrite the description
- **Large PRs** (>20 files): Summarize by directory/module

## Key Differences from `/create-pr`

| Aspect | `/create-pr` | `/update-pr` |
|--------|--------------|--------------|
| PR state | Creates new | Updates existing |
| Command | `gh pr create` | `gh pr edit` |
| First step | Check for commits | Find existing PR |
| Branch push | With `-u` flag | Simple push |
| Title | Always sets | Optional (with `--title`) |
| Footer | "Generated with AI" | "Description updated with AI" |
