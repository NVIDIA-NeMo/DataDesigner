---
name: commit
description: Commit current changes with a clear, descriptive message
argument-hint: [special instructions]
disable-model-invocation: true
---

# Commit Changes

Commit the current staged/unstaged changes with a well-crafted commit message.

## Arguments

`$ARGUMENTS` can be used for special instructions, such as:
- Guiding the message: "emphasize the breaking change"
- Adding context: "this fixes issue #123"
- Requesting amend: "amend the previous commit"

## Step 1: Gather Information

Run these commands in parallel to understand the changes:

1. **Git status**: `git status --porcelain` (check for staged/unstaged changes)
2. **Staged diff**: `git diff --cached` (what will be committed if there are staged changes)
3. **Unstaged diff**: `git diff` (what's modified but not staged)
4. **Recent commits**: `git log -5 --oneline` (to match existing commit style)

**Important checks:**
- If no changes exist (nothing staged or unstaged), inform the user there's nothing to commit
- If there are unstaged changes but nothing staged, stage all changes with `git add -A` (or ask user if they want to selectively stage)
- If there are both staged and unstaged changes, ask user if they want to include unstaged changes

## Step 2: Analyze Changes

Review the diff to understand:
- **What** changed (files, functions, features)
- **Why** it changed (bug fix, new feature, refactor, etc.)
- **Impact** (breaking changes, dependencies, configuration)

## Step 3: Generate Commit Message

### Message Guidelines

1. **Subject line**:
   - Use imperative mood ("add feature" not "added feature")
   - Keep under 50 characters (hard limit: 72)
   - Don't end with a period
   - Use lowercase for the first letter

2. **Body** (if needed for complex changes):
   - Separate from subject with blank line
   - Wrap at 72 characters
   - Explain *what* and *why*, not *how*
   - Use bullet points for multiple items

3. **Footer** (if needed):
   - Reference issues: `Fixes #123`, `Closes #456`

### Examples

**Simple change:**
```
add user authentication endpoint
```

**Change with body:**
```
add rate limiting to public endpoints

- Implement token bucket algorithm
- Add configurable limits per endpoint
- Include rate limit headers in responses

Closes #234
```

## Step 4: Create the Commit

1. **Stage changes** (if needed):
   ```bash
   git add -A
   ```
   Or stage specific files if user requested selective staging.

2. **Create commit** using HEREDOC for proper formatting:
   ```bash
   git commit -m "$(cat <<'EOF'
   <description>

   <body if needed>

   <footer if needed>
   EOF
   )"
   ```

3. **Verify** the commit was created:
   ```bash
   git log -1 --oneline
   ```

4. **Report** the commit hash and message to the user.

## Edge Cases

- **No changes**: Inform user there's nothing to commit
- **Amend requested**: Use `git commit --amend` (only if explicitly requested in arguments)
- **Large changes**: Suggest breaking into multiple commits if changes span unrelated areas
- **Merge conflicts**: Inform user and provide guidance on resolution
- **Pre-commit hooks fail**: Report the failure and suggest fixes
