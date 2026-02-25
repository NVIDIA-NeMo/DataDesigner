#!/bin/bash
# =============================================================================
# Ruff Linting Hook
# =============================================================================
# Runs ruff linting on modified Python files.
# Receives the file path as $1 from Claude Code.
# =============================================================================

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] <file>

Run ruff linting on a Python file.

Arguments:
  file              Path to the Python file to lint

Options:
  -h, --help        Show this help message and exit

Examples:
  $(basename "$0") src/main.py
  $(basename "$0") --help

Exit Codes:
  0    No linting issues found
  1    Linting issues detected
EOF
}

# Parse arguments
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

FILE="$1"

# Show help if no file argument provided
if [[ -z "$FILE" ]]; then
    show_help
    exit 0
fi

# Only process Python files
if [[ ! "$FILE" =~ \.py$ ]]; then
    exit 0
fi

echo "=== Ruff Lint Check ==="
uvx ruff check "$FILE"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "No linting issues found."
else
    echo "Linting issues detected. Consider running: uvx ruff check --fix $FILE"
fi
echo "======================="
exit $EXIT_CODE
