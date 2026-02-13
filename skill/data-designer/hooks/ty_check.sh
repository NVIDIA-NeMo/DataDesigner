#!/bin/bash
# =============================================================================
# Ty Type Checking Hook
# =============================================================================
# Runs ty type checking on modified Python files.
# Receives the file path as $1 from Claude Code.
# =============================================================================

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] <file>

Run ty type checking on a Python file.

Arguments:
  file              Path to the Python file to type check

Options:
  -h, --help        Show this help message and exit

Examples:
  $(basename "$0") src/main.py
  $(basename "$0") --help

Exit Codes:
  0    No type errors found
  1    Type errors detected
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

echo "=== Ty Type Check ==="
uvx ty check "$FILE"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "No type errors found."
fi
echo "====================="
exit $EXIT_CODE
