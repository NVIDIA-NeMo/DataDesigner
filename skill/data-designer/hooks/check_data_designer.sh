#!/bin/bash
# =============================================================================
# Data Designer Context Injection Hook
# =============================================================================
# This hook provides the agent with context about the data_designer library:
#   - Whether the library is installed
#   - The installed version
#   - The library's location on disk (useful for reading source code)
# =============================================================================

# Check if data_designer is installed
if ! uv run python -c "import data_designer" 2>/dev/null; then
    echo "=== Data Designer Context ==="
    echo "STATUS: NOT INSTALLED"
    echo "The data_designer library is not installed in the current environment."
    echo "To install, run: uv pip install data-designer"
    echo "============================="
    exit 0
fi

# Get version and library path
VERSION=$(uv run python -c "import importlib.metadata; print(importlib.metadata.version('data-designer'))" 2>/dev/null)
LIB_PATH=$(uv run python "$(dirname "$0")/../scripts/echo_data_designer_library_path.py" 2>/dev/null)

# Output formatted context for the agent
echo "=== Data Designer Library ==="
echo "STATUS: Installed"
echo "VERSION: ${VERSION}"
echo "LIBRARY_PATH: ${LIB_PATH}"
echo ""
echo "Use the discovery scripts (get_column_info.py, get_sampler_info.py, etc.)"
echo "to look up API details — prefer these over reading source code directly."
echo "============================="
