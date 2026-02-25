"""Print the absolute filesystem path to the installed data_designer package.

This script is used by agent hooks and other tools to inject context about the
data_designer library's source location for inspection or source exploration.

Usage:
    uv run echo_data_designer_library_path.py

Example output:
    /home/user/.venv/lib/python3.10/site-packages/data_designer
"""

import sys
from pathlib import Path


def main() -> None:
    """Print the data_designer library path and exit."""
    try:
        import data_designer.config as dd

        print(Path(dd.__file__).parent.parent)
    except ImportError:
        print(
            "Error: data_designer is not installed in the current environment.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
