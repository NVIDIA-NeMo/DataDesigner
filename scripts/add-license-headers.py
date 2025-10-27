# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from datetime import datetime
from pathlib import Path


def add_license_header_to_file(file_path: Path, license_header: str) -> bool:
    """Add license header to a single file. Returns True if header was added."""
    try:
        # Read file content
        content = file_path.read_text(encoding="utf-8")

        # Check if license header already exists
        if has_license_header(content):
            return False

        # Handle shebang lines
        lines = content.splitlines(keepends=True)
        insert_pos = 0

        # If file starts with shebang, insert after it
        if lines and lines[0].startswith("#!"):
            insert_pos = 1
            # Add empty line after shebang if there isn't one
            if len(lines) > 1 and not lines[1].strip() == "":
                license_header += "\n"

        # Insert license header
        if insert_pos < len(lines):
            lines.insert(insert_pos, license_header)
        else:
            lines.append(license_header)

        # Write back to file
        file_path.write_text("".join(lines), encoding="utf-8")
        return True

    except (UnicodeDecodeError, PermissionError) as e:
        print(f"  - Skipped {file_path} ({e})")
        return False


def has_license_header(file_content: str) -> bool:
    """Check if file already has a license header."""
    lines = file_content.splitlines()
    if not lines:
        return False

    # Check first few lines for license header patterns
    first_lines = lines[:10]  # Check first 10 lines
    license_pattern = r"SPDX\-License\-Identifier"

    for line in first_lines:
        if re.search(license_pattern, line, re.IGNORECASE):
            return True

    return False


def should_add_license_header(file_path: Path) -> bool:
    """Determine if a file should have a license header added."""
    # Skip certain files
    skip_patterns = [
        "__pycache__",
        ".pyc",
        ".pyo",
        ".pyd",
        ".so",
        ".egg-info",
        ".git",
        ".pytest_cache",
        "node_modules",
        ".venv",
        "venv",
    ]

    # Skip if file path contains any skip patterns
    file_str = str(file_path)
    for pattern in skip_patterns:
        if pattern in file_str:
            return False

    # Only process Python files
    if file_path.suffix != ".py":
        return False

    # Skip certain specific files
    skip_files = []

    # Allow __init__.py files that are not in the root of the SDK
    if file_path.name in skip_files:
        return False

    return True


def main(path: Path) -> tuple[int, int, int]:
    current_year = datetime.now().year
    LICENSE_HEADER = (
        f"# SPDX-FileCopyrightText: Copyright (c) {current_year} "
        "NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
    )

    # File patterns to process
    patterns = ["**/*.py"]

    processed_files = 0
    updated_files = 0
    skipped_files = 0

    for pattern in patterns:
        for file_path in path.glob(pattern):
            # Skip if not a file
            if not file_path.is_file():
                continue

            # Skip if file shouldn't have license header
            if not should_add_license_header(file_path):
                continue

            processed_files += 1

            # Add license header
            if add_license_header_to_file(file_path, LICENSE_HEADER):
                print(f"  |-- 📝 Adding license header to {file_path}")
                updated_files += 1
            else:
                skipped_files += 1

    return processed_files, updated_files, skipped_files


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent

    for folder in ["src", "tests", "scripts"]:
        print(f"📁 Adding license header to {repo_path / folder}")
        processed_files, updated_files, skipped_files = main(repo_path / folder)
        print(f"🚜 Processed {processed_files} files")
        print(f"✅ Updated {updated_files} files")
        print(f"⏩ Skipped {skipped_files} files")

    print("🏁 Done!")
