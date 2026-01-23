#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Publish script for DataDesigner
# Publishes all three subpackages to PyPI with the same version.
#
# Usage:
#   ./scripts/publish.sh 0.3.9rc1           # Full publish
#   ./scripts/publish.sh 0.3.9rc1 --dry-run # Dry run (no actual upload or tagging)

set -e

# ==============================================================================
# COLORS AND FORMATTING
# ==============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PACKAGE_DIRS=(
    "packages/data-designer-config"
    "packages/data-designer-engine"
    "packages/data-designer"
)

PYPIRC_FILE="$HOME/.pypirc"
EXPECTED_PYPI_USERNAME="data-designer-team"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

info() {
    echo -e "${BLUE}INFO:${NC} $1"
}

success() {
    echo -e "${GREEN}SUCCESS:${NC} $1"
}

warn() {
    echo -e "${YELLOW}WARNING:${NC} $1"
}

error() {
    echo -e "${RED}ERROR:${NC} $1" >&2
}

die() {
    error "$1"
    if [[ -n "$2" ]]; then
        echo -e "  ${YELLOW}$2${NC}" >&2
    fi
    exit 1
}

header() {
    echo ""
    echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}$1${NC}"
    echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
}

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

VERSION=""
DRY_RUN=false
ALLOW_BRANCH=false

usage() {
    echo "Usage: $0 <version> [--dry-run] [--allow-branch]"
    echo ""
    echo "Arguments:"
    echo "  version         Version to publish (e.g., 0.3.9 or 0.3.9rc1)"
    echo "  --dry-run       Run all checks but don't create tags or upload to PyPI"
    echo "  --allow-branch  Allow publishing from non-main branches"
    echo ""
    echo "Examples:"
    echo "  $0 0.3.9rc1                        # Full publish from main"
    echo "  $0 0.3.9rc1 --dry-run              # Dry run"
    echo "  $0 0.3.9rc1 --allow-branch         # Publish from current branch"
    exit 1
}

parse_args() {
    if [[ $# -lt 1 ]]; then
        usage
    fi

    VERSION="$1"
    shift

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --allow-branch)
                ALLOW_BRANCH=true
                shift
                ;;
            *)
                error "Unknown argument: $1"
                usage
                ;;
        esac
    done
}

# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

validate_version_format() {
    # Validate version matches X.Y.Z or X.Y.ZrcN pattern
    if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(rc[0-9]+)?$ ]]; then
        die "Invalid version format: $VERSION" \
            "Version must match pattern X.Y.Z or X.Y.ZrcN (e.g., 0.3.9 or 0.3.9rc1)"
    fi
    success "Version format is valid: $VERSION"
}

check_main_branch() {
    local current_branch
    current_branch=$(git branch --show-current)

    if [[ "$current_branch" != "main" ]]; then
        if [[ "$ALLOW_BRANCH" == true ]]; then
            warn "Not on main branch (allowed via --allow-branch): $current_branch"
        else
            die "Not on main branch" \
                "Current branch: $current_branch. Please checkout main or use --allow-branch"
        fi
    else
        success "On main branch"
    fi
}

check_clean_working_directory() {
    local status
    status=$(git status --porcelain)

    if [[ -n "$status" ]]; then
        die "Working directory has uncommitted changes" \
            "Please commit or stash your changes before publishing"
    fi
    success "Working directory is clean"
}

check_tag_does_not_exist() {
    local tag="v$VERSION"
    local existing_tag
    existing_tag=$(git tag -l "$tag")

    if [[ -n "$existing_tag" ]]; then
        die "Git tag already exists: $tag" \
            "Please choose a different version or delete the existing tag: git tag -d $tag"
    fi
    success "Git tag v$VERSION does not exist"
}

check_pypi_access() {
    if [[ ! -f "$PYPIRC_FILE" ]]; then
        die "PyPI config file not found: $PYPIRC_FILE" \
            "Please create ~/.pypirc with your PyPI credentials"
    fi

    # Check for [pypi] section with expected username
    if ! grep -q "^\[pypi\]" "$PYPIRC_FILE"; then
        die "No [pypi] section found in $PYPIRC_FILE" \
            "Please add a [pypi] section with your credentials"
    fi

    # Extract username from pypirc (handles various formats including leading whitespace)
    local username
    username=$(awk '/^\[pypi\]/,/^\[/ {if (/^[ \t]*username[ \t]*=/) {sub(/^[ \t]*username[ \t]*=[ \t]*/, ""); sub(/[ \t]*$/, ""); print; exit}}' "$PYPIRC_FILE")

    if [[ "$username" != "$EXPECTED_PYPI_USERNAME" ]]; then
        die "PyPI username mismatch" \
            "Expected username '$EXPECTED_PYPI_USERNAME' but found '$username' in $PYPIRC_FILE"
    fi
    success "PyPI access configured (username: $EXPECTED_PYPI_USERNAME)"
}

check_twine_works() {
    if ! uv run --with twine python -m twine --version > /dev/null 2>&1; then
        die "Twine is not working" \
            "Please ensure uv is installed and working"
    fi
    local twine_version
    twine_version=$(uv run --with twine python -m twine --version 2>/dev/null | head -1)
    success "Twine is available: $twine_version"
}

# ==============================================================================
# BUILD AND PUBLISH FUNCTIONS
# ==============================================================================

run_tests() {
    header "Running Tests"
    info "Executing make test..."

    if ! make test; then
        die "Tests failed" \
            "Please fix the failing tests before publishing"
    fi
    success "All tests passed"
}

build_packages() {
    header "Building Packages"

    info "Cleaning dist directories..."
    make clean-dist

    info "Building all packages..."
    if ! make build; then
        die "Package build failed" \
            "Please check the build output for errors"
    fi
    success "All packages built successfully"
}

create_git_tag() {
    header "Creating Git Tag"

    local tag="v$VERSION"

    if [[ "$DRY_RUN" == true ]]; then
        warn "[DRY RUN] Would create git tag: $tag"
        return
    fi

    info "Creating tag: $tag"
    git tag "$tag"
    success "Created git tag: $tag"
}

rebuild_with_tag() {
    header "Rebuilding with Tag"

    if [[ "$DRY_RUN" == true ]]; then
        warn "[DRY RUN] Would rebuild packages with tag for correct version embedding"
        return
    fi

    info "Cleaning dist directories..."
    make clean-dist

    info "Rebuilding all packages with tag..."
    if ! make build; then
        die "Package rebuild failed" \
            "Please check the build output for errors"
    fi
    success "All packages rebuilt with correct version"

    # Verify the built packages have the correct version
    info "Verifying built package versions..."
    for pkg_dir in "${PACKAGE_DIRS[@]}"; do
        local wheel_count
        wheel_count=$(ls "$pkg_dir/dist/"*.whl 2>/dev/null | wc -l)
        if [[ "$wheel_count" -eq 0 ]]; then
            die "No wheel found in $pkg_dir/dist/" \
                "Build may have failed silently"
        fi
        info "  Found wheel(s) in $pkg_dir/dist/"
    done
}

upload_to_pypi() {
    header "Uploading to PyPI"

    if [[ "$DRY_RUN" == true ]]; then
        warn "[DRY RUN] Would upload the following packages to PyPI:"
        for pkg_dir in "${PACKAGE_DIRS[@]}"; do
            echo "  From $pkg_dir/dist/:"
            ls -1 "$pkg_dir/dist/" 2>/dev/null | sed 's/^/    /'
        done
        return
    fi

    for pkg_dir in "${PACKAGE_DIRS[@]}"; do
        info "Uploading $pkg_dir..."
        (
            cd "$pkg_dir"
            if ! uv run --with twine python -m twine upload dist/*; then
                die "Failed to upload $pkg_dir" \
                    "Please check the twine output for errors"
            fi
        )
        success "Uploaded $pkg_dir"
    done

    success "All packages uploaded to PyPI"
}

push_git_tag() {
    header "Pushing Git Tag"

    local tag="v$VERSION"

    if [[ "$DRY_RUN" == true ]]; then
        warn "[DRY RUN] Would push git tag to origin: $tag"
        return
    fi

    info "Pushing tag to origin: $tag"
    if ! git push origin "$tag"; then
        die "Failed to push tag to origin" \
            "Please push the tag manually: git push origin $tag"
    fi
    success "Pushed git tag: $tag"
}

# ==============================================================================
# MAIN
# ==============================================================================

main() {
    parse_args "$@"

    header "DataDesigner Publish v$VERSION"

    if [[ "$DRY_RUN" == true ]]; then
        warn "DRY RUN MODE - No tags will be created or packages uploaded"
    fi

    # Pre-flight checks
    header "Pre-flight Checks"
    validate_version_format
    check_main_branch
    check_clean_working_directory
    check_tag_does_not_exist
    check_pypi_access
    check_twine_works

    # Run tests
    run_tests

    # Build packages (initial build to verify everything works)
    build_packages

    # Create git tag
    create_git_tag

    # Rebuild with tag for correct version embedding
    rebuild_with_tag

    # Upload to PyPI
    upload_to_pypi

    # Push git tag to remote
    push_git_tag

    # Final summary
    header "Publish Complete"
    if [[ "$DRY_RUN" == true ]]; then
        warn "DRY RUN completed successfully"
        info "To perform the actual publish, run without --dry-run:"
        echo "  $0 $VERSION"
    else
        success "Successfully published DataDesigner v$VERSION"
        echo ""
        echo "Packages published:"
        for pkg_dir in "${PACKAGE_DIRS[@]}"; do
            local pkg_name
            pkg_name=$(basename "$pkg_dir")
            echo "  - $pkg_name"
        done
        echo ""
        echo "View on PyPI:"
        echo "  https://pypi.org/project/data-designer-config/$VERSION/"
        echo "  https://pypi.org/project/data-designer-engine/$VERSION/"
        echo "  https://pypi.org/project/data-designer/$VERSION/"
    fi
}

main "$@"
