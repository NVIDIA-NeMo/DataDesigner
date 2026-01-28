#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Publish script for DataDesigner
# Publishes all three subpackages to PyPI with the same version.
#
# Usage:
#   ./scripts/publish.sh 0.3.9rc1             # Full publish
#   ./scripts/publish.sh 0.3.9rc1 --dry-run   # Dry run (build, check, no upload)
#   ./scripts/publish.sh 0.3.9rc1 --test-pypi # Upload to TestPyPI instead of PyPI

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

# PyPI repository URLs
PYPI_REPOSITORY="pypi"
TEST_PYPI_REPOSITORY="testpypi"
TEST_PYPI_URL="https://test.pypi.org/simple/"

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

abort() {
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
FORCE_TAG=false
TEST_PYPI=false

usage() {
    local exit_code="${1:-1}"
    echo "Usage: $0 <version> [options]"
    echo ""
    echo "Publish all DataDesigner packages to PyPI with synchronized versions."
    echo ""
    echo "Arguments:"
    echo "  version           Version to publish (e.g., 0.3.9 or 0.3.9rc1)"
    echo ""
    echo "Options:"
    echo "  -h, --help        Show this help message and exit"
    echo "  --dry-run         Build packages and run validation (twine check) but don't"
    echo "                    create tags or upload. Good for CI validation."
    echo "  --test-pypi       Upload to TestPyPI (test.pypi.org) instead of production PyPI."
    echo "                    Useful for testing the full upload flow safely."
    echo "  --allow-branch    Allow publishing from non-main branches"
    echo "  --force-tag       Overwrite existing git tag if it exists"
    echo ""
    echo "Examples:"
    echo "  $0 0.3.9rc1                        # Full publish to PyPI"
    echo "  $0 0.3.9rc1 --dry-run              # Validate only (build + twine check)"
    echo "  $0 0.3.9rc1 --test-pypi            # Upload to TestPyPI"
    echo "  $0 0.3.9rc1 --test-pypi --allow-branch  # Test from feature branch"
    echo "  $0 0.3.9rc1 --force-tag            # Overwrite existing tag"
    echo ""
    echo "Version format:"
    echo "  Valid:   0.3.9, 0.3.9rc1, 1.0.0rc2"
    echo "  Invalid: v0.3.9, 0.3.9-rc1, 0.3.9a1"
    exit "$exit_code"
}

parse_args() {
    # Check for help flag first
    for arg in "$@"; do
        if [[ "$arg" == "-h" ]] || [[ "$arg" == "--help" ]]; then
            usage 0
        fi
    done

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
            --test-pypi)
                TEST_PYPI=true
                shift
                ;;
            --allow-branch)
                ALLOW_BRANCH=true
                shift
                ;;
            --force-tag)
                FORCE_TAG=true
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
        abort "Invalid version format: $VERSION" \
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
            abort "Not on main branch" \
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
        abort "Working directory has uncommitted changes" \
            "Please commit or stash your changes before publishing"
    fi
    success "Working directory is clean"
}

check_tag_does_not_exist() {
    if [[ "$DRY_RUN" == true ]]; then
        info "Skipping git tag check (dry-run mode)"
        return
    fi

    local tag="v$VERSION"
    local existing_tag
    existing_tag=$(git tag -l "$tag")

    if [[ -n "$existing_tag" ]]; then
        if [[ "$FORCE_TAG" == true ]]; then
            warn "Git tag already exists: $tag (will be overwritten via --force-tag)"
        else
            abort "Git tag already exists: $tag" \
                "Please choose a different version or delete the existing tag: git tag -d $tag"
        fi
    else
        success "Git tag v$VERSION does not exist"
    fi
}

check_pypi_access() {
    if [[ "$DRY_RUN" == true ]]; then
        info "Skipping PyPI access check (dry-run mode)"
        return
    fi

    if [[ ! -f "$PYPIRC_FILE" ]]; then
        abort "PyPI config file not found: $PYPIRC_FILE" \
            "Please create ~/.pypirc with your PyPI credentials"
    fi

    if [[ "$TEST_PYPI" == true ]]; then
        # Check for testpypi section in pypirc
        if ! grep -q "\[testpypi\]" "$PYPIRC_FILE"; then
            abort "TestPyPI section not found in $PYPIRC_FILE" \
                "Please add a [testpypi] section to ~/.pypirc"
        fi
        success "TestPyPI access configured"
    else
        # Check that the expected username exists in pypirc
        if ! grep -q "$EXPECTED_PYPI_USERNAME" "$PYPIRC_FILE"; then
            abort "Expected username '$EXPECTED_PYPI_USERNAME' not found in $PYPIRC_FILE" \
                "Please add your PyPI credentials with username '$EXPECTED_PYPI_USERNAME'"
        fi
        success "PyPI access configured (username: $EXPECTED_PYPI_USERNAME)"
    fi
}

check_twine_works() {
    if ! uv run --with twine python -m twine --version > /dev/null 2>&1; then
        abort "Twine is not working" \
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
        abort "Tests failed" \
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
        abort "Package build failed" \
            "Please check the build output for errors"
    fi
    success "All packages built successfully"
}

check_packages_with_twine() {
    header "Validating Packages with Twine"

    info "Running twine check on all distribution files..."

    local check_failed=false
    for pkg_dir in "${PACKAGE_DIRS[@]}"; do
        local pkg_name
        pkg_name=$(basename "$pkg_dir")
        info "Checking $pkg_name..."

        if ! uv run --with twine python -m twine check "$pkg_dir/dist/"*; then
            error "twine check failed for $pkg_name"
            check_failed=true
        else
            success "  $pkg_name passed twine check"
        fi
    done

    if [[ "$check_failed" == true ]]; then
        abort "Package validation failed" \
            "Please fix the issues reported by twine check"
    fi
    success "All packages passed twine check"
}

create_git_tag() {
    header "Creating Git Tag"

    local tag="v$VERSION"

    if [[ "$DRY_RUN" == true ]]; then
        if [[ "$FORCE_TAG" == true ]]; then
            warn "[DRY RUN] Would force create git tag: $tag"
        else
            warn "[DRY RUN] Would create git tag: $tag"
        fi
        return
    fi

    if [[ "$FORCE_TAG" == true ]]; then
        info "Force creating tag: $tag"
        git tag -f "$tag"
    else
        info "Creating tag: $tag"
        git tag "$tag"
    fi

    if [[ "$TEST_PYPI" == true ]]; then
        success "Created git tag: $tag (local only, will not be pushed)"
    else
        success "Created git tag: $tag"
    fi
}

verify_package_versions() {
    # Verify the built packages have the correct version
    header "Verifying Package Versions"
    info "Checking that built packages match expected version: $VERSION"

    for pkg_dir in "${PACKAGE_DIRS[@]}"; do
        local pkg_name
        pkg_name=$(basename "$pkg_dir")

        # Check wheel exists
        local wheel_file
        wheel_file=$(ls "$pkg_dir/dist/"*.whl 2>/dev/null | head -1)
        if [[ -z "$wheel_file" ]]; then
            abort "No wheel found in $pkg_dir/dist/" \
                "Build may have failed silently"
        fi

        # Verify version in wheel filename
        # Use -E for extended regex (works on both macOS/BSD and Linux/GNU sed)
        local wheel_version
        wheel_version=$(basename "$wheel_file" | sed -E -n 's/.*-([0-9]+\.[0-9]+\.[0-9]+(rc[0-9]+)?)-.*/\1/p')
        if [[ "$wheel_version" != "$VERSION" ]]; then
            abort "Version mismatch in $pkg_name wheel" \
                "Expected: $VERSION, Found: $wheel_version in $(basename "$wheel_file")"
        fi
        info "  $pkg_name: $wheel_version ✓"
    done
    success "All package versions verified"
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
        abort "Package rebuild failed" \
            "Please check the build output for errors"
    fi
    success "All packages rebuilt with correct version"

    # Verify versions
    verify_package_versions
}

upload_to_pypi() {
    local target_repo="$PYPI_REPOSITORY"
    local target_name="PyPI"
    local repo_args=()

    if [[ "$TEST_PYPI" == true ]]; then
        target_repo="$TEST_PYPI_REPOSITORY"
        target_name="TestPyPI"
        repo_args=("--repository" "$target_repo")
    fi

    header "Uploading to $target_name"

    if [[ "$DRY_RUN" == true ]]; then
        warn "[DRY RUN] Would upload the following packages to $target_name:"
        for pkg_dir in "${PACKAGE_DIRS[@]}"; do
            echo "  From $pkg_dir/dist/:"
            ls -1 "$pkg_dir/dist/" 2>/dev/null | sed 's/^/    /'
        done
        return
    fi

    for pkg_dir in "${PACKAGE_DIRS[@]}"; do
        local pkg_name
        pkg_name=$(basename "$pkg_dir")
        info "Uploading $pkg_name to $target_name..."
        (
            cd "$pkg_dir"
            if ! uv run --with twine python -m twine upload "${repo_args[@]}" dist/*; then
                abort "Failed to upload $pkg_name to $target_name" \
                    "If some packages uploaded successfully, you may need to:\n  1. Wait for the failed package issue to be resolved\n  2. Re-run with --force-tag: ./scripts/publish.sh $VERSION --force-tag"
            fi
        )
        success "Uploaded $pkg_name to $target_name"
    done

    success "All packages uploaded to $target_name"
}

push_git_tag() {
    header "Pushing Git Tag"

    local tag="v$VERSION"

    if [[ "$DRY_RUN" == true ]]; then
        if [[ "$FORCE_TAG" == true ]]; then
            warn "[DRY RUN] Would force push git tag to origin: $tag"
        else
            warn "[DRY RUN] Would push git tag to origin: $tag"
        fi
        return
    fi

    if [[ "$TEST_PYPI" == true ]]; then
        info "Cleaning up local tag (TestPyPI mode - tag not pushed)"
        git tag -d "$tag" > /dev/null 2>&1 || true
        success "Deleted local git tag: $tag"
        return
    fi

    if [[ "$FORCE_TAG" == true ]]; then
        info "Force pushing tag to origin: $tag"
        if ! git push -f origin "$tag"; then
            abort "Failed to force push tag to origin" \
                "Please push the tag manually: git push -f origin $tag"
        fi
    else
        info "Pushing tag to origin: $tag"
        if ! git push origin "$tag"; then
            abort "Failed to push tag to origin" \
                "Please push the tag manually: git push origin $tag"
        fi
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
        info "Will build packages and run twine check for validation"
    fi
    if [[ "$TEST_PYPI" == true ]]; then
        warn "TEST PYPI MODE - Uploading to test.pypi.org instead of pypi.org"
    fi
    if [[ "$FORCE_TAG" == true ]]; then
        warn "FORCE TAG MODE - Existing tag will be overwritten"
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

    # Different flows for dry-run, TestPyPI, and production
    if [[ "$DRY_RUN" == true ]]; then
        # Dry run: build without tag (dev versions), validate, done
        build_packages
        check_packages_with_twine

    elif [[ "$TEST_PYPI" == true ]]; then
        # TestPyPI: create temporary tag first, build once with correct version
        # This is more efficient since the tag will be deleted anyway
        create_git_tag
        build_packages
        # Verify versions match expected
        verify_package_versions
        check_packages_with_twine
        upload_to_pypi
        push_git_tag  # This will delete the local tag for TestPyPI

    else
        # Production: build first to validate, then tag and rebuild for safety
        # This ensures we don't create a tag if the build process itself is broken
        build_packages
        check_packages_with_twine
        create_git_tag
        rebuild_with_tag
        check_packages_with_twine
        upload_to_pypi
        push_git_tag
    fi

    # Final summary
    header "Publish Complete"
    if [[ "$DRY_RUN" == true ]]; then
        success "DRY RUN completed successfully"
        echo ""
        echo "All validation checks passed:"
        echo "  - Version format validated"
        echo "  - Packages built successfully"
        echo "  - Packages passed twine check"
        echo ""
        info "Next steps:"
        echo "  - To publish to TestPyPI (recommended first):"
        echo "      $0 $VERSION --test-pypi"
        echo "  - To publish to production PyPI:"
        echo "      $0 $VERSION"
    elif [[ "$TEST_PYPI" == true ]]; then
        success "Successfully published DataDesigner v$VERSION to TestPyPI"
        echo ""
        echo "Packages published to TestPyPI:"
        for pkg_dir in "${PACKAGE_DIRS[@]}"; do
            local pkg_name
            pkg_name=$(basename "$pkg_dir")
            echo "  - $pkg_name"
        done
        echo ""
        echo "View on TestPyPI:"
        echo "  https://test.pypi.org/project/data-designer-config/$VERSION/"
        echo "  https://test.pypi.org/project/data-designer-engine/$VERSION/"
        echo "  https://test.pypi.org/project/data-designer/$VERSION/"
        echo ""
        echo "Test installation with:"
        echo "  pip install --index-url $TEST_PYPI_URL data-designer==$VERSION"
        echo ""
        info "If everything looks good, publish to production PyPI:"
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
