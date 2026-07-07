# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, distribution, packages_distributions
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

DEPENDENCY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
MODULE_DISTRIBUTION_OVERRIDES = {
    "IPython": "ipython",
    "PIL": "pillow",
    "dateutil": "python-dateutil",
    "jwt": "pyjwt",
    "yaml": "pyyaml",
}


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def dependency_name(specifier: str) -> str | None:
    match = DEPENDENCY_NAME.match(specifier)
    return normalize_name(match.group()) if match else None


def requirement_dependencies(
    specifiers: list[str],
    selected_extras: set[str] | None = None,
    *,
    allow_version_template: bool = False,
) -> dict[str, set[str]]:
    dependencies: dict[str, set[str]] = defaultdict(set)
    marker_extras = {"", *(selected_extras or set())}

    for specifier in specifiers:
        try:
            requirement = Requirement(specifier)
        except InvalidRequirement:
            if not allow_version_template or "{{ version }}" not in specifier:
                raise
            if name := dependency_name(specifier):
                dependencies[name]
            continue

        # Marker evaluation is intentionally relative to the current audit environment.
        if requirement.marker and not any(requirement.marker.evaluate({"extra": extra}) for extra in marker_extras):
            continue
        dependencies[normalize_name(requirement.name)].update(normalize_name(extra) for extra in requirement.extras)

    return dict(dependencies)


def declared_dependencies(pyproject_path: Path) -> dict[str, set[str]]:
    with pyproject_path.open("rb") as file:
        config = tomllib.load(file)

    project_dependencies = config.get("project", {}).get("dependencies", [])
    dynamic_dependencies = (
        config.get("tool", {})
        .get("hatch", {})
        .get("metadata", {})
        .get("hooks", {})
        .get("uv-dynamic-versioning", {})
        .get("dependencies", [])
    )
    dependencies = requirement_dependencies(project_dependencies)
    for name, extras in requirement_dependencies(dynamic_dependencies, allow_version_template=True).items():
        dependencies.setdefault(name, set()).update(extras)
    return dependencies


def imported_modules(source_root: Path, repository_root: Path) -> dict[str, list[str]]:
    imports: dict[str, set[str]] = defaultdict(set)
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        relative_path = str(path.relative_to(repository_root))
        # Include guarded and TYPE_CHECKING imports; the recipe re-verifies runtime use before fixing a gap.
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name.partition(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module.partition(".")[0]]
            else:
                continue
            for module in modules:
                if module not in sys.stdlib_module_names and module != "data_designer":
                    imports[module].add(relative_path)
    return {module: sorted(paths) for module, paths in sorted(imports.items())}


def resolve_distribution(
    module: str,
    candidates: list[str],
    declared: set[str],
    declared_anywhere: set[str],
) -> str | None:
    if override := MODULE_DISTRIBUTION_OVERRIDES.get(module):
        return normalize_name(override)

    normalized = sorted({normalize_name(candidate) for candidate in candidates})
    for pool in (declared, declared_anywhere):
        matches = [candidate for candidate in normalized if candidate in pool]
        if len(matches) == 1:
            return matches[0]
    return normalized[0] if len(normalized) == 1 else None


def installed_requirements(distribution_name: str, selected_extras: set[str]) -> dict[str, set[str]]:
    try:
        requirements = distribution(distribution_name).requires or []
    except PackageNotFoundError:
        return {}

    return requirement_dependencies(requirements, selected_extras)


def audit_repository(
    repository_root: Path,
    module_distributions: dict[str, list[str]] | None = None,
    requirement_map: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    package_dirs = sorted(path.parent for path in repository_root.glob("packages/*/pyproject.toml"))
    projects: dict[str, dict[str, Any]] = {}

    for package_dir in package_dirs:
        with (package_dir / "pyproject.toml").open("rb") as file:
            project_name = normalize_name(tomllib.load(file)["project"]["name"])
        projects[project_name] = {
            "path": str(package_dir.relative_to(repository_root)),
            "declarations": declared_dependencies(package_dir / "pyproject.toml"),
            "imports": imported_modules(package_dir / "src", repository_root),
        }

    declared_by: dict[str, set[str]] = defaultdict(set)
    for project_name, project in projects.items():
        for dependency in project["declarations"]:
            declared_by[dependency].add(project_name)

    distribution_map = module_distributions if module_distributions is not None else packages_distributions()
    declared_anywhere = set(declared_by)

    requirement_cache: dict[tuple[str, frozenset[str]], dict[str, set[str]]] = {}

    def requirements_for(distribution_name: str, selected_extras: set[str]) -> dict[str, set[str]]:
        if distribution_name in projects:
            return projects[distribution_name]["declarations"]
        if requirement_map is not None:
            return requirement_dependencies(requirement_map.get(distribution_name, []), selected_extras)
        cache_key = (distribution_name, frozenset(selected_extras))
        if cache_key not in requirement_cache:
            requirement_cache[cache_key] = installed_requirements(distribution_name, selected_extras)
        return requirement_cache[cache_key]

    def dependency_closure(distribution_name: str, selected_extras: set[str]) -> set[str]:
        closure = set()
        pending = [(distribution_name, selected_extras)]
        visited: set[tuple[str, frozenset[str]]] = set()
        while pending:
            current, extras = pending.pop()
            current_key = (current, frozenset(extras))
            if current_key in visited:
                continue
            visited.add(current_key)
            for dependency, dependency_extras in requirements_for(current, extras).items():
                closure.add(dependency)
                pending.append((dependency, dependency_extras))
        return closure

    results = []
    for project_name, project in projects.items():
        declarations = project["declarations"]
        declared = set(declarations)
        dependency_closures = {
            dependency: dependency_closure(dependency, extras) for dependency, extras in declarations.items()
        }
        resolved_imports: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"modules": set(), "files": set()})
        unresolved = []

        for module, files in project["imports"].items():
            distribution_name = resolve_distribution(
                module,
                distribution_map.get(module, []),
                declared,
                declared_anywhere,
            )
            if distribution_name is None:
                unresolved.append({"module": module, "files": files})
                continue
            resolved_imports[distribution_name]["modules"].add(module)
            resolved_imports[distribution_name]["files"].update(files)

        missing = []
        for distribution_name, usage in sorted(resolved_imports.items()):
            if distribution_name in declared:
                continue
            sibling_declarations = sorted(declared_by[distribution_name] - {project_name})
            guaranteed_by = sorted(
                dependency for dependency, closure in dependency_closures.items() if distribution_name in closure
            )
            missing.append(
                {
                    "dependency": distribution_name,
                    "modules": sorted(usage["modules"]),
                    "files": sorted(usage["files"]),
                    "declared_by": sibling_declarations,
                    "guaranteed_by": guaranteed_by,
                    "severity": "low" if guaranteed_by else "high",
                }
            )

        results.append(
            {
                "package": project_name,
                "path": project["path"],
                "declared": sorted(declared),
                "imported": sorted(resolved_imports),
                "missing": missing,
                "unresolved_modules": unresolved,
            }
        )

    return {"packages": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory package import/dependency gaps")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = audit_repository(args.root)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
