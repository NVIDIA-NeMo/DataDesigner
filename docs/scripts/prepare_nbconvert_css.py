# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import sys
from importlib.metadata import distribution
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

WHEEL_URL = (
    "https://files.pythonhosted.org/packages/81/fd/a8d457af68848a8e5c12dd1d84ff35c415db471af78280e0703db4efbb12/"
    "nbconvert-7.17.2-py3-none-any.whl"
)
WHEEL_SHA256 = "6a143c11c00bf831604da4b1be4347d36ea52367057eb920135d69bc2862fed2"
WHEEL_TEMPLATES = "nbconvert-7.17.2.data/data/share/jupyter/nbconvert/templates"
CSS_SHA256 = {
    "classic/static/style.css": "5865a609f4437b0464bc121cd567b619074e540a0515a3b82f222f764eb51e01",
    "lab/static/index.css": "917ff47850a7cc08fd0658026fda7672a85220aaab258e8849e891b37426f947",
    "lab/static/theme-dark.css": "795f2d5069737cbeb5cba01e6b5c7cadbde227c909e43004c5a60f58d5160aec",
    "lab/static/theme-light.css": "11bf3558fd3ed353a4c1401ac0c1730d01df073f6436d357c5bbf02a03bd6962",
}


def prepare_nbconvert_css() -> None:
    """Restore the pinned nbconvert build's HTML assets without replacing its Python code."""
    installed = distribution("nbconvert")
    templates = Path(sys.prefix) / "share" / "jupyter" / "nbconvert" / "templates"
    if all(
        (templates / name).is_file() and hashlib.sha256((templates / name).read_bytes()).hexdigest() == digest
        for name, digest in CSS_SHA256.items()
    ):
        print("nbconvert HTML stylesheets already present.")
        return

    record_entry = next(
        entry for entry in installed.files or () if entry.name == "RECORD" and entry.parent.name.endswith(".dist-info")
    )
    record_path = Path(installed.locate_file(record_entry))
    print("Downloading nbconvert HTML stylesheets from PyPI (256 KiB wheel)...")
    with urlopen(WHEEL_URL, timeout=30) as response:
        wheel = response.read()
    if hashlib.sha256(wheel).hexdigest() != WHEEL_SHA256:
        raise ValueError("nbconvert CSS wheel checksum mismatch")
    with ZipFile(io.BytesIO(wheel)) as archive:
        stylesheets = {name: archive.read(f"{WHEEL_TEMPLATES}/{name}") for name in CSS_SHA256}
    for name, content in stylesheets.items():
        if hashlib.sha256(content).hexdigest() != CSS_SHA256[name]:
            raise ValueError(f"nbconvert stylesheet checksum mismatch: {name}")

    paths = {
        name: os.path.relpath(templates / name, record_path.parent.parent).replace(os.sep, "/") for name in stylesheets
    }
    records = [
        row for row in csv.reader(io.StringIO(record_path.read_text(encoding="utf-8"))) if row[0] not in paths.values()
    ]
    for name, content in stylesheets.items():
        destination = templates / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        records.append([paths[name], f"sha256={digest}", str(len(content))])
    # Track restored assets so reinstalling or uninstalling nbconvert also removes them.
    with record_path.open("w", newline="", encoding="utf-8") as record:
        csv.writer(record).writerows(records)
    print("Restored nbconvert HTML stylesheets; installed Python code is unchanged.")


if __name__ == "__main__":
    prepare_nbconvert_css()
