#!/usr/bin/env python3
"""Extract code blocks from deep-research-trajectories.mdx for ExpandableCode.

Run from fern/ directory:
    python scripts/extract-code.py
"""

import re
from pathlib import Path

fern_dir = Path(__file__).resolve().parent.parent
mdx_path = fern_dir / "v0.5.0/pages/devnotes/deep-research-trajectories.mdx"

with open(mdx_path, "r") as f:
    content = f.read()


def extract_accordion_code(title: str) -> str:
    pattern = rf'<Accordion title="Full source: {re.escape(title)}">\s*```python\s*\n(.*?)\n```'
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).rstrip() if match else ""


for name in ["prepare_corpus.py", "retriever_mcp.py"]:
    code = extract_accordion_code(name)
    var_name = name.replace(".py", "").replace("-", "_") + "Code"
    out = f'''/** Full source for {name} - used in ExpandableCode on deep-research-trajectories page */

export const {var_name} = `{code}`;
'''
    out_path = fern_dir / f"components/devnotes/{name.replace('.py', '-code')}.ts"
    with open(out_path, "w") as f:
        f.write(out)
    print(f"Wrote {out_path} ({len(code)} chars)")
