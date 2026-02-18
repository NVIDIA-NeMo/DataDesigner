# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Literal

PAGER_FILENAME = "sample_records_browser.html"


def create_sample_records_pager(
    sample_records_dir: Path,
    num_records: int,
    *,
    num_columns: int | None = None,
    theme: Literal["dark", "light"] = "dark",
) -> None:
    """Generate sample_records_browser.html in the given directory.

    Creates a single HTML file that provides a paginated view over
    record_0.html through record_{num_records-1}.html.

    Args:
        sample_records_dir: Directory containing record_0.html, record_1.html, etc.
        num_records: Number of record files (0-based indices through num_records - 1).
        num_columns: Number of columns in the dataset (displayed in the subtitle).
        theme: Initial color theme — dark or light.
    """
    if num_records <= 0:
        return

    records = [{"path": f"record_{i}.html"} for i in range(num_records)]
    results_dir_name = html.escape(sample_records_dir.parent.name)
    title = f"Sample Records Browser - {results_dir_name}"

    pager_html = _build_pager_html(title=title, records=records, num_columns=num_columns, theme=theme)
    out_path = sample_records_dir / PAGER_FILENAME
    out_path.write_text(pager_html, encoding="utf-8")


def _build_pager_html(
    *,
    title: str,
    records: list[dict[str, str]],
    num_columns: int | None = None,
    theme: Literal["dark", "light"] = "dark",
) -> str:
    records_json = json.dumps(records)
    initial_theme_js = f'"{theme}"'
    github_icon = _GITHUB_ICON_SVG

    subtitle_parts = [f"Total Records: {len(records)}"]
    if num_columns is not None:
        subtitle_parts.append(f"Number of Columns: {num_columns}")
    subtitle = html.escape(", ".join(subtitle_parts))

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style id="theme-css"></style>
<style>
* {{
  box-sizing: border-box;
}}
html, body {{
  height: 100%;
}}
body {{
  margin: 0;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: var(--text);
  background: var(--body-bg);
}}
.app {{
  display: flex;
  flex-direction: column;
  height: 100vh;
  gap: 10px;
  padding: 10px;
}}
.topbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--topbar-bg);
  box-shadow: var(--shadow);
}}
.title-block h1 {{
  margin: 0;
  font-size: 20px;
  line-height: 1.2;
  letter-spacing: 0.3px;
}}
.github-link {{
  color: var(--muted);
  text-decoration: none;
  font-size: 13px;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  transition: color 120ms ease;
}}
.github-link:hover {{
  color: var(--text);
}}
.subtitle {{
  margin-top: 6px;
  color: var(--muted);
  font-size: 12px;
}}
.toolbar {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 10px;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--panel);
  box-shadow: var(--shadow);
}}
button, select {{
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--btn-bg);
  padding: 7px 12px;
  transition: 120ms ease;
}}
button {{
  cursor: pointer;
}}
button:hover:not(:disabled) {{
  border-color: var(--btn-hover-border);
  box-shadow: var(--btn-hover-glow);
}}
button:disabled {{
  opacity: 0.45;
  cursor: not-allowed;
}}
#theme-toggle {{
  margin-left: auto;
  font-size: 18px;
  padding: 4px 10px;
  line-height: 1;
}}
#frame-wrap {{
  flex: 1;
  min-height: 0;
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  background: var(--frame-bg);
  box-shadow: var(--shadow);
}}
iframe {{
  display: block;
  width: 100%;
  height: 100%;
  border: 0;
}}
@media (max-width: 800px) {{
  .topbar {{
    align-items: flex-start;
    flex-direction: column;
  }}
  .toolbar {{
    flex-wrap: wrap;
  }}
}}
</style>
</head>
<body>
  <div class="app">
    <div class="topbar">
      <div class="title-block">
        <h1>\U0001f3a8 NeMo Data Designer \u2013 Preview Record Browser</h1>
        <div class="subtitle">{subtitle}</div>
      </div>
      <a class="github-link" href="https://github.com/NVIDIA-NeMo/DataDesigner" target="_blank" rel="noopener noreferrer">
        {github_icon} GitHub
      </a>
    </div>
    <div class="toolbar">
      <select id="jump" aria-label="Jump to record"></select>
      <span id="counter" style="font-size:13px;color:var(--muted);min-width:80px;text-align:center;"></span>
      <button id="prev" aria-label="Previous record">\u2190 Prev</button>
      <button id="next" aria-label="Next record">Next \u2192</button>
      <button id="theme-toggle" aria-label="Toggle theme"></button>
    </div>
    <div id="frame-wrap">
      <iframe id="frame" title="sample record preview"></iframe>
    </div>
  </div>
  <script>
    const DARK_THEME = `:root {{
  color-scheme: dark;
  --panel: rgba(8, 20, 46, 0.88);
  --border: rgba(108, 160, 255, 0.32);
  --text: #e5efff;
  --muted: #97add2;
  --shadow: 0 16px 44px rgba(0, 6, 22, 0.58);
  --body-bg:
    radial-gradient(1100px 520px at 15% -20%, rgba(31,77,163,0.45) 0%, transparent 62%),
    radial-gradient(850px 460px at 95% -15%, rgba(38,73,170,0.42) 0%, transparent 60%),
    linear-gradient(180deg, #000612 0%, #03112a 45%, #010817 100%);
  --topbar-bg: linear-gradient(135deg, rgba(10,26,62,0.92) 0%, rgba(7,18,42,0.95) 100%);
  --btn-bg: linear-gradient(180deg, rgba(17,48,104,0.85) 0%, rgba(10,28,64,0.92) 100%);
  --btn-hover-border: rgba(124, 184, 255, 0.65);
  --btn-hover-glow: 0 0 0 3px rgba(59, 169, 255, 0.15);
  --frame-bg: rgba(2, 10, 26, 0.85);
}}`;

    const LIGHT_THEME = `:root {{
  color-scheme: light;
  --panel: rgba(255, 255, 255, 0.95);
  --border: rgba(0, 0, 0, 0.15);
  --text: #1a1a2e;
  --muted: #6c757d;
  --shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  --body-bg: linear-gradient(180deg, #e8ecf1 0%, #ffffff 45%, #f8f9fa 100%);
  --topbar-bg: linear-gradient(135deg, rgba(248,249,250,0.98) 0%, rgba(235,238,242,0.95) 100%);
  --btn-bg: linear-gradient(180deg, rgba(230,235,245,0.9) 0%, rgba(215,220,230,0.95) 100%);
  --btn-hover-border: rgba(0, 80, 180, 0.45);
  --btn-hover-glow: 0 0 0 3px rgba(0, 80, 180, 0.1);
  --frame-bg: rgba(245, 246, 248, 0.85);
}}`;

    const records = {records_json};
    let index = 0;
    let isDark = {initial_theme_js} === "dark";

    const frame = document.getElementById("frame");
    const prev = document.getElementById("prev");
    const next = document.getElementById("next");
    const jump = document.getElementById("jump");
    const themeStyle = document.getElementById("theme-css");
    const themeToggle = document.getElementById("theme-toggle");
    const counter = document.getElementById("counter");

    for (let i = 0; i < records.length; i += 1) {{
      const opt = document.createElement("option");
      opt.value = i;
      opt.textContent = `Record ${{i + 1}}`;
      jump.appendChild(opt);
    }}

    function applyPagerTheme() {{
      themeStyle.textContent = isDark ? DARK_THEME : LIGHT_THEME;
      themeToggle.textContent = isDark ? "\u2600\ufe0f" : "\U0001f319";
      themeToggle.title = isDark ? "Switch to light mode" : "Switch to dark mode";
    }}

    function applyThemeToIframe() {{
      if (frame.contentWindow) {{
        frame.contentWindow.postMessage({{type: "theme", dark: isDark}}, "*");
      }}
    }}

    frame.addEventListener("load", applyThemeToIframe);

    themeToggle.addEventListener("click", () => {{
      isDark = !isDark;
      applyPagerTheme();
      applyThemeToIframe();
    }});

    function show() {{
      frame.src = records[index].path;
      jump.value = String(index);
      prev.disabled = index === 0;
      next.disabled = index === records.length - 1;
      counter.textContent = `${{index + 1}} of ${{records.length}}`;
    }}

    prev.addEventListener("click", () => {{
      if (index > 0) {{
        index -= 1;
        show();
      }}
    }});

    next.addEventListener("click", () => {{
      if (index < records.length - 1) {{
        index += 1;
        show();
      }}
    }});

    jump.addEventListener("change", (event) => {{
      index = Number(event.target.value);
      show();
    }});

    document.addEventListener("keydown", (event) => {{
      if (event.key === "ArrowLeft" && index > 0) {{
        index -= 1;
        show();
      }} else if (event.key === "ArrowRight" && index < records.length - 1) {{
        index += 1;
        show();
      }}
    }});

    applyPagerTheme();
    show();
  </script>
</body>
</html>
"""


_GITHUB_ICON_SVG = (
    '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">'
    '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38'
    " 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28"
    "-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28"
    "-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02"
    ".08-2.12 0 0 .67-.21 2.2.82a7.63 7.63 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44"
    " 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25"
    ".54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16"
    ' 8c0-4.42-3.58-8-8-8z"/></svg>'
)
