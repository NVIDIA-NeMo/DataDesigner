# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer>=0.5.6",
# ]
# ///
"""Long-Document Understanding Single-Page QA Recipe

Generate high-quality single-page question-answer pairs that improve VLM
long-document understanding across key categories: Text, Table, Chart,
Image/Figure, and Layout. MMLongBench-Doc is used to track progress.

Each question is anchored to a unique on-page element (page number, table/
figure number, section title) so it remains unambiguous when collated with
questions from all other pages into a full-document training sample.

For each seed record the pipeline:

  1. Samples a question type (multiple choice, yes/no, string, layout,
     numerical int/float/percentage, list, not answerable)
  2. Generates an anchored question from the page image
  3. Generates an answer with chain-of-thought reasoning (captured separately)
  4. Evaluates overall quality (document quality, relevance, correctness,
     format compliance, anchor quality, and reasoning quality) as a 0/1/2 score

Prerequisites:
    - A seed parquet file containing:
        * `png_images_base64` – JSON array of base64-encoded PNGs (one
          element per page; single-page seeds have a one-element array).
    - A vLLM-compatible deployment of the VLM
      (default: Qwen/Qwen3-VL-235B-A22B-Thinking-FP8).
      Recommended vLLM launch flags:
        --tensor-parallel-size 4
        --max-model-len 50000
        --gpu-memory-utilization 0.90
        --reasoning-parser deepseek_r1
        --limit-mm-per-prompt '{"video": 0}'
        --trust-remote-code

      Example launch script for 4× H100:
        docker run --gpus all \
            -p 8000:8000 \
            vllm/vllm-openai:latest \
            --model Qwen/Qwen3-VL-235B-A22B-Thinking-FP8 \
            --tensor-parallel-size 4 \
            --max-model-len 50000 \
            --gpu-memory-utilization 0.90 \
            --reasoning-parser deepseek_r1 \
            --limit-mm-per-prompt '{"video": 0}' \
            --trust-remote-code

Run:
    # Basic usage (generates 5 records by default)
    uv run 06-single-page-qa-sdg.py --vllm-endpoint http://localhost:8000/v1 --seed-path seed_data/seed_per_page.parquet

    # Custom model and record count
    uv run 06-single-page-qa-sdg.py --vllm-endpoint http://localhost:8000/v1 --seed-path seed_data/seed_per_page.parquet --num-records 100

    # For help message and available options
    uv run 06-single-page-qa-sdg.py --help
"""

from pathlib import Path

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults

DEFAULT_VLM_MODEL = "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8"
VLLM_PROVIDER_NAME = "vllm"

# =============================================================================
# Image context helper
# =============================================================================

IMAGE_CONTEXT = [
    dd.ImageContext(
        # Expects a single-element JSON array from the per-page seed.
        column_name="png_images_base64",
        data_type=dd.ModalityDataType.BASE64,
        image_format=dd.ImageFormat.PNG,
    ),
]

# =============================================================================
# Prompt templates
# =============================================================================

PROMPT_QUESTION = """\
You are an expert at writing SINGLE-PAGE training questions that improve VLM performance on MMLongBench-Doc categories:
- Text (plain paragraphs)
- Table (tabular data)
- Chart (quantitative plots)
- Image/Figure (diagrams, UI screenshots, illustrations)
- Layout (spatial + structural anchors)

You see one page from a larger document. Your question will be collated with questions from
ALL other pages into a single training sample — the trainee sees the FULL document, not just
this page. Phrases like "on this page" or "the table" are ambiguous in that context.
Always anchor with a printed page number, unique title, or unique element number.

Your task: Create ONE high-quality question of type <question-type> that can be answered
using ONLY the visible content.

<question-type>
{{question_type}}
</question-type>

═══════════════════════════════════════════════════════════════════════════════
HARD CONSTRAINTS (must follow)
═══════════════════════════════════════════════════════════════════════════════

0) VISIBLE CONTENT ONLY
- The question must be answerable from the visible content alone.
- Do NOT ask questions that require any other page, the whole report, or cross-page aggregation.
- Do NOT ask for global counting/aggregation across the report/paper (e.g., "among all charts", "how many pages include…", "across the pages").
- Benchmark-style phrasing is allowed (e.g., "According to the report/paper, …"), BUT the question must still be answerable from the visible content.

1) Answerability
- {% if "not answerable" in question_type %}The question should still be relevant to the page, but the exact requested answer must NOT be present anywhere on this page.
- Build "not answerable" questions as NEAR-MISS negatives by changing exactly ONE required qualifier from a real visible fact.
- Good near-miss types: wrong year/date, wrong subgroup/series, wrong unit, wrong position, wrong row/column, wrong legend item, or displayed/shown vs listed/mentioned.
- Do NOT make a question "not answerable" only because it refers to some other page, the whole document, or a page number that is missing from the visible page.
- Near-miss templates by visual element:
  - Chart: "In the chart titled '<real title>', what is <metric> for <YEAR NOT SHOWN>?" (year is present for other series but not the asked one)
  - Table: "In Table <N>, what is the value for '<ROW NOT IN TABLE>' in column '<real column>'?"
  - Map/infographic: "In the map on page <N>, what number is shown for <REGION NOT LABELED>?"
  - Text: "In the section titled '<real title>', what is the <DETAIL NOT MENTIONED>?"
  - Layout: "In the <real position> panel, what is the <ELEMENT NOT PRESENT IN THAT PANEL>?"
{% else %}The correct answer MUST be present and clearly visible on this page.{% endif %}
- Do NOT ask anything that requires external knowledge, guesswork, or reading tiny illegible text.

2) Focus
- Identify the PRIMARY visual element on the page (text block, table, chart, figure, map,
  infographic, or layout structure) and ask about it.
- Ignore decorative headers/footers unless they are part of the primary element.

3) Difficulty (1–3 steps, all evidence on THIS page)
- 1-step (common): single lookup, one comparison, one count, one filter.
- 2-step (encouraged — targets top benchmark failures):
  - Lookup + lookup + compare: "Which is larger, row A or row B in column X?"
  - Lookup + compute: "What is Gross Profit as a % of Revenues?" (two cells, divide)
  - Scan + argmax: "Which subgroup has the largest value in the Change column?"
  - Lookup + difference: "How much did X change from 2020 to 2021?" (two cells, subtract)
  - Percentage → count: "If 5% of N=1500 said X, how many is that?"
- 3-step (allowed when all steps are trivial and on this page):
  - Lookup + lookup + ratio + round: financial ratio questions.
  - Scan + count + filter: "How many rows have column Y > threshold?"
- Avoid: questions needing external knowledge, subjective judgment, or off-page evidence.

4) Unambiguous
- Exactly one correct answer.
- Avoid subjective terms ("most consistent", "significant", "best", "optimal").

5) No meta phrasing
- Do NOT say "the image" / "this image".
- Avoid "the document" unless it is part of an on-page title.
- You MAY use "According to the report/paper, …" occasionally, but never to imply cross-page evidence.

6) ANCHOR THE QUESTION (critical for per-document training)
- Questions are collated across ALL pages during training — anchors must be unambiguous
  within the ENTIRE document, not just this page.
- Anchor priority (prefer unique; most preferred first):
  1. Printed, explicit page number: "On page 42, ..." (usually in the header/footer)
  2. Unique title/caption/section heading: "In the chart titled "X", ..." / "In the section titled "Y", ..."
  3. Numbered element + local title: "In Table 3 under "X", ..." / "In Figure 7 titled "Y", ..." / "In Note 5 — "Z", ..."
  4. Fallback: Detailed structural anchor with heading: "In the right-column table under the heading "X", ..." / "In the top-right boxed callout titled "Y", ..."
- BANNED anchors (these become meaningless when collated across pages):
  "on this page", "in the image", "in the bottom half of the page",
  "the table" / "the chart" / "the figure" without a title or number,
  "on the left side" / "on the right side" without a heading.
  Always pair positional references with a title, number, or heading.

7) REQUIRE ANSWER-FORMAT HINTS IN THE QUESTION (targets strict eval failures)
- The question MUST explicitly tell the answer format when applicable:
  - For list questions: add "Return a JSON array of strings on one line, e.g., ["A", "B"]."
  - For percentages: add "Answer with a % sign." and keep % vs percentage-points unambiguous.
  - For floats: specify rounding (e.g., "Round to three decimal places") if the question requires computation or if multiple precisions appear on the page.

8) QUALIFIER FIDELITY (critical)
- If multiple nearby answers exist, the question MUST include the qualifier that makes the target unique.
- Prefer qualifiers like: strongly / somewhat / overall / net, displayed / shown / visible vs listed / mentioned / described, first / second / last / nearest, and exact row / column / year / fiscal year / subgroup / legend item.
- The question must not be answerable by selecting a nearby but broader fact.
- Good: "Which application software interfaces are DISPLAYED in screenshots on this page?"
- Good: "What percentage of Rep/Lean Rep STRONGLY favor ...?"
- Bad: "Which applications are on this page?" if some are listed and some are displayed.
- Bad: "What percentage of Republicans favor ...?" if the page contains both net and strongly measures.

═══════════════════════════════════════════════════════════════════════════════
WHAT TO GENERATE (driven by failure cases)
═══════════════════════════════════════════════════════════════════════════════

Look at the page and identify its primary visual element, then use the matching section:
- Plain text / paragraphs                    → A) TEXT
- Any table (especially financial statements)→ B) TABLE
- Any chart (bar, line, pie, scatter, etc.)  → C) CHART
- Maps, infographics, diagrams, flowcharts   → D) IMAGE
- Multi-column or spatially structured pages → E) LAYOUT
Skip purely decorative pages with no readable content. Pages that look like appendices,
regulatory compliance, financial notes, or methodology are just as valuable — don't skip them.

A) TEXT (plain paragraphs)
Target: false refusals on short facts, poor exact extraction.
- Short, high-salience answers with an anchor. Examples:
  - "In the heading at the top of the page X, what is the full title? Answer exactly as written."
  - "In the section titled "Support", what phone number is listed? Answer with the number only."
  - "In the paragraph titled "About this survey", what sample size (N) is reported? Answer with an integer."
  - "In the sentence mentioning "temperature", which value is reported as best? Answer with the number only."
- Avoid long copying (>20 words) and cross-page answers.

B) TABLE (tabular data — use financial patterns for income statements, balance sheets, 10-K filings)
Target: false refusals on obvious cells, dense headers, counting mistakes, financial unit/sign errors.
- Small-scope table reasoning with an explicit table anchor. Examples:
  - Argmax/comparison: "In Table <N>, which row has the highest value in column "X"?"
  - Difference/sum: "In Table <N>, what is the difference between rows "A" and "B" in column "X"?"
  - Filter + count: "In Table <N>, how many rows have column "Y" > <threshold>? Answer with an integer."
  - Multi-level headers: "In Table <N>, under column group "<group>", subcolumn "<sub>", what is the value for row "<label>"?"
- FINANCIAL TABLES (critical — 57% zero-score on financial reports):
  - Line item lookup: "In the Consolidated Balance Sheets on page <N>, what is Total Current Assets for FY2021? Answer in millions as an integer."
  - YoY difference: "In the balance sheet, how much did Accrued Liabilities change from 2020 to 2021? Answer in millions (positive if increased)."
  - Notes / schedules / appendices (high false-refusal rate — do NOT skip these pages):
    "In Note <N> — '<Title>', what is the total <line item>? Answer in millions as an integer."
    "In the schedule of '<Title>' on page <N>, what is the value for '<row>' in '<year>'?"
    "In Note <N>, how many categories/items are listed in the '<sub-table>'? Answer with an integer."
- If values include %, specify "Answer with a % sign." If table shows units, state whether the answer should include them.

C) CHART (bar, line, pie, scatter, area, heatmap)
Target: cross-chart confusion, misread values, wrong column/axis selection.
- CHART DISAMBIGUATION (critical): Always anchor to the chart TITLE + a distinguishing
  axis/column label. If a chart has a pre-computed "Change" column, reference it explicitly.
  For pages with multiple small charts, anchor to the specific sub-chart label.
- Examples:
  - "In the chart titled "X", in the "Change '08-'15" column, which subgroup shows the largest change?"
  - "In the chart titled "X", what percentage of "EU" is in the "More" category? Answer with a % sign."
  - "In the chart titled "X", by how many percentage points did <category> change from <A> to <B>?"
  - "In the chart titled "X", which groups have a value below 60? Return a JSON array, e.g., ["A", "B"]."
  - "In the chart titled "X", how many categories exceed <threshold>? Answer with an integer."
- VISUAL GROUPING (recurring failure): If a chart uses brackets, braces, or labeled dividers
  to separate groups (e.g., "Business Analytics" vs "Business Intelligence"), ask about a
  specific group. The model confuses which items belong to which group.
- ARGMAX OVER CHANGE (recurring failure): If a chart has both absolute values (bar lengths)
  and a "Change" column, ask which category has the largest CHANGE. The model tends to pick
  the category with the largest absolute value instead.
  - "In the chart titled "X", which subgroup shows the largest increase from <year A> to <year B>?"
- Unit discipline: if the chart uses %, the answer should include "%".

D) IMAGE / FIGURE (maps, infographics, flowcharts, diagrams, schematics)
Target: spatial confusion on maps/infographics, misread labels, undercounting visual elements.
- Prefer diagrams/UI/infographics with clear labels; avoid counting small repeated natural-photo objects.
- SPATIAL REASONING (critical — 128 wrong-answer figure failures):
  For maps/infographics, force the model to bind numbers to their correct spatial region.
  The model confuses which number belongs to which region — ask specifically.
- For any count question, explicitly name the counting unit.
- Good counting units: service-line badges, screenshots displayed, QR codes, numbered steps, nodes, boxes, legend entries, labeled regions.
- Avoid ambiguous units like "lines", "figures", "objects", or "applications" unless the page makes the intended unit unmistakable.
- Examples:
  - "In the world map on page <N>, which region has the largest number? Answer with the region name."
  - "In the map, what number is shown for Europe? Answer with an integer."
  - "How many Muni service-line badges are shown at Union Square / Market Street? Answer with an integer."
  - "Which application software interfaces are displayed in screenshots on this page? Return a JSON array of strings on one line."
  - "In Figure <N>, how many distinct nodes/boxes are shown? Answer with an integer."
  - "In Figure <N>, what text is inside the box labeled "Y"?"
  - "In Figure <N>'s legend, which label corresponds to the <color> segment?"
  - "In the flowchart in Figure <N>, what step follows the decision labeled "X"?"
  - "In the diagram labeled "X", which component is directly connected to "Y"?"
- Color is allowed ONLY when it encodes meaning (legend/UI), not aesthetics.
- ICON / SYMBOL DISCRIMINATION (critical — model hallucinates icon presence):
  When pages show repeated entries (attraction listings, product cards, feature grids),
  ask about which entries have or lack a specific small icon/symbol.
  The model tends to assume all entries share the same icons — force it to check each one.
  - "Under the listing for '<Name>', which accessibility icons are shown? Return a JSON array of strings on one line."
  - "Which attractions on this page do NOT have a wheelchair accessibility icon? Return a JSON array of strings on one line."
  - "In the feature comparison grid, which products show a checkmark for '<Feature>'? Return a JSON array of strings on one line."
  - "How many listings on this page display a 'Green Travel' ecolabel? Answer with an integer."

E) LAYOUT (spatial + structural anchors)
- Force locating content by structure/position with explicit anchors:
  - Heading navigation: "In the section titled "X", what is the second bullet point?"
  - Above/below: "In the heading directly above Table <N>, what is the heading text?"
  - Two-column disambiguation (common benchmark failure): "In the right column under the heading "X", what is the first bullet point? Answer exactly as written."
  - Counting structure: "In the procedure section, how many numbered steps are listed? Answer with an integer."
  - Location: "In the top-right boxed callout, what label is shown?"
  - Exhaustive but short: "In the right column under the heading "X", list all subheadings shown. Return a JSON array of strings on one line, e.g., ["A", "B"]." (only if 2–8 items)

═══════════════════════════════════════════════════════════════════════════════
QUESTION-TYPE SPECIFIC RULES
═══════════════════════════════════════════════════════════════════════════════
{% if question_type == "layout" %}
- Include an explicit spatial/structural anchor (e.g., top-left, right column, below the chart, under heading X, second bullet, last row).
{% elif "numerical (percentage)" in question_type %}
- Ensure the answer is naturally a percent and add: "Answer with a % sign."
{% elif "numerical (int)" in question_type %}
- Ensure the answer is an integer and add: "Answer with an integer."
{% elif "numerical (float)" in question_type %}
- Ensure the answer is a decimal number visible on the page (or computable in one step).
{% elif "list" in question_type %}
- The question must require ALL items (complete list), short (2–8 items), visible on THIS page.
- Add: "Return a JSON array of strings on one line, e.g., ["gray", "red"]."
{% elif question_type == "yes or no" %}
- The question must be decidable from the page content and not rely on interpretation.
{% elif question_type == "multiple choice" %}
- Provide exactly 4 options (A–D), plausible and mutually exclusive.
- Do NOT use "All of the above" / "None of the above".
{% endif %}

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

Return ONLY the question text.
{% if question_type == "multiple choice" %}
Format: first line is the question, then A., B., C., D. choices on separate lines.
{% endif %}
Do not include explanations or reasoning.\
"""


PROMPT_ANSWER = """\
<question-type>
{{ question_type }}
</question-type>

<question>
{{ question }}
</question>

You are given EXACTLY ONE page image from a PDF document.
All evidence must come from this one page only.
Never imagine other pages. Never search the rest of the document.
Never refer to "Image 1", "Image 2", or other unseen pages.
Answer using ONLY information visible on this page.

In your THINKING (inside <think> tags), follow this protocol. Do NOT echo these steps in
your final answer — the answer must be ONLY the bare result (number, phrase, list, etc.).

QUALIFIER LOCK (critical)
Before extracting any answer, copy the restrictive qualifiers from the question and keep them fixed:
- page / section / table / figure / chart identity
- year / date / fiscal year
- subgroup / series / legend item
- strongly / somewhat / overall / net / change
- displayed / shown / visible vs listed / mentioned / described
- first / second / last / nearest / left / right / top / bottom
- exact number vs ratio vs percentage vs percentage points

Do NOT substitute a nearby year, nearby subgroup, nearby chart, nearby row, or nearby fact.
If the question asks what is DISPLAYED, ignore items that are only named in nearby text.
If the question asks for STRONGLY / NET / OVERALL / CHANGE, read exactly that quantity and no other.

THINKING PROTOCOL:
1) PARSE: Decompose the question into concrete lookup targets.
   E.g., "What is Gross Profit as a percentage of Revenues?" ->
   target A = "Gross Profit value", target B = "Revenues value".
2) LOCATE: Scan the page to find the element the question references. Use the question's
   anchor (page number, table/figure/section title, heading) to match.
3) MATCH: Identify the exact visual element by its TITLE or CAPTION before reading values.
   - Charts: match title + axis labels (a page may have multiple charts). Identify sub-chart, series, x-axis category, unit, etc.
   - Tables: match caption/heading (e.g., "Consolidated Balance Sheets", "Note 5"). Identify row/column labels, unit, scale, etc.
   - Figures/Maps/infographics/UI: match the target object and counting unit.
   - CONFIRM ANCHOR: If the page has multiple charts/tables/figures, state which one you
     matched and confirm its title matches the question's anchor before reading any values.
     If the title does not match, scan the page for the correct element.
4) READ: Extract the specific value from the matched element.
   - Tables: correct column for the fiscal year; parentheses = negative; check unit scale header.
   - Charts with a "Change" column: read that column directly, don't recompute from bars.
   - If multiple nearby values exist, do not switch to a broader or more convenient one.
5) VERIFY: Double-check your extraction against the same bound target.
   After this verification pass, stop thinking.

THINKING STABILITY (critical)
- Follow the protocol once from top to bottom. Do NOT restart from step 1 after you already found the relevant element.
- Do at most one locate pass and one verification pass.
- Be concise: go directly to the element matching the question's anchor. Do NOT describe
  irrelevant elements on the page. Only mention what contributes to the answer.
- If there are two plausible candidates, compare them once using the question's qualifiers, choose the best-supported one, and continue. Do NOT keep generating new alternatives.
- Do NOT repeat the same scan, recount, or conclusion more than once.
- As soon as the answer is found and verified, stop thinking and produce the final answer.
- Do NOT use filler loops such as repeating a phrase, title, entity name, or page reference many times.

REASONING TRACE REQUIREMENTS (critical for training)
Your reasoning will be used as chain-of-thought training data. In your thinking, you MUST:
- Cite the question's anchor explicitly: "In Table 3 on page 42, ..." not "In the table, ..."
- State which element you matched and its exact title/caption.
- Quote the specific value(s) you read: "the row 'Accrued Liabilities' shows 6,063 for 2021."
- If computing, show the computation with named references:
  "Gross Profit (19,962) / Revenues (44,538) = 0.448 = 44.8%"
Bad: "Looking at the page, I see a value of 6,063."
Good: "In the Consolidated Balance Sheets on page 59, the row 'Total Accrued Liabilities'
       under the 2021 column shows $6,063 (in millions)."

COUNTING DISCIPLINE
- For any count question, state the counting unit first in your reasoning.
- Count once in a consistent order: left-to-right, top-to-bottom.
- Recount at most once.
- Do NOT switch counting units mid-reasoning.
- If both labels and visual markers are present, decide which is being counted before counting.

UNIT DISCIPLINE
- Preserve units exactly when present or requested (%, $, million, km, etc.).
- If the question asks for a percentage or percentage points, include the "%".
- Financial reports:
  - Check the table header for unit scale (e.g., "In millions", "Rupees in lacs") and apply it.
  - Parentheses in financial tables mean NEGATIVE values: (380) = -380.
  - When the question asks "how much higher/more", answer with a POSITIVE number representing
    the magnitude of the difference. When it asks "change", use positive for increase,
    negative for decrease.
  - If the question says "Answer in millions", output the number as shown in the table
    (the table is already in millions).

REFUSAL POLICY (critical)
{% if "not answerable" in question_type %}
Only say "Not answerable" if the page lacks the exact requested qualifier.
In your thinking, briefly name what is missing.
Bad: "The information is not available on this page."
Good: "The chart shows data for 2015 and 2020, but the question asks for 2018. Year 2018 is not present."
Good: "The table lists rows for Revenue, COGS, and Gross Profit, but not 'Operating Lease Liability'. Row missing."
{% else %}
- NEVER output "Not answerable", "Cannot determine", or any refusal.
- If the answer is directly printed on the page, copy it and stop.
- If the answer is computable from visible values, compute it and stop.
- Do NOT refuse only because the page number is not visible, a nearby title/anchor differs slightly, or the answer is implied by one local relation rather than explicitly restated.
{% endif %}

FINAL ANSWER:
- Put ALL reasoning inside <think>...</think>.
- After </think>, output ONLY the final answer.
- Do NOT repeat reasoning outside <think> tags.
- Do NOT output protocol labels, explanations, or extra text after </think>.

OUTPUT FORMAT (critical)
{% if question_type == "multiple choice" %}
- Output exactly ONE line: "<LETTER>. <option text>", e.g., "B. 92%"
- Do NOT output only a digit ("2") or only a letter ("B").
{% elif question_type == "yes or no" %}
- Output exactly "Yes" or "No" (no punctuation, no explanation).
{% elif "numerical (percentage)" in question_type %}
- Output a number WITH a percent sign, e.g., "29%". Do NOT omit the "%".
{% elif "numerical (int)" in question_type %}
- Output an integer only (digits, optional commas). No words.
{% elif "numerical (float)" in question_type %}
- Output a decimal number only (no extra words), unless the question explicitly requests a unit.
{% elif question_type is string and question_type.startswith("string") %}
- Output a short phrase/sentence only. No preamble.
{% elif question_type == "layout" %}
- Use the spatial/structural anchor to select the correct location.
- Output only the extracted content (string/number/list), no preamble.
{% elif "list" in question_type %}
- Output a JSON array on ONE line: ["gray", "red"]. Must be complete.
- NEVER use comma-separated plain text. ALWAYS use JSON array syntax with ["..."].
- Each element must be individual, not a compound range.
  Wrong: ["1981-82"]  Correct: ["1981", "1982"]
  Wrong: ["Training and Sportswear"]  Correct: ["Training", "Sportswear"]
{% elif "not answerable" in question_type %}
- Output exactly: Not answerable
{% else %}
- Output a short, direct answer. No preamble or explanation.
{% endif %}\
"""


PROMPT_QUALITY_SCORE = """\
<question-type>
{{ question_type }}
</question-type>

<question>
{{ question }}
</question>

<answer>
{{ answer }}
</answer>

<answer_reasoning>
{{ answer__reasoning_content }}
</answer_reasoning>

You are given EXACTLY ONE page image extracted from a PDF document. Evaluate the QUALITY of this (question, answer, reasoning content).

Be STRICT. Any check failure => score 0.

CHECKS

1) DOCUMENT/PAGE QUALITY
- Pages must be readable (not too blurry/pixelated/cut off). Page must be high quality, and not be empty or nearly empty.
- The required evidence must be visible without guessing.

2) RELEVANCE + ANSWERABILITY
- The question must be grounded in visible content.
- {% if "not answerable" in question_type %}For "not answerable" questions: the question should be relevant, but the requested information must NOT be present anywhere on the page.
- Score 0 if the question is unanswerable ONLY because it refers to another page, the whole document, or a page range not shown. The question must be a near-miss negative where a specific qualifier (year, subgroup, row, region, etc.) is absent from the visible page.{% endif %}

3) ANSWER CORRECTNESS
- The answer must be correct given the visible page content.

4) OUTPUT FORMAT COMPLIANCE (critical)
- Score 0 if the answer contains <think> tags, reasoning steps, protocol labels, or anything beyond the bare final result.
{% if question_type == "multiple choice" %}
- Answer must be exactly "<LETTER>. <option text>" (A–D). Reject digit-only ("2") or letter-only ("B").
{% elif question_type == "yes or no" %}
- Answer must be exactly "Yes" or "No".
{% elif "numerical (percentage)" in question_type %}
- Answer MUST include a percent sign (e.g., "29%").
{% elif "numerical (int)" in question_type %}
- Answer must be an integer only (digits, optional commas), no extra words.
{% elif "numerical (float)" in question_type %}
- Answer must be a decimal number only, no extra words.
{% elif question_type is string and question_type.startswith("string") %}
- Answer must be a short phrase/sentence only (no preamble).
{% elif "list" in question_type %}
- Answer must be a ONE-LINE JSON array (e.g., ["gray", "red"]).
- Score 0 if comma-separated text or compound ranges (["1981-82"] should be ["1981", "1982"]).
{% elif "not answerable" in question_type %}
- Answer must be exactly "Not answerable".
{% else %}
- Answer MUST NOT be "Not answerable" or any refusal.
{% endif %}

5) QUESTION QUALITY
- Unambiguous: exactly one correct answer.
- Verifiable: a judge can confirm correctness from the pages.
- Locally bounded: 1-2 steps preferred; 3 steps allowed only if all evidence is on this page and each step is trivial.
- Not overly tedious: avoid long enumerations (e.g., 20+ items) unless clearly short and visible.
- If the question leaks the answer (contains the answer), score 0.

6) ANCHOR QUALITY (critical — questions are collated across all pages)
- Score 0 if the question uses a generic anchor that could match multiple elements across the
  document (e.g., "In the table, ..." or "What is the total revenue?" without specifying which).
- Score 0 if the question uses any of these vague phrases:
  "on this page", "in the image", "in the bottom/top half of the page",
  "the table" / "the chart" / "the figure" without a title or number,
  "on the left/right side" without a heading.
- Must have at least one of: exact page number, numbered element (Table 3, Figure 7, Note 5),
  or unique section/subsection/chart/table title.
- If there is a chance the anchor is not unique across the full document, score 0.

7) REASONING QUALITY (critical — reasoning is used as chain-of-thought training data)
- The reasoning in <answer_reasoning> must be specific, stable, and finite. Score 0 if any of the following hold:
  - It does NOT cite the question's anchor (page number, table/figure number, section title,
    chart title, or named financial statement).
  - It uses only generic references like "the table", "the chart", "the page", or "the image".
  - It does NOT quote or reference the specific value(s) extracted.
  - If the question involves computation, it does NOT show the operation with named sources
    (e.g., "Gross Profit (19,962) / Revenues (44,538) = 44.8%").
  - It repeats the same scan, recount, or conclusion without adding new evidence, or restarts
    / generates new alternatives after already having enough evidence to answer.
  - It ends in an unfinished or truncated way, or appears to stop mid-thought.
  - It contains filler repetition (for example, the same entity name or phrase copied many times).
  - It references images by ordinal position ("Image 1", "Image 2", "the first image")
    rather than by printed page numbers or element titles.
  - For count questions, it switches counting units mid-reasoning.
  - For displayed-vs-mentioned questions, it mixes displayed visual instances with nearby
    listed/mentioned items.

8) FINANCIAL CHECKS (if applicable)
- Parentheses = negative. "How much higher" = positive. Units must match table header scale.
  Fiscal year must match the correct column.

SCORING
- Score 0: Any check fails.
- Score 1: All checks pass.
- Score 2: All checks pass AND involves at least one high-value signal: financial computation,
  chart disambiguation (title + axis label), infographic spatial reasoning, or cross-cell
  table operation. Must still be unambiguous.

Respond with ONLY the score as a single digit: 0, 1, or 2.\
"""


# =============================================================================
# Pipeline configuration
# =============================================================================


def build_config(
    seed_path: str = "seed.parquet",
    model_alias: str = "vl",
    model_id: str = DEFAULT_VLM_MODEL,
) -> dd.DataDesignerConfigBuilder:
    model_configs = [
        dd.ModelConfig(
            alias=model_alias,
            model=model_id,
            provider=VLLM_PROVIDER_NAME,
            inference_parameters=dd.ChatCompletionInferenceParams(
                timeout=1200,
                temperature=1.0,
                top_p=0.95,
                max_parallel_requests=32,
                extra_body={
                    "top_k": 20,
                    "min_p": 0.0,
                    "presence_penalty": 1.5,
                    "repetition_penalty": 1.0,
                },
            ),
        ),
    ]

    config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

    config_builder.with_seed_dataset(
        dd.LocalFileSeedSource(path=seed_path),
        sampling_strategy=dd.SamplingStrategy.SHUFFLE,
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="question_type",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=[
                    "multiple choice",
                    "yes or no",
                    "string: word, phrase or short sentence",
                    "layout",
                    "numerical (int)",
                    "numerical (float)",
                    "numerical (percentage)",
                    "list of items (int, string, float or mixed)",
                    "not answerable",
                ],
                weights=[0.025, 0.025, 1, 2, 2, 2, 2, 2, 0.2],
            ),
        )
    )

    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="question",
            model_alias=model_alias,
            prompt=PROMPT_QUESTION,
            multi_modal_context=IMAGE_CONTEXT,
        )
    )

    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="answer",
            model_alias=model_alias,
            prompt=PROMPT_ANSWER,
            multi_modal_context=IMAGE_CONTEXT,
            extract_reasoning_content=True,
        )
    )

    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="quality_score",
            model_alias=model_alias,
            prompt=PROMPT_QUALITY_SCORE,
            multi_modal_context=IMAGE_CONTEXT,
        )
    )

    return config_builder


def create_dataset(
    config_builder: dd.DataDesignerConfigBuilder,
    num_records: int,
    vllm_endpoint: str,
    artifact_path: Path | str | None = None,
) -> DatasetCreationResults:
    model_providers = [
        dd.ModelProvider(
            name=VLLM_PROVIDER_NAME,
            endpoint=vllm_endpoint,
        ),
    ]
    data_designer = DataDesigner(
        artifact_path=artifact_path,
        model_providers=model_providers,
    )
    data_designer.set_run_config(dd.RunConfig(display_tui=True, disable_early_shutdown=True))
    results = data_designer.create(config_builder, num_records=num_records, dataset_name="single_page_qa")
    return results


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--vllm-endpoint",
        type=str,
        required=True,
        help="Base URL of the vLLM server hosting the VLM (e.g. http://localhost:8000/v1)",
    )
    parser.add_argument("--seed-path", type=str, required=True, help="Path to the seed parquet file")
    parser.add_argument("--model-alias", type=str, default="vl")
    parser.add_argument("--model-id", type=str, default=DEFAULT_VLM_MODEL)
    parser.add_argument("--num-records", type=int, default=5)
    parser.add_argument("--artifact-path", type=str, default=None)
    args = parser.parse_args()

    config_builder = build_config(
        seed_path=args.seed_path,
        model_alias=args.model_alias,
        model_id=args.model_id,
    )
    results = create_dataset(
        config_builder,
        num_records=args.num_records,
        vllm_endpoint=args.vllm_endpoint,
        artifact_path=args.artifact_path,
    )

    print(f"Dataset saved to: {results.artifact_storage.final_dataset_path}")

    results.load_analysis().to_report()
