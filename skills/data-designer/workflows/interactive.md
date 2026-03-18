# Interactive Workflow

This is an interactive, iterative design process. Do not disengage from the loop unless the user says they are satisfied.

1. **Learn** — Run `data-designer agent context`.
  - If no model aliases are configured, stop and ask the user.
  - Inspect schemas for every column, sampler type, validator, and processor you plan to use.
  - Never guess types or parameters — read the relevant config files first.
  - Always read `base.py` for inherited fields shared by all config objects.
2. **Clarify** — Ask the user clarifying questions to narrow down precisely what they want.
  - Use a question-asking UX tool if available.
  - Optimize for a great user experience: batch related questions together, keep the set short, provide concrete options/examples/defaults where possible, and use structured inputs (single-select, multi-select, forms, etc.) when they make answering easier.
  - Common things to make precise:
    - What the "axes of diversity" are — what should be well represented and diverse in the resulting dataset.
    - The kind and nature of any input data.
    - What variables should be randomized.
    - The schema of the final dataset.
    - The structure of any required structured output columns.
    - What facets of the output dataset are important to capture.
3. **Plan** — Determine columns, samplers, processors, validators, and other dataset features needed.
4. **Build** — Write the Python script with `load_config_builder()` (see Output Template in SKILL.md).
5. **Validate** — Run `data-designer validate <path>`. Address any warnings or errors and re-validate until it passes.
6. **Preview** — Run `data-designer preview <path> --save-results` to generate sample records as HTML files.
  - Note the sample records directory printed by the `data-designer preview` command
  - Run `python -m http.server 8741 --directory <sample-records-dir>` (in background). If port 8741 is taken, retry with port 0 and note the port it prints
  - **Verify the server started** by reading the background task output until you see `Serving HTTP on ...`. Get the port from that output line. Do not use curl, lsof, or any other probing method
  - Tell the user to open `http://localhost:<port>/sample_records_browser.html` to review them
  - When the user is done reviewing, stop the background server
7. **Iterate** — Ask the user for feedback. Edit the script, re-validate, re-preview, and serve again. Repeat until they are satisfied.
8. **Finalize** — Once the user is happy, tell them they can run the following command to create the dataset:
  - `data-designer create <path> --num-records <N>`.
  - Warn the user that generation can take a long time for large record counts (50+).
  - Do not run this command yourself. It requires model endpoints and can take a long time.
