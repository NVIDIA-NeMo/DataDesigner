# Interactive Workflow

This is an interactive, iterative design process. Do not disengage from the loop unless the user says they are satisfied.

1. **Learn** — Run `data-designer agent context`.
  - If no model aliases are configured, stop and tell the user to run `data-designer config` to set them up before proceeding.
  - Inspect schemas for every column, sampler type, validator, and processor you plan to use.
  - Never guess types or parameters — read the relevant config files first.
  - Always read `base.py` for inherited fields shared by all config objects.
2. **Clarify** — Ask the user clarifying questions to narrow down precisely what they want.
  - Optimize for a great user experience: prefer a structured question tool over plain text if one is available, batch related questions together, keep the set short, provide concrete options/examples/defaults where possible, and use structured inputs (single-select, multi-select, free text, etc.) when they make answering easier.
  - If multiple model aliases are available, ask which one(s) to use (or default to an alias with the appropriate `generation_type` for each column).
  - Common things to make precise:
    - What the "axes of diversity" are — what should be well represented and diverse in the resulting dataset.
    - The kind and nature of any input data.
    - What variables should be randomized.
    - The schema of the final dataset.
    - The structure of any required structured output columns.
    - What facets of the output dataset are important to capture.
3. **Plan** — Determine columns, samplers, processors, validators, and other dataset features needed. Present the plan to the user and ask if they want any changes before generating a preview.
4. **Build** — Write the Python script with `load_config_builder()` (see Output Template in SKILL.md).
5. **Validate** — Run `data-designer validate <path>`. Address any warnings or errors and re-validate until it passes.
6. **Preview** — Run `data-designer preview <path> --save-results` to generate sample records as HTML files.
  - Note the sample records directory printed by the `data-designer preview` command
  - Give the user a clickable link: `file://<sample-records-dir>/sample_records_browser.html`
7. **Iterate**
   - Ask the user for feedback.
   - Offer to review the records yourself and suggest improvements. If the user accepts, read `references/preview-review.md` for guidance.
   - Apply changes, re-validate, and re-preview. Repeat until the user is satisfied.
8. **Finalize** — Once the user is happy, tell them they can run the following command to create the dataset:
  - `data-designer create <path> --num-records <N> --dataset-name <name>`.
  - Caution the user that generation speed depends heavily on the dataset configuration and their inference setup.
  - Do not run this command yourself — the user should control when it runs.
