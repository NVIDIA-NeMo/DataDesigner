# Autopilot Workflow

In this mode, make reasonable design decisions autonomously based on the dataset description. Do not ask clarifying questions — infer sensible defaults and move straight through to a working preview.

1. **Learn** — Run `data-designer agent context`.
  - If no model aliases are configured, stop and tell the user to run `data-designer config` to set them up before proceeding.
  - Inspect schemas for every column, sampler type, validator, and processor you plan to use.
  - Never guess types or parameters — read the relevant config files first.
  - Always read `base.py` for inherited fields shared by all config objects.
2. **Infer** — Based on the dataset description, make reasonable decisions for:
  - Axes of diversity and what should be well represented.
  - Which variables to randomize.
  - The schema of the final dataset.
  - The structure of any structured output columns.
  - Briefly state the key decisions you made so the user can course-correct if needed.
3. **Plan** — Determine columns, samplers, processors, validators, and other dataset features needed.
4. **Build** — Write the Python script with `load_config_builder()` (see Output Template in SKILL.md).
5. **Validate** — Run `data-designer validate <path>`. Address any warnings or errors and re-validate until it passes.
6. **Preview** — Run `data-designer preview <path> --save-results` to generate sample records as HTML files.
  - Note the sample records directory printed by the `data-designer preview` command
  - Give the user a clickable link: `file://<sample-records-dir>/sample_records_browser.html`
7. **Create** — If the user specified a record count:
  - Run `data-designer create <path> --num-records <N> --dataset-name <name>`.
  - Generation speed depends heavily on the dataset configuration and the user's inference setup. For larger datasets, warn the user and ask for confirmation before running.
  - If no record count was specified, skip this step.
8. **Present** — Summarize what was built: columns, samplers used, key design choices. If the create command was run, share the results. Ask the user if they want any changes. If so, edit the script, re-validate, re-preview, and iterate.
