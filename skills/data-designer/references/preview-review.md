# Preview Review Guide

## Mindset

Quality is statistical, not per-record. Fix systemic issues that affect many records; don't chase cosmetic flaws in individual ones. But don't stop early — clear patterns of broken data or ignored instructions are worth fixing.

## Reading Sample Records

Load `dataset.parquet` from the preview results directory (printed as `Results path:` by the preview command, or the most recent `artifacts/preview_results_*/` directory). Use pandas to load the parquet file and print the records in a compact, reviewable format.

## What to Look For

The specifics depend on the dataset and its intended use. The categories below are common starting points — adapt based on what matters for this dataset.

### Diversity
- **Mode collapse**: are records clustering around the same patterns, topics, or phrasings?
- **Sampler effectiveness**: are samplers being used effectively to steer diversity in the dataset?
- **Structural monotony**: do LLM-generated columns follow the same template across records?

### Data Quality
- **Instruction compliance**: does generated content follow prompt constraints (step counts, format requirements, allowed values)?
- **Internal consistency**: does data within a record agree with itself?
- **Encoding integrity**: no garbled encoding, mojibake, or broken unicode.
- **Plausibility**: do examples look like they could come from the real domain, or are they obviously synthetic?

### Design Choices
- **Column types**: if a text column consistently produces structured data or code, use the appropriate specialized column type. If values come from a fixed set or known distribution, use a sampler instead of an LLM column.
- **Validation**: if output could be checked programmatically (syntax, schema conformance, value ranges), attach a validator.
- **Judge calibration** (if applicable): are scores consistent across similar-quality records? Does the judge catch visible problems? Consider the user's intent — uniformly high scores may be correct if the judge is a quality filter; a spread matters more if it's a training signal.
