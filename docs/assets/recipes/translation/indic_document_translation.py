# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer>=0.5.7",
#     "data-designer-nemotron-indic-translation",
# ]
# ///
"""Indic Document Translation Recipe

Translate English documents into any of the 22 scheduled Indic languages with the
``data-designer-nemotron-indic-translation`` plugin. Every row is translated, scored by an LLM judge
against the source (faithfulness, fluency, terminology, style, format preservation), and, when
rewrites are enabled, self-corrected until it clears the score threshold.

The seed dataset is a parquet or JSONL file with one document per row in a text column.

Optional features, all driven by flags on the same ``document-translation`` column:

    - Rewrites: ``--max-rewrites`` / ``--rewrite-threshold`` control the judge-and-rewrite loop.
    - Windowing: ``--enable-windowing`` translates long documents in overlapping token windows.
    - Romanization: ``--intermediate-language hi-Deva-IN --target-language hi-Latn-IN`` translates to
      native script first, then to Latin script.
    - Native numerals: ``--native-numeral-probability`` draws native-script digits for a share of rows.
    - Script remapping: ``--target-language mni-Beng-IN --target-language-script Mtei`` for Manipuri in Meetei Mayek.
    - Specialist translators: ``--translation-mode bodhan`` for ``bodhan-ai/indic-translate``.

Prerequisites:
    - A ``gemma`` model alias registered in your Data Designer model configs (``data-designer config``),
      pointing at ``google/gemma-4-31b-it``. It is the default translator and judge. The specialist routes
      add a ``bodhan`` alias for ``bodhan-ai/indic-translate``. Pass other aliases with ``--model-alias``
      and ``--judge-model-alias``.
    - A seed parquet/JSONL file with a text column (``text`` by default).

Run:
    # Translate to Hindi with gemma as both translator and judge
    uv run indic_document_translation.py --seed-path docs.parquet

    # Bengali target
    uv run indic_document_translation.py --seed-path docs.jsonl --target-language bn-Beng-IN

    # Specialist translator with a gemma judge, Bodo target
    uv run indic_document_translation.py --seed-path docs.jsonl --target-language brx-Deva-IN \\
        --translation-mode bodhan --model-alias bodhan --judge-model-alias gemma

    # Also write the translated dataset to a parquet or JSONL file
    uv run indic_document_translation.py --seed-path docs.parquet --output-path translated.jsonl

    # Romanized Hindi (native script first, then Latin script)
    uv run indic_document_translation.py --seed-path docs.parquet \\
        --intermediate-language hi-Deva-IN --target-language hi-Latn-IN

    # For help message and available options
    uv run indic_document_translation.py --help
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from data_designer_nemotron_indic_translation.config import DocumentTranslationConfig

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults

DEFAULT_MODEL_ALIAS = "gemma"
DEFAULT_MAX_REWRITES = 2
OUTPUT_COLUMN = "translation"
OUTPUT_SUFFIXES = (".parquet", ".jsonl")


def resolve_max_rewrites(max_rewrites: int | None, translation_mode: str, enable_windowing: bool) -> int:
    """Default to the judge-and-rewrite loop; bodhan mode and windowing run a single translation pass."""
    if max_rewrites is not None:
        return max_rewrites
    if translation_mode == "bodhan" or enable_windowing:
        return 0
    return DEFAULT_MAX_REWRITES


def build_config(
    seed_path: str,
    text_column: str = "text",
    source_language: str = "en-Latn-IN",
    target_language: str = "hi-Deva-IN",
    model_alias: str = DEFAULT_MODEL_ALIAS,
    judge_model_alias: str | None = None,
    translation_mode: str = "generic",
    max_rewrites: int | None = None,
    rewrite_threshold: int = 75,
    enable_windowing: bool = False,
    window_tokens: int = 2048,
    intermediate_language: str | None = None,
    target_language_script: str | None = None,
    native_numeral_probability: float = 0.0,
) -> dd.DataDesignerConfigBuilder:
    """Build a seed -> translate -> judge (-> rewrite) pipeline on one ``document-translation`` column."""
    judge_model_alias = judge_model_alias or model_alias
    max_rewrites = resolve_max_rewrites(max_rewrites, translation_mode, enable_windowing)

    config_builder = dd.DataDesignerConfigBuilder()
    registered_aliases = {model_config.alias for model_config in config_builder.model_configs}
    missing_aliases = sorted({model_alias, judge_model_alias} - registered_aliases)
    if missing_aliases:
        raise ValueError(
            f"Register these model alias(es) in your Data Designer model configs: {', '.join(missing_aliases)}. "
            f"Registered aliases: {', '.join(sorted(registered_aliases))}. "
            "Add them with `data-designer config` or by editing ~/.data-designer/model_configs.yaml."
        )
    config_builder.with_seed_dataset(dd.LocalFileSeedSource(path=seed_path))

    translation_kwargs: dict[str, object] = {}
    if intermediate_language:
        translation_kwargs["intermediate_language"] = intermediate_language
        translation_kwargs["max_rewrites_intermediate"] = max_rewrites
        translation_kwargs["rewrite_threshold_intermediate"] = rewrite_threshold
    if target_language_script:
        translation_kwargs["target_language_script"] = target_language_script
    if enable_windowing:
        translation_kwargs["enable_windowing"] = True
        translation_kwargs["window_tokens"] = window_tokens
        # Segment extraction needs a general-purpose model, which is the judge when the
        # translator is a specialist.
        translation_kwargs["window_merge_model_alias"] = judge_model_alias

    config_builder.add_column(
        DocumentTranslationConfig(
            name=OUTPUT_COLUMN,
            text_column=text_column,
            source_language=source_language,
            target_language=target_language,
            translation_mode=translation_mode,
            model_alias=model_alias,
            evaluator_model_alias=judge_model_alias,
            max_rewrites=max_rewrites,
            rewrite_threshold=rewrite_threshold,
            native_numeral_probability=native_numeral_probability,
            **translation_kwargs,
        )
    )

    return config_builder


def create_dataset(
    config_builder: dd.DataDesignerConfigBuilder,
    num_records: int,
    artifact_path: Path | str | None = None,
) -> DatasetCreationResults:
    data_designer = DataDesigner(artifact_path=artifact_path)
    return data_designer.create(config_builder, num_records=num_records, dataset_name="indic_document_translation")


def save_output(results: DatasetCreationResults, output_path: Path) -> None:
    """Write the full dataset (seed columns, translation, judge output) to a parquet or JSONL file."""
    dataset = results.load_dataset()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        dataset.to_parquet(output_path, index=False)
    else:
        # force_ascii=False writes Indic scripts as readable characters.
        dataset.to_json(output_path, orient="records", lines=True, force_ascii=False)


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Translate documents into Indic languages with LLM-judged quality scoring")
    parser.add_argument("--seed-path", type=str, required=True, help="Path to the seed parquet or JSONL file")
    parser.add_argument("--text-column", type=str, default="text", help="Seed column holding the source documents")
    parser.add_argument("--source-language", type=str, default="en-Latn-IN", help="BCP-47 code, e.g. en-Latn-IN")
    parser.add_argument("--target-language", type=str, default="hi-Deva-IN", help="BCP-47 code, e.g. bn-Beng-IN")
    parser.add_argument(
        "--model-alias",
        type=str,
        default=DEFAULT_MODEL_ALIAS,
        help="Translator model alias, as registered in your Data Designer model configs",
    )
    parser.add_argument(
        "--judge-model-alias", type=str, default=None, help="Judge model alias (default: same as --model-alias)"
    )
    parser.add_argument(
        "--translation-mode",
        choices=["generic", "bodhan"],
        default="generic",
        help="'generic' for instruction-tuned LLMs, 'bodhan' for bodhan-ai/indic-translate",
    )
    parser.add_argument(
        "--max-rewrites",
        type=int,
        default=None,
        help=f"Self-correction attempts per row (default: {DEFAULT_MAX_REWRITES}, or 0 for bodhan or windowing)",
    )
    parser.add_argument(
        "--rewrite-threshold", type=int, default=75, help="Judge score (0-100) below which a rewrite is triggered"
    )
    parser.add_argument(
        "--enable-windowing", action="store_true", help="Translate long documents in overlapping token windows"
    )
    parser.add_argument("--window-tokens", type=int, default=2048, help="Max tokens per window with --enable-windowing")
    parser.add_argument(
        "--intermediate-language",
        type=str,
        default=None,
        help="Pivot language for a two-stage translation, e.g. hi-Deva-IN before a hi-Latn-IN target",
    )
    parser.add_argument(
        "--target-language-script",
        type=str,
        default=None,
        help="ISO 15924 script to remap the final translation into, e.g. Mtei with --target-language mni-Beng-IN",
    )
    parser.add_argument(
        "--native-numeral-probability",
        type=float,
        default=0.0,
        help="Per-row chance (0 to 1) of writing numbers in native-script digits (with --translation-mode generic)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Also write the final dataset to this .parquet or .jsonl file",
    )
    parser.add_argument("--num-records", type=int, default=5)
    parser.add_argument("--artifact-path", type=str, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.output_path and Path(args.output_path).suffix not in OUTPUT_SUFFIXES:
        parser.error(f"--output-path must end in one of {', '.join(OUTPUT_SUFFIXES)}, got {args.output_path!r}")

    config_builder = build_config(
        seed_path=args.seed_path,
        text_column=args.text_column,
        source_language=args.source_language,
        target_language=args.target_language,
        model_alias=args.model_alias,
        judge_model_alias=args.judge_model_alias,
        translation_mode=args.translation_mode,
        max_rewrites=args.max_rewrites,
        rewrite_threshold=args.rewrite_threshold,
        enable_windowing=args.enable_windowing,
        window_tokens=args.window_tokens,
        intermediate_language=args.intermediate_language,
        target_language_script=args.target_language_script,
        native_numeral_probability=args.native_numeral_probability,
    )
    results = create_dataset(config_builder, num_records=args.num_records, artifact_path=args.artifact_path)

    print(f"Dataset saved to: {results.artifact_storage.final_dataset_path}")
    if args.output_path:
        save_output(results, Path(args.output_path))
        print(f"Output written to: {args.output_path}")

    scores = results.load_dataset()[f"{OUTPUT_COLUMN}_overall_score"]
    below_threshold = int((scores < args.rewrite_threshold).sum())
    print(f"Judge score: mean {scores.mean():.1f}, min {scores.min():.0f}, max {scores.max():.0f}")
    print(f"Rows below the rewrite threshold ({args.rewrite_threshold}): {below_threshold} of {len(scores)}")

    results.load_analysis().to_report()


if __name__ == "__main__":
    main()
