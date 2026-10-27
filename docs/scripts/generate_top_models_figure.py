# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib==3.9.4",
# ]
# ///
"""Regenerate the "Top Model Usage" telemetry figure.

Renders the ranked input-vs-output token breakdown shown in the README's
"Top models" section, styled to match the Data Designer devnote charts
(near-black canvas, NVIDIA-green duotone). The PNG is written to the Fern image
path rendered by the README and docs site:

    fern/images/top-models.png

The source telemetry export lives at docs/scripts/top-model-usage.csv with
columns: model name, input (context) tokens, output (generated) tokens, plus a
trailing "Other" aggregate row. Model aliases are normalized in the committed
CSV. Only the ten models with the most total tokens are shown; the rest are
combined with Other. The axis break spans 2.5T to 15T for this snapshot.

Run:
    # Regenerate from the committed CSV (zero args)
    uv run docs/scripts/generate_top_models_figure.py

    # Refresh from a new telemetry export
    uv run docs/scripts/generate_top_models_figure.py --csv ~/Downloads/new-export.csv

    # Options
    uv run docs/scripts/generate_top_models_figure.py --help
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter

# Repo root is two levels up from docs/scripts/.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "docs" / "scripts" / "top-model-usage.csv"
# Tracked figure path rendered by the README and Fern docs site.
TARGETS = (REPO_ROOT / "fern" / "images" / "top-models.png",)

# ---------------------------------------------------------------- palette ----
BG = "#0E0E0E"  # near-black canvas (matches DD devnote charts)
GREEN = "#76B900"  # NVIDIA green -> input (context) tokens
LIME = "#C5E86C"  # light NVIDIA-tint green -> output (generated) tokens
WHITE = "#FFFFFF"
SUBTLE = "#9A9A9A"
MODELNAME = "#ECECEC"
SPINE = "#4A4A4A"
INK = "#0E0E0E"  # dark ink for labels sitting on bright bars

B = 1e9  # render token counts in billions


def load_rows(csv_path: Path) -> list[tuple[str, float, float]]:
    """Parse the telemetry CSV into (name, input_tokens, output_tokens) rows."""
    rows: list[tuple[str, float, float]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for name, inp, out in reader:
            rows.append((name, float(inp.replace(",", "")), float(out.replace(",", ""))))
    return rows


def configure_matplotlib() -> None:
    """Pin rendering to deterministic settings so the asset is reproducible.

    Forces the Agg backend and matplotlib's bundled DejaVu Sans face rather than
    opportunistically selecting a system Helvetica/Arial. Combined with the
    pinned matplotlib version in the script metadata, this keeps the checked-in
    PNG byte-reproducible across machines and CI.
    """
    plt.switch_backend("Agg")
    rcParams["font.family"] = "DejaVu Sans"
    rcParams["font.size"] = 12


def fmt(v: float) -> str:
    """Compact billions/trillions label."""
    if v >= 1e12:
        return f"{v / 1e12:.2f}T"
    return f"{v / 1e9:.0f}B"


def rank_rows(rows: list[tuple[str, float, float]], limit: int = 10) -> list[tuple[str, float, float]]:
    """Rank models by total tokens and fold the remainder into the Other aggregate."""
    models = sorted((row for row in rows if row[0].lower() != "other"), key=lambda row: row[1] + row[2], reverse=True)
    remaining = [row for row in rows if row[0].lower() == "other"] + models[limit:]
    ranked = models[:limit]
    if remaining:
        ranked.append(("Other", sum(row[1] for row in remaining), sum(row[2] for row in remaining)))
    return ranked


def render(rows: list[tuple[str, float, float]], out_path: Path) -> None:
    """Render the top ten normalized models with the current snapshot's axis break."""
    data = rank_rows(rows)
    models = [row for row in data if row[0].lower() != "other"]
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(16, max(9.6, 4.0 + 0.55 * len(models))),
        dpi=200,
        sharey=True,
        gridspec_kw={"width_ratios": [3.2, 1.3], "wspace": 0.08},
    )
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.335, right=0.955, bottom=0.14, top=0.80)
    n = len(models)
    has_other = any(row[0].lower() == "other" for row in data)
    ys = list(range(n, 0, -1)) + ([-0.5] if has_other else [])
    labels = ["Other models" if name == "Other" else name for name, _, _ in data]
    for ax in (left, right):
        ax.set_facecolor(BG)
        for y, (name, inp, out) in zip(ys, data):
            alpha = 0.45 if name == "Other" else 1
            ax.barh(y, inp / B, height=0.58, color=GREEN, alpha=alpha, zorder=3, edgecolor=BG, linewidth=1)
            ax.barh(y, out / B, left=inp / B, height=0.58, color=LIME, alpha=alpha, zorder=3, edgecolor=BG, linewidth=1)
        ax.set_ylim(-1.25, n + 0.8)
        ax.xaxis.grid(True, color=WHITE, alpha=0.07)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=SUBTLE, length=0, pad=9, labelsize=10)
        ax.tick_params(axis="y", length=0, pad=12, labelsize=11)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(SPINE)
    left.set_xlim(0, 2500)
    right.set_xlim(15000, 22500)
    left.set_xticks([0, 500, 1000, 1500, 2000, 2500])
    right.set_xticks([16000, 18000, 20000, 22000])
    left.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / 1000:.1f}T"))
    right.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}T"))
    left.set_yticks(ys, labels=labels, color=MODELNAME)
    if has_other:
        left.get_yticklabels()[-1].set_color(SUBTLE)
        left.get_yticklabels()[-1].set_fontstyle("italic")
    right.tick_params(axis="y", left=False, labelleft=False)
    for y, (name, inp, out) in zip(ys, data):
        total = inp + out
        ax = right if total > 1e13 else left
        ax.text(
            total / B + (80 if ax is right else 30),
            y,
            fmt(total),
            color=SUBTLE if name == "Other" else MODELNAME,
            va="center",
            fontweight="bold",
            fontsize=12,
        )
        if inp > 1e13:
            left.text(1250, y, fmt(inp), ha="center", va="center", color=BG, fontweight="bold", fontsize=11)
        elif inp > 250e9:
            left.text(inp / 2e9, y, fmt(inp), ha="center", va="center", color=BG, fontweight="bold", fontsize=10)
        if out > 200e9:
            ax = right if inp > 1e13 else left
            ax.text(
                (inp + out / 2) / B, y, fmt(out), ha="center", va="center", color=BG, fontweight="bold", fontsize=10
            )
    for ax, x in ((left, 1), (right, 0)):
        for y in (0, 1):
            ax.plot(
                [x - 0.006, x + 0.006],
                [y - 0.009, y + 0.009],
                transform=ax.transAxes,
                color=SUBTLE,
                clip_on=False,
                linewidth=1.3,
            )
    fig.text(0.342, 0.925, "Top Model Usage", color=WHITE, fontsize=27, weight="bold")
    fig.text(0.342, 0.875, "Top 10 models ranked by total tokens", color=SUBTLE, fontsize=12)
    fig.add_artist(plt.Rectangle((0.322, 0.86), 0.005, 0.105, transform=fig.transFigure, facecolor=GREEN))
    for y, color, label in ((0.915, GREEN, "Input  ·  context tokens"), (0.875, LIME, "Output  ·  generated tokens")):
        fig.add_artist(plt.Rectangle((0.79, y), 0.016, 0.018, transform=fig.transFigure, facecolor=color))
        fig.text(0.815, y + 0.001, label, color=MODELNAME, fontsize=11)
    fig.text(0.64, 0.075, "Tokens processed", color=SUBTLE, ha="center", fontsize=12)
    fig.text(0.335, 0.027, "Remaining models grouped as Other. Axis break: 2.5T–15T.", color=SUBTLE, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help=f"Telemetry export CSV (default: {DEFAULT_CSV})")
    args = parser.parse_args()

    configure_matplotlib()
    rows = load_rows(args.csv)

    (primary,) = TARGETS
    render(rows, primary)

    for target in TARGETS:
        print(f"wrote {target.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
