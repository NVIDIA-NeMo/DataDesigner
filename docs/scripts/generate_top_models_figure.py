# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib==3.9.4",
# ]
# ///
"""Regenerate the "Top Model Usage" telemetry figure.

Renders the ranked input-vs-output token breakdown shown in the README's
"Top models (YTD)" section, styled to match the Data Designer devnote charts
(near-black canvas, NVIDIA-green duotone). The PNG is written to the Fern image
path rendered by the README and docs site:

    fern/images/top-models.png

The source telemetry export lives at docs/scripts/top-model-usage.csv with
columns: model name, input (context) tokens, output (generated) tokens, plus a
trailing "Other" aggregate row. Drop in a fresh export to refresh the figure.

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
from matplotlib.ticker import FuncFormatter, MaxNLocator

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
AXIS = "#B8B8B8"
MODELNAME = "#ECECEC"
GRID = "#FFFFFF"
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
    rcParams["font.size"] = 13


def fmt(v: float) -> str:
    """Compact billions/trillions label."""
    if v >= 1e12:
        return f"{v / 1e12:.2f}T"
    return f"{v / 1e9:.0f}B"


def render(rows: list[tuple[str, float, float]], out_path: Path) -> None:
    """Render the ranked stacked-bar figure to out_path."""
    # Split the "Other" aggregate out; sort named models by total descending.
    other = next((r for r in rows if r[0].lower() == "other"), None)
    models = [r for r in rows if r[0].lower() != "other"]
    models.sort(key=lambda r: r[1] + r[2], reverse=True)

    n = len(models)
    ypos = list(range(n, 0, -1))  # n, n-1, ... 1  (top -> down)
    labels = [m[0] for m in models]
    inputs = [m[1] for m in models]
    outputs = [m[2] for m in models]

    if other is not None:
        ypos.append(-0.6)  # gap below the named models
        labels.append("Other models")
        inputs.append(other[1])
        outputs.append(other[2])

    fig, ax = plt.subplots(figsize=(14.5, 9.2), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    xmax = max(i + o for i, o in zip(inputs, outputs)) / B
    bar_h = 0.62

    for idx, (y, inp, out) in enumerate(zip(ypos, inputs, outputs)):
        is_other = other is not None and idx == len(ypos) - 1
        a = 0.45 if is_other else 1.0

        ax.barh(y, inp / B, height=bar_h, color=GREEN, alpha=a, zorder=3, edgecolor=BG, linewidth=1.2)
        ax.barh(y, out / B, height=bar_h, left=inp / B, color=LIME, alpha=a, zorder=3, edgecolor=BG, linewidth=1.2)

        total = (inp + out) / B
        ax.text(
            total + xmax * 0.008,
            y,
            fmt(inp + out),
            va="center",
            ha="left",
            color=SUBTLE if is_other else WHITE,
            fontweight="bold",
            fontsize=13.5,
            zorder=5,
        )

        # In-segment value labels only where the segment is wide enough.
        if inp / B > xmax * 0.085:
            ax.text(
                (inp / B) / 2,
                y,
                fmt(inp),
                va="center",
                ha="center",
                color=INK,
                fontweight="bold",
                fontsize=11.5,
                alpha=a,
                zorder=5,
            )
        if out / B > xmax * 0.085:
            ax.text(
                inp / B + (out / B) / 2,
                y,
                fmt(out),
                va="center",
                ha="center",
                color=INK,
                fontweight="bold",
                fontsize=11.5,
                alpha=a,
                zorder=5,
            )

    # ------------------------------------------------------------- axes -----
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=12.5)
    is_other_flags = [False] * n + ([True] if other else [])
    for tick, is_other in zip(ax.get_yticklabels(), is_other_flags):
        tick.set_color(SUBTLE if is_other else MODELNAME)
        if is_other:
            tick.set_fontstyle("italic")

    ax.set_xlim(0, xmax * 1.13)
    ax.set_ylim(-1.3, n + 0.8)

    # Derive ticks from the data so the axis stays sane as totals grow; fmt()
    # promotes B -> T automatically, so the labels never need hand-editing.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: "0" if v <= 0 else fmt(v * B)))
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", colors=AXIS, length=0, pad=8, labelsize=11)
    ax.set_xlabel("Tokens processed", color=AXIS, fontsize=12.5, labelpad=12)

    ax.xaxis.grid(True, color=GRID, alpha=0.07, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(SPINE)
        ax.spines[s].set_linewidth(1.0)

    # ---------------------------------------------------------- titling -----
    fig.subplots_adjust(left=0.235, right=0.965, top=0.83, bottom=0.085)
    # Signature DD green left-accent rule (mirrors the .devnote-dek element).
    ax.add_patch(
        plt.Rectangle(
            (-0.018, 1.045),
            0.006,
            0.135,
            transform=ax.transAxes,
            facecolor=GREEN,
            edgecolor="none",
            clip_on=False,
            zorder=6,
        )
    )
    ax.text(
        0.012,
        1.145,
        "Top Model Usage",
        transform=ax.transAxes,
        color=WHITE,
        fontweight="bold",
        fontsize=26,
        ha="left",
        va="bottom",
    )
    ax.text(
        0.012,
        1.07,
        "Context vs. generated tokens across the most-used models",
        transform=ax.transAxes,
        color=SUBTLE,
        fontsize=13.5,
        ha="left",
        va="bottom",
    )

    # Manual legend, top-right of the plotting area.
    leg_x, leg_y = 0.99, 1.115
    legend = [(GREEN, "Input  ·  context tokens"), (LIME, "Output  ·  generated tokens")]
    for i, (c, lbl) in enumerate(legend):
        yy = leg_y - i * 0.052
        ax.add_patch(
            plt.Rectangle(
                (leg_x - 0.205, yy - 0.012),
                0.022,
                0.026,
                transform=ax.transAxes,
                facecolor=c,
                edgecolor="none",
                clip_on=False,
                zorder=6,
            )
        )
        ax.text(leg_x - 0.172, yy, lbl, transform=ax.transAxes, color=MODELNAME, fontsize=12, ha="left", va="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG, dpi=200, bbox_inches="tight", pad_inches=0.25)
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
