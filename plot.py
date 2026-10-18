"""Plot section hit@5 against index size for every storage variant.

Usage:
    uv run --group plot plot.py                          # results/full_corpus.json
    uv run --group plot plot.py index/results.json out.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Categorical slots 1-3 of the reference palette; they validate all-pairs in light mode.
FAMILIES = {
    "Full precision / scalar": ("#2a78d6", "o"),
    "Binary": ("#eb6834", "s"),
    "Binary + int8 rescore": ("#1baf7a", "D"),
}


def family(name: str) -> str:
    if name.startswith("binary +"):
        return "Binary + int8 rescore"
    return "Binary" if name == "binary" else "Full precision / scalar"


"""
Short label for a point: the variant name without the "(baseline)" suffix, or just
the multiplier for rescoring variants, e.g. "16x".
"""
def label(name: str) -> str:
    if name.startswith("binary +"):
        return name[name.index("(") + 1 : -1]
    return name.replace(" (baseline)", "")


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/full_corpus.json")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "results/quality_vs_size.png")
    results = json.loads(src.read_text())

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True, facecolor=SURFACE)
    panels = [("stored_mb", "Stored on disk (MB, log scale)"), ("memory_mb", "Needed in RAM for search (MB, log scale)")]

    for ax, (key, xlabel) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log")
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, length=0)
        ax.set_xlabel(xlabel, color=MUTED)

        rescore = sorted(
            (r for r in results if family(r["variant"]) == "Binary + int8 rescore"),
            key=lambda r: r["section_hit"],
        )
        ax.plot(
            [r[key] for r in rescore],
            [100 * r["section_hit"] for r in rescore],
            color=FAMILIES["Binary + int8 rescore"][0], linewidth=2, zorder=1,
        )
        for r in results:
            color, marker = FAMILIES[family(r["variant"])]
            x, y = r[key], 100 * r["section_hit"]
            ax.scatter(x, y, s=70, color=color, marker=marker, edgecolor=SURFACE, linewidth=2, zorder=3)
            scalar = family(r["variant"]) == "Full precision / scalar"
            # Scalar variants share one value and sit close together, so they are
            # labelled above their markers instead of to the right.
            text = f"{label(r['variant'])}\n{y:.1f}%" if scalar else f"{label(r['variant'])}  {y:.1f}%"
            ax.annotate(
                text,
                (x, y), xytext=(0, 18) if scalar else (9, 0), textcoords="offset points",
                va="center", ha="center" if scalar else "left", fontsize=9, color=INK,
            )
        ax.set_xlim(2, 400)

    axes[0].set_ylabel("Gold section in top 5 (%)", color=MUTED)
    axes[0].set_ylim(62, 84)
    handles = [
        plt.Line2D([], [], marker=m, color=c, linestyle="", markersize=8, label=n)
        for n, (c, m) in FAMILIES.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.97), labelcolor=INK)
    fig.suptitle(
        "Binary quantization with rescoring: near-fp32 retrieval with 3.6 MB in RAM",
        y=1.03, fontsize=13, fontweight="bold", color=INK,
    )
    fig.text(0.5, -0.02, "bge-small-en-v1.5, 74,316 chunks, 1,000 questions. Left: the rescoring variants also keep the 28.5 MB int8 copy.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
