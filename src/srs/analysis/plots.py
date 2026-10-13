"""Matplotlib figures for the RQs, saved as PNG and PDF with one consistent palette.

Colors follow the dataviz reference palette: magnitudes use the single-hue blue sequential
ramp; the FAR x ORR plane highlights Pareto-optimal models in blue against neutral gray
(dominated), with hollow markers as a second cue and every point labeled directly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
HIGHLIGHT = "#2a78d6"
NEUTRAL = "#8a8984"
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("srs_blue", SEQUENTIAL)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY, "axes.titlecolor": TEXT,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "pdf.fonttype": 42, "svg.hashsalt": "srs", "path.simplify": False,
})


def save(fig, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight", metadata={"Software": None})
    fig.savefig(paths[1], bbox_inches="tight", metadata={"CreationDate": None,
                                                         "Producer": None, "Creator": None})
    plt.close(fig)
    return paths


def heatmap(table: pd.DataFrame, title: str, stem: Path, cbar_label: str = "rate",
            vmin: float = 0.0, vmax: float = 1.0) -> list[Path]:
    """Rows x columns heatmap of rates with the value printed in each cell."""
    n_rows, n_cols = table.shape
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * n_cols, 1.0 + 0.42 * n_rows))
    im = ax.imshow(table.to_numpy(float), cmap=SEQ_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_cols), [str(c).replace("_", " ") for c in table.columns],
                  rotation=40, ha="right")
    ax.set_yticks(range(n_rows), [str(r) for r in table.index])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(n_rows):
        for j in range(n_cols):
            v = table.iat[i, j]
            if pd.notna(v):
                dark = (v - vmin) / (vmax - vmin) > 0.55
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color=SURFACE if dark else TEXT)
    ax.set_title(title, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(cbar_label)
    cb.outline.set_visible(False)
    return save(fig, stem)


OFFSETS = [(7, 6), (7, -12), (-7, 6), (-7, -12), (9, -3), (-9, -3), (0, 11), (0, -16),
           (14, 14), (-14, 14), (14, -20), (-14, -20)]


def _place_labels(fig, ax, points: pd.DataFrame) -> None:
    """Direct labels that avoid each other and the markers (greedy over candidate offsets)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    taken = []
    for _, p in points.iterrows():  # marker footprints
        x, y = ax.transData.transform((p["far_comply"], p["orr"]))
        taken.append((x - 6, y - 6, x + 6, y + 6))
    for _, p in points.sort_values(["orr", "far_comply"]).iterrows():
        for dx, dy in OFFSETS:
            ha = "left" if dx > 0 else "right" if dx < 0 else "center"
            txt = ax.annotate(p["model_id"], (p["far_comply"], p["orr"]), xytext=(dx, dy),
                              textcoords="offset points", fontsize=8, color=TEXT, ha=ha)
            bb = txt.get_window_extent(renderer)
            box = (bb.x0 - 1, bb.y0 - 1, bb.x1 + 1, bb.y1 + 1)
            if not any(box[0] < o[2] and o[0] < box[2] and box[1] < o[3] and o[1] < box[3]
                       for o in taken):
                taken.append(box)
                break
            txt.remove()
        else:  # nothing free: keep the first position
            ax.annotate(p["model_id"], (p["far_comply"], p["orr"]), xytext=OFFSETS[0],
                        textcoords="offset points", fontsize=8, color=TEXT)


def far_orr_plane(points: pd.DataFrame, title: str, stem: Path) -> list[Path]:
    """Models at (FAR, ORR) with 95% CI error bars; Pareto-optimal models highlighted."""
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.grid(True)
    ax.set_axisbelow(True)
    for _, p in points.sort_values("model_id").iterrows():
        color = HIGHLIGHT if p["pareto_optimal"] else NEUTRAL
        ax.errorbar(p["far_comply"], p["orr"],
                    xerr=[[p["far_comply"] - p["far_comply_lo"]],
                          [p["far_comply_hi"] - p["far_comply"]]],
                    yerr=[[p["orr"] - p["orr_lo"]], [p["orr_hi"] - p["orr"]]],
                    fmt="none", ecolor=color, elinewidth=1, alpha=0.7, capsize=0)
        ax.plot(p["far_comply"], p["orr"], "o", ms=8, mec=color, mew=2,
                mfc=color if p["pareto_optimal"] else SURFACE, zorder=3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    _place_labels(fig, ax, points)
    ax.set_xlabel("FAR (refuse-expected prompts not refused)")
    ax.set_ylabel("ORR (comply-expected prompts refused)")
    ax.set_title(title, loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=7, mec=HIGHLIGHT, mfc=HIGHLIGHT,
                          mew=2, label="Pareto-optimal"),
               plt.Line2D([], [], marker="o", ls="", ms=7, mec=NEUTRAL, mfc=SURFACE, mew=2,
                          label="dominated")]
    ax.legend(handles=handles, frameon=False, loc="upper right", labelcolor=TEXT_SECONDARY)
    return save(fig, stem)
