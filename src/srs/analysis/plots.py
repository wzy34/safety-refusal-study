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
from matplotlib.ticker import PercentFormatter  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
HIGHLIGHT = "#2a78d6"
NEUTRAL = "#8a8984"
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("srs_blue", SEQUENTIAL)
# Diverging blue <-> red through the neutral gray midpoint. Blue arm: sequential steps
# 650/500/400/250/100; red arm: the red pole's hue stepped to the same OKLab lightness.
DIVERGING = ["#104281", "#256abf", "#3987e5", "#86b6ef", "#cde2fb", "#f0efec",
             "#ffd4cf", "#ff8c85", "#ea3d40", "#c80a22", "#830011"]
DIV_CMAP = LinearSegmentedColormap.from_list("srs_blue_red", DIVERGING)
# Categorical slots in fixed order (validated adjacent-pair palette, light mode).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
               "#e34948"]

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
            vmin: float = 0.0, vmax: float = 1.0, diverging: bool = False,
            marks: pd.DataFrame | None = None, note: str | None = None) -> list[Path]:
    """Rows x columns heatmap with the value printed in each cell. `diverging` centers a
    blue-gray-red scale on zero and prints signed values; cells true in `marks` get a '*'."""
    n_rows, n_cols = table.shape
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * n_cols, 1.0 + 0.42 * n_rows))
    im = ax.imshow(table.to_numpy(float), cmap=DIV_CMAP if diverging else SEQ_CMAP,
                   vmin=vmin, vmax=vmax, aspect="auto")
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
                dark = (abs(v) / vmax if diverging else (v - vmin) / (vmax - vmin)) > 0.55
                star = "*" if marks is not None and bool(marks.iat[i, j]) else ""
                ax.text(j, i, f"{v:+.2f}{star}" if diverging else f"{v:.2f}{star}",
                        ha="center", va="center", fontsize=7, color=SURFACE if dark else TEXT)
    ax.set_title(title, loc="left", pad=14 if note else 6)
    if note:  # subtitle between the title and the plot
        ax.annotate(note, (0, 1), xycoords="axes fraction", xytext=(0, 4),
                    textcoords="offset points", va="bottom", fontsize=7, color=TEXT_SECONDARY)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(cbar_label)
    cb.outline.set_visible(False)
    return save(fig, stem)


def bump_chart(ranks: pd.DataFrame, family: dict[str, str], family_colors: dict[str, str],
               thinking: dict[str, bool], xlabels: list[str], title: str, note: str,
               stem: Path) -> list[Path]:
    """Model rank (rows of `ranks`) under each evaluator (columns), rank 1 at the top. Hue is
    the model family, a dashed line with hollow markers the thinking variant; models are
    labeled at both ends, tied models sharing one label."""
    n_models, n_cols = ranks.shape

    def end_labels(x: int) -> dict[float, str]:
        col = ranks.iloc[:, x]
        return {r: ", ".join(sorted(n)) for r, n in col.groupby(col).groups.items()}

    ends = {x: end_labels(x) for x in (0, n_cols - 1)}
    # Room for the end labels, in columns: ~0.065 in per character at 8 pt, plus the offset.
    col_w = 0.95
    pad = {x: (max(map(len, ends[x].values())) * 0.065 + 0.3) / col_w for x in ends}
    fig, ax = plt.subplots(figsize=(0.7 + col_w * (n_cols - 1 + pad[0] + pad[n_cols - 1]),
                                    1.4 + 0.42 * n_models))
    xs = list(range(n_cols))
    for model, row in ranks.iterrows():
        c, thk = family_colors[family[model]], thinking[model]
        ax.plot(xs, row.to_numpy(float), color=c, lw=1.6, ls=(0, (4, 2)) if thk else "-",
                zorder=2)
        ax.plot(xs, row.to_numpy(float), "o", ms=7, color=c, mec=c if thk else SURFACE,
                mew=1.6 if thk else 1.2, mfc=SURFACE if thk else c, zorder=3)
    for x, ha, dx in ((0, "right", -10), (n_cols - 1, "left", 10)):
        for r, label in ends[x].items():
            ax.annotate(label, (x, r), xytext=(dx, 0), textcoords="offset points", ha=ha,
                        va="center", fontsize=8, color=TEXT)
    ax.set_xticks(xs, xlabels)
    ax.set_yticks(range(1, n_models + 1))
    ax.set_ylim(n_models + 0.5, 0.5)
    ax.set_xlim(-pad[0], n_cols - 1 + pad[n_cols - 1])
    ax.set_ylabel("rank (1 = lowest rate)")
    ax.tick_params(length=0)
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", pad=14)
    ax.annotate(note, (0, 1), xycoords="axes fraction", xytext=(0, 4),
                textcoords="offset points", va="bottom", fontsize=7, color=TEXT_SECONDARY)
    present = {family[m] for m in ranks.index}
    handles = [plt.Line2D([], [], color=c, lw=1.6, label=f)
               for f, c in family_colors.items() if f in present]
    handles += [plt.Line2D([], [], color=NEUTRAL, lw=1.6, marker="o", ms=6, mfc=NEUTRAL,
                           mec=SURFACE, label="standard"),
                plt.Line2D([], [], color=NEUTRAL, lw=1.6, ls=(0, (4, 2)), marker="o", ms=6,
                           mfc=SURFACE, mec=NEUTRAL, label="thinking")]
    ax.legend(handles=handles, frameon=False, loc="upper center", ncol=len(handles),
              bbox_to_anchor=(0.5, -0.16), borderaxespad=0, labelcolor=TEXT_SECONDARY,
              fontsize=8, handlelength=2.4)
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


# Stability patterns are an ordered scale (complied -> mixed -> refused): diverging poles from
# DIVERGING with the neutral gray midpoint (validated: CVD dE 9.8, normal-vision dE 16.8).
PATTERN_COLORS = {"always_comply": DIVERGING[2], "mixed": NEUTRAL, "always_refuse": DIVERGING[8]}
PATTERN_LABELS = {"always_comply": "always answered", "mixed": "mixed",
                  "always_refuse": "always refused"}


HEADER_IN = 0.5  # figure-level title + note above two-panel figures, in inches


def _figure_header(fig, title: str, note: str | None) -> None:
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.08 / h, title, fontsize=10, color=TEXT, va="top")
    if note:
        fig.text(0.01, 1 - 0.3 / h, note, fontsize=7, color=TEXT_SECONDARY, va="top")


def stability_patterns(shares: pd.DataFrame, title: str, stem: Path,
                       note: str | None = None) -> list[Path]:
    """100% horizontal bars per model: share of prompts always answered / mixed / always
    refused across the sampled responses, one panel per expected behavior. The mixed share is
    labeled directly; segments are separated by a surface gap."""
    models = sorted(shares["model_id"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 0.9 + 0.36 * len(models)), sharey=True)
    for ax, (eb, label) in zip(axes, (("refuse", "Should refuse"), ("comply", "Should answer")),
                               strict=True):
        sub = shares[shares["expected_behavior"] == eb]
        n = int(sub.groupby("model_id")["n"].sum().max()) if len(sub) else 0
        for i, m in enumerate(models):
            left = 0.0
            for pat in PATTERN_COLORS:
                v = float(sub[(sub["model_id"] == m) & (sub["pattern"] == pat)]["share"].sum())
                if v > 0:
                    ax.barh(i, v, left=left, height=0.64, color=PATTERN_COLORS[pat],
                            edgecolor=SURFACE, linewidth=1.5)
                    if pat == "mixed" and v >= 0.06:
                        ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center",
                                fontsize=7, color=SURFACE)
                left += v
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_yticks(range(len(models)), models)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.tick_params(length=0)
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"{label} (up to {n} prompts per model)", loc="left", fontsize=9)
    _figure_header(fig, title, note)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=PATTERN_LABELS[p])
               for p, c in PATTERN_COLORS.items()]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.06), labelcolor=TEXT_SECONDARY, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1 - HEADER_IN / fig.get_figheight()), w_pad=3)
    return save(fig, stem)


def greedy_vs_sampled(summary: pd.DataFrame, title: str, stem: Path,
                      note: str | None = None) -> list[Path]:
    """Dumbbell per model: the rate from the greedy responses (hollow) vs the mean over the
    sampled responses (filled), for FAR and ORR on the stability subset."""
    s = summary.sort_values("model_id").reset_index(drop=True)
    dark = SEQUENTIAL[10]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 0.9 + 0.36 * len(s)), sharey=True)
    panels = (("far_comply", "FAR (should-refuse prompts answered)"),
              ("orr", "ORR (should-answer prompts refused)"))
    for ax, (metric, label) in zip(axes, panels, strict=True):
        ax.grid(True, axis="x")
        ax.set_axisbelow(True)
        for i, r in s.iterrows():
            g, v = r[f"{metric}_greedy"], r[f"{metric}_sampled"]
            ax.plot([g, v], [i, i], color=GRID, lw=2.5, solid_capstyle="round", zorder=1)
            ax.plot(g, i, "o", ms=8, mfc=SURFACE, mec=HIGHLIGHT, mew=2, zorder=3)
            ax.plot(v, i, "o", ms=8, mfc=dark, mec=SURFACE, mew=1.2, zorder=4)
        hi = float(s[[f"{metric}_greedy", f"{metric}_sampled"]].max().max())
        ax.set_xlim(0, max(0.1, hi * 1.15))
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_yticks(range(len(s)), s["model_id"])
        ax.set_ylim(len(s) - 0.5, -0.5)
        ax.tick_params(length=0)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.set_title(label, loc="left", fontsize=9)
    _figure_header(fig, title, note)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=7, mfc=SURFACE, mec=HIGHLIGHT, mew=2,
                          label="greedy (temperature 0)"),
               plt.Line2D([], [], marker="o", ls="", ms=7, mfc=dark, mec=SURFACE,
                          label="mean of sampled responses")]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.06), labelcolor=TEXT_SECONDARY, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1 - HEADER_IN / fig.get_figheight()), w_pad=3)
    return save(fig, stem)
