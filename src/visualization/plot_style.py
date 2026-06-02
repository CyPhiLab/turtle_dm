"""
Shared plot style constants and helpers for individual (publication-quality) figures.

All single-panel output plots (focal FID, individual metrics, sound analysis)
should call apply_individual_style() before saving to ensure consistent appearance.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── Dimensions & typography ──────────────────────────────────────────────────
INDIVIDUAL_FIGSIZE = (10, 5)   # width × height in inches
BASE_FONTSIZE      = 11        # tick labels, legend
LABEL_FONTSIZE     = 12        # axis labels, title

# IEEE conference single-column figure width is about 3.5 in.
PHASE2A_PANEL_FIGSIZE = (3.5, 2.3)
PHASE2A_GRID_COL_W    = 3.2
PHASE2A_GRID_ROW_H    = 2.35
PHASE2A_LABEL_FS      = 9
PHASE2A_TICK_FS       = 8
PHASE2A_LEGEND_FS     = 8
PHASE2A_FONT_FAMILY   = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]

# ── Robot / subject colours ──────────────────────────────────────────────────
ROBOT_COLORS: dict[str, str] = {
    "auv":    "#E87722",   # orange
    "turtle": "#0097A7",   # teal
    "diver":  "#2CA02C",   # green
    "shark":  "#9467BD",   # purple
}

AUV_C    = ROBOT_COLORS["auv"]
TURTLE_C = ROBOT_COLORS["turtle"]
DIVER_C  = ROBOT_COLORS["diver"]
SHARK_C  = ROBOT_COLORS["shark"]


def apply_individual_style(fig: plt.Figure, ax=None) -> None:
    """Apply consistent formatting to a single-panel figure.

    Call this immediately before fig.savefig().  Works with or without an
    explicitly passed axes object (falls back to all axes on *fig*).
    """
    if ax is None:
        axes = fig.get_axes()
    elif isinstance(ax, np.ndarray):
        axes = ax.flatten().tolist()
    elif isinstance(ax, (list, tuple)):
        axes = list(ax)
    else:
        axes = [ax]

    for a in axes:
        a.tick_params(axis="both", which="both", labelsize=BASE_FONTSIZE)

        xlabel = a.get_xlabel()
        ylabel = a.get_ylabel()
        title  = a.get_title()

        if xlabel:
            a.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
        if ylabel:
            a.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
        if title:
            a.set_title(title, fontsize=LABEL_FONTSIZE, fontweight="bold")

        for spine in a.spines.values():
            spine.set_visible(True)

        a.grid(True, alpha=0.25, linewidth=0.6)

        for line in a.get_lines():
            if line.get_linewidth() < 1.5:
                line.set_linewidth(1.5)

        leg = a.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontsize(BASE_FONTSIZE)
            if leg.get_title() and leg.get_title().get_text():
                leg.get_title().set_fontsize(BASE_FONTSIZE)

    fig.set_size_inches(*INDIVIDUAL_FIGSIZE, forward=True)
    fig.tight_layout()


def apply_phase2a_style(fig: plt.Figure, ax=None, figsize=None, clear_titles: bool = True) -> None:
    """Compact title-free styling for phase2a temporal figures."""
    apply_individual_style(fig, ax)
    if figsize is not None:
        fig.set_size_inches(*figsize, forward=True)

    if ax is None:
        axes = fig.get_axes()
    elif isinstance(ax, np.ndarray):
        axes = ax.flatten().tolist()
    elif isinstance(ax, (list, tuple)):
        axes = list(ax)
    else:
        axes = [ax]

    for a in axes:
        if clear_titles:
            a.set_title("")
        a.tick_params(axis="both", which="both", labelsize=PHASE2A_TICK_FS)

        for label in a.get_xticklabels() + a.get_yticklabels():
            label.set_fontfamily(PHASE2A_FONT_FAMILY[0])

        xlabel = a.get_xlabel()
        ylabel = a.get_ylabel()
        if xlabel:
            a.set_xlabel(xlabel, fontsize=PHASE2A_LABEL_FS, fontfamily=PHASE2A_FONT_FAMILY[0])
        if ylabel:
            a.set_ylabel(ylabel, fontsize=PHASE2A_LABEL_FS, fontfamily=PHASE2A_FONT_FAMILY[0])

        leg = a.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontsize(PHASE2A_LEGEND_FS)
                txt.set_fontfamily(PHASE2A_FONT_FAMILY[0])

        a.grid(True, alpha=0.18, linewidth=0.5)

    fig.tight_layout(pad=0.4)
