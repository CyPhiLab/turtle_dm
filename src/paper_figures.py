"""
Publication figures for the paper.

Outputs written to figures/paper/:
  panel.svg  — 2×2 grid: fish_count, signed_reef_distance,
               vertical_position, centroid_speed (raw, absolute time, mean±SEM)
  fid.svg    — fish_count FID: |seconds from transect midpoint| (raw)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.interpolate import interp1d

from config.paths import FIGURES_DIR, ensure_dirs_exist
from src.analysis.fid_model import FID_ALPHA, fid_logistic, fit_fid_model
from src.utils.trial_loader import load_trials
from src.visualization.plot_style import (
    apply_paper_style,
    PANEL_FIGSIZE,
    PAPER_FONT,
    AUV_C, TURTLE_C, DIVER_C, SHARK_C,
)

OUT_DIR = FIGURES_DIR / "paper"
FPS     = 30

# ── Figure-level configuration ────────────────────────────────────────────────

# Metrics shown in the panel figure, row-major 3×2 order.
PANEL_METRICS = [
    "fish_count",
    "shelter_distance",
    "vertical_position",
    "centroid_speed",
    "nnd",
    "spatial_entropy",
]

PANEL_YLABELS = {
    "fish_count":        "Fish count",
    "shelter_distance":  "Shelter distance (px)",
    "vertical_position": "Vertical position (px)",
    "centroid_speed":    "Centroid speed (px/frame)",
    "nnd":               "Mean Nearest Neighbour Distance (px)",
    "spatial_entropy":   "Spatial entropy (bits)",
}

# Shared absolute-time axis: ±40 s from transect midpoint.
ABS_TIME_GRID = np.linspace(-40, 40, 320)

# FID display axis: 0–40 s (absolute distance from midpoint).
# Internally, points are collected as signed time [-40, 0] for correct model fitting.
FID_GRID = np.linspace(0, 40, 200)

# Subjects in plot order: (trial_loader key, colour, legend label).
SUBJECTS = [
    ("auv",    AUV_C,    "Thruster AUV"),
    ("turtle", TURTLE_C, "Turtle AUV"),
    ("diver",  DIVER_C,  "Diver"),
    ("shark",  SHARK_C,  "Shark"),
]

# Subjects shown in the FID figure (shark excluded: only 1 transect).
FID_SUBJECTS = [(k, c, l) for k, c, l in SUBJECTS if k != "shark"]

# Subjects for which the FID model is fitted (shark excluded: only 1 transect).
FID_FIT_SUBJECTS = {k for k, _, _ in SUBJECTS if k != "shark"}


# ── Core data helper ──────────────────────────────────────────────────────────

def _abstime_matrix(trials: list, metric: str) -> np.ndarray | None:
    """Interpolate all trials onto ABS_TIME_GRID.

    Returns a (n_trials × len(ABS_TIME_GRID)) array, or None if no valid trials.
    """
    rows = []
    for t in trials:
        arr       = t["arrays"].get(metric)
        frames    = t["frames"]
        dist_f    = t["dist_frame"]
        end_f     = t.get("end_frame")
        trial_fps = t["fps"]
        if arr is None:
            continue
        mid_f = (dist_f + end_f) / 2.0 if end_f is not None else dist_f
        t_sec = (frames - mid_f) / trial_fps
        valid = ~np.isnan(arr)
        if valid.sum() < 4:
            continue
        fi = interp1d(t_sec[valid], arr[valid], bounds_error=False, fill_value=np.nan)
        rows.append(fi(ABS_TIME_GRID))
    return np.array(rows) if rows else None


def _mean_sem(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(mat, axis=0)
    n    = np.sum(~np.isnan(mat), axis=0).clip(1)
    sem  = np.nanstd(mat, axis=0) / np.sqrt(n)
    return mean, sem


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_panel_figure(trials_by_robot: dict, out_dir: Path) -> None:
    """4-panel 2×2 figure: raw absolute-time mean±SEM for each metric in PANEL_METRICS."""
    panel_w = PANEL_FIGSIZE[0]
    panel_h = PANEL_FIGSIZE[1]
    fig, axes = plt.subplots(3, 2, figsize=(panel_w * 2, panel_h * 3))
    axes_flat = axes.flatten()

    for i, metric in enumerate(PANEL_METRICS):
        ax = axes_flat[i]
        legend_plotted = False

        for robot_key, color, label in SUBJECTS:
            trials = trials_by_robot.get(robot_key) or []
            if not trials:
                continue
            mat = _abstime_matrix(trials, metric)
            if mat is None:
                continue
            mean, sem = _mean_sem(mat)
            vm = ~np.isnan(mean)
            ax.plot(ABS_TIME_GRID[vm], mean[vm], color=color, lw=2, label=label)
            ax.fill_between(
                ABS_TIME_GRID[vm],
                (mean - sem)[vm], (mean + sem)[vm],
                color=color, alpha=0.18,
            )
            legend_plotted = True

        ax.axvline(0, color="gray", lw=0.8, ls=":", alpha=0.6)
        ax.set_xlabel("Time from transect midpoint (s)")
        ax.set_ylabel(PANEL_YLABELS[metric])
        if i == 0 and legend_plotted:
            ax.legend(
                loc="upper right", ncol=2, frameon=False,
                handlelength=1.2, columnspacing=0.8, borderaxespad=0.2,
            )

    apply_paper_style(
        fig, axes_flat,
        figsize=(panel_w * 2, panel_h * 3),
        clear_titles=True,
    )
    for ax, lbl in zip(axes_flat, "abcdef"):
        ax.text(0.02, 0.97, f"{lbl}", transform=ax.transAxes,
                fontsize=10, va="top", ha="left",
                fontfamily=PAPER_FONT[0])
    out_path = out_dir / "panel.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


FID_METRICS = [
    ("fish_count",      "Fish count"),
    ("spatial_entropy", "Spatial entropy (bits)"),
]


def _collect_fid_points(trials: list, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Collect pooled signed-time FID points (x, y) for one robot and metric.

    x is signed time relative to transect midpoint (s), restricted to [-40, 0].
    """
    all_x, all_y = [], []
    for t in trials:
        arr       = t["arrays"].get(metric)
        frames    = t["frames"]
        dist_f    = t["dist_frame"]
        end_f     = t.get("end_frame")
        trial_fps = t["fps"]
        if arr is None:
            continue

        mid_f = (dist_f + end_f) / 2.0 if end_f is not None else dist_f
        t_sec = (frames - mid_f) / trial_fps
        valid = ~np.isnan(arr) & (t_sec >= -FID_GRID[-1]) & (t_sec <= 0.0)
        all_x.extend(t_sec[valid].tolist())
        all_y.extend(arr[valid].tolist())

    if not all_x:
        return np.array([]), np.array([])
    return np.array(all_x), np.array(all_y)


def plot_fid(trials_by_robot: dict, out_dir: Path) -> None:
    """1×2 FID figure: |seconds from transect midpoint| for each metric."""
    MAX_SCATTER = 4000
    rng = np.random.default_rng(42)
    panel_w, panel_h = PANEL_FIGSIZE
    n_metrics = len(FID_METRICS)

    fig, axes = plt.subplots(1, n_metrics, figsize=(panel_w * n_metrics, panel_h))

    for ax, (metric, ylabel) in zip(axes, FID_METRICS):
        print(f"\n[FID Model] metric={metric}, alpha={FID_ALPHA:.2f}")
        for robot_key, color, label in FID_SUBJECTS:
            trials = trials_by_robot.get(robot_key) or []
            if not trials:
                continue

            sx, sy = _collect_fid_points(trials, metric)
            if sx.size == 0:
                continue
            # Display axis is |t|: negate signed x so 0 = midpoint, 40 = far from robot.
            dx = -sx

            if len(dx) > MAX_SCATTER:
                idx = rng.choice(len(dx), size=MAX_SCATTER, replace=False)
                ax.scatter(dx[idx], sy[idx], color=color, alpha=0.12, s=5,
                           linewidths=0, rasterized=True)
            else:
                ax.scatter(dx, sy, color=color, alpha=0.12, s=5,
                           linewidths=0, rasterized=True)

            bin_w = FID_GRID[1] - FID_GRID[0]
            means, sems = [], []
            for gx in FID_GRID:
                vals = sy[np.abs(dx - gx) < bin_w]
                if len(vals) >= 3:
                    means.append(np.nanmean(vals))
                    sems.append(np.nanstd(vals, ddof=1) / np.sqrt(len(vals)))
                else:
                    means.append(np.nan)
                    sems.append(np.nan)

            means = np.array(means)
            sems  = np.array(sems)
            vm    = ~np.isnan(means)
            if vm.sum() >= 3:
                ax.plot(FID_GRID[vm], means[vm], color=color, lw=2.5, label=label)
                ax.fill_between(
                    FID_GRID[vm],
                    (means - sems)[vm], (means + sems)[vm],
                    color=color, alpha=0.18,
                )

            fit = fit_fid_model(sx, sy, alpha=FID_ALPHA)
            if robot_key not in FID_FIT_SUBJECTS:
                continue
            if fit is None:
                print(f"  {label}: fit failed (insufficient or unstable data), n={len(sx)}")
                continue

            fit_curve = fid_logistic(-FID_GRID, fit.k, fit.x0, fit.L0, fit.Ld)
            vf = np.isfinite(fit_curve)
            if vf.any():
                ax.plot(
                    FID_GRID[vf], fit_curve[vf],
                    color=color, ls="--", lw=1.0,
                    path_effects=[
                        pe.Stroke(linewidth=2.5, foreground="white", alpha=0.8),
                        pe.Normal(),
                    ],
                )

            x_fid_txt = f"{-fit.x_fid:.2f}s" if fit.x_fid is not None else fit.x_fid_status
            print(
                f"  {label}: n={len(sx)} | "
                f"L0(empirical)={fit.L0:.4f} | Ld(empirical)={fit.Ld:.4f} | "
                f"k(fitted)={fit.k:.4f} | x0(fitted)={fit.x0:.2f}s | x_FID={x_fid_txt}"
            )

        ax.set_xlabel("Time from transect midpoint (s)")
        ax.set_ylabel(ylabel)
        ax.axvline(0, color="gray", lw=0.8, ls=":", alpha=0.6)
        legend_loc = "upper center" if metric == FID_METRICS[0][0] else "lower center"
        ax.legend(
            loc=legend_loc, ncol=3, frameon=False,
            handlelength=1.2, columnspacing=0.8, borderaxespad=0.2,
        )

    apply_paper_style(fig, axes, figsize=(panel_w * n_metrics, panel_h))
    for ax, lbl in zip(axes, "ab"):
        ax.text(0.02, 0.97, f"{lbl}", transform=ax.transAxes,
                fontsize=10, va="top", ha="left",
                fontfamily=PAPER_FONT[0])
    out_path = out_dir / "fid.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dirs_exist(OUT_DIR)
    print("Loading trials...")
    auv    = load_trials("auv")
    turtle = load_trials("turtle")
    diver  = load_trials("diver")
    shark  = load_trials("shark")
    print(f"  AUV: {len(auv)}  Turtle: {len(turtle)}  Diver: {len(diver)}  Shark: {len(shark)}")

    trials_by_robot = {
        "auv":    auv,
        "turtle": turtle,
        "diver":  diver,
        "shark":  shark,
    }

    plot_panel_figure(trials_by_robot, OUT_DIR)
    plot_fid(trials_by_robot, OUT_DIR)
    print(f"\nDone. Figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
