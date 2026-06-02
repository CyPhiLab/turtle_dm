"""
Per-frame fish metric computations shared across multiple plot scripts.
"""

import numpy as np
import pandas as pd

from src.utils.mask_utils import MASK_WIDTH, MASK_HEIGHT, SCALE_X, SCALE_Y


def compute_all_metrics_from_df(
    df: pd.DataFrame,
    signed_reef_map: np.ndarray | None = None,
    rolling_window: int = 5,
) -> dict[str, np.ndarray]:
    """
    Compute per-frame metrics for all frames in *df*.

    Parameters
    ----------
    df              : normalised per-frame DataFrame (output of auto_read_csv)
    signed_reef_map : 2-D float array where ocean pixels are positive (distance
                      to reef edge) and reef pixels are negative (negative
                      distance to ocean edge).  When provided, computes
                      ``signed_reef_distance``.  If None, the metric is omitted.
    rolling_window  : smoothing window (in frames)
    """
    from src.utils.csv_utils import loc_cols as _loc_cols

    lc = _loc_cols(df)
    n  = len(df)

    def _smooth(arr: np.ndarray) -> np.ndarray:
        return (
            pd.Series(arr)
            .rolling(rolling_window, center=True, min_periods=1)
            .mean()
            .values
        )

    # ── Vectorised position extraction ────────────────────────────────
    J = len(lc)
    all_xy = np.full((n, J, 2), np.nan)
    for j, col in enumerate(lc):
        parsed = df[col].astype(str).str.extract(r'\(([^,]+),([^)]+)\)', expand=True)
        all_xy[:, j, 0] = pd.to_numeric(parsed[0], errors='coerce').values
        all_xy[:, j, 1] = pd.to_numeric(parsed[1], errors='coerce').values

    valid = ~np.isnan(all_xy[:, :, 0])  # (n, J) bool — fish present

    arrays: dict[str, np.ndarray] = {}

    # ── fish_count ────────────────────────────────────────────────────
    arrays["fish_count"] = _smooth(df["fish_count"].values.astype(float))

    # ── vertical_position ─────────────────────────────────────────────
    ys_masked = np.where(valid, all_xy[:, :, 1], np.nan)
    with np.errstate(all="ignore"):
        arrays["vertical_position"] = _smooth(np.nanmean(ys_masked, axis=1))

    # ── signed_reef_distance ──────────────────────────────────────────
    # Positive = fish in open ocean (distance to reef boundary).
    # Negative = fish on reef (negative distance to ocean boundary).
    if signed_reef_map is not None:
        scaled_x = np.where(valid, all_xy[:, :, 0] / SCALE_X, 0.0)
        scaled_y = np.where(valid, all_xy[:, :, 1] / SCALE_Y, 0.0)
        mxs_safe = np.clip(scaled_x.astype(int), 0, MASK_WIDTH  - 1)
        mys_safe = np.clip(scaled_y.astype(int), 0, MASK_HEIGHT - 1)
        sr_vals  = signed_reef_map[mys_safe, mxs_safe].astype(float)
        sr_vals  = np.where(valid, sr_vals, np.nan)
        with np.errstate(all="ignore"):
            arrays["signed_reef_distance"] = _smooth(np.nanmean(sr_vals, axis=1))

    # ── centroid_speed ────────────────────────────────────────────────
    all_pos: list[np.ndarray] = [all_xy[i][valid[i]] for i in range(n)]
    cspeed = np.full(n, np.nan)
    for i in range(1, n):
        curr, prev = all_pos[i], all_pos[i - 1]
        if len(curr) and len(prev):
            cspeed[i] = float(np.linalg.norm(
                np.mean(curr, axis=0) - np.mean(prev, axis=0)
            ))
    arrays["centroid_speed"] = _smooth(cspeed)

    return arrays
