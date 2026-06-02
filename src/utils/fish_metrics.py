"""
Per-frame fish metric computations shared across multiple plot scripts.
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.stats import entropy as sp_entropy

from config.constants import SHELTER_LEFT, SHELTER_RIGHT, REEF_EDGE_THRESHOLD
from src.utils.mask_utils import MASK_WIDTH, MASK_HEIGHT, SCALE_X, SCALE_Y

REFUGE_CORE_BINS_X = 24
REFUGE_CORE_BINS_Y = 14
REFUGE_CORE_MASS = 0.50


def fill_no_fish_frames(arr: np.ndarray, lookback: int = 2) -> np.ndarray:
    """Fill NaN entries via forward-fill within a *lookback*-frame window.

    Uses pandas ffill for speed; if *lookback* is finite the result is limited
    to at most *lookback* consecutive filled values (matching the old semantics).
    """
    s = pd.Series(arr)
    return s.ffill(limit=lookback).values


def compute_refuge_core_occupancy(
    all_xy: np.ndarray,
    valid: np.ndarray,
    reference_mask: np.ndarray,
    bins_x: int = REFUGE_CORE_BINS_X,
    bins_y: int = REFUGE_CORE_BINS_Y,
    core_mass: float = REFUGE_CORE_MASS,
) -> np.ndarray:
    """
    Learn a refuge core from reference-frame occupancy and score each frame by
    the fraction of fish that remain inside that core.

    The refuge core is defined as the smallest set of 2-D histogram bins that
    contains ``core_mass`` of all reference detections. This avoids anchoring
    the metric to a manually selected shelter point or radius.
    """
    ref_valid = valid & reference_mask[:, np.newaxis]
    ref_pts = all_xy[ref_valid]
    if len(ref_pts) < 10:
        return np.full(all_xy.shape[0], np.nan)

    x_edges = np.linspace(0.0, 3840.0, bins_x + 1)
    y_edges = np.linspace(0.0, 2160.0, bins_y + 1)
    hist, _, _ = np.histogram2d(
        ref_pts[:, 0],
        ref_pts[:, 1],
        bins=[x_edges, y_edges],
    )

    flat = hist.ravel()
    positive = flat > 0
    if not np.any(positive):
        return np.full(all_xy.shape[0], np.nan)

    order = np.argsort(flat)[::-1]
    cum = np.cumsum(flat[order])
    cutoff = core_mass * flat.sum()
    keep_n = max(1, int(np.searchsorted(cum, cutoff, side="left") + 1))
    core_flat = np.zeros_like(flat, dtype=bool)
    core_flat[order[:keep_n]] = True
    core_mask = core_flat.reshape(bins_x, bins_y)

    xs = np.clip(all_xy[:, :, 0], 0.0, 3839.999)
    ys = np.clip(all_xy[:, :, 1], 0.0, 2159.999)
    x_idx = np.clip(np.digitize(xs, x_edges) - 1, 0, bins_x - 1)
    y_idx = np.clip(np.digitize(ys, y_edges) - 1, 0, bins_y - 1)

    in_core = np.zeros(valid.shape, dtype=bool)
    valid_idx = np.where(valid)
    in_core[valid_idx] = core_mask[x_idx[valid_idx], y_idx[valid_idx]]

    fish_counts = valid.sum(axis=1).astype(float)
    with np.errstate(all="ignore"):
        return np.where(fish_counts > 0, in_core.sum(axis=1) / fish_counts, np.nan)


# ── Vectorised batch computation ──────────────────────────────────────────────

def compute_all_metrics_from_df(
    df: pd.DataFrame,
    shelters: list[tuple[int, int]],
    ocean_mask: np.ndarray,
    reef_map: np.ndarray,
    rolling_window: int = 5,
    reference_mask: np.ndarray | None = None,
    signed_reef_map: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Compute every metric for all frames in *df* and return a dict of smoothed
    arrays keyed by metric name.

    All fish positions are extracted once using vectorised pandas string ops
    into a (n_frames, max_fish, 2) array.  Most metrics are then pure numpy
    with no Python loops over frames.  Only NND, convex-hull, position-jitter,
    y-skewness, spatial-entropy, and centroid-speed still use per-frame loops
    (their geometry is inherently sequential or irregular-shaped).

    Parameters
    ----------
    df              : normalised per-frame DataFrame (output of auto_read_csv)
    shelters        : list of (x, y) shelter coordinates for this GoPro
    ocean_mask      : 2-D boolean array (MASK_HEIGHT × MASK_WIDTH)
    reef_map        : 2-D float array of reef-distance values
    rolling_window  : smoothing window (in frames)
    reference_mask  : bool array marking frames used to learn data-driven refuge
                      occupancy. If None, the metric is omitted.
    signed_reef_map : 2-D float array where ocean pixels are positive (distance
                      to reef edge) and reef pixels are negative (negative distance
                      to ocean edge).  When provided, computes
                      ``signed_reef_distance`` in addition to the standard
                      ``reef_distance``.  If None, the metric is omitted.
    """
    from src.utils.csv_utils import loc_cols as _loc_cols, size_cols as _size_cols

    lc = _loc_cols(df)
    sc = _size_cols(df)
    n  = len(df)

    if not shelters:
        shelters = [SHELTER_LEFT, SHELTER_RIGHT]
    shelters_arr = np.array(shelters, dtype=float).reshape(-1, 2)  # (S, 2)

    def _smooth(arr: np.ndarray) -> np.ndarray:
        return (
            pd.Series(arr)
            .rolling(rolling_window, center=True, min_periods=1)
            .mean()
            .values
        )

    # ── Vectorised position extraction ────────────────────────────────
    # Parse all "(cx,cy)" strings at once per column using regex.
    # Result: all_xy shape (n_frames, n_fish_slots, 2), NaN where no fish.
    J = len(lc)
    all_xy = np.full((n, J, 2), np.nan)
    for j, col in enumerate(lc):
        parsed = df[col].astype(str).str.extract(r'\(([^,]+),([^)]+)\)', expand=True)
        all_xy[:, j, 0] = pd.to_numeric(parsed[0], errors='coerce').values
        all_xy[:, j, 1] = pd.to_numeric(parsed[1], errors='coerce').values

    valid = ~np.isnan(all_xy[:, :, 0])          # (n, J) bool — fish present
    fish_counts_raw = valid.sum(axis=1).astype(float)  # (n,)

    arrays: dict[str, np.ndarray] = {}

    # ── fish_count ────────────────────────────────────────────────────
    arrays["fish_count"] = _smooth(df["fish_count"].values.astype(float))

    # ── vertical_position — fully vectorised ──────────────────────────
    ys_masked = np.where(valid, all_xy[:, :, 1], np.nan)
    with np.errstate(all="ignore"):
        arrays["vertical_position"] = _smooth(np.nanmean(ys_masked, axis=1))

    # ── shelter_distance — fully vectorised ───────────────────────────
    # all_xy: (n, J, 2)  shelters_arr: (S, 2)
    # diffs: (n, J, S, 2)
    xy_exp = all_xy[:, :, np.newaxis, :]            # (n, J, 1, 2)
    sh_exp = shelters_arr[np.newaxis, np.newaxis, :, :]  # (1, 1, S, 2)
    dists_to_shelters = np.sqrt(((xy_exp - sh_exp) ** 2).sum(axis=3))  # (n, J, S)
    min_shelter_dist  = dists_to_shelters.min(axis=2)                   # (n, J)
    min_shelter_dist  = np.where(valid, min_shelter_dist, np.nan)
    with np.errstate(all="ignore"):
        sd_arr = np.nanmean(min_shelter_dist, axis=1)   # (n,)
    sd_arr = fill_no_fish_frames(sd_arr)
    arrays["shelter_distance"] = _smooth(sd_arr)

    # ── school_spread — fully vectorised ──────────────────────────────
    xs_masked = np.where(valid, all_xy[:, :, 0], np.nan)
    with np.errstate(all="ignore"):
        std_x = np.nanstd(xs_masked, axis=1)
        std_y = np.nanstd(ys_masked, axis=1)
    spread = np.where(fish_counts_raw >= 2, (std_x + std_y) / 2.0, np.nan)
    arrays["school_spread"] = _smooth(spread)

    # ── shelter_clustering — fully vectorised ─────────────────────────
    CLUSTER_THRESHOLD = 400.0
    near = np.where(valid, min_shelter_dist < CLUSTER_THRESHOLD, False)  # (n, J)
    with np.errstate(all="ignore"):
        sclust = np.where(fish_counts_raw > 0,
                          near.sum(axis=1) / fish_counts_raw.clip(1), np.nan)
    arrays["shelter_clustering"] = _smooth(sclust)

    # ── refuge_core_occupancy — baseline-derived spatial refuge ───────────
    if reference_mask is not None:
        refuge_occ = compute_refuge_core_occupancy(
            all_xy,
            valid,
            np.asarray(reference_mask, dtype=bool),
        )
        arrays["refuge_core_occupancy"] = _smooth(refuge_occ)

    # ── mask-based metrics — fully vectorised ─────────────────────────
    # Scale 4K coordinates to mask coordinates
    scaled_x = np.where(valid, all_xy[:, :, 0] / SCALE_X, 0.0)
    scaled_y = np.where(valid, all_xy[:, :, 1] / SCALE_Y, 0.0)
    mxs_safe = np.clip(scaled_x.astype(int), 0, MASK_WIDTH  - 1)
    mys_safe = np.clip(scaled_y.astype(int), 0, MASK_HEIGHT - 1)

    in_ocean_all = ocean_mask[mys_safe, mxs_safe]  # (n, J) bool
    in_ocean_all = in_ocean_all & valid             # mask invalid slots

    rdists_all = reef_map[mys_safe, mxs_safe].astype(float)  # (n, J)
    rdists_all = np.where(valid, rdists_all, np.nan)

    fish_in_ocean = in_ocean_all.sum(axis=1).astype(float)
    with np.errstate(all="ignore"):
        vacancy = np.where(fish_counts_raw > 0,
                           fish_in_ocean / fish_counts_raw.clip(1), np.nan)
        reef_dist_arr = np.nanmean(rdists_all, axis=1)
        # exposure: reef-dist for ocean fish, 0 for reef fish
        exposure_vals = np.where(in_ocean_all, rdists_all,
                                 np.where(valid, 0.0, np.nan))
        exposure_arr  = np.nanmean(exposure_vals, axis=1)

    edge_mask = valid & (rdists_all > 0) & (rdists_all <= REEF_EDGE_THRESHOLD)
    reef_edge_arr = edge_mask.sum(axis=1).astype(float)

    arrays["fish_in_ocean"]   = _smooth(fish_in_ocean)
    arrays["shelter_vacancy"] = _smooth(vacancy)
    arrays["reef_distance"]   = _smooth(reef_dist_arr)
    arrays["exposure_index"]  = _smooth(exposure_arr)
    arrays["reef_edge_count"] = _smooth(reef_edge_arr)

    # ── signed_reef_distance (optional) ──────────────────────────────
    # Positive = fish in open ocean (distance to reef boundary).
    # Negative = fish on reef (negative distance to ocean boundary).
    if signed_reef_map is not None:
        sr_vals = signed_reef_map[mys_safe, mxs_safe].astype(float)  # (n, J)
        sr_vals = np.where(valid, sr_vals, np.nan)
        with np.errstate(all="ignore"):
            signed_reef_arr = np.nanmean(sr_vals, axis=1)
        arrays["signed_reef_distance"] = _smooth(signed_reef_arr)


    if sc:
        size_mat = np.full((n, len(sc)), np.nan)
        for k, col in enumerate(sc):
            size_mat[:, k] = pd.to_numeric(df[col], errors="coerce").values
        size_mat[size_mat <= 0] = np.nan
        size_valid = ~np.isnan(size_mat)
        size_count = size_valid.sum(axis=1)
        with np.errstate(all="ignore"):
            arrays["mean_fish_size"]   = _smooth(np.nanmean(size_mat, axis=1))
            arrays["median_fish_size"] = _smooth(np.nanmedian(size_mat, axis=1))
            arrays["total_fish_size"]  = _smooth(np.nansum(size_mat, axis=1))
            size_mean = np.nanmean(size_mat, axis=1)
            size_std  = np.nanstd(size_mat,  axis=1)
            cv = np.where((size_count >= 2) & (size_mean > 0),
                          size_std / size_mean, np.nan)
            arrays["fish_size_cv"] = _smooth(cv)

    # ── derived ───────────────────────────────────────────────────────
    arrays["total_displacement"] = arrays["shelter_distance"] * arrays["fish_count"]

    # ── Remaining per-frame loops (irregular geometry) ────────────────
    # Build list of valid position arrays once for the loop-based metrics.
    # Avoids repeated string parsing — uses the already-parsed all_xy array.
    all_pos: list[np.ndarray] = []
    for i in range(n):
        mask_i = valid[i]
        pts = all_xy[i][mask_i]  # (N_i, 2)
        all_pos.append(pts)

    # ── nnd ───────────────────────────────────────────────────────────
    nnd_arr = np.full(n, np.nan)
    for i, pts in enumerate(all_pos):
        if len(pts) >= 2:
            D = cdist(pts, pts)
            np.fill_diagonal(D, np.inf)
            nnd_arr[i] = float(D.min(axis=1).mean())
    arrays["nnd"] = _smooth(nnd_arr)

    # ── convex_hull ───────────────────────────────────────────────────
    hull_arr = np.full(n, np.nan)
    for i, pts in enumerate(all_pos):
        if len(pts) >= 3:
            try:
                hull_arr[i] = float(ConvexHull(pts).volume)
            except Exception:
                pass
    arrays["convex_hull"] = _smooth(hull_arr)

    # ── position_jitter ───────────────────────────────────────────────
    jitter = np.full(n, np.nan)
    for i in range(1, n):
        curr, prev = all_pos[i], all_pos[i - 1]
        if len(curr) >= 2 and len(prev) >= 2:
            jitter[i] = float(cdist(curr, prev).min(axis=1).mean())
    arrays["position_jitter"] = _smooth(jitter)

    # ── y_skewness ────────────────────────────────────────────────────
    yskew = np.full(n, np.nan)
    for i, pts in enumerate(all_pos):
        if len(pts) >= 3:
            ys = pts[:, 1]
            m, s = np.mean(ys), np.std(ys)
            yskew[i] = float(np.mean(((ys - m) / s) ** 3)) if s >= 1e-6 else 0.0
    arrays["y_skewness"] = _smooth(yskew)

    # ── spatial_entropy ───────────────────────────────────────────────
    N_BINS = 8
    sent = np.full(n, np.nan)
    for i, pts in enumerate(all_pos):
        if len(pts) < 2:
            continue
        xs = np.clip(pts[:, 0] / 3840, 0, 1)
        ys = np.clip(pts[:, 1] / 2160, 0, 1)
        hist, _, _ = np.histogram2d(xs, ys, bins=N_BINS, range=[[0, 1], [0, 1]])
        flat = hist.flatten()
        flat = flat[flat > 0]
        sent[i] = float(sp_entropy(flat / flat.sum(), base=2)) if len(flat) else 0.0
    arrays["spatial_entropy"] = _smooth(sent)

    # ── centroid_speed ────────────────────────────────────────────────
    cspeed = np.full(n, np.nan)
    for i in range(1, n):
        curr, prev = all_pos[i], all_pos[i - 1]
        if len(curr) and len(prev):
            cspeed[i] = float(np.linalg.norm(
                np.mean(curr, axis=0) - np.mean(prev, axis=0)
            ))
    arrays["centroid_speed"] = _smooth(cspeed)

    return arrays
