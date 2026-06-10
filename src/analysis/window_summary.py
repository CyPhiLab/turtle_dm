"""
Window assignment, per-transect median summaries, and delta computation.

Windows are defined on normalised transect progress (0 = entry, 1 = exit):
    start:  [0.000, 0.250)
    middle: [0.375, 0.625]
    end:    (0.750, 1.000]

Frames outside these ranges are excluded from the primary analysis.
"""

import pandas as pd


_WINDOWS = {
    "start":  (0.000, 0.250),
    "middle": (0.375, 0.625),
    "end":    (0.750, 1.000),
}


def assign_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'window' column; rows outside defined windows are dropped."""
    conditions = []
    for name, (lo, hi) in _WINDOWS.items():
        mask = (df["progress"] >= lo) & (df["progress"] <= hi)
        conditions.append(df.loc[mask].assign(window=name))
    return pd.concat(conditions, ignore_index=True)


def compute_window_summaries(df_windowed: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(transect, condition, metric, window) median.

    Returns a tidy DataFrame with columns:
        transect_id, condition, metric_name, window, value_summary
    """
    return (
        df_windowed
        .groupby(
            ["transect_id", "condition", "metric_name", "window"],
            observed=True,
        )["value"]
        .median()
        .reset_index(name="value_summary")
    )


def compute_transect_deltas(window_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot window summaries to wide format and compute the response contrast.

    delta_middle_vs_ends = middle - 0.5 * (start + end)

    Transects missing any of the three windows for a given metric are excluded
    from that metric's analysis; exclusions are logged to the console.

    Returns a DataFrame with columns:
        transect_id, condition, metric_name,
        start, middle, end, delta_middle_vs_ends
    """
    wide = (
        window_df
        .pivot_table(
            index=["transect_id", "condition", "metric_name"],
            columns="window",
            values="value_summary",
            observed=True,
        )
        .reset_index()
    )
    wide.columns.name = None

    n_before = len(wide)
    incomplete = wide[wide[["start", "middle", "end"]].isna().any(axis=1)]
    if len(incomplete):
        for _, row in incomplete.iterrows():
            missing = [w for w in ("start", "middle", "end") if pd.isna(row[w])]
            print(
                f"  Excluding {row['transect_id']} / {row['metric_name']}: "
                f"missing window(s) {missing}"
            )

    wide = wide.dropna(subset=["start", "middle", "end"])
    n_after = len(wide)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after} incomplete transect-metric rows.")

    wide["delta_middle_vs_ends"] = wide["middle"] - 0.5 * (wide["start"] + wide["end"])
    return wide
