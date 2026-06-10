"""
Statistical tests on per-transect delta_middle_vs_ends values.

Primary analysis:
    auv_minus_turtle_delta      Welch t-test + permutation test (BH-corrected across metrics)

Secondary:
    turtle_delta_vs_zero        one-sample t-test
    auv_delta_vs_zero           one-sample t-test

Exploratory:
    diver_delta_vs_zero         one-sample t-test
    diver_minus_turtle_delta    Welch t-test
    diver_minus_auv_delta       Welch t-test

Nonparametric sensitivity checks are computed alongside each parametric test.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ci_mean(vals: np.ndarray, alpha: float = 0.05):
    """95% CI for the mean via t-distribution."""
    n = len(vals)
    if n < 2:
        return np.nan, np.nan
    se  = stats.sem(vals)
    h   = se * stats.t.ppf(1 - alpha / 2, df=n - 1)
    return float(np.mean(vals) - h), float(np.mean(vals) + h)


def _ci_difference(x: np.ndarray, y: np.ndarray, alpha: float = 0.05):
    """95% CI for mean(x) - mean(y) using Welch SE."""
    nx, ny  = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan, np.nan
    vx, vy  = np.var(x, ddof=1), np.var(y, ddof=1)
    se      = np.sqrt(vx / nx + vy / ny)
    df      = (vx / nx + vy / ny) ** 2 / (
                  (vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)
              )
    h       = se * stats.t.ppf(1 - alpha / 2, df=df)
    diff    = float(np.mean(x) - np.mean(y))
    return diff - h, diff + h


def _permutation_test(x: np.ndarray, y: np.ndarray, n_perm: int = 10_000, seed: int = 0):
    """Two-sided permutation test for mean(x) - mean(y)."""
    rng      = np.random.default_rng(seed)
    x, y     = np.asarray(x), np.asarray(y)
    observed = np.mean(x) - np.mean(y)
    pooled   = np.concatenate([x, y])
    nx       = len(x)
    permuted = np.array([
        np.mean(rng.permutation(pooled)[:nx]) - np.mean(rng.permutation(pooled)[nx:])
        for _ in range(n_perm)
    ])
    p = float(np.mean(np.abs(permuted) >= abs(observed)))
    return observed, p


# ── One-sample tests ──────────────────────────────────────────────────────────

def one_sample_delta_tests(delta_df: pd.DataFrame) -> pd.DataFrame:
    """One-sample t-test of delta_middle_vs_ends vs 0, per condition × metric.

    Wilcoxon signed-rank test included as a sensitivity check.
    """
    rows = []
    test_map = {
        "turtle": "turtle_delta_vs_zero",
        "auv":    "auv_delta_vs_zero",
        "diver":  "diver_delta_vs_zero",
    }
    priority = {
        "turtle_delta_vs_zero": "secondary",
        "auv_delta_vs_zero":    "secondary",
        "diver_delta_vs_zero":  "exploratory",
    }

    for metric in delta_df["metric_name"].unique():
        for condition, test_name in test_map.items():
            vals = (
                delta_df.loc[
                    (delta_df["metric_name"] == metric) &
                    (delta_df["condition"] == condition),
                    "delta_middle_vs_ends",
                ]
                .dropna()
                .values
            )
            n = len(vals)
            if n < 2:
                continue

            t_stat, p_val = stats.ttest_1samp(vals, popmean=0.0)
            ci_lo, ci_hi  = _ci_mean(vals)

            # Wilcoxon (sensitivity check)
            try:
                _, p_wilcox = stats.wilcoxon(vals)
            except ValueError:
                p_wilcox = np.nan

            rows.append({
                "metric_name":          metric,
                "test_name":            test_name,
                "condition_1":          condition,
                "condition_2":          None,
                "n_1":                  n,
                "n_2":                  None,
                "mean_delta_1":         float(np.mean(vals)),
                "mean_delta_2":         None,
                "median_delta_1":       float(np.median(vals)),
                "median_delta_2":       None,
                "effect_size":          float(np.mean(vals)),
                "ci_lower":             ci_lo,
                "ci_upper":             ci_hi,
                "t_statistic":          float(t_stat),
                "p_value":              float(p_val),
                "p_value_fdr":          np.nan,
                "p_wilcoxon":           float(p_wilcox) if not np.isnan(p_wilcox) else np.nan,
                "test_type":            "one_sample_t + wilcoxon_sensitivity",
                "primary_or_exploratory": priority[test_name],
                "notes":                "diver n=4; interpret cautiously" if condition == "diver" else "",
            })

    return pd.DataFrame(rows)


# ── Two-sample tests ──────────────────────────────────────────────────────────

def between_condition_tests(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Welch t-test + Mann-Whitney + permutation test for each condition pair × metric.

    Pairs: auv vs turtle (primary), diver vs turtle, diver vs auv (exploratory).
    """
    rows = []
    pairs = [
        ("auv",   "turtle", "auv_minus_turtle_delta",   "primary"),
        ("diver", "turtle", "diver_minus_turtle_delta",  "exploratory"),
        ("diver", "auv",    "diver_minus_auv_delta",     "exploratory"),
    ]

    for metric in delta_df["metric_name"].unique():
        mdf = delta_df[delta_df["metric_name"] == metric]

        for cond1, cond2, test_name, priority in pairs:
            x = mdf.loc[mdf["condition"] == cond1, "delta_middle_vs_ends"].dropna().values
            y = mdf.loc[mdf["condition"] == cond2, "delta_middle_vs_ends"].dropna().values

            if len(x) < 2 or len(y) < 2:
                continue

            t_stat, p_t   = stats.ttest_ind(x, y, equal_var=False)
            _, p_mwu       = stats.mannwhitneyu(x, y, alternative="two-sided")
            _, p_perm      = _permutation_test(x, y)
            ci_lo, ci_hi   = _ci_difference(x, y)

            rows.append({
                "metric_name":            metric,
                "test_name":              test_name,
                "condition_1":            cond1,
                "condition_2":            cond2,
                "n_1":                    len(x),
                "n_2":                    len(y),
                "mean_delta_1":           float(np.mean(x)),
                "mean_delta_2":           float(np.mean(y)),
                "median_delta_1":         float(np.median(x)),
                "median_delta_2":         float(np.median(y)),
                "effect_size":            float(np.mean(x) - np.mean(y)),
                "ci_lower":               float(ci_lo),
                "ci_upper":               float(ci_hi),
                "t_statistic":            float(t_stat),
                "p_value":                float(p_t),
                "p_value_fdr":            np.nan,
                "p_wilcoxon":             np.nan,
                "p_mann_whitney":         float(p_mwu),
                "p_permutation":          float(p_perm),
                "test_type":              "welch_t + mann_whitney + permutation",
                "primary_or_exploratory": priority,
                "notes":                  "diver n=4; interpret cautiously" if "diver" in (cond1, cond2) else "",
            })

    return pd.DataFrame(rows)


# ── FDR correction ────────────────────────────────────────────────────────────

def apply_fdr(results_df: pd.DataFrame) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction to primary tests.

    Only auv_minus_turtle_delta rows are corrected; all others retain NaN.
    """
    primary_mask = results_df["test_name"] == "auv_minus_turtle_delta"
    p_primary    = results_df.loc[primary_mask, "p_value"].values

    if len(p_primary) > 0:
        _, p_fdr, _, _ = multipletests(p_primary, alpha=0.05, method="fdr_bh")
        results_df.loc[primary_mask, "p_value_fdr"] = p_fdr

    return results_df
