"""
Orchestrates the full statistical analysis pipeline.

Usage:
    python3 -m analysis.run_analysis

Outputs written to outputs/:
    window_level_metrics.csv
    transect_delta_metrics.csv
    simple_statistical_results.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from src.analysis.preprocess import load_frame_metrics
from src.analysis.window_summary import assign_windows, compute_window_summaries, compute_transect_deltas
from src.analysis.delta_tests import one_sample_delta_tests, between_condition_tests, apply_fdr

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs"


_METRIC_LABELS = {
    "fish_count":        "Fish count",
    "shelter_distance":  "Shelter distance",
    "centroid_speed":    "Centroid speed",
    "nnd":               "Mean NND",
    "spatial_entropy":   "Spatial entropy",
    "vertical_position": "Vertical position",
}


def _write_latex_tables(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Write two LaTeX tables to outputs/:
        table_within_condition.tex   — one-sample delta vs zero per condition
        table_auv_vs_turtle.tex      — primary between-robot comparison
    """
    float_fmt = "{:.2f}"
    p_fmt     = lambda p: "$<0.001$" if p < 0.001 else f"{p:.3f}"

    # ── Within-condition table ────────────────────────────────────────────
    _w = (
        results_df[results_df["test_name"].isin(
            ["turtle_delta_vs_zero", "auv_delta_vs_zero", "diver_delta_vs_zero"]
        )]
        .sort_values(["condition_1", "p_value"])
    )
    tbl_within = pd.DataFrame({
        "Condition": pd.Categorical(
            _w["condition_1"].str.capitalize().values,
            categories=["Auv", "Diver", "Turtle"], ordered=True,
        ),
        "Metric":    _w["metric_name"].map(_METRIC_LABELS).values,
        "$\\Delta$": _w["mean_delta_1"].values,
        "95\\% CI":  [f"[{r.ci_lower:.2f}, {r.ci_upper:.2f}]" for r in _w.itertuples()],
        "$p$":       _w["p_value"].values,
    }).set_index(["Condition", "Metric"])
    tex_within = tbl_within.style.format({
        "$\\Delta$": float_fmt,
        "$p$":       p_fmt,
    }).to_latex(
        multirow_align="c",
        hrules=True,
        caption="Within-condition response contrast "
                "($\\Delta = x_{\\text{middle}} - \\tfrac{1}{2}(x_{\\text{start}} + x_{\\text{end}})$).",
        label="tab:within_condition",
    )
    path_within = output_dir / "table_within_condition.tex"
    path_within.write_text(tex_within)
    print(f"  Saved: {path_within}")

    # ── AUV vs Turtle table ───────────────────────────────────────────────
    _p = (
        results_df[results_df["test_name"] == "auv_minus_turtle_delta"]
        .sort_values("p_value")
    )
    tbl_primary = pd.DataFrame({
        "Metric":              _p["metric_name"].map(_METRIC_LABELS).values,
        "AUV $\\Delta$":      _p["mean_delta_1"].values,
        "Turtle $\\Delta$":   _p["mean_delta_2"].values,
        "Difference":         _p["effect_size"].values,
        "95\\% CI":           [f"[{r.ci_lower:.2f}, {r.ci_upper:.2f}]" for r in _p.itertuples()],
        "$p$":                _p["p_value"].values,
        "$p_{\\text{FDR}}$": _p["p_value_fdr"].values,
    }).set_index("Metric")
    tex_primary = tbl_primary.style.format({
        "AUV $\\Delta$":      float_fmt,
        "Turtle $\\Delta$":   float_fmt,
        "Difference":         float_fmt,
        "$p$":                p_fmt,
        "$p_{\\text{FDR}}$": p_fmt,
    }).to_latex(
        hrules=True,
        caption="AUV versus turtle robot: difference in middle-vs-ends response contrast. "
                "Difference $=$ AUV $\\Delta -$ Turtle $\\Delta$. "
                "$p_{\\text{FDR}}$: Benjamini--Hochberg corrected.",
        label="tab:auv_vs_turtle",
    )
    path_primary = output_dir / "table_auv_vs_turtle.tex"
    path_primary.write_text(tex_primary)
    print(f"  Saved: {path_primary}")


def run_all(output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load frame-level metrics ───────────────────────────────────────
    print("Loading trials...")
    frame_df = load_frame_metrics()
    print(f"  {len(frame_df):,} frame-metric rows loaded.\n")

    # ── 2. Assign windows and compute per-transect summaries ──────────────
    print("Assigning windows...")
    windowed     = assign_windows(frame_df)
    window_df    = compute_window_summaries(windowed)
    window_path  = output_dir / "window_level_metrics.csv"
    window_df.to_csv(window_path, index=False)
    print(f"  Saved: {window_path}")

    # ── 3. Compute per-transect deltas ────────────────────────────────────
    print("\nComputing transect deltas...")
    delta_df   = compute_transect_deltas(window_df)
    delta_path = output_dir / "transect_delta_metrics.csv"
    delta_df.to_csv(delta_path, index=False)
    print(f"  Saved: {delta_path}")

    # Sanity-check transect counts per condition × metric
    counts = (
        delta_df.groupby(["condition", "metric_name"], observed=True)["transect_id"]
        .nunique()
        .unstack("metric_name")
    )
    print("\n  Transects per condition × metric:")
    print(counts.to_string())

    # ── 4. Statistical tests ──────────────────────────────────────────────
    print("\nRunning statistical tests...")
    one_sample  = one_sample_delta_tests(delta_df)
    two_sample  = between_condition_tests(delta_df)
    results_df  = pd.concat([one_sample, two_sample], ignore_index=True)
    results_df  = apply_fdr(results_df)

    # Enforce consistent column order from spec
    col_order = [
        "metric_name", "test_name", "condition_1", "condition_2",
        "n_1", "n_2",
        "mean_delta_1", "mean_delta_2",
        "median_delta_1", "median_delta_2",
        "effect_size", "ci_lower", "ci_upper",
        "t_statistic", "p_value", "p_value_fdr",
        "p_wilcoxon", "p_mann_whitney", "p_permutation",
        "test_type", "primary_or_exploratory", "notes",
    ]
    for col in col_order:
        if col not in results_df.columns:
            results_df[col] = None
    results_df = results_df[col_order]

    results_path = output_dir / "simple_statistical_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"  Saved: {results_path}")

    # ── 5. Print summary of primary results ──────────────────────────────
    primary = results_df[results_df["test_name"] == "auv_minus_turtle_delta"].copy()
    primary = primary.sort_values("p_value")
    print("\nPrimary results (AUV vs Turtle, sorted by p-value):")
    print(
        primary[["metric_name", "effect_size", "ci_lower", "ci_upper",
                  "p_value", "p_value_fdr"]]
        .to_string(index=False, float_format="{:.4f}".format)
    )

    # ── 6. Print within-condition results ─────────────────────────────────
    within = results_df[results_df["test_name"].isin(
        ["turtle_delta_vs_zero", "auv_delta_vs_zero", "diver_delta_vs_zero"]
    )].copy()
    within = within.sort_values(["condition_1", "p_value"])
    print("\nWithin-condition results (delta vs zero, sorted by condition then p-value):")
    print(
        within[["condition_1", "metric_name", "mean_delta_1", "ci_lower", "ci_upper", "p_value"]]
        .rename(columns={"condition_1": "condition", "mean_delta_1": "mean_delta"})
        .to_string(index=False, float_format="{:.4f}".format)
    )

    # ── 7. Write LaTeX tables ─────────────────────────────────────────────
    _write_latex_tables(results_df, output_dir)

    print(f"\nDone. All outputs in {output_dir}")


if __name__ == "__main__":
    run_all()
