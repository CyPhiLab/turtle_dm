# Simplified First-Pass Instructions for Fish Disturbance Statistics in Python

## Goal

Implement a simple, defensible first-pass statistical analysis for comparing fish disturbance responses across three conditions:

```text
turtle robot: 17 transects
AUV:          18 transects
diver:         4 transects
```

The diver condition is not a robot condition. Treat it as a small-sample reference or positive-control disturbance condition, not as a fully powered treatment.

The two primary questions are:

```text
1. Does each condition show a stronger response in the middle of the transect than at the ends?

2. Is the middle-transect response different between the turtle robot and the AUV?
```

The primary comparison is turtle versus AUV. Diver comparisons are secondary and should be interpreted cautiously.

The statistical unit is the transect, not the video frame.

Do not test frame-level values as independent samples.

## Expected Input Data

Start with a frame-level table:

```text
transect_id
condition
frame
time_seconds
progress
metric_name
value
```

where:

```text
transect_id: unique transect identifier, e.g. T01, A01, D01
condition: turtle, auv, or diver
frame: video frame index
time_seconds: timestamp in seconds
progress: normalized transect progress from 0 to 1
metric_name: behavioral metric name
value: metric value at that frame
```

If `progress` is missing, compute it within each transect:

```python
df["progress"] = (
    df.groupby("transect_id")["time_seconds"]
      .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
)
```

If robot/diver position is available, use position-based progress instead of time-based progress.

## Define Transect Windows

Assign each frame to one of three windows based on normalized transect progress:

```text
start:  0.000 <= progress < 0.250
middle: 0.375 <= progress <= 0.625
end:    0.750 < progress <= 1.000
```

Frames outside these windows are ignored for the primary analysis.

The gaps reduce sensitivity to transition regions.

Use these windows because they are defined from the transect geometry, not from the observed fish response.

## Summarize Each Transect

For each combination of:

```text
transect_id
condition
metric_name
window
```

compute one scalar summary.

Use the median by default:

```python
window_df = (
    df_windowed
    .groupby(["transect_id", "condition", "metric_name", "window"], observed=True)
    ["value"]
    .median()
    .reset_index(name="value_summary")
)
```

The resulting table should look like:

```text
transect_id   condition   metric_name   window    value_summary
T01           turtle      entropy       start     0.72
T01           turtle      entropy       middle    0.44
T01           turtle      entropy       end       0.69
A01           auv         entropy       start     0.68
A01           auv         entropy       middle    0.31
A01           auv         entropy       end       0.62
D01           diver       entropy       start     0.65
D01           diver       entropy       middle    0.20
D01           diver       entropy       end       0.48
```

Each transect should have one start, one middle, and one end value for each metric.

If a transect is missing one of the three windows for a metric, exclude that transect from that metric’s contrast analysis and log the exclusion.

## Compute the Main Per-Transect Response Contrast

For each transect and metric, compute:

```text
delta = middle - 0.5 * (start + end)
```

This is the middle-versus-ends response.

In Python:

```python
wide = (
    window_df
    .pivot_table(
        index=["transect_id", "condition", "metric_name"],
        columns="window",
        values="value_summary"
    )
    .reset_index()
)

wide = wide.dropna(subset=["start", "middle", "end"])

wide["delta_middle_vs_ends"] = (
    wide["middle"] - 0.5 * (wide["start"] + wide["end"])
)
```

This table is now the main statistical dataset.

Each row is one transect-metric pair.

The sample size for a metric is the number of transects, not the number of frames.

## Interpret the Sign of Delta

The sign of `delta_middle_vs_ends` depends on the metric.

Examples:

```text
spatial entropy:
negative delta means entropy is lower in the middle, consistent with stronger spatial restriction

distance from shelter:
negative delta means fish are closer to shelter in the middle

shelter occupancy:
positive delta means more fish are near/in shelter in the middle

activity or optical flow:
negative delta may indicate freezing-like immobility
positive delta may indicate agitation/escape response

nearest-neighbor distance:
positive delta may indicate scattering
negative delta may indicate tighter shoaling

fish count:
negative delta may indicate fewer visible fish in the middle
```

Do not assume all metrics should have the same sign.

## Primary Statistical Tests

Run tests on `delta_middle_vs_ends`.

### Test 1: Within-Condition Middle Response

For each metric and condition, test whether:

```text
delta_middle_vs_ends differs from 0
```

Use a one-sample t-test as the default.

```python
from scipy import stats

vals = metric_condition_df["delta_middle_vs_ends"].dropna()
t_stat, p_value = stats.ttest_1samp(vals, popmean=0.0)
```

Also compute:

```text
mean delta
median delta
standard deviation
standard error
95% confidence interval
n transects
```

For the diver condition, report this only as descriptive or exploratory because `n = 4`.

### Test 2: Turtle Versus AUV

For each metric, compare the per-transect deltas between turtle and AUV:

```text
delta_auv versus delta_turtle
```

Use Welch’s two-sample t-test as the default:

```python
turtle = metric_df.loc[metric_df["condition"] == "turtle", "delta_middle_vs_ends"]
auv = metric_df.loc[metric_df["condition"] == "auv", "delta_middle_vs_ends"]

t_stat, p_value = stats.ttest_ind(auv, turtle, equal_var=False)
```

The effect size is:

```text
mean(delta_auv) - mean(delta_turtle)
```

Compute a confidence interval for this difference using the Welch standard error.

This is the primary between-robot statistic.

### Test 3: Diver Comparisons

For each metric, optionally compare:

```text
delta_diver versus delta_turtle
delta_diver versus delta_auv
```

Use Welch’s t-test, but mark these as exploratory.

Because the diver sample size is only 4 transects, always show the raw points and confidence intervals.

Do not make strong claims based only on diver p-values.

## Optional Nonparametric Tests

Because sample sizes are modest and some metrics may not be normally distributed, also compute nonparametric tests as sensitivity checks.

Within-condition test against zero:

```python
stats.wilcoxon(vals)
```

Only use Wilcoxon if there are enough nonzero values.

Between-condition test:

```python
stats.mannwhitneyu(auv, turtle, alternative="two-sided")
```

These should be reported as sensitivity checks, not necessarily as the main tests.

For the primary turtle-vs-AUV comparison, a permutation test is also useful.

```python
import numpy as np

def permutation_test_mean_difference(x, y, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)

    observed = np.mean(x) - np.mean(y)
    pooled = np.concatenate([x, y])
    n_x = len(x)

    permuted = np.empty(n_perm)

    for i in range(n_perm):
        rng.shuffle(pooled)
        permuted[i] = np.mean(pooled[:n_x]) - np.mean(pooled[n_x:])

    p = np.mean(np.abs(permuted) >= abs(observed))
    return observed, p
```

Here `x` can be AUV deltas and `y` turtle deltas.

The permutation test is easy to explain and does not rely on normality, but it still assumes transects are exchangeable between groups. Because condition is confounded with order, interpret it cautiously.

## Multiple Comparisons

Apply Benjamini-Hochberg false discovery rate correction across the primary turtle-vs-AUV tests.

```python
from statsmodels.stats.multitest import multipletests

reject, pvals_fdr, _, _ = multipletests(
    pvals,
    alpha=0.05,
    method="fdr_bh"
)
```

Keep diver comparisons separate from the primary FDR correction unless the paper frames them as formal hypothesis tests.

## Recommended Output Tables

Save the window-level summaries:

```text
outputs/window_level_metrics.csv
```

Save the per-transect contrast table:

```text
outputs/transect_delta_metrics.csv
```

Save statistical results:

```text
outputs/simple_statistical_results.csv
```

The statistical result table should include:

```text
metric_name
test_name
condition_1
condition_2
n_1
n_2
mean_delta_1
mean_delta_2
median_delta_1
median_delta_2
effect_size
ci_lower
ci_upper
t_statistic
p_value
p_value_fdr
test_type
primary_or_exploratory
notes
```

Use these test names:

```text
turtle_delta_vs_zero
auv_delta_vs_zero
diver_delta_vs_zero
auv_minus_turtle_delta
diver_minus_turtle_delta
diver_minus_auv_delta
```

Mark:

```text
primary:
auv_minus_turtle_delta

secondary:
turtle_delta_vs_zero
auv_delta_vs_zero

exploratory:
diver_delta_vs_zero
diver_minus_turtle_delta
diver_minus_auv_delta
```

Depending on paper framing, turtle and AUV within-condition tests may also be primary.


## Minimal Python Function Structure

Suggested files:

```text
analysis/
  run_simple_analysis.py
  preprocess.py
  window_summary.py
  delta_tests.py
```

Suggested functions:

```python
def load_frame_metrics(path):
    ...

def add_progress_if_missing(df):
    ...

def assign_windows(df):
    ...

def compute_window_summaries(df):
    ...

def compute_transect_deltas(window_df):
    ...

def one_sample_delta_tests(delta_df):
    ...

def between_condition_delta_tests(delta_df):
    ...

def apply_fdr(results_df):
    ...


def run_all(input_path, output_dir):
    ...
```

## Suggested Analysis Flow

1. Load frame-level metric data.
2. Verify condition labels:
   - turtle
   - auv
   - diver
3. Verify expected transect counts:
   - 17 turtle
   - 18 AUV
   - 4 diver
4. Normalize progress within transect if needed.
5. Assign start/middle/end windows.
6. Compute one window summary per transect, metric, and window.
7. Pivot to wide format.
8. Compute per-transect:
   - `delta_middle_vs_ends`
9. For each metric:
   - test turtle delta against zero
   - test AUV delta against zero
   - test AUV delta versus turtle delta
   - optionally test diver delta against zero
   - optionally test diver versus turtle and diver versus AUV
10. Apply FDR correction to primary turtle-vs-AUV metric tests.
12. Save all tables.


