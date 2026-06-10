"""
Load trial data and convert to a long-format per-frame DataFrame.

Uses trial dicts from load_trials() directly — no re-parsing of CSVs.
Progress is computed from dist_frame / end_frame already present in each dict.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from src.utils.trial_loader import load_trials

METRICS = [
    "fish_count",
    "shelter_distance",
    "centroid_speed",
    "nnd",
    "spatial_entropy",
    "vertical_position",
]

CONDITIONS = ["turtle", "auv", "diver"]

# Expected transect counts — warn but don't crash on mismatch.
_EXPECTED_COUNTS = {"turtle": 17, "auv": 18, "diver": 4}


def load_frame_metrics(conditions: list[str] = CONDITIONS) -> pd.DataFrame:
    """Load all trials and return a long-format per-frame DataFrame.

    Columns:
        transect_id, condition, frame, time_seconds, progress,
        metric_name, value
    """
    rows = []

    for condition in conditions:
        trials = load_trials(condition)

        expected = _EXPECTED_COUNTS.get(condition)
        if expected is not None and len(trials) != expected:
            print(
                f"Warning: expected {expected} {condition} trials, "
                f"got {len(trials)}."
            )
        print(f"  {condition}: {len(trials)} trials loaded")

        for t in trials:
            trial_id  = t["trial_id"]
            fps       = t["fps"]
            frames    = t["frames"]
            dist_f    = t["dist_frame"]
            end_f     = t["end_frame"]

            # Progress over the full buffered clip (0 = clip start, 1 = clip end).
            # With symmetric buffers the transect midpoint lands near 0.5,
            # so the window thresholds align with pre-buffer / passage / post-buffer.
            clip_lo   = float(frames.min())
            clip_hi   = float(frames.max())
            clip_span = clip_hi - clip_lo

            time_seconds = (frames - dist_f) / fps
            progress     = (frames - clip_lo) / clip_span if clip_span > 0 else np.full(len(frames), np.nan)

            for metric in METRICS:
                arr = t["arrays"].get(metric)
                if arr is None:
                    continue
                for i, frame in enumerate(frames):
                    rows.append({
                        "transect_id":  trial_id,
                        "condition":    condition,
                        "frame":        int(frame),
                        "time_seconds": float(time_seconds[i]),
                        "progress":     float(progress[i]),
                        "metric_name":  metric,
                        "value":        float(arr[i]),
                    })

    df = pd.DataFrame(rows)
    df["condition"]   = pd.Categorical(df["condition"],   categories=CONDITIONS, ordered=False)
    df["metric_name"] = pd.Categorical(df["metric_name"], categories=METRICS,     ordered=False)
    return df
