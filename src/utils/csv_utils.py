"""
CSV loading utilities for fish detection data.

Fish detection CSVs use variable-width rows — rows can have more fish
location columns than the header defines, so we compute the true max
field count before parsing.
"""

from pathlib import Path

import warnings
import numpy as np
import pandas as pd


def parse_location(loc_str: str) -> tuple[float, float] | None:
    """Parse a '(x,y)' string into a (float, float) tuple, or None on failure."""
    try:
        x, y = str(loc_str).strip("() ").split(",")
        return float(x), float(y)
    except Exception:
        return None


def read_fish_csv(csv_path: Path | str) -> pd.DataFrame:
    """
    Read a fish_detections.csv with variable-width rows into a DataFrame.

    Handles rows that contain more fish location columns than the header by
    auto-expanding column names.  Adds a 'frame_num' integer column parsed
    from 'frame_ID', and returns rows sorted by frame number.
    """
    csv_path = Path(csv_path)
    with open(csv_path) as f:
        lines = f.readlines()

    if not lines:
        return pd.DataFrame()

    max_fields    = max(len(line.split(",")) for line in lines if line.strip())
    header_fields = len(lines[0].split(","))
    extra         = max_fields - header_fields
    extra_cols    = [f"fish_location_{header_fields - 1 + i}" for i in range(1, extra + 1)]
    col_names     = lines[0].strip().split(",") + extra_cols

    df = pd.read_csv(csv_path, names=col_names, skiprows=1,
                     on_bad_lines="warn", low_memory=False)
    df["frame_num"] = df["frame_ID"].str.extract(r"(\d+)").astype(int)
    return df.sort_values("frame_num").reset_index(drop=True)


def loc_cols(df: pd.DataFrame) -> list[str]:
    """Return all fish_location_* column names present in *df*."""
    return [c for c in df.columns if c.startswith("fish_location_")]


def size_cols(df: pd.DataFrame) -> list[str]:
    """Return all fish_size_px_* column names present in *df*."""
    return [c for c in df.columns if c.startswith("fish_size_px_")]


# ── Bounding-box format (annotated_video / 30-FPS data) ───────────────────────

def read_bbox_csv(csv_path: Path | str) -> pd.DataFrame:
    """
    Read a per-detection bounding-box CSV and normalise it to the same
    per-frame schema used by read_fish_csv.

    Input columns:  frame, x1, y1, x2, y2, confidence, class_id, class_name, track_id
    Output columns: frame_ID (frame_NNNN), fish_count, fish_location_1..N (centroid "(cx,cy)"),
                    fish_size_px_1..N (bbox area), frame_num

    Detections for each frame are sorted by descending confidence so that
    fish_location_1 is always the highest-confidence detection.
    """
    csv_path = Path(csv_path)
    raw = pd.read_csv(csv_path)

    # Compute centroid and bbox area per detection
    raw["cx"] = (raw["x1"] + raw["x2"]) / 2.0
    raw["cy"] = (raw["y1"] + raw["y2"]) / 2.0
    raw["area"] = (raw["x2"] - raw["x1"]) * (raw["y2"] - raw["y1"])

    # Sort by frame then descending confidence so slot-1 = best detection
    raw = raw.sort_values(["frame", "confidence"], ascending=[True, False])

    rows = []
    for frame_num, grp in raw.groupby("frame", sort=True):
        n = len(grp)
        row: dict = {
            "frame_ID":   f"frame_{int(frame_num):04d}",
            "frame_num":  int(frame_num),
            "fish_count": n,
        }
        for i, (_, det) in enumerate(grp.iterrows(), start=1):
            row[f"fish_location_{i}"] = f"({det['cx']:.2f},{det['cy']:.2f})"
            row[f"fish_size_px_{i}"]  = det["area"]
        rows.append(row)

    return pd.DataFrame(rows).sort_values("frame_num").reset_index(drop=True)


def auto_read_csv(csv_path: Path | str) -> pd.DataFrame:
    """Detect the CSV format from its header and dispatch to the correct reader.

    - Old per-frame format (header starts with 'frame_ID') → read_fish_csv
    - New per-detection bbox format (header starts with 'frame,x1') → read_bbox_csv
    """
    csv_path = Path(csv_path)
    with open(csv_path) as f:
        first_line = f.readline()
    if first_line.startswith("frame_ID"):
        return read_fish_csv(csv_path)
    return read_bbox_csv(csv_path)
