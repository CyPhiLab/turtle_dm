"""
CSV loading utilities for fish detection data.

Fish detection CSVs use variable-width rows — rows can have more fish
location columns than the header defines, so we compute the true max
field count before parsing.
"""

from pathlib import Path

import pandas as pd


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
                     on_bad_lines="warn", low_memory=False).copy()
    df["frame_num"] = df["frame_ID"].str.extract(r"(\d+)")[0].astype(int)
    return df.sort_values("frame_num").reset_index(drop=True)


def loc_cols(df: pd.DataFrame) -> list[str]:
    """Return all fish_location_* column names present in *df*."""
    return [c for c in df.columns if c.startswith("fish_location_")]
