"""
Trial data loaders for fish behavioural analysis.

Public API
----------
load_trials(robot_type, fps=30)
    Load trials for any robot type: 'turtle', 'auv', 'diver', or 'shark'.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from config.paths import ANNOTATED_VIDEO_DIR
from config.constants import ROLLING_WINDOW
from config.transects import (
    TURTLE_TRANSECTS, AUV_TRANSECTS,
    DIVER_TRANSECTS, SHARK_TRANSECTS,
)
from src.utils.mask_utils import get_signed_reef_map
from src.utils.csv_utils import read_fish_csv, loc_cols
from src.utils.fish_metrics import compute_all_metrics_from_df


def _load_trials(segs: list, src_dir: Path, robot: str, buffer_s: int, shelter_pt: tuple | None = None) -> list:
    """
    Shared multi-video concatenation and metric computation for all robot types.

    Each segment in *segs* follows the standard format:
        {"id": str, "videos": [...], "starts": [...], "ends": [...]}
    An optional "prev_video" key triggers a pre-buffer from the preceding clip.
    """
    signed_reef_map = get_signed_reef_map("GP1")
    fps       = 30
    buffer_f  = buffer_s * fps
    rolling_w = max(1, ROLLING_WINDOW * fps)

    trials = []
    for seg in segs:
        videos     = seg["videos"]
        starts     = seg["starts"]
        ends       = seg["ends"]
        trial_id   = seg["id"]
        prev_video = seg.get("prev_video")

        all_dfs: list[pd.DataFrame] = []
        video_offset = 0

        # If the video starts too close to the transect, we pull data from the end of the previous video.
        if prev_video is not None and starts[0] * fps < buffer_f:
            prev_csv = src_dir / prev_video / "fish_detections.csv"
            if prev_csv.exists():
                prev_df  = read_fish_csv(prev_csv)
                prev_max = int(prev_df["frame_num"].max())
                shortfall = buffer_f - starts[0] * fps
                prev_lo   = max(0, prev_max - shortfall + 1)
                sliced    = prev_df[prev_df["frame_num"] >= prev_lo].copy()
                sliced["frame_num"] = sliced["frame_num"] - (prev_max + 1)
                all_dfs.append(sliced)
                print(f"Pre-buffered {len(sliced)} frames from {prev_video} for trial {trial_id!r}.")

        for v_idx, (video_id, v_start, v_end) in enumerate(zip(videos, starts, ends)):
            csv_path = src_dir / video_id / "fish_detections.csv"
            if not csv_path.exists():
                break
            df     = read_fish_csv(csv_path)
            buf_lo = max(0, int(v_start * fps) - buffer_f) if v_idx == 0 else 0
            buf_hi = (int(v_end * fps) + buffer_f
                      if v_idx == len(videos) - 1
                      else int(df["frame_num"].max()))
            sliced = df[(df["frame_num"] >= buf_lo) & (df["frame_num"] <= buf_hi)].copy()
            sliced["frame_num"] = sliced["frame_num"] + video_offset
            all_dfs.append(sliced)
            video_offset += int(df["frame_num"].max()) + 1
        else:
            # All videos loaded successfully
            raw = (pd.concat(all_dfs, ignore_index=True)
                     .sort_values("frame_num")
                     .reset_index(drop=True))
            if not loc_cols(raw) or len(raw) < 10:
                continue
            frames     = raw["frame_num"].values
            dist_frame = float(starts[0] * fps)
            last_max   = int(read_fish_csv(src_dir / videos[-1] / "fish_detections.csv")["frame_num"].max())
            end_frame  = float(video_offset - (last_max + 1) + ends[-1] * fps)
            arrays = compute_all_metrics_from_df(
                raw,
                signed_reef_map=signed_reef_map,
                shelter_pt=shelter_pt,
                rolling_window=rolling_w,
            )
            trials.append({
                "trial_id":   trial_id,
                "robot":      robot,
                "fps":        fps,
                "dist_frame": dist_frame,
                "end_frame":  end_frame,
                "frames":     frames,
                "arrays":     arrays,
                "n_frames":   len(frames),
            })
            continue
        # for-loop broke early (missing video) → skip segment
        print(f"Warning: skipping trial {trial_id!r} — missing CSV for video {video_id!r}.")

    return trials


# ── Public loader ────────────────────────────────────────────────────────────

# (segs, src_dir, buffer_s)  — src_dir is relative to ANNOTATED_VIDEO_DIR
# (segs, subdir, buffer_s, shelter_pt)
_SHELTER_DEFAULT = (1800, 1200)
_SHELTER_DIVER   = (1600, 1100)

_ROBOT_CONFIG = {
    "turtle": (TURTLE_TRANSECTS, "GP1",   30, _SHELTER_DEFAULT),
    "auv":    (AUV_TRANSECTS,    "GP1",   30, _SHELTER_DEFAULT),
    "diver":  (DIVER_TRANSECTS,  "diver", 40, _SHELTER_DIVER),
    "shark":  (SHARK_TRANSECTS,  "GP1",   40, _SHELTER_DEFAULT),
}


def load_trials(robot_type: str) -> list:
    """Load trials for *robot_type* ('turtle', 'auv', 'diver', or 'shark')."""
    if robot_type not in _ROBOT_CONFIG:
        raise ValueError(f"Unknown robot_type {robot_type!r}. Choose from {list(_ROBOT_CONFIG)}.")
    segs, subdir, buffer_s, shelter_pt = _ROBOT_CONFIG[robot_type]
    src_dir = ANNOTATED_VIDEO_DIR / subdir
    if not src_dir.exists():
        return []
    return _load_trials(segs, src_dir, robot_type, buffer_s=buffer_s, shelter_pt=shelter_pt)
