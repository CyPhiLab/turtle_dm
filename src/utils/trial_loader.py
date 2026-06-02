"""
Trial data loaders for fish behavioural analysis.

Public API
----------
load_all_trials_from_timestamps(robot_type, fps=30)
    Load 2024-11-15 GP1 trials from annotated_video/GP1/.

load_diver_trials()
    Load diver transect trials from annotated_video/diver/ (30 FPS bbox format).

load_shark_trials()
    Load shark transect trials from annotated_video/GP1/ (30 FPS).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from config.paths import ANNOTATED_VIDEO_DIR, DATA_DIR
from config.constants import (
    ROLLING_WINDOW,
)
from config.transects import DIVER_TRANSECTS, SHARK_VIDEO_IDS, SHARK_TRANSECT_SECS
from src.utils.mask_utils import get_signed_reef_map
from src.utils.csv_utils import auto_read_csv, loc_cols
from src.utils.fish_metrics import compute_all_metrics_from_df






# ── Timestamp-based loader (2024-11-15 GP1 dataset) ──────────────────────────

def load_all_trials_from_timestamps(robot_type: str, fps: int = 30) -> list:
    """
    Load trials for the 2024-11-15 GP1 dataset.

    fps=30 (default) reads from annotated_video/GP1/ (30 FPS bounding-box CSVs).
    fps=1            reads from data/             (1 FPS legacy per-frame CSVs).

    Uses TURTLE_TRANSECTS / AUV_TRANSECTS from timestamp_loader as the source
    of truth.  dist_frame = starts[0] * fps (transect start in frames),
    end_frame = ends[-1] * fps (transect end, absolute in concatenated timeline).
    A 30-second buffer on each side provides pre/post context.

    Returns a list of trial dicts with the same keys as load_all_trials, plus
    end_frame and baseline_buffer.
    """
    from config.transects import TURTLE_TRANSECTS, AUV_TRANSECTS

    segs       = TURTLE_TRANSECTS if robot_type == "turtle" else AUV_TRANSECTS
    signed_reef_map = get_signed_reef_map("GP1")

    BUFFER_F   = 30 * fps
    rolling_w  = max(1, ROLLING_WINDOW * fps)
    src_dir    = ANNOTATED_VIDEO_DIR / "GP1" if fps != 1 else DATA_DIR

    trials = []
    for seg in segs:
        videos     = seg["videos"]
        starts     = seg["starts"]
        ends       = seg["ends"]
        trial_id   = seg["id"]
        prev_video = seg.get("prev_video")

        all_dfs: list[pd.DataFrame] = []
        video_offset = 0

        # Optional: pre-buffer from the preceding video
        if prev_video is not None and starts[0] * fps < BUFFER_F:
            prev_csv = src_dir / prev_video / "fish_detections.csv"
            if prev_csv.exists():
                prev_df  = auto_read_csv(prev_csv)
                prev_max = int(prev_df["frame_num"].max())
                shortfall = BUFFER_F - starts[0] * fps
                prev_lo   = max(0, prev_max - shortfall + 1)
                sliced    = prev_df[prev_df["frame_num"] >= prev_lo].copy()
                sliced["frame_num"] = sliced["frame_num"] - (prev_max + 1)
                all_dfs.append(sliced)

        for v_idx, (video_id, v_start, v_end) in enumerate(zip(videos, starts, ends)):
            csv_path = src_dir / video_id / "fish_detections.csv"
            if not csv_path.exists():
                break
            df     = auto_read_csv(csv_path)
            buf_lo = max(0, int(v_start * fps) - BUFFER_F) if v_idx == 0 else 0
            buf_hi = (int(v_end * fps) + BUFFER_F
                      if v_idx == len(videos) - 1
                      else int(df["frame_num"].max()))
            sliced = df[(df["frame_num"] >= buf_lo) & (df["frame_num"] <= buf_hi)].copy()
            sliced["frame_num"] = sliced["frame_num"] + video_offset
            all_dfs.append(sliced)
            video_offset += int(df["frame_num"].max()) + 1
        else:
            raw = (pd.concat(all_dfs, ignore_index=True)
                     .sort_values("frame_num")
                     .reset_index(drop=True))
            if not loc_cols(raw):
                continue
            frames     = raw["frame_num"].values
            dist_frame = float(starts[0] * fps)
            last_max   = int(auto_read_csv(src_dir / videos[-1] / "fish_detections.csv")["frame_num"].max())
            end_frame  = float(video_offset - (last_max + 1) + ends[-1] * fps)
            arrays = compute_all_metrics_from_df(
                raw,
                signed_reef_map=signed_reef_map,
                rolling_window=rolling_w,
            )
            trials.append({
                "trial_id":   trial_id,
                "robot":      robot_type,
                "dist_frame": dist_frame,
                "end_frame":  end_frame,
                "frames":     frames,
                "arrays":          arrays,
                "n_frames":        len(frames),
            })
            continue
        # for-loop broke early → skip this segment (missing video)
    return trials
def load_diver_trials() -> list:
    """Load diver transect trials from annotated_video/diver/ at 30 FPS.

    Uses DIVER_TRANSECTS as the source of truth for start/end times, including
    the cross-chapter diver_2 trial. Shelter is fixed at DIVER_SHELTER.
    """
    signed_reef_map = get_signed_reef_map("GP1")
    src_dir = ANNOTATED_VIDEO_DIR / "diver"
    FPS = 30
    BUFFER_F = 40 * FPS
    rolling_w = max(1, ROLLING_WINDOW * FPS)

    trials = []
    if not src_dir.exists():
        return trials

    for seg in DIVER_TRANSECTS:
        videos = seg["videos"]
        starts = seg["starts"]
        ends = seg["ends"]
        trial_id = seg["id"]

        all_dfs: list[pd.DataFrame] = []
        video_offset = 0

        for v_idx, (video_id, v_start, v_end) in enumerate(zip(videos, starts, ends)):
            csv_path = src_dir / video_id / "fish_detections.csv"
            if not csv_path.exists():
                break

            df = auto_read_csv(csv_path)
            buf_lo = max(0, int(v_start * FPS) - BUFFER_F) if v_idx == 0 else 0
            buf_hi = (
                int(v_end * FPS) + BUFFER_F
                if v_idx == len(videos) - 1
                else int(df["frame_num"].max())
            )
            sliced = df[(df["frame_num"] >= buf_lo) & (df["frame_num"] <= buf_hi)].copy()
            sliced["frame_num"] = sliced["frame_num"] + video_offset
            all_dfs.append(sliced)
            video_offset += int(df["frame_num"].max()) + 1
        else:
            raw = (
                pd.concat(all_dfs, ignore_index=True)
                .sort_values("frame_num")
                .reset_index(drop=True)
            )
            if not loc_cols(raw):
                continue

            frames = raw["frame_num"].values
            dist_frame = float(starts[0] * FPS)
            last_max = int(auto_read_csv(src_dir / videos[-1] / "fish_detections.csv")["frame_num"].max())
            end_frame = float(video_offset - (last_max + 1) + ends[-1] * FPS)
            arrays = compute_all_metrics_from_df(
                raw,
                signed_reef_map=signed_reef_map,
                rolling_window=rolling_w,
            )
            trials.append({
                "trial_id":   trial_id,
                "robot":      "diver",
                "dist_frame": dist_frame,
                "end_frame":  end_frame,
                "fps":        FPS,
                "frames":     frames,
                "arrays":     arrays,
                "n_frames":   len(frames),
            })
    return trials


def load_shark_trials(fps: int = 30) -> list:
    """Load shark transect trials from annotated_video/GP1/ at 30 FPS.

    Uses SHARK_TRANSECT_SECS (absolute seconds from video start) to slice a
    ±40 s window around each transect.  dist_frame/end_frame are in 30 FPS
    frame numbers; fps=30 stored in each trial dict.
    """
    signed_reef_map = get_signed_reef_map("GP1")
    BUFFER_F   = 40 * fps          # 40 s buffer at fps
    rolling_w  = max(1, ROLLING_WINDOW * fps)
    src_dir    = ANNOTATED_VIDEO_DIR / "GP1"

    trials = []
    for key, (start_s, end_s) in SHARK_TRANSECT_SECS.items():
        video_id = SHARK_VIDEO_IDS.get(key)
        if video_id is None:
            continue
        csv_path = src_dir / video_id / "fish_detections.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping shark trial {key}")
            continue

        dist_f = float(start_s * fps)
        end_f  = float(end_s   * fps)
        buf_lo = max(0, int(dist_f) - BUFFER_F)
        buf_hi = int(end_f) + BUFFER_F

        raw = auto_read_csv(csv_path)
        raw = (raw[(raw["frame_num"] >= buf_lo) & (raw["frame_num"] <= buf_hi)]
               .copy()
               .reset_index(drop=True))
        if not loc_cols(raw) or len(raw) < 10:
            continue

        frames = raw["frame_num"].values
        arrays = compute_all_metrics_from_df(
            raw,
            signed_reef_map=signed_reef_map,
            rolling_window=rolling_w,
        )
        trials.append({
            "trial_id":   key,
            "robot":      "shark",
            "dist_frame": dist_f,
            "end_frame":  end_f,
            "fps":        fps,
            "frames":     frames,
            "arrays":     arrays,
            "n_frames":   len(frames),
        })
    return trials
