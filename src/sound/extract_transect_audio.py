"""
Extract transect-aligned audio from raw GoPro MP4 files.

For each transect defined in TURTLE_TRANSECTS and AUV_TRANSECTS, this script
extracts audio from both GoPro 1 and GoPro 4 raw videos for the disturbance
window ± 30 s (matching the fish-analysis buffer), applies spectral noise
subtraction, and saves cleaned WAVs to sound/transects/.

The two cameras provide independent audio measurements per transect, keeping
the same dual-GoPro approach used for the original clip-based sound analysis.

Usage:
    python -m src.sound.extract_transect_audio \\
        --gp1-dir /Volumes/Transcend_A/2024-11-USVI/2024-11-15/GoPro1-turtle-dm \\
        --gp4-dir /Volumes/Transcend_A/2024-11-USVI/2024-11-15/GoPro4-turtle-dm

Output files: sound/transects/{trial_id}_{robot_type}_{gp1|gp4}_sound.wav
    e.g.  T1_turtle_gp1_sound.wav,  A3_auv_gp4_sound.wav
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from moviepy import VideoFileClip
from scipy import signal

from config.transects import TURTLE_TRANSECTS, AUV_TRANSECTS, CONTROL_SEGMENTS
from config.paths import TRANSECT_SOUND_DIR

# Seconds of buffer on each side of transect — matches the fish-analysis BUFFER_F.
BUFFER_S = 30.0

# Seconds from the start of each raw video used as the wave-noise reference.
NOISE_REF_S = 20.0

# Video-ID suffix used by each camera in this dataset.
GP1_SUFFIX = "534"
GP4_SUFFIX = "407"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gp4_video_id(gp1_video_id: str) -> str:
    """Derive the GoPro 4 video ID from a GoPro 1 video ID."""
    return gp1_video_id.replace(GP1_SUFFIX, GP4_SUFFIX)


def _find_mp4(video_dir: Path, video_id: str) -> Path | None:
    for ext in ("MP4", "mp4"):
        p = video_dir / f"{video_id}.{ext}"
        if p.exists():
            return p
    return None


def _read_audio_segment(mp4_path: Path, t_start: float, t_end: float) -> tuple[np.ndarray, int]:
    """Extract [t_start, t_end] seconds from *mp4_path*, return (mono_float64, sr)."""
    clip = VideoFileClip(str(mp4_path))
    t0 = max(0.0, t_start)
    t1 = min(clip.duration, t_end)
    audio = clip.audio.subclipped(t0, t1)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    audio.write_audiofile(str(tmp_path), logger=None)
    audio.close()
    clip.close()
    y, sr = sf.read(tmp_path)
    tmp_path.unlink(missing_ok=True)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y.astype(np.float64), sr


def _spectral_subtract(audio: np.ndarray, noise: np.ndarray, sr: int) -> np.ndarray:
    """Subtract wave-noise from audio via spectral subtraction (same as sound_extract.py)."""
    nperseg = 2048
    _, _, stft_audio = signal.stft(audio, fs=sr, nperseg=nperseg)
    _, _, stft_noise = signal.stft(noise,  fs=sr, nperseg=nperseg)
    mag    = np.abs(stft_audio)
    phase  = np.angle(stft_audio)
    noise_mean = np.mean(np.abs(stft_noise), axis=1, keepdims=True)
    _, cleaned = signal.istft(
        np.maximum(mag - noise_mean, 0.0) * np.exp(1j * phase),
        fs=sr, nperseg=nperseg,
    )
    return cleaned[:len(audio)]


# ── Per-transect extraction ───────────────────────────────────────────────────

def extract_transect(
    transect: dict,
    robot_type: str,
    gp1_dir: Path,
    gp4_dir: Path,
) -> None:
    """Extract and save cleaned audio for one transect from both cameras.

    For cross-video transects the audio segments are concatenated before
    noise filtering.  The noise reference is always taken from the first 20 s
    of the first video used by that transect.
    """
    trial_id = transect["id"]
    videos   = transect["videos"]
    starts   = transect["starts"]
    ends     = transect["ends"]

    cameras = [
        ("gp1", gp1_dir, lambda vid: vid),
        ("gp4", gp4_dir, _gp4_video_id),
    ]

    for cam_label, video_dir, id_fn in cameras:
        out_path = TRANSECT_SOUND_DIR / f"{trial_id}_{robot_type}_{cam_label}_sound.wav"
        if out_path.exists():
            print(f"  skip (exists): {out_path.name}")
            continue

        segments: list[np.ndarray] = []
        sr_out: int | None = None
        noise_audio: np.ndarray | None = None

        for v_idx, (video_id, v_start, v_end) in enumerate(zip(videos, starts, ends)):
            mp4_path = _find_mp4(video_dir, id_fn(video_id))
            if mp4_path is None:
                print(f"  ✘ not found: {video_dir / id_fn(video_id)}.MP4 — skipping {trial_id} {cam_label}")
                segments = []
                break

            # Noise reference from the opening of the first video.
            if noise_audio is None:
                noise_audio, sr_out = _read_audio_segment(mp4_path, 0.0, NOISE_REF_S)

            # Time window: add pre-buffer before first segment, post-buffer after last.
            t_start = (v_start - BUFFER_S) if v_idx == 0 else 0.0
            t_end   = (v_end   + BUFFER_S) if v_idx == len(videos) - 1 else float("inf")

            y, sr = _read_audio_segment(mp4_path, t_start, t_end)
            if sr_out is None:
                sr_out = sr
            segments.append(y)

        if not segments:
            continue

        combined = np.concatenate(segments) if len(segments) > 1 else segments[0]
        cleaned  = _spectral_subtract(combined, noise_audio, sr_out)
        sf.write(str(out_path), cleaned, sr_out)
        print(f"  ✔ saved: {out_path.name}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract transect-aligned audio from raw GoPro MP4 files."
    )
    parser.add_argument(
        "--gp1-dir", required=True, type=Path,
        help="Directory containing raw GoPro 1 MP4 files "
             "(e.g. /Volumes/Transcend_A/2024-11-USVI/2024-11-15/GoPro1-turtle-dm)",
    )
    parser.add_argument(
        "--gp4-dir", required=True, type=Path,
        help="Directory containing raw GoPro 4 MP4 files",
    )
    args = parser.parse_args()

    TRANSECT_SOUND_DIR.mkdir(parents=True, exist_ok=True)

    all_transects = (
        [("turtle",  t) for t in TURTLE_TRANSECTS] +
        [("auv",     t) for t in AUV_TRANSECTS] +
        [("control", t) for t in CONTROL_SEGMENTS]
    )
    print(f"Extracting audio for {len(all_transects)} segments "
          f"({len(TURTLE_TRANSECTS)} turtle + {len(AUV_TRANSECTS)} AUV "
          f"+ {len(CONTROL_SEGMENTS)} control) × 2 cameras...")

    for robot_type, transect in all_transects:
        print(f"\n{transect['id']} ({robot_type})")
        extract_transect(transect, robot_type, args.gp1_dir, args.gp4_dir)

    print(f"\nDone. Audio files written to {TRANSECT_SOUND_DIR}")


if __name__ == "__main__":
    main()
