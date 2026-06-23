"""
Sound analysis for transect-aligned audio clips.

Loads cleaned WAVs from sound/transects/, groups files by robot type
(AUV vs Turtle vs Control — each represented by both GP1 and GP4 recordings),
and generates spectrogram / RMS / SPL comparison plots for all three groups.

Run extract_transect_audio.py first to populate sound/transects/.

Outputs are written to figures/sound/transects/ to keep them distinct
from the original clip-based figures in figures/sound/.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from scipy import signal
import soundfile as sf
import matplotlib.pyplot as plt

from config.paths import FIGURES_DIR, TRANSECT_SOUND_DIR, ensure_dirs_exist
from src.visualization.plot_style import (
    apply_paper_style,
    PANEL_FIGSIZE,
    AUV_C, TURTLE_C, SHARK_C,
)

OUT_DIR = FIGURES_DIR / "sound" / "transects"

CONTROL_C = SHARK_C  # purple — distinct from AUV (orange) and Turtle (teal)

# Shared time axis: ±40 s from transect midpoint (mirrors paper_figures.py).
ABS_TIME_GRID = np.linspace(-40, 40, 320)

# (key in filename, colour, legend label)
GROUPS = [
    ("turtle",  TURTLE_C,  "Turtle"),
    ("auv",     AUV_C,     "AUV"),
    ("control", CONTROL_C, "Control"),
]

_PLOTS = [
    ("rms",            "times", "rms",      "Time from transect midpoint (s)", "RMS Amplitude"),
    ("spl",            "times", "spl",       "Time from transect midpoint (s)", "SPL (dB)"),
    ("mean_spectrum",  "freqs", "spectrum",  "Frequency (Hz)",                  "Mean Amplitude (dB)"),
    ("mean_amplitude", "times", "temporal",  "Time from transect midpoint (s)", "Mean Amplitude (dB)"),
]


# ── Audio utilities ───────────────────────────────────────────────────────────

def _load_audio(path):
    y, sr = sf.read(path)
    if y.ndim > 1:
        print(f"Warning: {path} has {y.shape[1]} channels; converting to mono by averaging.")
        y = np.mean(y, axis=1)
    return y.astype(np.float64), sr


def _stft_db(y, sr, nperseg=2048, noverlap=1536):
    frequencies, times, Zxx = signal.stft(
        y, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap,
        boundary=None, padded=False,
    )
    magnitude = np.abs(Zxx)
    ref = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10) / ref)
    return frequencies, times, Zxx, magnitude_db


def _frame_rms(y, sr, frame_length=2048, hop_length=512):
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop_length]
    if len(frames) == 0:
        frames = y[:frame_length][None, :]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    times_rms = (np.arange(len(rms)) * hop_length + frame_length / 2) / sr
    return rms, times_rms


def _extract_features(file_path):
    y, sr = _load_audio(file_path)
    duration = len(y) / sr
    center   = duration / 2.0
    freqs, times, _, magnitude_db = _stft_db(y, sr)
    rms, times_rms = _frame_rms(y, sr)
    spl = 20 * np.log10(rms / 1e-6 + 1e-10)
    return {
        "freqs":            freqs,
        "times":            times     - center,
        "times_rms":        times_rms - center,
        "mean_spectrum":    np.mean(magnitude_db, axis=1),
        "mean_temporal":    np.mean(magnitude_db, axis=0),
        "rms":              rms,
        "spl":              spl,
        "mean_rms_scalar":  np.mean(rms),
        "mean_spl_scalar":  np.mean(spl),
    }




def aggregate_group(file_list):
    features = [_extract_features(f) for f in file_list]
    common_freqs = features[0]["freqs"]

    rms_all      = np.vstack([np.interp(ABS_TIME_GRID, f["times_rms"], f["rms"],           left=np.nan, right=np.nan) for f in features])
    spl_all      = np.vstack([np.interp(ABS_TIME_GRID, f["times_rms"], f["spl"],           left=np.nan, right=np.nan) for f in features])
    temporal_all = np.vstack([np.interp(ABS_TIME_GRID, f["times"],     f["mean_temporal"], left=np.nan, right=np.nan) for f in features])
    spectrum_all = np.vstack([np.interp(common_freqs,  f["freqs"],     f["mean_spectrum"])                             for f in features])

    scalars_rms = np.array([f["mean_rms_scalar"] for f in features])
    scalars_spl = np.array([f["mean_spl_scalar"] for f in features])

    # Centre-window (±5 s) per-trial means
    center_mask = np.abs(ABS_TIME_GRID) <= 5.0
    center_rms = np.array([np.nanmean(row[center_mask]) for row in rms_all])
    center_spl = np.array([np.nanmean(row[center_mask]) for row in spl_all])

    return {
        "freqs":                  common_freqs,
        "times":                  ABS_TIME_GRID,
        "rms_mean":               np.nanmean(rms_all,      axis=0),
        "rms_std":                np.nanstd(rms_all,       axis=0),
        "spl_mean":               np.nanmean(spl_all,      axis=0),
        "spl_std":                np.nanstd(spl_all,       axis=0),
        "temporal_mean":          np.nanmean(temporal_all, axis=0),
        "temporal_std":           np.nanstd(temporal_all,  axis=0),
        "spectrum_mean":          np.mean(spectrum_all,    axis=0),
        "spectrum_std":           np.std(spectrum_all,     axis=0),
        "mean_rms_scalar_mean":   np.mean(scalars_rms),
        "mean_rms_scalar_std":    np.std(scalars_rms),
        "mean_spl_scalar_mean":   np.mean(scalars_spl),
        "mean_spl_scalar_std":    np.std(scalars_spl),
        "center_rms_scalar_mean": np.mean(center_rms),
        "center_rms_scalar_std":  np.std(center_rms),
        "center_spl_scalar_mean": np.mean(center_spl),
        "center_spl_scalar_std":  np.std(center_spl),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dirs_exist(OUT_DIR)

    all_wavs = sorted(TRANSECT_SOUND_DIR.glob("*_sound.wav"))
    if not all_wavs:
        raise FileNotFoundError(
            f"No WAVs found in {TRANSECT_SOUND_DIR}. "
            "Run extract_transect_audio.py first."
        )

    file_groups = {
        key: [f for f in all_wavs if f"_{key}_" in f.name]
        for key, _, _ in GROUPS
    }
    for key, _, label in GROUPS:
        print(f"{label} files: {len(file_groups[key])}")

    aggregated = {
        key: aggregate_group(files)
        for key, files in file_groups.items() if file_groups[key]
    }
    for key, _, label in GROUPS:
        if key not in aggregated:
            continue
        g = aggregated[key]
        print(f"{label} Mean RMS: {g['mean_rms_scalar_mean']:.6f} ± {g['mean_rms_scalar_std']:.6f}")
        print(f"{label} Mean SPL: {g['mean_spl_scalar_mean']:.2f} ± {g['mean_spl_scalar_std']:.2f} dB")
        print(f"{label} Center (±5 s) RMS: {g['center_rms_scalar_mean']:.6f} ± {g['center_rms_scalar_std']:.6f}")
        print(f"{label} Center (±5 s) SPL: {g['center_spl_scalar_mean']:.2f} ± {g['center_spl_scalar_std']:.2f} dB")

    for fname, x_key, y_key, xlabel, ylabel in _PLOTS:
        fig, ax = plt.subplots()
        for key, color, label in GROUPS:
            if key not in aggregated:
                continue
            g = aggregated[key]
            x = g[x_key]
            mean, std = g[f"{y_key}_mean"], g[f"{y_key}_std"]
            valid = ~np.isnan(mean)
            ax.plot(x[valid], mean[valid], label=label, color=color)
            ax.fill_between(x[valid], (mean - std)[valid], (mean + std)[valid],
                            color=color, alpha=0.20)
        if x_key == "times":
            ax.axvline(0, color="gray", lw=0.8, ls=":", alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper center", ncol=3, frameon=False,
                  handlelength=1.2, columnspacing=0.8, borderaxespad=0.2)
        apply_paper_style(fig, ax, figsize=PANEL_FIGSIZE, clear_titles=False)
        fig.savefig(OUT_DIR / f"{fname}.svg", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {OUT_DIR / fname}.svg")

    print(f"\nDone. Figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
