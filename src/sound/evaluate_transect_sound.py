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

import matplotlib.pyplot as plt

from config.paths import FIGURES_DIR, TRANSECT_SOUND_DIR, ensure_dirs_exist
from src.sound.evaluate_sound import aggregate_group, plot_mean_std
from src.visualization.plot_style import (
    apply_phase2a_style,
    PHASE2A_PANEL_FIGSIZE,
    AUV_C, TURTLE_C, SHARK_C,
)

OUT_DIR = FIGURES_DIR / "sound" / "transects"

CONTROL_C = SHARK_C  # purple — distinct from AUV (orange) and Turtle (teal)

# (key in filename, colour, legend label)
GROUPS = [
    ("turtle",  TURTLE_C, "Turtle"),
    ("auv",     AUV_C,    "AUV"),
    ("control", CONTROL_C,"Control"),
]

_PLOTS = [
    ("rms",            "times_rms", "rms",      "Time (normalised)", "RMS Amplitude",       "RMS Amplitude per Frame"),
    ("spl",            "times_rms", "spl",       "Time (normalised)", "SPL (dB)",            "Sound Pressure Level (SPL) per Frame"),
    ("mean_spectrum",  "freqs",     "spectrum",  "Frequency (Hz)",    "Mean Amplitude (dB)", "Mean Spectrum (Averaged over Time)"),
    ("mean_amplitude", "times",     "temporal",  "Time (normalised)", "Mean Amplitude (dB)", "Mean Temporal Energy (Averaged over Frequency)"),
]


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

    aggregated = {key: aggregate_group(files) for key, files in file_groups.items() if file_groups[key]}

    for key, _, label in GROUPS:
        if key not in aggregated:
            continue
        g = aggregated[key]
        print(f"{label} Mean RMS: {g['mean_rms_scalar_mean']:.6f} ± {g['mean_rms_scalar_std']:.6f}")
        print(f"{label} Mean SPL: {g['mean_spl_scalar_mean']:.2f} ± {g['mean_spl_scalar_std']:.2f} dB")

    for fname, x_key, y_key, xlabel, ylabel, title in _PLOTS:
        fig, ax = plt.subplots()
        for key, color, label in GROUPS:
            if key not in aggregated:
                continue
            g = aggregated[key]
            plot_mean_std(ax, g[x_key], g[f"{y_key}_mean"], g[f"{y_key}_std"], label, color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # ax.set_title(title)
        ax.legend(loc="upper center", ncol=3, frameon=False,
                  handlelength=1.2, columnspacing=0.8, borderaxespad=0.2)
        apply_phase2a_style(fig, ax, figsize=PHASE2A_PANEL_FIGSIZE, clear_titles=False)
        fig.savefig(OUT_DIR / f"{fname}.svg", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {OUT_DIR / fname}.svg")

    print(f"\nDone. Figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
