import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.paths import FIGURES_SOUND, ensure_dirs_exist
from src.visualization.plot_style import (
    apply_phase2a_style,
    PHASE2A_PANEL_FIGSIZE,
    AUV_C,
    TURTLE_C,
)



def load_audio(raw_noise_audio):
    y, sr = sf.read(raw_noise_audio)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y.astype(np.float64), sr


def stft_db(y, sr, nperseg=2048, noverlap=1536):
    frequencies, times, Zxx = signal.stft(
        y,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False
    )
    magnitude = np.abs(Zxx)
    ref = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10) / ref)
    return frequencies, times, Zxx, magnitude_db


def frame_rms(y, sr, frame_length=2048, hop_length=512):
    if len(y) < frame_length:
        pad = frame_length - len(y)
        y = np.pad(y, (0, pad), mode="constant")

    frames = np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop_length]

    if len(frames) == 0:
        frames = y[:frame_length][None, :]

    rms = np.sqrt(np.mean(frames**2, axis=1))
    times_rms = (np.arange(len(rms)) * hop_length + frame_length / 2) / sr
    return rms, times_rms


def extract_features(file_path):
    y, sr = load_audio(file_path)
    freqs, times, D, magnitude_db = stft_db(y, sr)
    mean_spectrum = np.mean(magnitude_db, axis=1)
    mean_temporal = np.mean(magnitude_db, axis=0)
    rms, times_rms = frame_rms(y, sr)
    spl = 20 * np.log10(rms + 1e-10)

    return {
        "freqs": freqs,
        "times": times,
        "times_rms": times_rms,
        "mean_spectrum": mean_spectrum,
        "mean_temporal": mean_temporal,
        "rms": rms,
        "spl": spl,
        "mean_rms_scalar": np.mean(rms),
        "mean_spl_scalar": np.mean(spl),
    }


def interp_to_common_axis(x, y, x_common):
    return np.interp(x_common, x, y)


def aggregate_group(file_list):
    features = [extract_features(f) for f in file_list]

    common_freqs = features[0]["freqs"]

    n_rms = min(len(f["rms"]) for f in features)
    n_spec = min(len(f["mean_temporal"]) for f in features)

    common_times_rms = np.linspace(0, 1, n_rms)
    common_times_spec = np.linspace(0, 1, n_spec)

    rms_all = np.vstack([
        interp_to_common_axis(
            f["times_rms"] / f["times_rms"][-1],
            f["rms"],
            common_times_rms
        )
        for f in features
    ])
    spl_all = np.vstack([
        interp_to_common_axis(
            f["times_rms"] / f["times_rms"][-1],
            f["spl"],
            common_times_rms
        )
        for f in features
    ])
    temporal_all = np.vstack([
        interp_to_common_axis(
            f["times"] / f["times"][-1],
            f["mean_temporal"],
            common_times_spec
        )
        for f in features
    ])
    spectrum_all = np.vstack([
        interp_to_common_axis(f["freqs"], f["mean_spectrum"], common_freqs)
        for f in features
    ])

    mean_rms_scalar_all = np.array([f["mean_rms_scalar"] for f in features])
    mean_spl_scalar_all = np.array([f["mean_spl_scalar"] for f in features])

    return {
        "freqs": common_freqs,
        "times_rms": common_times_rms,
        "times": common_times_spec,
        "rms_mean": np.mean(rms_all, axis=0),
        "rms_std": np.std(rms_all, axis=0),
        "spl_mean": np.mean(spl_all, axis=0),
        "spl_std": np.std(spl_all, axis=0),
        "temporal_mean": np.mean(temporal_all, axis=0),
        "temporal_std": np.std(temporal_all, axis=0),
        "spectrum_mean": np.mean(spectrum_all, axis=0),
        "spectrum_std": np.std(spectrum_all, axis=0),
        "mean_rms_scalar_mean": np.mean(mean_rms_scalar_all),
        "mean_rms_scalar_std": np.std(mean_rms_scalar_all),
        "mean_spl_scalar_mean": np.mean(mean_spl_scalar_all),
        "mean_spl_scalar_std": np.std(mean_spl_scalar_all),
    }


def plot_mean_std(ax, x, mean, std, label, color):
    ax.plot(x, mean, label=label, color=color)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.20)


def spectrogram_averages_compare(
    gopro1_files,
    gopro4_files,
    file1_label="GoPro 1",
    file2_label="GoPro 4",
    out_dir: Path | None = None,
):
    """Compare two groups of audio files and save spectrogram/RMS/SPL plots.

    out_dir defaults to figures/sound/ (FIGURES_SOUND from config.paths).
    """
    if out_dir is None:
        out_dir = FIGURES_SOUND
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    group1 = aggregate_group(gopro1_files)
    group2 = aggregate_group(gopro4_files)

    print(f"{file1_label} Mean RMS Amplitude: {group1['mean_rms_scalar_mean']:.6f} ± {group1['mean_rms_scalar_std']:.6f}")
    print(f"{file2_label} Mean RMS Amplitude: {group2['mean_rms_scalar_mean']:.6f} ± {group2['mean_rms_scalar_std']:.6f}")
    print(f"{file1_label} Mean SPL: {group1['mean_spl_scalar_mean']:.2f} ± {group1['mean_spl_scalar_std']:.2f} dB")
    print(f"{file2_label} Mean SPL: {group2['mean_spl_scalar_mean']:.2f} ± {group2['mean_spl_scalar_std']:.2f} dB")

    _plots = [
        ("rms",           "times_rms", "rms",      "Time (s)",       "RMS Amplitude",          "RMS Amplitude per Frame"),
        ("spl",           "times_rms", "spl",      "Time (s)",       "SPL (dB)",               "Sound Pressure Level (SPL) per Frame"),
        ("mean_spectrum", "freqs",     "spectrum", "Frequency (Hz)", "Mean Amplitude (dB)",    "Mean Spectrum (Averaged over Time)"),
        ("mean_amplitude","times",     "temporal", "Time (s)",       "Mean Amplitude (dB)",    "Mean Temporal Energy (Averaged over Frequency)"),
    ]

    for fname, x_key, y_key, xlabel, ylabel, title in _plots:
        fig, ax = plt.subplots()
        plot_mean_std(ax, group1[x_key], group1[f"{y_key}_mean"], group1[f"{y_key}_std"], file1_label, TURTLE_C)
        plot_mean_std(ax, group2[x_key], group2[f"{y_key}_mean"], group2[f"{y_key}_std"], file2_label, AUV_C)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # ax.set_title(title)
        ax.legend(loc="upper center", ncol=2, frameon=False,
                  handlelength=1.2, columnspacing=0.8, borderaxespad=0.2)
        apply_phase2a_style(fig, ax, figsize=PHASE2A_PANEL_FIGSIZE, clear_titles=False)
        fig.savefig(out_dir / f"{fname}.svg", bbox_inches="tight")
        plt.close(fig)
