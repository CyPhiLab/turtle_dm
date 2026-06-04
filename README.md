# turtle_dm

Analysis of fish behaviour from GoPro video detections, producing per-frame metrics and figures. We compare two robot platforms, [Crush the sea turtle robot](https://arxiv.org/abs/2602.21389) and [CUREE](https://ieeexplore.ieee.org/abstract/document/10161282) (A Curious Underwater Robot for Ecosystem Exploration), to assess the disturbance to local fauna near a coral reef hotspot in the US Virgin Islands. We also compare to the response to a diver and to a shark.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

## Disturbance metrics

Generate paper figures (outputs to `figures/paper/`):

```bash
python src/paper_figures.py
```

All metrics are computed per frame, then smoothed with a rolling window.

| Metric | Description |
|---|---|
| `fish_count` | Number of fish detected in the frame |
| `vertical_position` | Mean y-coordinate of all fish |
| `signed_reef_distance` | Mean signed distance of each fish to the reef boundary: positive in open ocean, negative on reef |
| `centroid_speed` | Frame-to-frame displacement of the school centroid |



## Sound analysis

The sound pipeline requires the raw GoPro MP4 files from an external drive. Run the steps in order:

**1. Extract transect audio** — pulls ±30 s windows around each transect from both GoPro 1 and GoPro 4, applies spectral noise subtraction, and saves cleaned WAVs to `sound/transects/`:

```bash
python -m src.sound.extract_transect_audio \
    --gp1-dir /path/to/GoPro1 \
    --gp4-dir /path/to/GoPro4
```

**2. Evaluate transect sound** — loads the cleaned WAVs, groups by robot type (AUV / Turtle / Control), and writes spectrogram, RMS, SPL, and mean-spectrum figures to `figures/sound/transects/`:

```bash
python -m src.sound.evaluate_transect_sound
```

## Structure

```
config/   — constants, paths, transect definitions, ocean masks
data/     — per-clip fish detection CSVs tagged with SAM3
src/      — analysis code and figure generation
figures/  — output figures
```


