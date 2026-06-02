"""
Domain constants shared across the project.
All hard-coded scientific parameters live here so they can be updated in one place.
"""

# ── Shelter pixel coordinates (4K frame space) ────────────────────────────────
SPLIT_X       = 3840 / 2          # x boundary that determines which shelter a fish belongs to
SHELTER_LEFT  = (850,  850)        # shelter used when fish x < SPLIT_X (GP4)
SHELTER_RIGHT = (2220, 900)        # shelter used when fish x >= SPLIT_X (GP4)

# Per-GoPro shelter overrides.  When a gopro key is present, all fish are
# assigned to the nearest shelter in the list (ignoring SPLIT_X).
_SHELTERS_BY_GOPRO: dict[str, list[tuple[int, int]]] = {
    "GP1": [(1800, 1200)],
}

# Diver experiments (2024-11-14) used GP1 but the shelter appeared at a
# different pixel location due to camera positioning.
DIVER_SHELTER: tuple[int, int] = (1600, 1100)


def shelters_for_gopro(gopro: str) -> list[tuple[int, int]] | None:
    """Return shelter coords for *gopro* or None to use the default SPLIT_X logic."""
    for key, shelters in _SHELTERS_BY_GOPRO.items():
        if key in gopro:
            return shelters
    return None

# ── Ocean mask dimensions ─────────────────────────────────────────────────────
MASK_WIDTH  = 2048
MASK_HEIGHT = 1152
FISH_WIDTH  = 3840   # full-resolution 4K frame width
FISH_HEIGHT = 2160   # full-resolution 4K frame height
SCALE_X     = FISH_WIDTH  / MASK_WIDTH    # 1.875
SCALE_Y     = FISH_HEIGHT / MASK_HEIGHT   # 1.875

# ── Disturbance frames ────────────────────────────────────────────────────────
# Key format: "<depth>_<trial>" e.g. "190_6" extracted from the directory name.

AUV_DEFAULT_DISTURBANCE = 38  # frame number used for AUV trials not listed below

AUV_DISTURBANCE_FRAMES: dict[str, int] = {
    "190_6":  5,
    "200_1": 20,
}

TURTLE_DISTURBANCE_FRAMES: dict[str, int] = {
    "100_1": 57,
    "110_1": 51,
    "110_2": 43,
    "120_1": 52,
    "120_2": 46,
    "120_3": 60,
    "130_1": 50,
    "130_2": 54,
}

# ── New-subject disturbance frames (GP1, 2024-11-14 / 2024-11-15) ─────────────
# Frame at which the disturbance reaches the center of the clip (1 FPS).
# Keys match the subject-specific suffix in the trial directory name.

SHARK_DISTURBANCE_FRAMES: dict[str, int] = {
    "shark_1": 43,   # GX160534 10:07–10:14, 40s buffers, center = (607+614)/2 - 567
}

TURTLE_ANIMAL_DISTURBANCE_FRAMES: dict[str, int] = {
    "turtle_animal_1": 30,  # GX190534 7:13–7:30, 30s buffers, center = (433+450)/2 - 403
}

# ── Reef-edge detection ───────────────────────────────────────────────────────
REEF_EDGE_THRESHOLD = 50  # mask-space pixels (~94 4K-px); fish within this
                          # distance of the reef boundary are "edge-hugging"

# ── Data processing ───────────────────────────────────────────────────────────
ROLLING_WINDOW    = 5   # frames; used for smoothing time-series
BASELINE_BUFFER   = 10  # frames before disturbance excluded from baseline
