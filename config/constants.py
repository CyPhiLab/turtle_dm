"""
Domain constants shared across the project.
All hard-coded scientific parameters live here so they can be updated in one place.
"""

# ── Ocean mask dimensions ──────────────────────────────────────────────────
MASK_WIDTH  = 2048
MASK_HEIGHT = 1152
FISH_WIDTH  = 3840   # full-resolution 4K frame width
FISH_HEIGHT = 2160   # full-resolution 4K frame height
SCALE_X     = FISH_WIDTH  / MASK_WIDTH    # 1.875
SCALE_Y     = FISH_HEIGHT / MASK_HEIGHT   # 1.875

# ── Data processing ─────────────────────────────────────────────────────────────────
ROLLING_WINDOW = 1   # frames; used for smoothing time-series
