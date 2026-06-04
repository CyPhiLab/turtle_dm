"""
Centralized path definitions for the turtle_dm_analysis project.
"""

from pathlib import Path

# Project root — the directory containing this config/ package.
PROJECT_ROOT = Path(__file__).parent.parent

# Annotated video frames (fish_detections.csv per video)
ANNOTATED_VIDEO_DIR = PROJECT_ROOT / "data"

# Output directories
FIGURES_DIR            = PROJECT_ROOT / "figures"
FIGURES_SOUND          = FIGURES_DIR / "sound"
TRANSECT_SOUND_DIR     = PROJECT_ROOT / "data" / "sound"
FIGURES_SOUND_TRANSECT = FIGURES_DIR / "sound" / "transects"


def ensure_dirs_exist(*dirs):
    """Create directories if they don't already exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
