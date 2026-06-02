"""
Ocean-mask utilities.

Decodes the project's COCO RLE ocean mask once at import time and
exports the ready-to-use binary array together with the coordinate
scaling factors needed to map 4K fish positions into mask space.
"""

import json
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask

from config.constants import MASK_WIDTH, MASK_HEIGHT, SCALE_X, SCALE_Y

# Mask JSON files live in config/ alongside the rest of the project configuration.
_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"

# Per-GoPro mask file names; GP4 is the legacy default.
_MASK_FILES: dict[str, str] = {
    "GP1": "ocean_mask_GP1.JSON",
    "GP4": "ocean_mask_GP4.JSON",
}
_DEFAULT_MASK = "ocean_mask_GP4.JSON"

_mask_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_signed_cache: dict[str, np.ndarray] = {}


def _decode_mask(path: Path) -> np.ndarray:
    """Decode a COCO RLE ocean mask JSON → binary array (H×W, uint8)."""
    with open(path) as f:
        data = json.load(f)
    rle = data[0]["sam"]["predictions"][0]["rle_mask"]
    return coco_mask.decode(rle)   # 1 = ocean, 0 = reef


def _reef_distance_map(ocean_mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(ocean_mask).astype(np.float32)


def get_masks(gopro: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (ocean_mask, reef_distance_map) for the given GoPro label.

    Results are cached after first load.  Pass *gopro=None* or an unknown key
    to get the GP4 default.
    """
    # Find the correct file
    mask_file = _DEFAULT_MASK
    if gopro:
        for key, fname in _MASK_FILES.items():
            if key in gopro:
                mask_file = fname
                break

    if mask_file not in _mask_cache:
        mask_path = _CONFIG_DIR / mask_file
        om = _decode_mask(mask_path)
        rdm = _reef_distance_map(om)
        _mask_cache[mask_file] = (om, rdm)

    return _mask_cache[mask_file]


def get_signed_reef_map(gopro: str | None = None) -> np.ndarray:
    """Return a signed reef-distance map for the given GoPro.

    Pixels **inside the ocean** have positive values (distance to nearest reef
    edge).  Pixels **inside the reef** have negative values (negative of their
    distance to the nearest ocean edge).  The zero crossing is the reef/ocean
    boundary.

    This lets you distinguish fish that are sheltering on the reef (negative)
    from fish in open water (positive), rather than treating both as positive
    reef-distances.

    Results are cached after first computation.
    """
    from scipy.ndimage import distance_transform_edt

    mask_file = _DEFAULT_MASK
    if gopro:
        for key, fname in _MASK_FILES.items():
            if key in gopro:
                mask_file = fname
                break

    if mask_file not in _signed_cache:
        om, _ = get_masks(gopro)
        edt_ocean = distance_transform_edt(om).astype(np.float32)
        edt_reef  = distance_transform_edt(~om.astype(bool)).astype(np.float32)
        _signed_cache[mask_file] = edt_ocean - edt_reef

    return _signed_cache[mask_file]


def load_ocean_mask() -> np.ndarray:
    """Decode the default (GP4) COCO RLE ocean mask → binary numpy array."""
    om, _ = get_masks("GP4")
    return om


# Pre-loaded singletons (GP4 default) — kept for backward compatibility.
OCEAN_MASK: np.ndarray
REEF_DISTANCE_MAP: np.ndarray
OCEAN_MASK, REEF_DISTANCE_MAP = get_masks("GP4")

__all__ = [
    "OCEAN_MASK", "REEF_DISTANCE_MAP",
    "MASK_WIDTH", "MASK_HEIGHT", "SCALE_X", "SCALE_Y",
    "load_ocean_mask", "get_masks", "get_signed_reef_map",
]
