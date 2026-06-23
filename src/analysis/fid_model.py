"""Flight initiation distance (FID) logistic model utilities.

Model
-----
The logistic profile is:

    y(x) = Ld + (L0 - Ld) / (1 + exp(-k * (x - x0)))

On the signed FID axis (x = t in seconds, -40 to 0):
  - x = -40 → far from robot (baseline)            → y ≈ L0
  - x =   0 → transect midpoint (closest approach) → y ≈ Ld
  - x0 is the inflection point of the sigmoid (free parameter)
  - k  controls the steepness (expected < 0 so y transitions from L0 at left to Ld at right)

L0 and Ld are estimated empirically:
  - L0 : mean of y where x < L0_WINDOW_HI  (far tail near -40 s, ~baseline)
  - Ld : mean of y where x > Ld_WINDOW_LO  (near 0 s, ~peak disturbance)

k and x0 are then fitted jointly with L0/Ld held fixed.

x_FID is computed analytically from:
    (L0 - Ld) / (1 + exp(-k*(x_FID - x0))) + Ld = alpha * L0
    => x_FID = x0 - ln(L0*(1-alpha) / (alpha*L0 - Ld)) / k
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


FID_ALPHA = 0.9

# Windows (in signed seconds) used to estimate L0 and Ld from raw data.
# x < L0_WINDOW_HI → near -40 s (far from robot, baseline)
# x > Ld_WINDOW_LO → near  0 s (transect midpoint, peak disturbance)
L0_WINDOW_HI = -35.0
Ld_WINDOW_LO =  -0.5

_MIN_POINTS = 3  # minimum raw points required in each estimation window


@dataclass
class FIDFitResult:
    """Container for per-robot/per-metric FID fit outputs."""

    L0: float           # empirical baseline (far-tail mean)
    Ld: float           # empirical disturbance (near-centre mean)
    k: float            # fitted steepness
    x0: float           # fitted sigmoid inflection point (s)
    x_fid: float | None
    x_fid_status: str


def fid_logistic(x: np.ndarray, k: float, x0: float,
                 L0: float, Ld: float) -> np.ndarray:
    """Logistic profile y(x) = Ld + (L0-Ld) / (1 + exp(-k*(x-x0)))."""
    x = np.asarray(x, dtype=float)
    return Ld + (L0 - Ld) / (1.0 + np.exp(-k * (x - x0)))


def compute_x_fid(L0: float, Ld: float, k: float, x0: float,
                  alpha: float = FID_ALPHA) -> tuple[float | None, str]:
    """Compute x_FID analytically.

    Solves: (L0-Ld)/(1+exp(-k*(x_FID-x0))) + Ld = alpha*L0
    => x_FID = x0 - ln(L0*(1-alpha) / (alpha*L0 - Ld)) / k
    """
    if not all(np.isfinite(v) for v in (L0, Ld, k, x0)):
        return None, "undefined (non-finite parameters)"
    if np.isclose(k, 0.0):
        return None, "undefined (k ≈ 0)"
    denom = alpha * L0 - Ld
    if np.isclose(denom, 0.0):
        return None, "undefined (alpha*L0 ≈ Ld)"
    inner = L0 * (1.0 - alpha) / denom
    if inner <= 0.0 or not np.isfinite(inner):
        return None, "undefined (invalid log domain)"
    x_fid = x0 - np.log(inner) / k
    if not np.isfinite(x_fid):
        return None, "undefined (non-finite result)"
    return float(x_fid), "ok"


def fit_fid_model(x: np.ndarray, y: np.ndarray,
                  alpha: float = FID_ALPHA) -> FIDFitResult | None:
    """Estimate L0/Ld empirically, fit k and x0, compute x_FID analytically.

    Parameters
    ----------
    x : folded time (seconds, 0–40)
    y : metric values (same length as x)
    alpha : threshold fraction of L0 that defines x_FID (default 0.9)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < _MIN_POINTS * 2:
        return None

    xv = x[valid]
    yv = y[valid]

    # ── Empirical L0 and Ld ──────────────────────────────────────────────────
    L0_vals = yv[xv < L0_WINDOW_HI]
    Ld_vals = yv[xv > Ld_WINDOW_LO]

    if len(L0_vals) < _MIN_POINTS or len(Ld_vals) < _MIN_POINTS:
        return None

    L0 = float(np.mean(L0_vals))
    Ld = float(np.mean(Ld_vals))

    # ── Fit k and x0 with L0 and Ld held fixed ───────────────────────────────
    def _model(x_arr, k, x0):
        return fid_logistic(x_arr, k, x0, L0, Ld)

    # Initial guess: inflection at midpoint of data range, k negative (L0 at left, Ld at right).
    x0_init = -20.0
    try:
        popt, _ = curve_fit(_model, xv, yv, p0=[-0.2, x0_init])
    except Exception:
        return None

    k, x0 = float(popt[0]), float(popt[1])
    x_fid, status = compute_x_fid(L0, Ld, k, x0, alpha=alpha)
    return FIDFitResult(L0=L0, Ld=Ld, k=k, x0=x0, x_fid=x_fid, x_fid_status=status)


# Exposed for reference.
FID_GRID_MIN = -40.0
FID_GRID_MAX =   0.0

