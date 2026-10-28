"""Small, independently testable numerical contracts and PC selection."""
from dataclasses import dataclass, asdict
import math
import numpy as np


class QualityGateError(RuntimeError):
    """No valid scientific result should be emitted for this processing unit."""


def safe_divide_numpy(num, den, epsilon=1e-6):
    """Preserve missingness: invalid divisions are masked, never replaced by zero."""
    a, b = np.broadcast_arrays(np.asarray(num, float), np.asarray(den, float))
    valid = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > epsilon)
    out = np.zeros(a.shape, dtype=float)
    with np.errstate(all="ignore"):
        np.divide(a, b, out=out, where=valid)
    valid &= np.isfinite(out)
    return np.ma.array(out, mask=~valid)


def illumination_cosine(slope_deg, aspect_deg, solar_azimuth_deg, zenith_deg):
    """Aspect and azimuth clockwise from north; no reflectance threshold."""
    slope, aspect, az, z = np.deg2rad(
        [slope_deg, aspect_deg, solar_azimuth_deg, zenith_deg]
    )
    return float(np.cos(slope)*np.cos(z) +
                 np.sin(slope)*np.sin(z)*np.cos(az-aspect))


@dataclass(frozen=True)
class PCChoice:
    component: int                 # 0-based, descending eigenvalue order
    sign: int                      # orient target absorption contrast positive
    cosine: float
    pair_energy: float
    eigenvalue: float
    loadings: list

    def record(self):
        return asdict(self)


def select_component(eigenvalues, eigenvectors, bands, positive, negative,
                     min_cosine=0.55, min_pair_energy=0.55,
                     min_abs_loading=0.20, min_gap=0.05):
    """Feature-oriented PCA, not a fixed PC number or a gold classifier.

    Eigensystem must have row eigenvectors in descending eigenvalue order.
    Require opposite signs on a physically motivated reflectance pair, adequate
    contrast alignment, adequate pair loading energy, and a non-degenerate PC.
    Gates are configurable engineering defaults, not field calibration.
    A component's sign is arbitrary; choose it to make positive-negative bright.
    """
    vals, vecs = np.asarray(eigenvalues, float), np.asarray(eigenvectors, float)
    n = len(bands)
    if (vals.shape != (n,) or vecs.shape != (n, n) or
        not np.isfinite(vals).all() or not np.isfinite(vecs).all()):
        raise QualityGateError("Invalid eigensystem")
    if vals[0] <= 0 or np.any(vals < -1e-10) or np.any(np.diff(vals) > 1e-10):
        raise QualityGateError("Non-positive or unsorted eigenvalues")
    if not np.allclose(vecs @ vecs.T, np.eye(n), atol=1e-5):
        raise QualityGateError("Eigenvectors are not orthonormal")
    p, q = bands.index(positive), bands.index(negative)
    prototype = np.zeros(n); prototype[p] = 1/math.sqrt(2); prototype[q] = -1/math.sqrt(2)
    candidates = []
    for i, v in enumerate(vecs):
        if vals[i] <= max(vals[0]*1e-8, 1e-12):
            continue
        gap = min((abs(vals[i]-x)/max(vals[i], x, 1e-12)
                   for j, x in enumerate(vals) if j != i), default=1)
        alignment = float(v @ prototype)
        energy = float(v[p]**2 + v[q]**2)
        if (v[p]*v[q] >= 0 or min(abs(v[p]), abs(v[q])) < min_abs_loading
            or abs(alignment) < min_cosine or energy < min_pair_energy
            or gap < min_gap):
            continue
        sign = 1 if alignment >= 0 else -1
        candidates.append(PCChoice(i, sign, abs(alignment), energy,
                                   float(vals[i]), (v*sign).tolist()))
    if not candidates:
        raise QualityGateError("No stable PC matches the target loading signature")
    return max(candidates, key=lambda x: (x.cosine, x.pair_energy))
