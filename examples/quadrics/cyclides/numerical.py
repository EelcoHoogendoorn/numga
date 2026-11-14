"""Numeric root finding and image-plane offsets for ray tracing on the 3-sphere."""

from __future__ import annotations

import numpy as np

from numga.extensor.extensor import Extensor


def pixel_offsets(image_shape: tuple[int, int], fov: float) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole pixel offsets along right and up on the plane one unit ahead form two row-major
    `[n_pixels]` arrays."""
    height, width = image_shape
    horizontal = (2 * (np.arange(width) + 0.5) / width - 1)[None, :]
    vertical = (1 - 2 * (np.arange(height) + 0.5) / height)[:, None] * height / width
    right, up = np.broadcast_arrays(horizontal * np.tan(fov / 2), vertical * np.tan(fov / 2))
    return right.ravel(), up.ravel()


def nearest_hit_angles(coefficients: list[Extensor]) -> Extensor:
    """Ascending polynomial coefficients in the half-angle tangent `u` yield the smallest positive
    hit angle (`[...] Scalar+`) along each great circle, or `inf` on a miss.

    Solving the companion matrix of the reversed polynomial in the reciprocal root `r` maps each
    real root through `arctan2` onto `[0, 2 * pi]`, ordering hits past (`r < 0`) and at (`r == 0`)
    the antipode along the full circle.
    """
    stacked = np.stack(np.broadcast_arrays(*(c.to_array() for c in coefficients)), axis=-1)
    degree = stacked.shape[-1] - 1
    companion = np.zeros(stacked.shape[:-1] + (degree, degree))
    companion[..., 1:, :-1] = np.eye(degree - 1)
    companion[..., :, -1] = -stacked[..., degree:0:-1] / stacked[..., :1]
    roots = np.linalg.eigvals(companion)
    angles = 2 * np.arctan2(1.0, roots.real)
    # Near-double roots carry round-off imaginary parts on grazing hits.
    real = np.abs(roots.imag) <= 1e-6 * (1 + np.abs(roots.real))
    nearest = np.where(real, angles, np.inf).min(axis=-1)
    return coefficients[0].context.multivector.scalar() * nearest
