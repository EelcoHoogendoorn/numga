"""Gravitational lensing by point masses in the plane of the sky.

Sky directions are intrinsic points (`Point-`) in units of the total mass's Einstein angle. Each mass deflects a sightline towards itself by its mass times the inverse separation, and the local derivative map adds one reflection per mass to the identity, keeping trace two everywhere and vanishing in area ratio along the critical curve.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA2D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                    # Scalar+
# Sky directions are intrinsic planar points (`Antivector-`).
Point = ga.gatype.antivector()                                                 # Antivector-
LocalMap = ga.gatype((Point, Point))                                           # Point- <- Point-
Exposure = tuple[Callable[[Point], Scalar], Callable[[Point], Scalar]]


# --- math -----------------------------------------------------------------------------
def source_direction(observed: Point, positions: Point, masses: np.ndarray) -> Point:
    """Each mass in `positions` (`[n_masses] Point-`) bends `observed` (`[...] Point-`) by its mass times the inverse separation to reach the source direction (`[...] Point-`)."""
    separation = observed[..., None] - positions                               # [..., ...] Point-
    return observed - (masses * separation.inverse()).sum(axis=-1)             # [...] Point-


def local_map(observed: Point, positions: Point, masses: np.ndarray) -> LocalMap:
    """The sky-to-source Jacobian (`[...] Point- <- Point-`) sums the identity with one scaled separation-line reflection per mass."""
    inverse_separation = (observed[..., None] - positions).inverse()           # [..., ...] Point-
    # Differentiating separation.inverse() gives minus the step sandwiched between two inverse separations.
    return Point + (masses * (inverse_separation * Point * inverse_separation)).sum(axis=-1)   # [...] Point- <- Point-


def source_brightness(directions: Point, centre: Point, width: float) -> Scalar:
    """A round Gaussian source centred at `centre` with angular standard deviation `width` has unit peak brightness (`[...] Scalar+`)."""
    offset = directions - centre                                               # [...] Point-
    return (-(offset | offset) / (2 * width**2)).exp()                         # [...] Scalar+


def path_exposures(path: Point, positions: Point, masses: np.ndarray, width: float) -> Iterator[Exposure]:
    """Yields `(unlensed, lensed)` sky brightness maps along each source centre in `path` (`[n_steps] Point-`)."""
    for centre in path:
        yield (
            lambda directions, centre=centre: source_brightness(directions, centre, width),
            lambda directions, centre=centre: source_brightness(source_direction(directions, positions, masses), centre, width),
        )


# --- plumbing -------------------------------------------------------------------------
def sky_grid(half_width: float, n_pixels: int) -> Point:
    """Pixel centres across `[-half_width, half_width]` form a square grid of intrinsic sky directions (`[n_pixels, n_pixels] Point-`)."""
    angles = (np.arange(n_pixels) + 0.5) * (2 * half_width / n_pixels) - half_width
    return (mv.x * angles[None, :] + mv.y * angles[:, None]).dual()            # [n_pixels, n_pixels] Point-
