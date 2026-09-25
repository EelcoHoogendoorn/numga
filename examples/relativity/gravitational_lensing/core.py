"""Point masses bending light, in the geometric algebra of the plane of the sky.

A direction on the sky is a vector, in units of the Einstein angle of the total mass. Each mass
deflects a sightline towards itself by the inverse of the separation, weighted by its mass, so the
direction a sightline reaches at the source is `observed - (masses * separation.inverse()).sum(axis=-1)`.

The local map carries a small displacement of the sightline to the displacement it makes at the
source. The change of a vector's inverse is a sandwich: to first order
`(separation + small).inverse() - separation.inverse()` is `-(separation.inverse() >> small)`. Each
mass adds to the identity its own reflection in the line of the separation, scaled by the inverse
square distance. A reflection has no trace, so the local map's trace is two everywhere, and a map of
the plane with trace two is undone by `(2 * Vector - local) / area`, with `area` the ratio its
outermorphism carries areas by. Its two stretches are one plus and one minus the square root of
`1 - area`.

Where the area ratio is zero one direction collapses: the critical curve on the sky. The lens
carries it to the caustic at the source. A source crossing the caustic gains or loses a pair of
images, one on each side of the critical curve. Surface brightness is conserved along a ray, so each
sky direction shows the source's brightness at the direction it reaches, and larger images carry
more light; the total light over the source's own is the magnification.

In the complex notation of lensing theory a direction reads as a complex number, the sum of the
reflections as the shear, each mass over the square of the conjugate separation, and the area ratio
as one less the shear's squared modulus. Reference: M. Dominik, The binary gravitational lens and
its extreme cases (1999), section 2, https://arxiv.org/abs/astro-ph/9903014
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext

ga = Algebra("x+y+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Area = ga.gatype.bivector()
LocalMap = ga.gatype((Vector, Vector))                                         # Vector <- Vector


# --- math -----------------------------------------------------------------------------
def deflected(observed: Vector, positions: Vector, masses: np.ndarray) -> Vector:
    """The source direction each observed direction reaches."""
    separation = observed[..., None] - positions                               # [..., masses] Vector
    return observed - (masses * separation.inverse()).sum(axis=-1)             # [...] Vector


def local_map(observed: Vector, positions: Vector, masses: np.ndarray) -> LocalMap:
    """The map from a small displacement of each observed direction to the displacement it makes at
    the source: the identity plus each mass's reflection in the line of its separation."""
    separation = observed[..., None] - positions                               # [..., masses] Vector
    return Vector + (masses * (separation.inverse() >> Vector)).sum(axis=-1)   # [...] Vector <- Vector


def brightness(directions: Vector, centre: Vector, width: float) -> Scalar:
    """A round Gaussian source of unit peak brightness and the given angular standard deviation."""
    offset = directions - centre                                               # [...] Vector
    return (-(offset | offset) / (2 * width**2)).exp()                         # [...] Scalar


# --- plumbing -------------------------------------------------------------------------
def sky(half_width: float, resolution: int) -> Vector:
    """Directions at the centres of a square of pixels about the optical axis."""
    angles = (np.arange(resolution) + 0.5) * (2 * half_width / resolution) - half_width
    return mv.x * angles[None, :] + mv.y * angles[:, None]                    # [rows, columns] Vector

