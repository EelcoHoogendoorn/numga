"""Refraction through a rippled interface and its map to a flat receiver.

Positions and ray directions are antivectors; the wavevectors of the ripples and the plane of the water
surface are covectors, vectors, read against a position or a direction by pairing, `wavevector & position`.
The surface is the level set of its height above the ripples, so its plane at a point is
`mv.z - gradient`, and a ray refracts across that plane by Snell's law. Horizontal positions, on the
water and on the receiver, are the antivectors that contain `z`.
"""

from __future__ import annotations

import jax
import numpy as np

from numga.algebras import VGA3D as ga
from numga.backend.jax import JaxContext

jax.config.update("jax_enable_x64", True)
mv = JaxContext(ga, dtype=np.float64).multivector
Scalar = ga.gatype.scalar()
# Positions and directions are antivectors; wavevectors and planes are covectors, vectors.
Direction = ga.gatype.antivector()
Plane = ga.gatype.vector()
Horizontal = Direction.derive.containing("z")
Wavevector = Plane.derive.within("x y")

# The direction straight up.
up = mv.z.dual()


# --- math -----------------------------------------------------------------------------
def rays(
    position: Horizontal, wavevectors: Wavevector, amplitudes: np.ndarray,
    phases: np.ndarray, index_ratio: float,
) -> tuple[Direction, Plane, Direction]:
    """Surface positions, unit surface planes and transmitted directions for vertical illumination."""
    phase = (wavevectors & position[..., None]) - phases        # [..., waves] Scalar
    height = (amplitudes * phase.cos()).sum(axis=-1)            # [...] Scalar
    gradient = -(wavevectors * (amplitudes * phase.sin())).sum(axis=-1)   # [...] Wavevector
    # The plane of the surface, `z - height == 0`, and the unit direction across it.
    plane = (mv.z - gradient).normalized()                      # [...] Plane
    normal = plane.dual()                                       # [...] Direction

    # Snell's law scales the part of the ray along the surface; unit length fixes the part across it.
    incoming = -up                                              # Direction
    tangent = index_ratio * (incoming - normal * (plane & incoming))   # [...] Direction
    outgoing = tangent - normal * (1 - tangent.scalar_norm_squared()).square_root()   # [...] Direction
    surface = position + up * height                            # [...] Direction
    return surface, plane, outgoing


def receiver(
    position: Horizontal, wavevectors: Wavevector, amplitudes: np.ndarray,
    phases: np.ndarray, depth: float, index_ratio: float,
) -> Horizontal:
    """Horizontal landing positions on the receiver below the refracting surface."""
    surface, _, outgoing = rays(position, wavevectors, amplitudes, phases, index_ratio)
    # The distance along the ray down to the receiver, at height minus the depth.
    distance = (-depth - (mv.z & surface)) / (mv.z & outgoing)  # [...] Scalar
    landing = surface + distance * outgoing                     # [...] Direction
    return landing.cast(Horizontal)
