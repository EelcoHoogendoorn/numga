"""Snell refraction across a rippled air-water interface maps incoming rays onto a flat horizontal receiver.

Three-dimensional geometric algebra models spatial displacements and ray directions as intrinsic antivectors, while wavevectors and local tangent planes are extrinsic vectors. Pairing a horizontal wavevector with a horizontal position via `wavevector & position` evaluates the ripple phase, and subtracting the slope 1-form from the vertical 1-form gives the local interface plane `z - gradient`. Scaling the tangential part of an incident ray by the refractive index ratio implements Snell's law directly on antivector directions, and projecting refracted rays onto the horizontal receiver defines a differentiable map whose determinant vanishes along caustic folds.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra
from numga.algebras import VGA3D as ga
from numga.backend.jax import JaxContext

mv = JaxContext(ga).multivector
Scalar = ga.gatype.scalar()                                                       # Scalar+
# Spatial displacements and ray directions are intrinsic antivectors; planes and wavevectors are extrinsic vectors.
Direction = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                        # Vector+
Wavevector = Plane.derive.within(Algebra("x+y+"))                                 # Vector+
Horizontal = Wavevector.derive.dual                                               # Antivector-

# Unit upward vertical direction.
up = mv.z.dual()                                                                  # [] Direction-


# --- math -----------------------------------------------------------------------------
def refracted_rays(
    position: Horizontal, wavevectors: Wavevector, amplitudes: np.ndarray,
    phases: np.ndarray, index_ratio: float,
) -> tuple[Direction, Plane, Direction]:
    """Vertical illumination hitting the rippled surface at horizontal `position` (`[...] Horizontal-`)
    refracts across unit surface planes (`[...] Plane+`) from surface points (`[...] Direction-`) into
    transmitted unit directions (`[...] Direction-`)."""
    wave_phases = (wavevectors & position[..., None]) - phases                    # [..., n_waves] Scalar+
    height = (amplitudes * wave_phases.cos()).sum(axis=-1)                        # [...] Scalar+
    slope_gradient = -(wavevectors * (amplitudes * wave_phases.sin())).sum(axis=-1)  # [...] Wavevector+
    # Level-set 1-form of `z - height == 0` normalized to unit length, and its dual upward normal.
    surface_plane = (mv.z - slope_gradient).normalized()                          # [...] Plane+
    surface_normal = surface_plane.dual()                                         # [...] Direction-

    # Snell's law scales the component tangential to the interface, and unit norm fixes the downward normal component.
    incoming = -up                                                                # [] Direction-
    tangent_part = index_ratio * (incoming - surface_normal * (surface_plane & incoming))  # [...] Direction-
    outgoing = tangent_part - surface_normal * (1 - tangent_part.scalar_norm_squared()).square_root()  # [...] Direction-
    surface_point = position + up * height                                        # [...] Direction-
    return surface_point, surface_plane, outgoing


def receiver_landing(
    position: Horizontal, wavevectors: Wavevector, amplitudes: np.ndarray,
    phases: np.ndarray, depth: float, index_ratio: float,
) -> Horizontal:
    """Tracing refracted rays from horizontal `position` (`[...] Horizontal-`) down to `z = -depth`
    yields their horizontal landing positions (`[...] Horizontal-`) on the receiver plane."""
    surface_point, _, outgoing = refracted_rays(position, wavevectors, amplitudes, phases, index_ratio)
    ray_distance = (-depth - (mv.z & surface_point)) / (mv.z & outgoing)          # [...] Scalar+
    landing_point = surface_point + ray_distance * outgoing                       # [...] Direction-
    return landing_point.cast(Horizontal)                                         # [...] Horizontal-
