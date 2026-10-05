"""Light's wave surface in a crystal, from extensors in spacetime algebra.

A material is a map from the field, a bivector, to its excitation, built by leaving the field open:
`dielectric` weights the observer's electric planes by their permittivities and the magnetic planes
by one reluctivity. A potential A gives the field `k ^ A`, and `potential_map` is the map from A to
what is left of Maxwell's other equation, `(k ^ material(k ^ A)).dual()`, a map `Vector <- Vector`.
It always loses the potential along k; a wave is a second potential it loses, which its action on
volumes, its outermorphism on trivectors, detects. The adjugate of that action holds two factors of
k and one number more, the Fresnel quartic, `polynomial`, zero where light can travel, and
`radial_sheets` finds its two roots along each direction, the two polarizations.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Antibivector = ga.gatype.antibivector()
Antivector = ga.gatype.antivector()
Constitutive = ga.gatype((Antibivector, Bivector))             # Antibivector <- Bivector
WaveMap = ga.gatype((Vector, Vector))                         # Vector <- Vector
axes = stack([mv.x, mv.y, mv.z])
ROOT_SIGNS = np.array([-1.0, 1.0])


# --- math -----------------------------------------------------------------------------
def dielectric(permittivity: np.ndarray, reluctivity: np.ndarray) -> Constitutive:
    """A reciprocal dielectric at rest along t, with positive principal electric weights.

    Permittivity has shape [..., 3]; reluctivity is the scalar inverse permeability.
    """
    # Weight the electric planes, leave the magnetic planes at their common weight,
    # and dualize the response to obtain the excitation.
    planes = axes ^ mv.t                                                     # [3] Bivector
    electric = (permittivity * planes * (planes | Bivector)).sum(axis=-1)     # [...] Bivector <- Bivector
    # The sandwich by t keeps the magnetic planes, those without t, and turns the electric ones
    # around: half the sum with the identity keeps the magnetic planes alone.
    magnetic = (Bivector + (mv.t >> Bivector)) / 2                            # [] Bivector <- Bivector
    return (electric + reluctivity * magnetic).dual()                         # [...] Antibivector <- Bivector


def potential_map(medium: Constitutive, wavevector: Vector) -> WaveMap:
    """Map a potential to its excitation residual; wavevector always lies in its kernel."""
    field = wavevector ^ Vector                                              # [...] Bivector <- Vector
    return (wavevector ^ medium(field)).dual()                                # [...] Vector <- Vector


def polynomial(medium: Constitutive, observer: Vector, wavevector: Vector) -> Scalar:
    """The Fresnel quartic, for a nonzero frequency measured by the unit timelike observer."""
    wave = potential_map(medium, wavevector)                                  # [...] Vector <- Vector
    adjugate = wave.outermorphism(Antivector).adjugate()                      # [...] Vector <- Vector
    frequency = observer | wavevector                                         # [...] Scalar
    return -(observer | adjugate(observer)) / frequency.squared()             # [...] Scalar


def radial_sheets(medium: Constitutive, directions: Vector, frequency: float) -> Vector:
    """Both wavevector sheets of a rest dielectric along unit spatial directions.

    The quartic is even in the spatial radius. Three samples determine its quadratic
    in radius squared, whose two positive roots are the two polarization sheets.
    """
    radius_squared = np.arange(3)
    probes = mv.t * frequency + directions[..., None] * np.sqrt(radius_squared)  # [..., samples] Vector
    values = polynomial(medium[..., None], mv.t, probes)                      # [..., samples] Scalar
    constant, unit, twice = (values[..., i] for i in range(3))                # [...] Scalar each
    quadratic = (twice - 2 * unit + constant) / 2                             # [...] Scalar
    linear = unit - constant - quadratic                                      # [...] Scalar
    # Where the sheets coincide the discriminant vanishes, up to round-off of either sign.
    discriminant = (linear.squared() - 4 * quadratic * constant).abs().square_root()  # [...] Scalar
    radii_squared = (-linear[..., None] + discriminant[..., None] * ROOT_SIGNS) / (2 * quadratic[..., None])  # [..., sheets] Scalar
    return directions[..., None] * radii_squared.square_root()                # [..., sheets] Vector


# --- plumbing -------------------------------------------------------------------------
def sphere(latitudes: int, longitudes: int) -> Vector:
    """A sphere of directions, made by two turns in spacetime's spatial planes."""
    polar = np.linspace(0, np.pi, latitudes)
    azimuth = np.linspace(0, 2 * np.pi, longitudes)
    meridian = (mv.zx * (polar[:, None] / 2)).exp() >> mv.z                   # [latitudes, 1] Vector
    return (mv.xy * (azimuth[None, :] / 2)).exp() >> meridian                 # [latitudes, longitudes] Vector


def circle(samples: int) -> Vector:
    angles = np.linspace(0, 2 * np.pi, samples)
    return (mv.zx * (angles / 2)).exp() >> mv.z                               # [samples] Vector
