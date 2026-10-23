"""Fresnel wave surfaces: light's wave surface in a crystal, from extensors in spacetime algebra.

A material is a map from the field, a bivector, to its excitation, built by multiplying by the space
of fields: `dielectric` weights the observer's electric planes by their permittivities and the magnetic planes
by one reluctivity. A potential A gives the field `k ^ A`, and `potential_map` is the map from A to
what is left of Maxwell's other equation, the current the potential would need, `k ^ material(k ^ A)`:
from potentials, extrinsic, to currents, intrinsic, `Antivector <- Vector`, with no metric but the
material's. It always loses the potential along k; a wave is a second potential it loses, which its action
on volumes detects, its outermorphism from the volume of three potentials to the join of their currents.
Read back on the observer's rest space, `rest_space & volumes(rest_space)`, it holds two factors of the
frequency and one number more, the Fresnel quartic, `polynomial`, zero where light can travel, and
`radial_sheets` finds its two roots along each direction, the two polarizations.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, Orientation, stack
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Antibivector = ga.gatype.antibivector()
Trivector = ga.gatype.trivector()
# Potentials are vectors, extrinsic; the currents they drive are antivectors, intrinsic.
Antivector = ga.gatype.antivector()
Constitutive = ga.gatype((Antibivector, Bivector))             # Antibivector <- Bivector
WaveMap = ga.gatype((Antivector, Vector))                     # Antivector <- Vector
# A coupling of the excitation to the field itself, with a sense of its own: measured against a handedness.
Axion = ga.gatype(ga.subspace.scalar(), orientation=Orientation.intrinsic)
axes = stack([mv.x, mv.y, mv.z])
root_signs = np.array([-1.0, 1.0])


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
    magnetic = (Bivector + mv.t * Bivector * mv.t.inverse()) / 2              # [] Bivector <- Bivector
    return (electric + reluctivity * magnetic).dual()                         # [...] Antibivector <- Bivector


def axion(coupling: Axion) -> Constitutive:
    """The axion part of a material: an excitation proportional to the field itself. Field and
    excitation share their blades in spacetime and differ in orientation, so the coupling is an
    intrinsic scalar, and the mirror image of the material has the opposite coupling. In Maxwell's other
    equation it meets the field's own wavevector, `k ^ (coupling * (k ^ A))`, which vanishes: light
    does not see it."""
    return coupling * Bivector                                                    # [...] Antibivector <- Bivector


def potential_map(medium: Constitutive, wavevector: Vector) -> WaveMap:
    """Map a potential to the current it would need, its excitation's residual; the wavevector always
    lies in its kernel."""
    field = wavevector ^ Vector                                              # [...] Bivector <- Vector
    return wavevector ^ medium(field)                                         # [...] Antivector <- Vector


def polynomial(medium: Constitutive, rest_space: Trivector, wavevector: Vector) -> Scalar:
    """The Fresnel quartic, for a nonzero frequency measured by the observer whose unit rest space is
    `rest_space`; the volume enters twice, so its sense drops out."""
    wave = potential_map(medium, wavevector)                                  # [...] Antivector <- Vector
    # The potential map's action on volumes: from the volume of three potentials to the join of their
    # currents.
    volumes = wave.outermorphism(Trivector)                                   # [...] intrinsic Vector <- Trivector
    # The frequency the observer measures: the wavevector paired with its rest space.
    frequency = wavevector & rest_space                                       # [...] Scalar
    # The rest space reads the join of the currents of its own three directions.
    return (rest_space & volumes(rest_space)) / frequency.squared()           # [...] Scalar


def radial_sheets(medium: Constitutive, directions: Vector, frequency: float) -> Vector:
    """Both wavevector sheets of a rest dielectric along unit spatial directions.

    The quartic is even in the spatial radius. Three samples determine its quadratic
    in radius squared, whose two positive roots are the two polarization sheets.
    """
    radius_squared = np.arange(3)
    probes = mv.t * frequency + directions[..., None] * np.sqrt(radius_squared)  # [..., samples] Vector
    values = polynomial(medium[..., None], mv.xyz, probes)                    # [..., samples] Scalar
    constant, unit, twice = (values[..., i] for i in range(3))                # [...] Scalar each
    quadratic = (twice - 2 * unit + constant) / 2                             # [...] Scalar
    linear = unit - constant - quadratic                                      # [...] Scalar
    # Where the sheets coincide the discriminant vanishes, up to round-off of either sign.
    discriminant = (linear.squared() - 4 * quadratic * constant).abs().square_root()  # [...] Scalar
    radii_squared = (-linear[..., None] + discriminant[..., None] * root_signs) / (2 * quadratic[..., None])  # [..., sheets] Scalar
    return directions[..., None] * radii_squared.square_root()                # [..., sheets] Vector


# --- plumbing -------------------------------------------------------------------------
def sphere(latitudes: int, longitudes: int) -> Vector:
    """A sphere of directions, made by two turns in spacetime's spatial planes."""
    polar = np.linspace(0, np.pi, latitudes)
    azimuth = np.linspace(0, 2 * np.pi, longitudes)
    meridian = (mv.zx * (-polar[:, None] / 2)).exp() >> mv.z                  # [latitudes, 1] Vector
    return (mv.xy * (-azimuth[None, :] / 2)).exp() >> meridian                # [latitudes, longitudes] Vector


def circle(samples: int) -> Vector:
    angles = np.linspace(0, 2 * np.pi, samples)
    return (mv.zx * (-angles / 2)).exp() >> mv.z                              # [samples] Vector
