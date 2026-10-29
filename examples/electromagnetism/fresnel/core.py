"""Fresnel wave surfaces: light's wave surface in a crystal, from extensors in spacetime algebra.

A material is a map from the field, a bivector, to its excitation, an antibivector, a sum of dyads: each
patch that reads a part of the field, times its reading of the field, weighted by its material constant. The
planes each axis of space sweeps through time read the electric field and carry the permittivities; the
patches of the planes of space read the magnetic field and share one reluctivity. A potential A gives the
field `k ^ A`, and `potential_map` is the map from A to what is left of Maxwell's other equation, the current
the potential would need, `k ^ material(k ^ A)`: from potentials, extrinsic, to currents, intrinsic,
`Antivector <- Vector`, with no metric but the material's. It always loses the potential along k; a wave is a
second potential it loses, which its action on volumes detects, its outermorphism from the volume of three
potentials to the join of their currents. Read back on the observer's rest space,
`rest_space & volumes(rest_space)`, it holds two factors of the frequency `k & time` and one number more, the
Fresnel quartic, `polynomial`, zero where light can travel, and `radial_sheets` finds its two roots along each
wavefront, the two polarizations. An axion term, a coupling times the field itself, `coupling * Bivector`,
meets the field's own wavevector in Maxwell's other equation, `k ^ (coupling * (k ^ A))`, which vanishes:
light does not see it.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                     # Scalar+
Vector = ga.gatype.vector()                                     # Vector+
Bivector = ga.gatype.bivector()                                 # Bivector+
Antibivector = ga.gatype.antibivector()                         # Antibivector-
Trivector = ga.gatype.trivector()                               # Trivector+
# Potentials are vectors, extrinsic; the currents they drive are antivectors, intrinsic.
Antivector = ga.gatype.antivector()                             # Antivector-
Constitutive = ga.gatype((Antibivector, Bivector))              # Antibivector- <- Bivector+
WaveMap = ga.gatype((Antivector, Vector))                       # Antivector- <- Vector+
# A coupling of the excitation to the field itself, with a sense of its own: an intrinsic scalar.
AxionCoupling = Scalar.derive.flipped_orientation               # Scalar-

# The observer's direction in time and its axes of space, and the planes of space across each axis.
time = -mv.t.dual()                                             # [] Antivector-
axes = stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))           # [3] Antivector-
magnetic_planes = stack((mv.yz, mv.zx, mv.xy))                  # [3] Bivector+
# The planes each axis sweeps through time, which read the electric field, and the patches of the planes of
# space, which read the magnetic field: intrinsic, both.
swept_planes = axes & time                                      # [3] Antibivector-
patches = magnetic_planes.dual()                                # [3] Antibivector-
# The observer's rest space, the volume of its three spatial potentials.
rest_space = mv.xyz                                             # [] Trivector+
root_signs = np.array([-1, 1])


# --- math -----------------------------------------------------------------------------
def dielectric(permittivity: np.ndarray, reluctivity: np.ndarray) -> Constitutive:
    """A reciprocal dielectric at rest along `time`, from its permittivity along each axis `[..., 3]` and
    its one reluctivity, the inverse permeability `[...]`: the dyads of the patches that read the field, each
    weighted by its material constant."""
    # Each dyad is a patch times its reading of the field. Time squares to minus one, so the electric dyads
    # enter with a minus.
    electric = -(permittivity * swept_planes * (Bivector & swept_planes)).sum(axis=-1)
    magnetic = (patches * (Bivector & patches)).sum(axis=-1)
    return electric + reluctivity * magnetic                                  # [...] Antibivector- <- Bivector+


def potential_map(medium: Constitutive, wavevector: Vector) -> WaveMap:
    """The map from a potential to the current it would need, its excitation's residual in Maxwell's other
    equation; the wavevector always lies in its kernel."""
    field = wavevector ^ Vector                                               # [...] Bivector+ <- Vector+
    return wavevector ^ medium(field)                                         # [...] Antivector- <- Vector+


def polynomial(medium: Constitutive, time: Antivector, rest_space: Trivector, wavevector: Vector) -> Scalar:
    """The Fresnel quartic, for a nonzero frequency measured by the observer whose unit direction in time is
    `time` and whose unit rest space is `rest_space`."""
    wave = potential_map(medium, wavevector)                                  # [...] Antivector- <- Vector+
    # The potential map's action on volumes: from the volume of three potentials to the join of their
    # currents.
    volumes = wave.outermorphism(Trivector)                                   # [...] Vector- <- Trivector+
    # The frequency the observer measures: the wavevector read along its direction in time.
    frequency = wavevector & time                                             # [...] Scalar+
    # The rest space reads the join of the currents of its own three potentials.
    return (rest_space & volumes(rest_space)) / frequency.squared()           # [...] Scalar+


def radial_sheets(medium: Constitutive, wavefronts: Vector, frequency: float) -> Vector:
    """Both wavevector sheets of a rest dielectric, along unit wavefronts, planes of space.

    The quartic is even in the wavenumber. Three samples determine its quadratic in the wavenumber squared,
    whose two positive roots are the two polarization sheets.
    """
    wavenumber_squared = np.arange(3)
    probes = mv.t * frequency + wavefronts[..., None] * np.sqrt(wavenumber_squared)  # [..., samples] Vector+
    values = polynomial(medium[..., None], time, rest_space, probes)              # [..., samples] Scalar+
    constant, unit, twice = (values[..., i] for i in range(3))                # [...] Scalar+, Scalar+, Scalar+
    quadratic = (twice - 2 * unit + constant) / 2                             # [...] Scalar+
    linear = unit - constant - quadratic                                      # [...] Scalar+
    # Where the sheets coincide the discriminant vanishes, up to round-off of either sign.
    discriminant = (linear.squared() - 4 * quadratic * constant).abs().square_root()  # [...] Scalar+
    wavenumbers_squared = (-linear[..., None] + discriminant[..., None] * root_signs) / (2 * quadratic[..., None])  # [..., sheets] Scalar+
    return wavefronts[..., None] * wavenumbers_squared.square_root()          # [..., sheets] Vector+


# --- plumbing -------------------------------------------------------------------------
def sphere(n_latitudes: int, n_longitudes: int) -> Vector:
    """A sphere of unit wavefronts, planes of space, made by two turns in spacetime's spatial planes."""
    polar = np.linspace(0, np.pi, n_latitudes)
    azimuth = np.linspace(0, 2 * np.pi, n_longitudes)
    meridian = (mv.zx * (-polar[:, None] / 2)).exp() >> mv.z                  # [latitudes, 1] Vector+
    return (mv.xy * (-azimuth[None, :] / 2)).exp() >> meridian                # [latitudes, longitudes] Vector+


def circle(n_samples: int) -> Vector:
    """A circle of unit wavefronts, turned in the xz plane."""
    angles = np.linspace(0, 2 * np.pi, n_samples)
    return (mv.zx * (-angles / 2)).exp() >> mv.z                              # [samples] Vector+
