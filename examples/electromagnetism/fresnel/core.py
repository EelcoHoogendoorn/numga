"""Fresnel wave surfaces of anisotropic dielectrics in spacetime algebra.

A linear medium maps each electromagnetic field bivector to an excitation antibivector through a
weighted sum of area-patch dyads `patch * (Bivector & patch)`. For a candidate wavevector `k` and
potential `A`, the source-free Maxwell equation requires the driven 4-current `k ^ medium(k ^ A)`
to vanish. Gauge potentials parallel to `k` always lie in the kernel of this potential map
`Antivector <- Vector`, and a propagating wave exists when a second independent potential lies in
the kernel so the induced 3-volume outermorphism `wave.outermorphism(Trivector)` vanishes. Pairing
that volume map on the observer's rest space divides out `(k & time).squared()` and isolates the
metric-free Fresnel quartic, while a chiral axion term `coupling * Bivector` drops out identically
because `k ^ (coupling * (k ^ A)) == 0`.
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
# Potentials are extrinsic vectors; the 4-currents they drive are intrinsic antivectors.
Antivector = ga.gatype.antivector()                             # Antivector-
Constitutive = ga.gatype((Antibivector, Bivector))              # Antibivector- <- Bivector+
WaveMap = ga.gatype((Antivector, Vector))                       # Antivector- <- Vector+
# A chiral coupling of excitation to field carries intrinsic scalar orientation.
AxionCoupling = ga.gatype.antipseudoscalar()                    # Scalar-

# Unit timelike direction, spatial axis directions, and spatial coordinate planes of the rest observer.
time = -mv.t.dual()                                             # [] Antivector-
axes = stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))           # [3] Antivector-
magnetic_planes = stack((mv.yz, mv.zx, mv.xy))                  # [3] Bivector+
# Intrinsic worldsheet patches read electric components; intrinsic spatial patches read magnetic components.
swept_planes = axes & time                                      # [3] Antibivector-
patches = magnetic_planes.dual()                                # [3] Antibivector-
rest_space = mv.xyz                                             # [] Trivector+
root_signs = np.array([-1, 1])


# --- math -----------------------------------------------------------------------------
def dielectric(permittivity: np.ndarray, reluctivity: np.ndarray) -> Constitutive:
    """Electric swept-plane dyads weighted by `permittivity` (`[..., 3]`) and magnetic spatial-patch dyads weighted by `reluctivity` (`[...]`) sum to the rest-frame constitutive map (`[...] Antibivector- <- Bivector+`)."""
    # Time squares to -1, so electric swept-plane dyads enter with a minus sign.
    electric = -(permittivity * swept_planes * (Bivector & swept_planes)).sum(axis=-1)  # [...] Antibivector- <- Bivector+
    magnetic = (patches * (Bivector & patches)).sum(axis=-1)                            # [] Antibivector- <- Bivector+
    return electric + reluctivity * magnetic                                            # [...] Antibivector- <- Bivector+


def potential_map(medium: Constitutive, wavevector: Vector) -> WaveMap:
    """Wedging a potential with `wavevector` and applying `medium` maps each candidate potential to the 4-current it would require (`[...] Antivector- <- Vector+`)."""
    field = wavevector ^ Vector                                               # [...] Bivector+ <- Vector+
    return wavevector ^ medium(field)                                         # [...] Antivector- <- Vector+


def fresnel_quartic(
    medium: Constitutive,
    observer_time: Antivector,
    observer_space: Trivector,
    wavevector: Vector,
) -> Scalar:
    """Volume action of the potential map on `observer_space`, normalized by squared observer frequency, gives the Fresnel quartic (`[...] Scalar+`)."""
    wave = potential_map(medium, wavevector)                                  # [...] Antivector- <- Vector+
    # A physical wave adds a second null direction beyond the gauge wavevector, so 3-volumes collapse.
    volumes = wave.outermorphism(Trivector)                                   # [...] Vector- <- Trivector+
    measured_frequency = wavevector & observer_time                           # [...] Scalar+
    return (observer_space & volumes(observer_space)) / measured_frequency.squared()  # [...] Scalar+


def sphere(n_latitudes: int, n_longitudes: int) -> Vector:
    """Polar and azimuthal rotor turns sweep a unit spatial plane across a sphere of wavefronts (`[n_latitudes, n_longitudes] Vector+`)."""
    polar = np.linspace(0, np.pi, n_latitudes)
    azimuth = np.linspace(0, 2 * np.pi, n_longitudes)
    meridian_rotors = (mv.zx * (-polar[:, None] / 2)).exp()                   # [n_latitudes, 1] Rotor+
    meridian = meridian_rotors >> mv.z                                        # [n_latitudes, 1] Vector+
    equatorial_rotors = (mv.xy * (-azimuth[None, :] / 2)).exp()               # [1, n_longitudes] Rotor+
    return equatorial_rotors >> meridian                                      # [n_latitudes, n_longitudes] Vector+


def circle(n_samples: int) -> Vector:
    """Rotor turns in the xz plane trace a circle of unit spatial wavefronts (`[n_samples] Vector+`)."""
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    rotors = (mv.zx * (-angles / 2)).exp()                                    # [n_samples] Rotor+
    return rotors >> mv.z                                                     # [n_samples] Vector+


def radial_sheets(medium: Constitutive, wavefronts: Vector, frequency: float) -> Vector:
    """Quadratic interpolation in the squared wavenumber yields both polarization sheets (`[..., 2] Vector+`) along unit spatial `wavefronts` (`[...] Vector+`)."""
    wavenumber_squared = np.arange(3)
    probes = mv.t * frequency + wavefronts[..., None] * np.sqrt(wavenumber_squared)  # [..., 3] Vector+
    values = fresnel_quartic(medium[..., None], time, rest_space, probes)     # [..., 3] Scalar+
    constant, unit, twice = (values[..., i] for i in range(3))                # [...] Scalar+, Scalar+, Scalar+
    quadratic = (twice - 2 * unit + constant) / 2                             # [...] Scalar+
    linear = unit - constant - quadratic                                      # [...] Scalar+
    # Where the sheets coincide the discriminant vanishes up to round-off of either sign.
    discriminant = (linear.squared() - 4 * quadratic * constant).abs().square_root()  # [...] Scalar+
    wavenumbers_squared = (-linear[..., None] + discriminant[..., None] * root_signs) / (2 * quadratic[..., None])  # [..., 2] Scalar+
    return wavefronts[..., None] * wavenumbers_squared.square_root()          # [..., 2] Vector+
