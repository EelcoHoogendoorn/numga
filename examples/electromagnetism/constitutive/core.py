"""Constitutive maps send an electromagnetic field bivector to an excitation antibivector via `excitation = medium(field)`.

A medium at rest responds to each part of the field an observer distinguishes with a dyad formed from the patch that reads that part, times its reading of the field, weighted by the material constant. The planes each spatial axis sweeps through time read the electric field and carry the permittivities, while the patches of the spatial planes read the magnetic field and carry the inverse permeabilities. Glass weights all three dyads of each kind alike, a crystal weights its electric dyads apart, and a ferrite weights its magnetic ones; the metric enters only through the material map itself. For a moving medium, the Lorentz boost carries its rest response via `boost >> medium(boost << Field)`, and an axion coupling adds a term proportional to the field itself. Pairing the excitation with a second field forms the constitutive bilinear form on fields, while the field and excitation together form the stress-energy map from spacetime directions to energy-momentum currents. A plane wave is a field that both source-free Maxwell equations annihilate along its wavevector.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA

mv = NumpyContext(STA).multivector
Scalar = STA.gatype.scalar()                                                    # Scalar+
Covector = STA.gatype.vector()                                                  # Vector+
Field = STA.gatype.bivector()                                                   # Bivector+
Excitation = STA.gatype.antibivector()                                          # Antibivector-
Direction = STA.gatype.antivector()                                             # Antivector-
Constitutive = STA.gatype((Excitation, Field))                                  # Antibivector- <- Bivector+
# An axion coupling is handed: an intrinsic scalar, with the orientation flipped.
AxionCoupling = STA.gatype.antipseudoscalar()                                   # Scalar-
StressEnergy = STA.gatype((Direction, Direction))                               # Antivector- <- Antivector-
WaveMap = STA.gatype((STA.gatype.odd(), Field))                                 # Odd+ <- Bivector+

t, z = mv.t, mv.z                                                               # Vector+, Vector+
# The observer's direction in time and spatial axes, and the spatial planes across each axis.
time = -t.dual()                                                                # [] Antivector-
axes = stack((mv.x.dual(), mv.y.dual(), z.dual()))                              # [3] Antivector-
magnetic_planes = stack((mv.yz, mv.zx, mv.xy))                                  # [3] Bivector+
# The planes each axis sweeps through time read the electric field, and the patches of the spatial planes
# read the magnetic field.
swept_planes = axes & time                                                      # [3] Antibivector-
patches = magnetic_planes.dual()                                                # [3] Antibivector-
# Time squares to minus one, so the electric dyads enter with a minus sign that makes the Lagrangian
# equal the electric field squared minus the magnetic field squared.
electric_dyads = -swept_planes * (Field & swept_planes)                         # [3] Antibivector- <- Bivector+
magnetic_dyads = patches * (Field & patches)                                    # [3] Antibivector- <- Bivector+
vacuum = Field.dual()                                                           # [] Antibivector- <- Bivector+
# Angular frequency of the travelling waves across one period.
omega = 1.0


# --- math -----------------------------------------------------------------------------
def medium(permittivities: np.ndarray, inverse_permeabilities: np.ndarray) -> Constitutive:
    """Weighted electric and magnetic dyads sum to the rest-frame constitutive map (`[...] Antibivector- <- Bivector+`) for permittivities and inverse permeabilities (`[..., 3]`)."""
    return (permittivities * electric_dyads + inverse_permeabilities * magnetic_dyads).sum(axis=-1)


def moving_medium(rest_medium: Constitutive, fractions: np.ndarray) -> Constitutive:
    """Lorentz boosts along `z` at speed `fractions` carry `rest_medium` into the observer's frame (`[...] Antibivector- <- Bivector+`)."""
    boosts = ((z ^ t) * (np.arctanh(fractions) / 2)).exp()                      # [...] Even+
    return boosts >> rest_medium(boosts << Field)                               # [...] Antibivector- <- Bivector+


def stress_energy(field: Field, excitation: Excitation) -> StressEnergy:
    """The Minkowski stress-energy map (`[...] Antivector- <- Antivector-`) sends each spacetime direction to the energy-momentum current of `field` and `excitation`."""
    # An electric plane pairs with its own dual to minus its square, so pairing the field with its
    # excitation gives minus twice the Lagrangian.
    negative_lagrangian = (field & excitation) / 2                              # [...] Scalar+
    # No metric enters: the current combines the Lagrangian along the direction with the interior readout
    # of the field wedged against the excitation.
    return negative_lagrangian * Direction - ((Direction & field) ^ excitation)


def wave_map(wavevectors: Covector, media: Constitutive) -> WaveMap:
    """The two exterior source-free Maxwell equations along `wavevectors` occupy separate odd grades (`[...] Odd+ <- Bivector+`)."""
    # Dualizing the excitation equation places it in grade 1 while `wavevectors ^ Field` sits in grade 3,
    # keeping the two equations from cancelling.
    return (wavevectors ^ Field) + (wavevectors ^ media).dual()


def wave_residuals(wavevectors: Covector, media: Constitutive) -> Scalar:
    """The smallest singular value (`[...] Scalar+`) of the wave map vanishes along wavevectors that carry a plane wave."""
    return wave_map(wavevectors, media).svdvals()[..., -1]


def field_modes(wavevectors: Covector, media: Constitutive) -> Field:
    """The trailing right singular vector of the wave map spans the null-space wave field (`[...] Bivector+`)."""
    _, _, fields = wave_map(wavevectors, media).svd()
    return fields[..., -1]


def travelling_fields(speeds: np.ndarray, fields: Field, distances: np.ndarray, n_frames: int) -> Iterator[Field]:
    """Yields the superposed travelling wave field (`[n_samples] Bivector+`) across `distances` over `n_frames` evenly spaced times of one period."""
    # Scale each mode to unit electric-field strength as read by an observer at rest (`field & time`).
    units = fields / (fields & time).norm()                                     # [2] Bivector+
    for tau in np.linspace(0.0, 2.0 * np.pi / omega, n_frames, endpoint=False):
        phases = np.cos(omega * (distances[:, None] / speeds - tau))            # [n_samples, 2]
        yield (units * phases).sum(axis=-1)                                     # [n_samples] Bivector+


# --- plumbing -------------------------------------------------------------------------
def local_minima(residual: Scalar, threshold: float) -> np.ndarray:
    """Interior speed samples (`[n_speeds - 2, ...]`) reach a wave mode where `residual` forms a local minimum below `threshold`."""
    left, middle, right = residual[:-2], residual[1:-1], residual[2:]
    return (middle <= left) & (middle <= right) & (middle < threshold)


def minimum_speeds(speeds: np.ndarray, residual: Scalar) -> np.ndarray:
    """The minimum of `residual` along the leading sample axis selects the phase speed (`[...]`) from `speeds`."""
    return speeds[residual.argmin(axis=0)]
