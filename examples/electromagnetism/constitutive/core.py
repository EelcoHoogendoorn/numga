"""Constitutive maps send an electromagnetic field bivector to an excitation antibivector via `excitation = medium(field)`.

A medium at rest responds to each part of the field an observer distinguishes with a dyad formed from the patch that reads that part, times its reading of the field, weighted by the material constant. The planes each spatial axis sweeps through time read the electric field and carry the permittivities, while the patches of the spatial planes read the magnetic field and carry the inverse permeabilities. Glass weights all three dyads of each kind alike, a crystal weights its electric dyads apart, and a ferrite weights its magnetic ones; the metric enters only through the material map itself. For a moving medium, the Lorentz boost carries its rest response via `boost >> medium(boost << Field)`, and an axion coupling adds a term proportional to the field itself. Pairing the excitation with a second field forms the constitutive bilinear form on fields, while the field and excitation together form the stress-energy map from spacetime directions to energy-momentum currents. A plane wave is a field that both source-free Maxwell equations annihilate along its wavevector.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, dim, stack
from numga.algebras import STA

mv = NumpyContext(STA, units=True).multivector

force_dim = dim.mass * dim.length / dim.time**2
field_dim = force_dim / (dim.current * dim.length)
permeability_dim = force_dim / dim.current**2
energy_density_dim = force_dim / dim.length**2

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

# Coherent unit constants keep every quantity O(1) while tracking all electromagnetic dimensions.
c = mv.scalar(1.0, unit=dim.length / dim.time)                                  # [] Scalar+, length/time
mu_0 = mv.scalar(1.0, unit=permeability_dim)                                    # [] Scalar+, length*mass/(current^2*time^2)
eps_0 = 1 / (mu_0 * c**2)                                                       # [] Scalar+, current^2*time^4/(length^3*mass)

t, z = mv.t, mv.z                                                               # Vector+, Vector+
# Time covector scaled by 1 / c so `omega * t_s - k * z` pairs angular frequency with wavenumber.
t_s = t / c                                                                     # [] Vector+, time/length
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
vacuum = Field.dual() / mu_0                                                    # [] Antibivector- <- Bivector+, 1/permeability
# Reference wavenumber and angular frequency of the travelling waves across one period.
wavenumber = mv.scalar(1.0, unit=1 / dim.length)                                # [] Scalar+, 1/length
omega = c * wavenumber                                                          # [] Scalar+, 1/time


# --- math -----------------------------------------------------------------------------
def medium(permittivities: Scalar, inverse_permeabilities: Scalar) -> Constitutive:
    """Weighted electric and magnetic dyads sum to the rest-frame constitutive map (`[...] Antibivector- <- Bivector+`) for `permittivities` and `inverse_permeabilities` (`[..., 3]`)."""
    return (c**2 * permittivities * electric_dyads + inverse_permeabilities * magnetic_dyads).sum(axis=-1)


def moving_medium(rest_medium: Constitutive, velocities: Scalar) -> Constitutive:
    """Lorentz boosts along `z` at `velocities` carry `rest_medium` into the observer's frame (`[...] Antibivector- <- Bivector+`)."""
    boosts = ((z ^ t) * ((velocities / c).arctanh() / 2)).exp()                 # [...] Even+
    return boosts >> rest_medium(boosts << Field)                               # [...] Antibivector- <- Bivector+


def stress_energy(field: Field, excitation: Excitation) -> StressEnergy:
    """The Minkowski stress-energy map (`[...] Antivector- <- Antivector-`) sends each spacetime direction to the energy-momentum current of `field` and `excitation`."""
    # An electric plane pairs with its own dual to minus its square, so pairing the field with its
    # excitation gives minus twice the Lagrangian.
    negative_lagrangian = (field & excitation) / 2                              # [...] Scalar+, energy/length^3
    # No metric enters: the current combines the Lagrangian along the direction with the interior readout
    # of the field wedged against the excitation.
    return negative_lagrangian * Direction - ((Direction & field) ^ excitation)


def wave_map(wavevectors: Covector, media: Constitutive) -> WaveMap:
    """The two exterior source-free Maxwell equations along `wavevectors` occupy separate odd grades (`[...] Odd+ <- Bivector+`)."""
    # Scaling the excitation equation by `mu_0` puts both Maxwell equations on the common wavenumber dimension
    # while keeping them in grades 1 and 3 so they cannot cancel.
    return (wavevectors ^ Field) + mu_0 * (wavevectors ^ media).dual()


def wave_residuals(wavevectors: Covector, media: Constitutive) -> Scalar:
    """The smallest singular value (`[...] Scalar+`, `1/length`) of the wave map vanishes along wavevectors that carry a plane wave."""
    return wave_map(wavevectors, media).svdvals()[..., -1]


def field_modes(wavevectors: Covector, media: Constitutive) -> Field:
    """The trailing right singular vector of the wave map spans the null-space wave field (`[...] Bivector+`)."""
    _, _, fields = wave_map(wavevectors, media).svd()
    return fields[..., -1]


def travelling_fields(speeds: Scalar, fields: Field, distances: Scalar, n_frames: int) -> Iterator[Field]:
    """Yields the superposed travelling wave field (`[n_samples] Bivector+`, `mass/(current*time^2)`) across `distances` over `n_frames` evenly spaced times of one period."""
    # Scale each mode to unit field strength in `|E| / c` as read by an observer at rest (`field & time`).
    units = (fields / (fields & time).norm()) * field_dim                       # [2] Bivector+, mass/(current*time^2)
    period = 2.0 * np.pi / omega                                                # [] Scalar+, time
    for step_fraction in np.linspace(0.0, 1.0, n_frames, endpoint=False):
        tau = step_fraction * period                                            # [] Scalar+, time
        phases = (omega * (distances[:, None] / speeds - tau)).cos()            # [n_samples, 2] Scalar+
        yield (units * phases).sum(axis=-1)                                     # [n_samples] Bivector+, mass/(current*time^2)


# --- plumbing -------------------------------------------------------------------------
def local_minima(residual: Scalar, threshold: Scalar) -> np.ndarray:
    """Interior speed samples (`[n_speeds - 2, ...]`) reach a wave mode where `residual` forms a local minimum below `threshold`."""
    left, middle, right = residual[:-2], residual[1:-1], residual[2:]
    return (middle <= left) & (middle <= right) & (middle < threshold)


def minimum_speeds(speeds: Scalar, residual: Scalar) -> Scalar:
    """The minimum of `residual` along the leading sample axis selects the phase speed (`[...] Scalar+`, `length/time`) from `speeds`."""
    return speeds[residual.argmin(axis=0)]
