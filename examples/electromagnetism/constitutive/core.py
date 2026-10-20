"""Constitutive maps: a material's response to the electromagnetic field, a map from the field, a
bivector, to the excitation it causes, an antibivector, `excitation = medium(field)`.

A medium at rest weights each of the observer's electric planes, a spatial axis against the time
direction, by its permittivity, and each magnetic plane, the dual of an electric one, by its inverse
permeability: glass weights all three alike, a crystal its electric planes apart, a ferrite its magnetic
ones. Seen moving, a medium is its rest map carried by the boost, `boost >> medium(boost << Bivector)`,
and an axion response adds a term proportional to the field itself. Pairing the excitation with a second
field gives the constitutive form, a form on pairs of fields, and the field and excitation together give
the stress-energy map, from directions to currents. A plane wave is a field that both source-free Maxwell
equations along its wavevector send to nothing.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA

mv = NumpyContext(STA).multivector
Scalar = STA.gatype.scalar()
Vector = STA.gatype.vector()
Bivector = STA.gatype.bivector()
Antibivector = STA.gatype.antibivector()
Antivector = STA.gatype.antivector()
Constitutive = STA.gatype((Antibivector, Bivector))          # Antibivector <- Bivector
# An axion coupling is a pseudoscalar: a scalar with the orientation of the dual.
AxionCoupling = Scalar.derive.flipped_orientation
StressEnergy = STA.gatype((Antivector, Antivector))          # Antivector <- Antivector
WaveMap = STA.gatype((STA.gatype.odd(), Bivector))           # Odd <- Bivector

t, x, y, z = mv.vector(np.eye(4))
# The observer's electric planes, each spatial axis against its time direction, and its magnetic planes,
# their duals.
electric_planes = stack((x, y, z)) ^ t                       # [3] Bivector
magnetic_planes = electric_planes.dual()                     # [3] Bivector


# --- math -----------------------------------------------------------------------------
def medium(permittivities: np.ndarray, inverse_permeabilities: np.ndarray) -> Constitutive:
    """A medium at rest, from its permittivity along each axis and its inverse permeability across each
    `[..., 3]`: every electric plane weighted by its permittivity, every magnetic plane by its inverse
    permeability, and the dual of the sum."""
    # Electric planes square to one and magnetic planes to minus one: each dyad is a projector.
    electric = (permittivities * electric_planes * (electric_planes | Bivector)).sum(axis=-1)
    magnetic = -(inverse_permeabilities * magnetic_planes * (magnetic_planes | Bivector)).sum(axis=-1)
    return (electric + magnetic).dual()                                       # [...] Constitutive


def stress_energy(field: Bivector, excitation: Antibivector) -> StressEnergy:
    """Electromagnetic stress-energy: from a direction in spacetime, an antivector, to the current of
    energy and momentum along it, intrinsic like the direction. A unit timelike observer's slice of space
    `t`, paired with the current along its direction in time, `t & current(t.dual())`, is the energy
    density it measures. In matter this is the electromagnetic Minkowski current, excluding the material's
    own stress-energy.
    """
    lagrangian = (field & excitation) / 2
    # The Lagrangian along the direction, less the field read along the direction wedged with the
    # excitation: no metric enters.
    return lagrangian * Antivector - ((Antivector & field) ^ excitation)


def wave_map(wavevectors: Vector, media: Constitutive) -> WaveMap:
    """Both exterior Maxwell residuals, packed into separate grades (Odd <- Bivector)."""
    # Dualizing the excitation residual keeps the two equations from cancelling.
    return (wavevectors ^ Bivector) + (wavevectors ^ media).dual()


def residuals(wavevectors: Vector, media: Constitutive) -> Scalar:
    """How far the nearest field is from a wave along each wavevector: the wave map's smallest singular
    value, zero where a wave travels."""
    return wave_map(wavevectors, media).svdvals()[..., -1]


def field_modes(wavevectors: Vector, media: Constitutive) -> Bivector:
    """The field each wave carries: the wave map's last right singular vector."""
    _, _, fields = wave_map(wavevectors, media).svd()
    return fields[..., -1]


# --- the step outside the algebra -------------------------------------------------------
def dips(speeds: np.ndarray, residual: Scalar, threshold: float) -> np.ndarray:
    """Where the residuals along the speeds, the first axis, fall to a local minimum under the threshold:
    the speeds at which waves travel, as a mask over the interior speeds `[speeds - 2, ...]`."""
    values = residual.to_array()
    left, middle, right = values[:-2], values[1:-1], values[2:]
    return (middle <= left) & (middle <= right) & (middle < threshold)


def deepest(speeds: np.ndarray, residual: Scalar) -> np.ndarray:
    """The speed at which the residual along the speeds, the first axis, is smallest `[...]`."""
    return speeds[residual.to_array().argmin(axis=0)]
