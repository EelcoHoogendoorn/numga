"""Constitutive maps: a material's response to the electromagnetic field, a map from the field, a
bivector, to the excitation it causes, an antibivector, `excitation = medium(field)`.

A medium at rest responds to each part of the field an observer tells apart with a dyad: the patch that
reads that part, times its reading of the field, weighted by the material constant. The planes each axis of
space sweeps through time read the electric field and carry the permittivities; the patches of the planes of
space read the magnetic field and carry the inverse permeabilities. Glass weights all three of each kind
alike, a crystal its electric dyads apart, a ferrite its magnetic ones; the metric enters only as the
material map itself. Seen moving, a medium is its rest map carried by the boost,
`boost >> medium(boost << Bivector)`, and an axion response adds a term proportional to the field itself.
Pairing the excitation with a second field gives the constitutive form, a form on pairs of fields, and the
field and excitation together give the stress-energy map, from directions to currents. A plane wave is a
field that both source-free Maxwell equations along its wavevector send to nothing.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA

mv = NumpyContext(STA).multivector
Scalar = STA.gatype.scalar()                                # Scalar+
Vector = STA.gatype.vector()                                # Vector+
Bivector = STA.gatype.bivector()                            # Bivector+
Antibivector = STA.gatype.antibivector()                    # Antibivector-
Antivector = STA.gatype.antivector()                        # Antivector-
Constitutive = STA.gatype((Antibivector, Bivector))         # Antibivector- <- Bivector+
# An axion coupling is handed: an intrinsic scalar, with the orientation flipped.
AxionCoupling = STA.gatype.antipseudoscalar()               # Antipseudoscalar-
StressEnergy = STA.gatype((Antivector, Antivector))         # Antivector- <- Antivector-
WaveMap = STA.gatype((STA.gatype.odd(), Bivector))          # Odd+ <- Bivector+

t, z = mv.t, mv.z
# The observer's direction in time and its axes of space, and the planes of space across each axis.
time = -mv.t.dual()                                         # [] Antivector-
axes = stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))       # [3] Antivector-
magnetic_planes = stack((mv.yz, mv.zx, mv.xy))              # [3] Bivector+
# The planes each axis sweeps through time, which read the electric field, and the patches of the planes of
# space, which read the magnetic field: intrinsic, both.
swept_planes = axes & time                                  # [3] Antibivector-
patches = magnetic_planes.dual()                            # [3] Antibivector-


# --- math -----------------------------------------------------------------------------
def medium(permittivities: np.ndarray, inverse_permeabilities: np.ndarray) -> Constitutive:
    """A medium at rest, from its permittivity along each axis and its inverse permeability across each
    `[..., 3]`: the dyads of the patches that read the field, each weighted by its material constant."""
    # Each dyad is a patch times its reading of the field. Time squares to minus one, so the electric dyads
    # enter with a minus: the sign that makes the Lagrangian the electric field squared less the magnetic.
    electric = -(permittivities * swept_planes * (Bivector & swept_planes)).sum(axis=-1)
    magnetic = (inverse_permeabilities * patches * (Bivector & patches)).sum(axis=-1)
    return electric + magnetic                                                # [...] Antibivector- <- Bivector+


def stress_energy(field: Bivector, excitation: Antibivector) -> StressEnergy:
    """Electromagnetic stress-energy: from a direction in spacetime, an antivector, to the current of
    energy and momentum along it, intrinsic like the direction. A unit timelike observer's slice of space
    `t`, paired with the current along its direction in time, `t & current(-t.dual())`, is the energy
    density it measures. In matter this is the electromagnetic Minkowski current, excluding the material's
    own stress-energy.
    """
    # An electric plane pairs with its own dual to minus its square, so the field paired with its
    # excitation is minus twice the Lagrangian.
    negative_lagrangian = (field & excitation) / 2
    # Minus the Lagrangian along the direction, less the field read along the direction wedged with the
    # excitation: no metric enters.
    return negative_lagrangian * Antivector - ((Antivector & field) ^ excitation)


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


# --- plumbing -------------------------------------------------------------------------
def dips(residual: Scalar, threshold: float) -> np.ndarray:
    """Where the residuals along the trial speeds, the first axis, fall to a local minimum under the
    threshold: the speeds at which waves travel, as a mask over the interior speeds `[speeds - 2, ...]`."""
    left, middle, right = residual[:-2], residual[1:-1], residual[2:]
    return (middle <= left) & (middle <= right) & (middle < threshold)


def deepest(speeds: np.ndarray, residual: Scalar) -> np.ndarray:
    """The speed at which the residual along the speeds, the first axis, is smallest `[...]`."""
    return speeds[residual.argmin(axis=0)]
