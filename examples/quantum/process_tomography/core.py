"""Quantum process tomography: learning a qubit's noisy evolution from prepared states and measured
probabilities, with the states of a qubit as the points of a ball in projective space.

Every outcome probability is an affine function of the state, and an affine function on space is a plane. A state
is modelled as a point of weight one, its weight read by the plane at infinity, `mv.w & state`; an effect is a
plane, and an outcome's probability is the pairing `effect & state`. A channel is a map from points to points: a
coherent turn is a rotor about the centre, and noise mixes a state with its images, linearly in the point.

Four preparations in general position determine such a map. Each outcome's dual is the point where the other three
outcomes' planes meet, each preparation's dual the plane through the other three preparations: the reciprocal
frames are meets and joins, and no scalar product enters. The recovered outputs times the readings of their
preparations' dual planes are dyads, and their sum is the whole channel.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D as ga
from examples import instantiate

mv = NumpyContext(ga).multivector
# Meshes over the algebra's points, such as the tetrahedron the four preparations span.
Mesh = instantiate("examples.util.mesh", ga).Mesh
Scalar = ga.gatype.scalar()                                                  # Scalar+
Bivector = ga.gatype.bivector()                                              # Bivector+
# A state is a point of the Bloch ball, of weight one; an effect is a plane, which reads states.
Point = ga.gatype.antivector()                                               # Antivector-
Effect = ga.gatype.vector()                                                  # Vector+
Channel = ga.gatype((Point, Point))                                          # Point- <- Point-
# The join of two states, the line through them.
Line = ga.gatype.antibivector()                                              # Antibivector-

# The centre of the ball, the completely mixed state, and the north pole, towards which relaxation pulls.
centre = mv.point([0, 0, 0])                                                 # [] Point-
north = mv.point([0, 0, 1])                                                  # [] Point-
# Each of four preparations or outcomes, with the indices of the other three.
other_idx = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


# --- math -----------------------------------------------------------------------------
def noisy_gate(
    angles: np.ndarray, phase_flip: np.ndarray, loss: np.ndarray, rotation_plane: Bivector,
) -> Channel:
    """A coherent turn after independent phase noise and relaxation towards the north pole."""
    # With probability phase_flip, the state is replaced by its half turn about the field.
    dephasing = Point + phase_flip * ((mv.xy >> Point) - Point)              # [channels] Point- <- Point-
    # A decayed state is at north: the plane at infinity reads the state's weight, and north receives it.
    decay = north * (mv.w & Point)                                           # [] Point- <- Point-
    # The part of a state across the field: the part a half turn about the field reverses.
    transverse = (Point - (mv.xy >> Point)) / 2                              # [] Point- <- Point-
    # A fraction loss of the way to north; an unobserved decay keeps the amplitude sqrt(1 - loss) of the coherence
    # across the field, more than the 1 - loss a mixture of decayed and untouched states would keep.
    damping = Point + loss * (decay - Point) + (np.sqrt(1 - loss) - (1 - loss)) * transverse   # [channels] Point- <- Point-
    rotations = (rotation_plane * (-angles / 2)).exp()                       # [channels] Rotor+
    # A map applied to another map composes them; the rotor then turns every output.
    return rotations >> damping(dephasing)                                   # [channels] Point- <- Point-


def probabilities(device: Channel, prepared: Point, effects: Effect) -> Scalar:
    """Each outcome's probability for each preparation passed through each channel."""
    outputs = device[..., None](prepared)                                    # [channels, preparations] Point-
    return effects & outputs[..., None]                                      # [channels, preparations, outcomes] Scalar+


def reconstruct(prepared: Point, effects: Effect, measured: Scalar) -> Channel:
    """The channel that gives the measured probabilities on four preparations in general position, extended to every
    state by linearity."""
    # The point where the other three effects' planes meet, scaled so its own effect reads one there.
    meets = effects[other_idx[:, 0]] ^ effects[other_idx[:, 1]] ^ effects[other_idx[:, 2]]   # [outcomes] Antivector+
    measurement_dual = meets / (effects & meets)                             # [outcomes] Point-
    outputs = (measured * measurement_dual).sum(axis=-1)                     # [channels, preparations] Point-
    # The plane through the other three preparations, scaled to read one on its own preparation.
    joins = prepared[other_idx[:, 0]] & prepared[other_idx[:, 1]] & prepared[other_idx[:, 2]]   # [preparations] Vector-
    preparation_dual = joins / (joins & prepared)                            # [preparations] Effect+
    return (outputs * (preparation_dual & Point)).sum(axis=-1)               # [channels] Point- <- Point-


def powers(device: Channel, n_uses: int) -> Iterator[Channel]:
    """One use of the channel, then two, three, and so on, as composed maps."""
    combined = device                                                        # [channels] Point- <- Point-
    for _ in range(n_uses):
        yield combined
        combined = device(combined)                                          # [channels] Point- <- Point-


# --- plumbing -------------------------------------------------------------------------
def sphere(n_latitudes: int, n_longitudes: int) -> Point:
    """Pure states on a closed latitude-longitude mesh of the Bloch sphere: north, tilted down and turned about the
    field."""
    inclinations = np.linspace(0, np.pi, n_latitudes + 1)
    azimuths = np.linspace(0, 2 * np.pi, n_longitudes + 1)
    tilts = (mv.zx * (-inclinations / 2)).exp()                              # [latitudes + 1] Rotor+
    turns = (mv.xy * (-azimuths / 2)).exp()                                  # [longitudes + 1] Rotor+
    return turns[None, :] >> (tilts[:, None] >> north)                       # [latitudes + 1, longitudes + 1] Point-
