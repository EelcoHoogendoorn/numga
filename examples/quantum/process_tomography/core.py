"""Quantum process tomography recovers a qubit's noisy channel from prepared states and measured outcome probabilities.

Every outcome probability depends affinely on the state, and an affine function on three-dimensional projective space
is a plane. A qubit state is modelled as a unit-weight point inside the Bloch ball whose weight is read by the plane
at infinity `w & state`, a detector outcome is modelled as an effect plane, and its Born probability is the pairing
`effect & state`. Coherent turns act by rotors around the centre, while dephasing and relaxation act linearly on the
projective point.

Four preparations in general position span the ball and determine the linear channel everywhere. Each outcome's dual
point is the meet of the other three outcome planes, and each preparation's dual plane is the join of the other three
preparations, so reciprocal frames arise from meets and joins without any metric inner product. Summing the dyads of
recovered output states against dual preparation planes gives the channel directly as a `Point <- Point` map.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Antiscalar = ga.gatype.antiscalar()                                          # Antiscalar-
Bivector = ga.gatype.bivector()                                              # Bivector+
# A state is a unit-weight point of the Bloch ball; an effect is a plane that reads states.
Point = ga.gatype.antivector()                                               # Antivector-
Effect = ga.gatype.vector()                                                  # Vector+
Line = ga.gatype.antibivector()                                              # Antibivector-
Channel = ga.gatype((Point, Point))                                          # Point- <- Point-

# The centre of the ball (the completely mixed state) and the north pole toward which relaxation pulls.
centre = mv.point([0, 0, 0])                                                 # [] Point-
north = mv.point([0, 0, 1])                                                  # [] Point-
# Indices of the complementary triple for each of four spanning preparations or outcomes.
other_indices = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


# --- math -----------------------------------------------------------------------------
def noisy_gate(
    angles: np.ndarray, phase_flip: np.ndarray, loss: np.ndarray, rotation_plane: Bivector,
) -> Channel:
    """Independent phase noise, relaxation toward the north pole, and a coherent turn form a gate channel (`[...] Point- <- Point-`)."""
    # Phase flip replaces the state with its half turn about the z axis with probability phase_flip.
    dephasing = Point + phase_flip * ((mv.xy >> Point) - Point)              # [...] Point- <- Point-
    # Relaxation sends decayed weight to the north pole, read off by pairing with the plane at infinity.
    decay = north * (mv.w & Point)                                           # [] Point- <- Point-
    # A half turn about the z axis reverses the transverse spin components while fixing the axis.
    transverse = (Point - (mv.xy >> Point)) / 2                              # [] Point- <- Point-
    # Unobserved decay retains transverse coherence amplitude sqrt(1 - loss) rather than classical 1 - loss.
    damping = Point + loss * (decay - Point) + (np.sqrt(1 - loss) - (1 - loss)) * transverse  # [...] Point- <- Point-
    rotations = (rotation_plane * (-angles / 2)).exp()                       # [...] Rotor+
    return rotations >> damping(dephasing)                                   # [...] Point- <- Point-


def outcome_probabilities(channel: Channel, states: Point, effects: Effect) -> Scalar:
    """Pairing `effects` against mapped `states` gives the Born outcome probabilities (`[...] Scalar+`)."""
    outputs = channel[..., None](states)                                     # [...] Point-
    return effects & outputs[..., None]                                      # [...] Scalar+


def reconstructed_channel(preparations: Point, effects: Effect, measurements: Scalar) -> Channel:
    """Reciprocal meets of `effects` and joins of `preparations` recover the linear channel (`[...] Point- <- Point-`)."""
    # Each outcome's dual point lies on the meet of the other three effect planes, scaled so its own effect reads one.
    meets = effects[other_indices[:, 0]] ^ effects[other_indices[:, 1]] ^ effects[other_indices[:, 2]]  # [4] Antivector+
    measurement_dual = meets / (effects & meets)                             # [4] Point-
    recovered_outputs = (measurements * measurement_dual).sum(axis=-1)       # [..., 4] Point-
    # Each preparation's dual plane joins the other three preparations, scaled to read one on its own preparation.
    joins = preparations[other_indices[:, 0]] & preparations[other_indices[:, 1]] & preparations[other_indices[:, 2]]  # [4] Vector-
    preparation_dual = joins / (joins & preparations)                        # [4] Effect+
    return (recovered_outputs * (preparation_dual & Point)).sum(axis=-1)     # [...] Point- <- Point-


def channel_powers(channel: Channel, n_uses: int) -> Iterator[Channel]:
    """Yields the composed channels (`[...] Point- <- Point-`) for one through `n_uses` applications of `channel`."""
    channel_power = channel                                                  # [...] Point- <- Point-
    for _ in range(n_uses):
        yield channel_power
        channel_power = channel(channel_power)                               # [...] Point- <- Point-


# --- plumbing -------------------------------------------------------------------------
def pure_sphere(n_latitudes: int, n_longitudes: int) -> Point:
    """Inclination tilts and azimuthal turns sweep `north` across a closed sphere grid (`[n_latitudes + 1, n_longitudes] Point-`)."""
    inclinations = np.linspace(0, np.pi, n_latitudes + 1)
    azimuths = np.linspace(0, 2 * np.pi, n_longitudes, endpoint=False)
    tilts = (mv.zx * (-inclinations / 2)).exp()                              # [n_latitudes + 1] Rotor+
    turns = (mv.xy * (-azimuths / 2)).exp()                                  # [n_longitudes] Rotor+
    meridian = tilts >> north                                                # [n_latitudes + 1] Point-
    return turns[None, :] >> meridian[:, None]                               # [n_latitudes + 1, n_longitudes] Point-
