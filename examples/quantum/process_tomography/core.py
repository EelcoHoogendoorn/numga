"""Learn a qubit's noisy evolution from prepared states and measured probabilities.

A state is a scalar plus a vector, half of one plus its Bloch vector. Its scalar part
keeps the normalization, so both a rotation and a relaxation towards a preferred state
are linear maps on State. Unobserved alternatives act by sandwiches whose results add.

Four preparations spanning State determine such a map. A measurement reads a state by
its scalar product with an effect; a dual frame undoes the overlap between these
readouts. Reconstructing the outputs, then pairing them with the dual preparations,
gives the whole channel as a sum of dyads, with the input state left open.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Rotor = ga.gatype.rotor()
State = ga.gatype.self_reverse()                     # 1 x y z
Channel = ga.gatype((State, State))                  # State <- State
ONE = mv.scalar([1.0])                              # [] Scalar


# --- math -----------------------------------------------------------------------------
def noisy_gate(
    angles: np.ndarray, phase_flip: np.ndarray, loss: np.ndarray, rotation_plane: Bivector,
) -> Channel:
    """A coherent turn after independent phase noise and relaxation towards the north pole."""
    # These alternatives are not observed: their output states add, rather than their amplitudes.
    phase_paths = stack((ONE * np.sqrt(1 - phase_flip), mv.xy * np.sqrt(phase_flip)), axis=-1)  # [channels, paths] Rotor
    dephasing = (phase_paths >> State).sum(axis=-1)   # [channels] State <- State

    north = (ONE + mv.z) / 2                        # [] State
    south = (ONE - mv.z) / 2                        # [] State
    # The first path preserves north and attenuates south; the second transfers south to north.
    # The square roots are amplitudes, whose squares give the loss probability.
    loss_paths = stack((north + south * np.sqrt(1 - loss),
                        (mv.x * south) * np.sqrt(loss)), axis=-1)  # [channels, paths] Scalar + Vector + Bivector
    damping = (loss_paths >> State).sum(axis=-1)      # [channels] State <- State
    rotations = (rotation_plane * (-angles / 2)).exp()  # [channels] Rotor
    # A map applied to another map composes them; the rotor then turns every output.
    return rotations >> damping(dephasing)          # [channels] State <- State


def dual_frame(states: State) -> State:
    """Weights that recover any state from its scalar products with a spanning set of states."""
    # Each dyad measures overlap with one preparation and sends it back along that preparation.
    overlap = (states * states.scalar_product(State)).sum(axis=-1)  # [...] State <- State
    return overlap[..., None].solve(states)         # [..., preparations] State


def probabilities(device: Channel, prepared: State, effects: State) -> Scalar:
    """Each outcome's probability for each preparation passed through each channel."""
    outputs = device[..., None](prepared)           # [channels, preparations] State
    return 2 * effects.scalar_product(outputs[..., None])  # [channels, preparations, outcomes] Scalar


def reconstruct(prepared: State, effects: State, measured: Scalar) -> Channel:
    """Recover the output states from their probabilities, then the channel from its outputs."""
    # The measurement's scalar-product readout includes the Born-rule factor of two.
    measurement_dual = dual_frame(2 * effects)       # [outcomes] State
    outputs = (measured * measurement_dual).sum(axis=-1)  # [channels, preparations] State
    preparation_dual = dual_frame(prepared)          # [preparations] State
    # Leaving the input open extends the measured action to every state by linearity.
    return (outputs * preparation_dual.scalar_product(State)).sum(axis=-1)  # [channels] State <- State


def powers(device: Channel, steps: int) -> Iterator[Channel]:
    """One use of the channel, then two, three, and so on, as composed maps."""
    combined = device                              # [channels] State <- State
    for _ in range(steps):
        yield combined
        combined = device(combined)                # [channels] State <- State


# --- plumbing -------------------------------------------------------------------------
def sphere(latitudes: int, longitudes: int) -> State:
    """Pure states on a closed latitude-longitude mesh of the Bloch sphere."""
    inclinations = np.linspace(0, np.pi, latitudes + 1)
    azimuths = np.linspace(0, 2 * np.pi, longitudes + 1)
    tilts = ((mv.z ^ mv.x) * (-inclinations / 2)).exp()  # [latitudes + 1] Rotor
    turns = (mv.xy * (-azimuths / 2)).exp()          # [longitudes + 1] Rotor
    directions = turns[None, :] >> (tilts[:, None] >> mv.z)  # [latitudes + 1, longitudes + 1] Vector
    return (ONE + directions) / 2                  # [latitudes + 1, longitudes + 1] State
