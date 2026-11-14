"""Four physical qubits encode two logical qubits in an algebra of four positive-square and four negative-square
generators so that two commuting checks catch any single-qubit bit or phase error.

Real linear operators on the sixteen real amplitudes of four qubits live directly in the 256-dimensional algebra, and
density-operator states live beside them in the same space. Every pattern of bit and phase flips across the four qubits
corresponds up to sign to one of the 256 basis blades. Flipping all four qubits and phase-flipping all four commute
with each other and with the bit and phase flips of two logical qubits, and the states fixed by both checks form the
code. Any single-qubit error flips at least one check outcome so the corrupted run is rejected, while expectations read
directly as `n_amplitudes * operator.scalar_product(state)`.
"""

from __future__ import annotations

from functools import reduce
from operator import mul

import numpy as np

from numga import Algebra, NumpyContext, stack

ga = Algebra("a+b+c+d+e-f-g-h-")
mv = NumpyContext(ga).multivector

Scalar = ga.gatype.scalar()                                                    # Scalar+
# Real linear operators on sixteen four-qubit amplitudes, including density-operator states.
Operator = ga.gatype.full()                                                    # Full+
# Multiplying scalar_product by 16 recovers the operator trace over the sixteen real amplitudes.
n_amplitudes = 16
identity = mv.scalar()                                                         # [] Scalar+

# Bit flips and phase flips square to +1, anticommute on the same qubit, and commute across distinct qubits.
flips = stack([mv.b, -mv.bce, -mv.bcdef, -mv.ah])                              # [4] Operator+
phase_flips = stack([mv.be, mv.cf, mv.dg, -mv.abcdefg])                        # [4] Operator+
# Each qubit undergoes the identity, a bit flip, a phase flip, or their product.
errors = stack([identity.broadcast_to(flips.shape), flips, phase_flips, flips * phase_flips], axis=-1)  # [4, 4] Operator+
# Multiplying the four +1 phase-flip projectors prepares the all-zero state.
ground = reduce(mul, (1 + phase_flips) / 2)                                    # [] Operator+

# One check flips all four qubits and the other phase-flips all four.
checks = stack([reduce(mul, flips), reduce(mul, phase_flips)])                 # [2] Operator+
# Pairing both checks with all four sign patterns builds four orthogonal syndrome projectors, with (+1, +1) as the code.
outcomes = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])                      # [4, 2]
halves = (1 + checks * outcomes) / 2                                           # [4, 2] Operator+
projectors = halves[:, 0] * halves[:, 1]                                       # [4] Operator+
code = projectors[0]                                                           # [] Operator+

# Pairwise bit flips and phase flips commute with both checks and act on the two encoded qubits.
logical_flips = stack([flips[0] * flips[1], flips[0] * flips[2]])              # [2] Operator+
logical_phase_flips = stack([phase_flips[0] * phase_flips[2], phase_flips[0] * phase_flips[1]])  # [2] Operator+
logical_operators = stack([logical_flips, logical_phase_flips], axis=-1)       # [2, 2] Operator+


# --- math -----------------------------------------------------------------------------
def turned_state(
    initial_state: Operator,
    qubit_flips: Operator,
    qubit_phase_flips: Operator,
    angles: np.ndarray,
) -> Operator:
    """Conjugation by the exponential of each qubit's flip times its phase flip turns `initial_state` from 0 toward 1 by `angles` (`[...] Operator+`)."""
    # Multi-blade generators squaring to -1 need not be grade-2 bivectors, so turn.inverse() need not equal ~turn.
    turns = ((qubit_flips * qubit_phase_flips) * (angles / 2)).exp()           # [..., 2] Operator+
    turn = turns[..., 0] * turns[..., 1]                                       # [...] Operator+
    return turn * initial_state * turn.inverse()                               # [...] Operator+


def encoded_state(angles: np.ndarray) -> Operator:
    """Projecting the four-qubit ground state into the code at unit trace and turning both logical qubits encodes `angles` (`[...] Operator+`)."""
    projected_ground = code * ground * code                                    # [] Operator+
    code_ground = projected_ground / (n_amplitudes * identity.scalar_product(projected_ground))  # [] Operator+
    return turned_state(code_ground, logical_flips, logical_phase_flips, angles)  # [...] Operator+


def single_error_readings(state: Operator) -> Scalar:
    """Conjugating `state` by each single-qubit error yields both check expectations across four qubits and four error kinds (`[4, 4, 2] Scalar+`)."""
    corrupted_states = errors * state * errors.inverse()                       # [4, 4] Operator+
    return n_amplitudes * checks.scalar_product(corrupted_states[..., None])   # [4, 4, 2] Scalar+


def error_patterns(n_qubits: int, rates: np.ndarray) -> tuple[Operator, np.ndarray]:
    """Independent depolarizing noise on `n_qubits` qubits pairs every multi-qubit error blade (`[...] Operator+`) with its probability (`[...]`) across `rates`."""
    kinds = np.indices((errors.shape[-1],) * n_qubits)                         # [n_qubits, ...]
    chances = np.stack([1 - rates, rates / 3, rates / 3, rates / 3], axis=-1)  # [..., 4]
    patterns = reduce(mul, (errors[qubit, kind] for qubit, kind in enumerate(kinds)))  # [...] Operator+
    weights = reduce(mul, (chances[:, kind] for kind in kinds))                # [...]
    return patterns, weights


def pattern_average(values: Scalar, weights: np.ndarray) -> Scalar:
    """Weighting `values` by `weights` sums over the trailing error-pattern axes to give the mean at each noise rate (`[...] Scalar+`)."""
    return (values * weights).sum(axis=tuple(range(1 - weights.ndim, 0)))      # [...] Scalar+


def code_performance(stored_state: Operator, rates: np.ndarray) -> tuple[Scalar, Scalar]:
    """Depolarizing noise on all four physical qubits gives the fraction of runs passing both checks and the conditional infidelity of accepted runs (`[...] Scalar+`)."""
    patterns, weights = error_patterns(errors.shape[0], rates)
    corrupted_states = patterns * stored_state * patterns.inverse()            # [4, 4, 4, 4] Operator+
    # Expecting the code projector gives the probability that both checks read +1.
    acceptance = n_amplitudes * code.scalar_product(corrupted_states)          # [4, 4, 4, 4] Scalar+
    # Because a pure density operator equals its own projector, its expectation in the corrupted state equals the fidelity.
    fidelity = n_amplitudes * stored_state.scalar_product(corrupted_states)    # [4, 4, 4, 4] Scalar+
    kept_fraction = pattern_average(acceptance, weights)                       # [...] Scalar+
    return kept_fraction, 1 - pattern_average(fidelity, weights) / kept_fraction  # [...] Scalar+, Scalar+


def bare_infidelity(angles: np.ndarray, rates: np.ndarray) -> Scalar:
    """Depolarizing noise on two unencoded qubits at `angles` gives their unconditional infidelity at each rate (`[...] Scalar+`)."""
    n_bare = angles.shape[-1]
    bare_state = turned_state(ground, flips[:n_bare], phase_flips[:n_bare], angles)  # [] Operator+
    patterns, weights = error_patterns(n_bare, rates)
    corrupted_states = patterns * bare_state * patterns.inverse()              # [4, 4] Operator+
    fidelity = n_amplitudes * bare_state.scalar_product(corrupted_states)      # [4, 4] Scalar+
    return 1 - pattern_average(fidelity, weights)                              # [...] Scalar+
