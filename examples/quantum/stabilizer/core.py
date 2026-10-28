"""A four-qubit stabilizer code, the smallest that catches bit flips and phase flips alike: two qubits kept in
four, so that any single qubit going wrong is noticed, in Cl(4, 4).

The whole of Cl(4, 4) is at work: the real linear operators on the sixteen real amplitudes of four qubits are the
algebra itself, and so are the states, as density operators. Every pattern of bit and phase flips is, up to sign,
one of its two hundred and fifty-six blades. Two checks, flipping all four qubits and phase-flipping all four,
commute with each other and with the flips and phase flips of two logical qubits; the states both checks leave alone
are the code. Any single error changes the outcome of at least one check, so it is noticed and the run discarded.
A measurement's expectation is the trace of the operator times the state, the algebra's own scalar product of the
two times the sixteen amplitudes, `n_amplitudes * operator.scalar_product(state)`.
"""

from __future__ import annotations

from functools import reduce
from operator import mul

import numpy as np

from numga import Algebra, NumpyContext, stack

ga = Algebra("a+b+c+d+e-f-g-h-")
mv = NumpyContext(ga).multivector

Scalar = ga.gatype.scalar()                                                    # Scalar+
# The operators on the sixteen real amplitudes of four qubits, the states among them as density operators.
Operator = ga.gatype.full()                                                    # Operator+
# The amplitudes of four qubits: the trace of an operator is this many times its scalar product with the identity.
n_amplitudes = 16
identity = mv.scalar()                                                         # [] Operator+

# These square-one blades anticommute within each qubit's pair and commute across qubits.
flips = stack([mv.b, -mv.bce, -mv.bcdef, -mv.ah])                              # [qubits] Operator+
phase_flips = stack([mv.be, mv.cf, mv.dg, -mv.abcdefg])                        # [qubits] Operator+
# What can happen to each qubit: nothing, a bit flip, a phase flip, or both.
errors = stack([identity.broadcast_to(flips.shape), flips, phase_flips, flips * phase_flips], axis=-1)   # [qubits, kinds] Operator+
# All four qubits 0: the state every phase flip leaves alone, the product of their projectors.
ground = reduce(mul, (1 + phase_flips) / 2)                                    # [] Operator+

# The two checks: flipping all four qubits, and phase-flipping all four.
checks = stack([reduce(mul, flips), reduce(mul, phase_flips)])                 # [checks] Operator+
# Each outcome of the two checks, and the projector onto the states that give it; the first is the code.
outcomes = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])                      # [outcomes, checks]
halves = (1 + checks * outcomes) / 2                                           # [outcomes, checks] Operator+
projectors = halves[:, 0] * halves[:, 1]                                       # [outcomes] Operator+
code = projectors[0]                                                           # [] Operator+

# The flips and phase flips of the two logical qubits: pairs of qubits, commuting with both checks.
logical_flips = stack([flips[0] * flips[1], flips[0] * flips[2]])              # [logical] Operator+
logical_phase_flips = stack([phase_flips[0] * phase_flips[2], phase_flips[0] * phase_flips[1]])   # [logical] Operator+


# --- math -----------------------------------------------------------------------------
def turned(start: Operator, qubit_flips: Operator, qubit_phase_flips: Operator, angles: np.ndarray) -> Operator:
    """Each qubit turned from 0 towards 1 by the exponential of its flip times its phase flip, the state conjugated
    by the turn.

    These products square to minus one and generate turns of the sixteen amplitudes; they need not be bivectors of
    Cl(4, 4), and their reverse need not be their inverse.
    """
    turns = ((qubit_flips * qubit_phase_flips) * (angles / 2)).exp()           # [..., logical] Operator+
    turn = turns[..., 0] * turns[..., 1]                                       # [...] Operator+
    return turn * start * turn.inverse()                                       # [...] Operator+


def encoded(angles: np.ndarray) -> Operator:
    """Two logical qubits at the given angles, stored in the code: the ground state projected into the code, at unit
    trace, then turned."""
    projected = code * ground * code                                           # [] Operator+
    start = projected / (n_amplitudes * identity.scalar_product(projected))    # [] Operator+
    return turned(start, logical_flips, logical_phase_flips, angles)           # [...] Operator+


def noise(n_qubits: int, rates: np.ndarray) -> tuple[Operator, np.ndarray]:
    """Every pattern of errors on the first so many qubits, and its probability at each rate: each qubit
    is left alone, or hit by one of the three errors with equal chance."""
    kinds = np.indices((errors.shape[-1],) * n_qubits)                         # [qubits, kinds...]
    chances = np.stack([1 - rates, rates / 3, rates / 3, rates / 3], axis=-1)  # [rates, kinds]
    patterns = reduce(mul, (errors[qubit, kind] for qubit, kind in enumerate(kinds)))   # [kinds...] Operator+
    weights = reduce(mul, (chances[:, kind] for kind in kinds))                # [rates, kinds...]
    return patterns, weights


def averaged(values: Scalar, weights: np.ndarray) -> Scalar:
    """The average over error patterns, the trailing axes of the values, at each rate."""
    return (values * weights).sum(axis=tuple(range(1 - weights.ndim, 0)))      # [rates] Scalar+


def detected(stored: Operator, rates: np.ndarray) -> tuple[Scalar, Scalar]:
    """The acceptance probability and the unconditional fidelity with the stored state.

    Rejected states have no weight in the code, so dividing fidelity by acceptance gives the fidelity
    conditional on both checks passing.
    """
    patterns, weights = noise(errors.shape[0], rates)
    corrupted = patterns * stored * patterns.inverse()                         # [kinds...] Operator+
    # The chance that both checks pass: the expectation of the code's projector.
    acceptance = n_amplitudes * code.scalar_product(corrupted)                 # [kinds...] Scalar+
    # The fidelity with the pure stored state: the expectation of that state, its own projector.
    fidelity = n_amplitudes * stored.scalar_product(corrupted)                 # [kinds...] Scalar+
    return averaged(acceptance, weights), averaged(fidelity, weights)          # [rates] Scalar+, Scalar+


def unprotected(angles: np.ndarray, rates: np.ndarray) -> Scalar:
    """The fidelity of two bare qubits with their intended state under the same noise."""
    # As many bare qubits as the code stores.
    n_bare = angles.shape[-1]
    stored = turned(ground, flips[:n_bare], phase_flips[:n_bare], angles)      # [] Operator+
    patterns, weights = noise(n_bare, rates)
    corrupted = patterns * stored * patterns.inverse()                         # [kinds, kinds] Operator+
    return averaged(n_amplitudes * stored.scalar_product(corrupted), weights)  # [rates] Scalar+
