"""Two qubits kept in four, so that any single qubit going wrong is noticed: the smallest stabilizer code
that catches bit flips and phase flips alike, in the spinors of Cl(4, 4).

The whole of Cl(4, 4) is at work: the real-amplitude four-qubit states used here have sixteen real spinor
components, and the real linear operators on them are the algebra itself. Every pattern of bit and
phase flips is, up to sign, one of its two hundred and fifty-six blades. Two checks, flipping all four
qubits and phase-flipping all four, commute with each other and with the flips and phase flips of two logical
qubits; the states both checks leave alone are the code. Any single error changes the outcome of at least
one check, so it is noticed and the run discarded.
"""

from __future__ import annotations

from functools import reduce
from operator import mul

import numpy as np

from numga import Algebra, NumpyContext, stack

ga = Algebra("a+b+c+d+e-f-g-h-")
mv = NumpyContext(ga).multivector
exact = ga.exact.multivector

Scalar = ga.gatype.scalar()
Full = ga.gatype.full()
State = ga.gatype.from_blades("1 b c d h bc bd cd bh ch dh bcd bch bdh cdh bcdh")

# The four commuting square-one elements give four projectors whose product is idempotent.
# Left multiples of ideal form a sixteen-dimensional space closed under left multiplication.
# The State blades times ideal form a basis of that space: a choice of spinor representatives.
ideal = (1 + exact.a) * (1 + exact.be) * (1 + exact.cf) * (1 + exact.dg) / 16
embedding = State * ideal                                                      # [] Full <- State
# Read the representatives back out; the factor 16 undoes their coefficient in ideal.
readout = (16 * Full).cast(State)                                              # [] State <- Full
action = readout(Full * embedding)                                             # [] State <- (Full, State)
# For this ideal, efgh and the cdeh readout make the State basis orthonormal.
# This positive pairing measures overlap; the flips and turns preserve it.
pairing = 16 * exact.cdeh.scalar_product(embedding.reverse() * exact.efgh * embedding)  # [] Scalar <- (State, State)

# These square-one blades anticommute within each qubit's pair and commute across qubits.
# The phase flips leave ground fixed; products of the bit flips generate the sixteen states.
flips = stack([mv.b, -mv.bce, -mv.bcdef, -mv.ah])                              # [qubits] Full
phase_flips = stack([mv.be, mv.cf, mv.dg, -mv.abcdefg])                        # [qubits] Full
# What can happen to each qubit: nothing, a bit flip, a phase flip, or both.
errors = stack([mv.scalar().broadcast_to(flips.shape), flips, phase_flips, flips * phase_flips], axis=-1)  # [qubits, kinds] Full
# All four qubits 0.
ground = mv.scalar().cast(State)                                               # [] State

# The two checks: flipping all four qubits, and phase-flipping all four.
checks = stack([reduce(mul, flips), reduce(mul, phase_flips)])                 # [checks] Full
# Each outcome of the two checks, and the projector onto the states that give it; the first is the code.
outcomes = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])                      # [outcomes, checks]
halves = (1 + checks * outcomes) / 2                                           # [outcomes, checks] Full
projectors = halves[:, 0] * halves[:, 1]                                       # [outcomes] Full
code = projectors[0]                                                           # [] Full

# The flips and phase flips of the two logical qubits: pairs of qubits, commuting with both checks.
logical_flips = stack([flips[0] * flips[1], flips[0] * flips[2]])              # [logical] Full
logical_phase_flips = stack([phase_flips[0] * phase_flips[2], phase_flips[0] * phase_flips[1]])  # [logical] Full


# --- math -----------------------------------------------------------------------------
def expectation(operators: Full, states: State) -> Scalar:
    """The expectation of each operator on each state."""
    return pairing(states, action(operators, states)) / pairing(states, states)  # [...] Scalar


def turned(start: State, qubit_flips: Full, qubit_phase_flips: Full, angles: np.ndarray) -> State:
    """Turn each qubit from 0 towards 1 by exponentiating its flip times its phase flip.

    These products square to minus one and generate norm-preserving turns of the spinor state space;
    they need not be bivectors of Cl(4, 4).
    """
    turns = ((qubit_flips * qubit_phase_flips) * (angles / 2)).exp()           # [..., 2] Full
    return action(turns[..., 0] * turns[..., 1], start)                        # [...] State


def encoded(angles: np.ndarray) -> State:
    """Two logical qubits at the given angles, stored in the code."""
    start = action(code, ground)                                               # [] State
    start = start / pairing(start, start).square_root()
    return turned(start, logical_flips, logical_phase_flips, angles)           # [...] State


def noise(qubits: int, rates: np.ndarray) -> tuple[Full, np.ndarray]:
    """Every pattern of errors on the first so many qubits, and its probability at each rate: each qubit
    is left alone, or hit by one of the three errors with equal chance."""
    kinds = np.indices((errors.shape[-1],) * qubits)                           # [qubits, kinds...]
    chances = np.stack([1 - rates, rates / 3, rates / 3, rates / 3], axis=-1)  # [rates, kinds]
    patterns = reduce(mul, (errors[qubit, kind] for qubit, kind in enumerate(kinds)))  # [kinds...] Full
    weights = reduce(mul, (chances[:, kind] for kind in kinds))                # [rates, kinds...]
    return patterns, weights


def averaged(values: Scalar, weights: np.ndarray) -> Scalar:
    """The average over error patterns, the trailing axes of the values, at each rate."""
    return (values * weights).sum(axis=tuple(range(1 - weights.ndim, 0)))      # [rates] Scalar


def detected(stored: State, rates: np.ndarray) -> tuple[Scalar, Scalar]:
    """The acceptance probability and unconditional fidelity with the normalized stored state.

    Rejected states have zero overlap with the code, so dividing fidelity by acceptance gives the
    fidelity conditional on both checks passing.
    """
    patterns, weights = noise(errors.shape[0], rates)
    corrupted = action(patterns, stored)                                       # [kinds...] State
    # The chance that both checks pass is the part of the state left in the code.
    passed = pairing(corrupted, action(code, corrupted))                       # [kinds...] Scalar
    overlap_squared = pairing(stored, corrupted).squared()                     # [kinds...] Scalar
    return averaged(passed, weights), averaged(overlap_squared, weights)       # [rates] Scalar each


def unprotected(angles: np.ndarray, rates: np.ndarray) -> Scalar:
    """The fidelity of two bare qubits with their intended state under the same noise."""
    stored = turned(ground, flips[:2], phase_flips[:2], angles)                # [] State
    patterns, weights = noise(2, rates)
    overlap_squared = pairing(stored, action(patterns, stored)).squared()      # [kinds, kinds] Scalar
    return averaged(overlap_squared, weights)                                 # [rates] Scalar
