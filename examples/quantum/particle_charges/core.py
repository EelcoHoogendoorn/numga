"""Gauge actions and charge observables on one chiral Spin(10) spinor.

A minimal left ideal of the ten-direction algebra models the chiral spinor so left multiplication by any even element maps states to states. Five commuting projectors select an ideal of 32 real coefficients forming sixteen physical amplitudes paired by the ten-direction pseudoscalar.
"""

from __future__ import annotations

from functools import reduce
from operator import mul

import numpy as np

from numga import Algebra, NumpyContext

ga = Algebra("a+b+c+d+e+f+g+h+i+j+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                # Scalar+
Bivector = ga.gatype.bivector()                                            # Bivector+
Even = ga.gatype.even()                                                    # Even+
Full = ga.gatype.full()                                                    # Full+
# One independent blade from each group of blades the ideal identifies.
State = Full.derive.within(Algebra("b+c+d+f+j+"))                          # Full+
StateMap = ga.gatype((State, State))                                       # State+ <- State+

# Five commuting square-one factors select a minimal left ideal with 32 real coefficients.
ideal = (1 + mv.a) * (1 + mv.bcde) * (1 + mv.bcfg) * (1 + mv.bdfh) * (1 + mv.cdfi) / 32  # [] Full+
# Each chart blade appears with weight 1/32 inside the ideal.
readout = (32 * Full).cast(State)                                          # [] State+ <- Full+
# Right multiplication by the ideal carries any element into it; the chart reads its coefficients.
projection = readout(Full * ideal)                                         # [] State+ <- Full+
# Left multiplication keeps the ideal invariant, turning each even element into a state map.
action = projection(Even * State)                                          # [] State+ <- (Even+, State+)
# The unit volume squares to minus one and commutes with every even element, turning each coefficient pair by a quarter turn.
pseudoscalar = mv.abcdefghij                                               # [] Even+


# --- math
def sector_occupations(planes: Bivector) -> StateMap:
    """Each commuting plane compares its spinor quarter turn against the pseudoscalar to count occupation (`[...] State+ <- State+`)."""
    signs = -pseudoscalar * planes                                         # [...] Even+
    return (State - action(signs)) / 2                                     # [...] State+ <- State+


def sector_projectors(planes: Bivector, occupations: np.ndarray) -> StateMap:
    """Four commuting sign factors isolate each two-dimensional particle sector (`[...] State+ <- State+`)."""
    signs = -pseudoscalar * planes                                         # [5] Even+
    factors = (1 + signs[None, :4] * (1 - 2 * occupations[:, :4])) / 2     # [..., 4] Even+
    selectors = reduce(mul, (factors[:, index] for index in range(4)))     # [...] Even+
    return action(selectors)                                               # [...] State+ <- State+


def sector_value(observable: StateMap, projectors: StateMap) -> Scalar:
    """A commuting observable takes a constant trace ratio on each particle sector (`[...] Scalar+`)."""
    return observable(projectors).trace() / projectors.trace()             # [...] Scalar+


def transition_strength(generators: StateMap, projectors: StateMap) -> Scalar:
    """The squared gauge blocks sum the transition strength between every pair of particle sectors (`[..., ...] Scalar+`)."""
    blocks = projectors[:, None, None](generators[None, None, :](projectors[None, :, None]))  # [..., ..., ...] State+ <- State+
    # Two real directions paired by the pseudoscalar carry one physical amplitude.
    return blocks.scalar_norm_squared().contract().sum(axis=-1) / 2        # [..., ...] Scalar+
