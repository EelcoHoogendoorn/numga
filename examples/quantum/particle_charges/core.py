"""A Spin(10) spinor and the gauge actions that distinguish its particle sectors.

A left ideal of the algebra of ten directions holds the spinor: left multiplication by any even element
keeps it there, so every internal rotation, and every charge built from them, acts on it as a map from
states to states. The ideal has 32 real coefficients, sixteen amplitudes, each a pair of coefficients that the pseudoscalar turns into each other,
and the blades within `b c d f j` serve as its chart.
"""

from __future__ import annotations

from numga import Algebra, NumpyContext

ga = Algebra("a+b+c+d+e+f+g+h+i+j+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                # Scalar+
Even = ga.gatype.even()                                                    # Even+
Full = ga.gatype.full()                                                    # Full+
# One independent blade from each group of blades the ideal identifies.
State = Full.derive.within("b c d f j")                                    # Full+
StateMap = ga.gatype((State, State))                                       # State+ <- State+

# These commuting square-one elements select a minimal left ideal with 32 real coefficients.
# The chart retains one independent blade from each group identified by the projector.
ideal = (1 + mv.a) * (1 + mv.bcde) * (1 + mv.bcfg) * (1 + mv.bdfh) * (1 + mv.cdfi) / 32
readout = (32 * Full).cast(State)                                          # [] State+ <- Full+
# Right multiplication by the ideal carries any element into it; the chart reads its coefficients.
projection = readout(Full * ideal)                                         # [] State+ <- Full+
# Left multiplication on the ideal, on its independent spinor coefficients.
action = projection(Even * State)                                          # [] State+ <- (Even+, State+)
# The unit volume of the ten directions: it squares to minus one and commutes with every even element, a quarter
# turn of each pair of the spinor's coefficients alike.
pseudoscalar = mv.abcdefghij                                               # [] Even+


# --- math
def sector_value(observable: StateMap, projectors: StateMap) -> Scalar:
    """The value of a commuting observable on each particle sector."""
    return observable(projectors).trace() / projectors.trace()             # [...] Scalar+


def transition_strength(generators: StateMap, projectors: StateMap) -> Scalar:
    """The summed squared gauge action between each pair of particle sectors."""
    blocks = projectors[:, None, None](generators[None, None, :](projectors[None, :, None]))
    # The pair of coefficients the pseudoscalar turns into each other carries one physical amplitude.
    return blocks.scalar_norm_squared().contract().sum(axis=-1) / 2        # [target, source] Scalar+
