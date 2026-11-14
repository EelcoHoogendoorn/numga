"""Products in the split algebra Cl(2, 2) represent wedge, contraction and every linear map on planar multivectors.

Two pairs of null vectors add basis vectors to an exterior state as factors and remove them. Mixed products of an adding and a removing vector act as dyads on vectors and as derivations on higher-grade blades, while a four-component left ideal carries all sixteen linear maps on the exterior algebra under left multiplication.
"""

from functools import reduce
from operator import mul

from numga import Algebra, NumpyContext, stack

ga = Algebra("x+y+u-v-")
context = NumpyContext(ga)
mv = context.multivector
Full = ga.gatype.full()                                     # Full+
State = Full.derive.within(Algebra("x+y+"))                 # Full+
Vector = ga.gatype.vector().derive.within(Algebra("x+y+"))  # Vector+
StateMap = ga.gatype((State, State))                        # State+ <- State+
n_state_components = len(State.output_subspace)

basis = stack([mv.x, mv.y])                                 # [2] Vector+
negative_basis = stack([mv.u, mv.v])                        # [2] Vector+
creators = (basis + negative_basis) / 2                     # [2] Vector+
removers = (basis - negative_basis) / 2                     # [2] Vector+

# Removing either basis vector from this state gives zero; the two empty-state projectors commute.
projector = reduce(mul, removers * creators)                # [] Even+
embedder = State * projector                                # [] Full+ <- State+
# Each empty-state projector contributes a factor of one half that reading out rescales away.
state_readout = (n_state_components * Full).cast(State)     # [] State+ <- Full+
representations = state_readout(Full * embedder)            # [] State+ <- (Full+, State+)

# Removing y and adding x sends y to x and extends to blades by the derivation product rule.
shear_generator = creators[0] * removers[1]                 # [] Even+
# Each diagonal product keeps the blades containing its basis vector, so their sum counts grade.
grade_count = (creators * removers).sum(axis=0)             # [] Even+
# Adding both basis vectors or removing them in reverse order swaps the scalar and area blades.
exchange = creators[0] * creators[1] + removers[1] * removers[0]  # [] Even+


def map_multivector(state_map: StateMap) -> Full:
    """Trace pairing against all sixteen represented blades recovers the multivector (`[] Full+`) whose left action equals `state_map`."""
    pairing = Full.scalar_product(Full)                     # [] Scalar+ <- (Full+, Full+)
    trace_rhs = representations(Full, state_map).trace(0, 2) / n_state_components  # [] Full+
    return pairing.solve(trace_rhs)
