"""Wedge, contraction and all maps on planar multivectors as products in Cl(2,2).

Two pairs of null vectors add basis vectors to a state as factors and remove them. Their mixed
products act as dyads on vectors and as derivations on blades. A four-component left ideal carries
the sixteen independent maps, with composition given by the geometric product.
"""

from functools import reduce
from operator import mul

import numpy as np

from numga import Algebra, NumpyContext, concatenate, stack

ga = Algebra("x+y+u-v-")
context = NumpyContext(ga)
mv = context.multivector
Full = ga.gatype.full()                                     # Full+
State = Full.derive.within("x y")                           # Full+
Vector = ga.gatype.vector().derive.within("x y")            # Vector+
StateMap = ga.gatype((State, State))                        # State+ <- State+
n_state_components = len(State.output_subspace)


# --- math -----------------------------------------------------------------------------
basis = stack([mv.x, mv.y])                                 # [2] Vector+
partners = stack([mv.u, mv.v])                              # [2] Vector+
creators = (basis + partners) / 2                           # [2] Vector+
removers = (basis - partners) / 2                           # [2] Vector+

# Removing either basis vector from this state gives zero. The two empty-state projectors commute.
projector = reduce(mul, removers * creators)                # [] Even+
embedder = State * projector                                # [] Full+ <- State+
# Keep the independent coefficients; the projector contributes a factor of one half per basis vector.
state_readout = (n_state_components * Full).cast(State)     # [] State+ <- Full+
representations = state_readout(Full * embedder)           # [] State+ <- (Full+, State+)
# Removing y and then adding x sends y to x; its exponential shears y towards x, and carries the
# scalar, both vectors and their oriented area along.
shear_generator = creators[0] * removers[1]                 # [] Even+


def encode(state_map: StateMap) -> Full:
    """The multivector whose left action is the given map on all four state components."""
    # Tracing against every represented blade gives its scalar-product pairing with the answer.
    pairing = Full.scalar_product(Full)                     # [] Scalar+ <- (Full+, Full+)
    return pairing.solve(representations(Full, state_map).trace(0, 2) / n_state_components)


# --- plumbing -------------------------------------------------------------------------
def grid_samples(basis: Vector, extent: float, n_lines: int) -> Vector:
    """Horizontal and vertical grid segments, with axes [lines, endpoints]."""
    offsets = np.linspace(-extent, extent, n_lines)
    ends = np.array([-extent, extent])
    horizontal = basis[0] * ends[None, :] + basis[1] * offsets[:, None]
    vertical = basis[1] * ends[None, :] + basis[0] * offsets[:, None]
    return concatenate([horizontal, vertical], axis=0)
