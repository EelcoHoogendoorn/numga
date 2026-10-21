"""Wedge, contraction and all maps on planar multivectors as products in Cl(2,2).

Two pairs of null vectors add and remove directions from a state. Their mixed products
act as dyads on vectors and as derivations on blades. A four-component left ideal carries
the sixteen independent maps, with composition given by the geometric product.
"""

from functools import reduce
from operator import mul

import numpy as np

from numga import Algebra, NumpyContext, concatenate, stack

ga = Algebra("x+y+u-v-")
context = NumpyContext(ga)
mv = context.multivector
Full = ga.gatype.full()
State = Full.derive.within("x y")
Vector = ga.gatype.vector().derive.within("x y")
StateMap = ga.gatype((State, State))
state_components = len(State.output_subspace)


# --- math -----------------------------------------------------------------------------
directions = stack([mv.x, mv.y])
partners = stack([mv.u, mv.v])
creators = (directions + partners) / 2
removers = (directions - partners) / 2

# Removing a direction from this state gives zero. The two empty-direction projectors commute.
projector = reduce(mul, removers * creators)
embedder = State * projector
# Keep the independent coefficients; the projector contributes a factor of one half per direction.
state_readout = (state_components * Full).cast(State)
representations = state_readout(Full * embedder)


def encode(state_map: StateMap) -> Full:
    """The multivector whose left action is the given map on all four state components."""
    # Tracing against every represented blade gives its scalar-product pairing with the answer.
    pairing = Full.scalar_product(Full)
    return pairing.solve(representations(Full, state_map).trace(0, 2) / state_components)


def shear(amount: np.ndarray) -> StateMap:
    """Shear y towards x, acting on the scalar, both directions and their oriented area."""
    generator = creators[0] * removers[1]
    return representations((generator * amount).exp())


# --- plumbing -------------------------------------------------------------------------
def grid_samples(directions: Vector, extent: float, lines: int) -> Vector:
    """Horizontal and vertical grid segments, with axes [lines, endpoints]."""
    offsets = np.linspace(-extent, extent, lines)
    ends = np.array([-extent, extent])
    horizontal = directions[0] * ends[None, :] + directions[1] * offsets[:, None]
    vertical = directions[1] * ends[None, :] + directions[0] * offsets[:, None]
    return concatenate([horizontal, vertical], axis=0)
