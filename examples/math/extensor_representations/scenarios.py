"""Extensor representations: the smallest faithful real representations of nondegenerate Clifford algebras.

Commuting blades select a left ideal. Multiplication acts on that ideal, and a
blade chart expresses the representation as an extensor on a smaller real space.
"""

import numpy as np

from numga import Algebra, Extensor, GAType, NumpyContext

from examples.math.extensor_representations.core import spinor_layout

# Every signature of every dimension up to this one is represented.
max_dimension = 6


# --- math -----------------------------------------------------------------------------
def representation(positive: int, negative: int) -> tuple[Algebra, GAType, Extensor]:
    """The algebra of the given signature, its space of states, and its representation: a map from any
    multivector to its map on states."""
    ga = Algebra((positive, negative, 0))
    mv = ga.exact.multivector
    Full = ga.gatype.full()
    Spinor, involutions = spinor_layout(ga)
    multiplicity = 2 ** len(involutions)

    # Each commuting blade that squares to one halves the state space. Their
    # projectors commute too, so their product selects the common positive part.
    projector = mv.scalar()
    for blade in involutions:
        projector = projector * (1 + mv.blade(blade)) / 2

    # Multiplied by Spinor, the space of states, the projector carries their independent real coefficients
    # into the ideal.
    # Left multiplication preserves it: a * (state * projector) stays in the ideal.
    state_embedding = Spinor * projector                                 # [] Full+ <- Spinor+

    # Every chart blade retains exactly 1 / multiplicity of its coefficient in
    # the embedding. Casting back and rescaling therefore recovers the state.
    state_readout = (multiplicity * Full).cast(Spinor)                   # [] Spinor+ <- Full+
    # Multiplied by Full as well, embedding, multiplying on the left and reading out is a map from any
    # multivector to its map on states.
    representations = state_readout(Full * state_embedding)              # [] Spinor+ <- (Full+, Spinor+)
    return ga, Spinor, representations


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    for dimension in range(max_dimension + 1):
        for positive in range(dimension, -1, -1):
            ga, Spinor, representations = representation(positive, dimension - positive)
            n_state_components = len(Spinor.output_subspace)
            print(f"Cl({positive},{dimension - positive}): {ga.blade_count:3d} coefficients -> "
                  f"{n_state_components:2d} x {n_state_components:<2d} real")

            # checks
            # The representation of a product is the composition of the representations, and of one the
            # identity on states.
            context = NumpyContext(ga)
            basis = context.multivector.full(np.eye(ga.blade_count))
            blade_representations = representations(basis)
            composition = blade_representations[:, None](blade_representations[None, :])
            product = representations(basis[:, None] * basis[None, :])
            identity = representations(context.multivector.scalar())
            np.testing.assert_array_equal((composition - product).kernel, 0)
            np.testing.assert_array_equal((identity - Spinor).kernel, 0)
            # The blade images are orthogonal signed permutations, so none of the algebra is lost,
            # including either split block. Generator relations alone do not test this.
            coefficients = blade_representations.kernel.reshape(ga.blade_count, -1)
            np.testing.assert_array_equal(coefficients @ coefficients.T, n_state_components * np.eye(ga.blade_count))


if __name__ == "__main__":
    main()
