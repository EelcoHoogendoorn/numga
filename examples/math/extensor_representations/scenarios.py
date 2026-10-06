"""Smallest faithful real representations of nondegenerate Clifford algebras.

Commuting blades select a left ideal. Multiplication acts on that ideal, and a
blade chart expresses the action as an extensor on a smaller real space.

Run with python -m examples.math.extensor_representations.scenarios --dimensions 6.
"""

import numpy as np

from numga import Algebra, Extensor, NumpyContext

from examples.math.extensor_representations.core import spinor_layout


# --- math -----------------------------------------------------------------------------
def main(positive: int, negative: int) -> Extensor:
    ga = Algebra((positive, negative, 0))
    mv = ga.exact.multivector
    Full = ga.gatype.full()
    Spinor, involutions = spinor_layout(ga)
    multiplicity = 2 ** len(involutions)
    state_dimension = len(Spinor.output_subspace)

    # Each commuting blade that squares to one halves the state space. Their
    # projectors commute too, so their product selects the common positive part.
    projector = mv.scalar()
    for blade in involutions:
        projector = projector * (1 + mv.blade(blade)) / 2

    # The open Spinor slot carries independent real coefficients into the ideal.
    # Left multiplication preserves it: a * (state * projector) stays in the ideal.
    state_embedding = Spinor * projector

    # Every chart blade retains exactly 1 / multiplicity of its coefficient in
    # the embedding. Casting back and rescaling therefore recovers the state.
    state_readout = (multiplicity * Full).cast(Spinor)
    # Keep Full open as well: embedding, multiplying on the left and reading out
    # converts any multivector into a Spinor map.
    embedder = state_readout(Full * state_embedding)

    # --- checks
    context = NumpyContext(ga)
    basis = context.multivector.full(np.eye(ga.blade_count))
    matrices = embedder(basis)
    composition = matrices[:, None](matrices[None, :])
    product = embedder(basis[:, None] * basis[None, :])
    identity = embedder(context.multivector.scalar())
    np.testing.assert_array_equal((composition - product).kernel, 0)
    np.testing.assert_array_equal((identity - Spinor).kernel, 0)

    # The blade images are orthogonal signed permutations, so none of the algebra
    # is lost, including either split block. Generator relations alone do not test this.
    coefficients = matrices.kernel.reshape(ga.blade_count, -1)
    np.testing.assert_array_equal(coefficients @ coefficients.T, state_dimension * np.eye(ga.blade_count))

    print(f"Cl({positive},{negative}): {ga.blade_count:3d} coefficients -> "
          f"{state_dimension:2d} x {state_dimension:<2d} real")
    return embedder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, default=6)
    arguments = parser.parse_args()
    for dimension in range(arguments.dimensions + 1):
        for positive in range(dimension, -1, -1):
            main(positive, dimension - positive)
