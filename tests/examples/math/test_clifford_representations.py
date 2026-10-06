"""Faithful Clifford representations have the smallest real state spaces through dimension six."""

import numpy as np
import pytest

from numga import NumpyContext

from examples.math.extensor_representations.scenarios import main


@pytest.mark.parametrize("positive, negative, size", [
    (0, 0, 1),
    (1, 0, 2), (0, 1, 2),
    (2, 0, 2), (1, 1, 2), (0, 2, 4),
    (3, 0, 4), (2, 1, 4), (1, 2, 4), (0, 3, 8),
    (4, 0, 8), (3, 1, 4), (2, 2, 4), (1, 3, 8), (0, 4, 8),
    (5, 0, 16), (4, 1, 8), (3, 2, 8), (2, 3, 8), (1, 4, 16), (0, 5, 8),
    (6, 0, 16), (5, 1, 16), (4, 2, 8), (3, 3, 8), (2, 4, 16), (1, 5, 16), (0, 6, 8),
])
def test_smallest_faithful_real_representation(positive, negative, size):
    """main checks every blade product, the unit, and faithfulness, including both split blocks."""
    embedder = main(positive, negative)
    algebra_slot, state_slot = embedder.input_subspaces
    assert algebra_slot.same_support(embedder.algebra.subspace.full())
    assert state_slot.same_support(embedder.output_subspace)
    assert len(embedder.output_subspace) == size


@pytest.mark.parametrize("positive, negative", [
    (2, 0), (3, 0), (3, 1), (0, 3), (4, 0), (5, 0),
])
def test_trace_recovers_every_blade_and_projects_arbitrary_actions(positive, negative):
    embedder = main(positive, negative)
    ga = embedder.algebra
    algebra_slot, state_slot = embedder.input_subspaces
    Full, Spinor = ga.gatype(algebra_slot), ga.gatype(state_slot)
    state_dimension = len(state_slot)
    context = NumpyContext(ga)
    basis = context.multivector.full(np.eye(ga.blade_count))

    # Pairing the action with every represented algebra element recovers the
    # complete multivector, including both components of a split algebra.
    pairing = (1 * Full).scalar_product(Full)
    actions = embedder(basis)
    readouts = embedder(Full, actions).trace(0, 2) / state_dimension
    recovered = pairing.solve(readouts)
    np.testing.assert_array_equal((recovered - basis).kernel, 0)

    # A single state-coordinate projector need not lie in the represented algebra.
    # Decoding and re-encoding projects it; the remainder pairs to zero with every
    # represented action, and projecting a second time changes nothing.
    one = ga.exact.multivector.scalar()
    arbitrary = one * one.scalar_product(Spinor)
    readout = embedder(Full, arbitrary).trace(0, 2) / state_dimension
    projected = embedder(pairing.solve(readout))
    residual = arbitrary - projected
    orthogonality = embedder(Full, residual).trace(0, 2)
    np.testing.assert_array_equal(orthogonality.kernel.values, 0)

    projected_readout = embedder(Full, projected).trace(0, 2) / state_dimension
    projected_twice = embedder(pairing.solve(projected_readout))
    np.testing.assert_array_equal((projected_twice - projected).kernel.values, 0)
