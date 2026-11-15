"""Minimal faithful real Clifford representations recover every blade by trace pairing and project general state maps."""

import numpy as np
import pytest

from numga import Algebra, NumpyContext

from examples.math.extensor_representations.core import readout, state_representations


@pytest.mark.parametrize("n_positive, n_negative, n_expected_components", [
    (0, 0, 1),
    (1, 0, 2), (0, 1, 2),
    (2, 0, 2), (1, 1, 2), (0, 2, 4),
    (3, 0, 4), (2, 1, 4), (1, 2, 4), (0, 3, 8),
    (4, 0, 8), (3, 1, 4), (2, 2, 4), (1, 3, 8), (0, 4, 8),
    (5, 0, 16), (4, 1, 8), (3, 2, 8), (2, 3, 8), (1, 4, 16), (0, 5, 8),
    (6, 0, 16), (5, 1, 16), (4, 2, 8), (3, 3, 8), (2, 4, 16), (1, 5, 16), (0, 6, 8),
])
def test_smallest_faithful_real_representation(n_positive: int, n_negative: int, n_expected_components: int) -> None:
    """Every signature through dimension six acts faithfully on its minimal real state space."""
    ga = Algebra((n_positive, n_negative, 0))
    mv = ga.exact.multivector
    Full = ga.gatype.full()
    State, representations = state_representations(ga)
    algebra_axis, state_axis = representations.input_subspaces
    n_state_components = len(State.output_subspace)

    assert algebra_axis.same_support(Full.output_subspace)
    assert state_axis.same_support(representations.output_subspace)
    assert n_state_components == n_expected_components
    assert not (representations(Full, representations) - representations(Full * Full)).kernel.output_nonzero_indices()
    assert not (representations(mv.scalar()) - State).kernel.output_nonzero_indices()

    if ga.dimension > 0:
        pairing = Full.scalar_product(Full)
        trace_pairing = representations(Full, representations).trace(0, 3)
        round_trip_error = pairing.solve(trace_pairing / n_state_components) - Full
        assert not round_trip_error.kernel.output_nonzero_indices()
    else:
        assert not (representations(mv.scalar(), mv.scalar()) - mv.scalar()).kernel.output_nonzero_indices()


@pytest.mark.parametrize("n_positive, n_negative", [
    (2, 0), (3, 0), (3, 1), (0, 3), (4, 0), (5, 0),
])
def test_trace_recovers_every_blade_and_projects_arbitrary_maps(n_positive: int, n_negative: int) -> None:
    """Trace readout recovers basis blades, restricts cleanly to subspaces, and projects rank-one state maps."""
    ga = Algebra((n_positive, n_negative, 0))
    Full = ga.gatype.full()
    Vector = ga.gatype.vector()
    State, representations = state_representations(ga)
    context = NumpyContext(ga)
    basis = context.multivector.full(np.eye(ga.blade_count))

    # Trace readout recovers every basis multivector, including both blocks of a split algebra.
    recovered_basis = readout(representations, representations(basis), Full)
    np.testing.assert_array_equal((recovered_basis - basis).kernel, 0)

    # Restricting the target space reads out only that subspace's blades.
    vector_sample = context.multivector.basis().sum(axis=0)
    mixed_sample = vector_sample + context.multivector.scalar()
    recovered_vector = readout(representations, representations(mixed_sample), Vector)
    np.testing.assert_array_equal((recovered_vector - vector_sample).kernel, 0)

    # Decoding and re-encoding projects an arbitrary rank-one state map orthogonally under the trace pairing.
    unit_scalar = ga.exact.multivector.scalar()
    arbitrary_map = unit_scalar * unit_scalar.scalar_product(State)
    projected_multivector = readout(representations, arbitrary_map, Full)
    projected_map = representations(projected_multivector)
    remainder = arbitrary_map - projected_map
    orthogonality = representations(Full, remainder).trace(0, 2)
    assert not orthogonality.kernel.output_nonzero_indices()

    reprojected_multivector = readout(representations, projected_map, Full)
    reprojected_map = representations(reprojected_multivector)
    assert not (reprojected_map - projected_map).kernel.output_nonzero_indices()
