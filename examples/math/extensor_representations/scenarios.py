"""Signature table and verification scenarios for minimal real Clifford extensor representations."""

from numga import Algebra

from examples.math.extensor_representations.core import readout, state_representations

max_dimension = 6
signatures = tuple(
    (n_positive, dimension - n_positive)
    for dimension in range(max_dimension + 1)
    for n_positive in range(dimension, -1, -1)
)
representative_signatures = ("x+y+z+", "x+y+z+t-")


def main() -> None:
    """Prints the minimal real representation dimensions through dimension six and verifies exact algebra laws."""
    for n_positive, n_negative in signatures:
        ga = Algebra((n_positive, n_negative, 0))
        mv = ga.exact.multivector
        Full = ga.gatype.full()
        State, representations = state_representations(ga)               # [] State+ <- (Full+, State+)
        n_state_components = len(State.output_subspace)
        print(
            f"Cl({n_positive},{n_negative}): {ga.blade_count:3d} coefficients -> "
            f"{n_state_components:2d} x {n_state_components:<2d} real"
        )

        # checks
        # Composition of representations matches multiplication on the whole algebra, and one maps to the state identity.
        composition_error = representations(Full, representations) - representations(Full * Full)  # [] State+ <- (Full+, Full+, State+)
        identity_error = representations(mv.scalar()) - State            # [] State+ <- State+
        assert not composition_error.kernel.output_nonzero_indices()
        assert not identity_error.kernel.output_nonzero_indices()
        # Trace pairing recovers the identity on every positive-dimensional algebra, including both split blocks.
        if ga.dimension > 0:
            pairing = Full.scalar_product(Full)                          # [] Scalar+ <- (Full+, Full+)
            trace_pairing = representations(Full, representations).trace(0, 3)  # [] Scalar+ <- (Full+, Full+)
            round_trip_error = pairing.solve(trace_pairing / n_state_components) - Full  # [] Full+ <- Full+
            blade_sample = mv.blade(ga.pseudoscalar_mask)                # [] Full+
            readout_error = readout(representations, representations(blade_sample), Full) - blade_sample  # [] Full+
            assert not round_trip_error.kernel.output_nonzero_indices()
            assert not readout_error.kernel.output_nonzero_indices()
        else:
            unit_action_error = representations(mv.scalar(), mv.scalar()) - mv.scalar()  # [] Scalar+
            assert not unit_action_error.kernel.output_nonzero_indices()


if __name__ == "__main__":
    main()
