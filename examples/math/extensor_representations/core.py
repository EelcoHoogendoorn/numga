"""Minimal faithful real representations of nondegenerate Clifford algebras act as extensors on projected blade charts."""

from numga import Algebra, Extensor, GAType, Orientation


def readout(representations: Extensor, state_map: Extensor, Space: GAType) -> Extensor:
    """Trace pairing against `representations` orthogonally projects `state_map` onto `Space` (`[] Space+`)."""
    pairing = Space.scalar_product(Space)                                # [] Scalar+ <- (Space+, Space+)
    n_state_components = len(representations.output_subspace)
    # Composing with the representations and tracing over the state isolates the scalar product with the sought element.
    return pairing.solve(representations(Space, state_map).trace(0, 2) / n_state_components)  # [] Space+


def state_layout(algebra: Algebra) -> tuple[GAType, tuple[int, ...]]:
    """Commuting unit-square blades partition `algebra` into cosets whose representatives span the minimal real state space `State+`."""
    involutions = ()
    coset_group = {0}
    is_split = algebra.dimension % 2 == 1 and algebra.pseudoscalar_squared == 1
    for blade in algebra.blade_masks:
        if blade in coset_group or algebra.geometric_product(blade, blade).coefficient != 1:
            continue
        if any(
            algebra.geometric_product(blade, other).coefficient
            != algebra.geometric_product(other, blade).coefficient
            for other in involutions
        ):
            continue
        # Leaving the central volume unconstrained retains both simple blocks of a split odd-dimensional algebra.
        if is_split and (blade ^ algebra.pseudoscalar_mask) in coset_group:
            continue
        involutions += (blade,)
        coset_group |= {blade ^ other for other in coset_group}

    # Right multiplication by the projector fills each coset, so one blade per coset gives independent state coordinates.
    representatives = {min(blade ^ other for other in coset_group) for blade in algebra.blade_masks}
    State = algebra.gatype(algebra.subspace.from_masks(representatives), orientation=Orientation.extrinsic)
    return State, involutions


def state_representations(algebra: Algebra) -> tuple[GAType, Extensor]:
    """Left multiplication on the minimal projected left ideal represents `algebra` on `State+` (`[] State+ <- (Full+, State+)`)."""
    mv = algebra.exact.multivector
    Full = algebra.gatype.full()
    State, involutions = state_layout(algebra)
    n_coset_blades = 2 ** len(involutions)

    # Commuting unit-square blades yield commuting projectors whose product selects their common positive ideal.
    projector = mv.scalar()                                              # [] Scalar+
    for blade in involutions:
        projector = projector * (1 + mv.blade(blade)) / 2

    # Left multiplication preserves the right ideal of the projector, and rescaling restores the projected weight.
    state_embedding = State * projector                                  # [] Full+ <- State+
    state_readout = (n_coset_blades * Full).cast(State)                  # [] State+ <- Full+
    representations = state_readout(Full * state_embedding)              # [] State+ <- (Full+, State+)
    return State, representations
