"""Extensor representations: blade charts for faithful Clifford actions on their smallest real state
spaces."""

from numga import Algebra, Extensor, GAType, Orientation


# --- math -----------------------------------------------------------------------------
def readout(representations: Extensor, state_map: Extensor, Space: GAType) -> Extensor:
    """The element of Space whose representation is the orthogonal projection of a map on states
    onto the representations of Space, under the trace pairing.

    Each blade's coefficient is the map composed with the representation of that blade's inverse,
    traced over the state and divided by the number of state components. The representation of an
    element of Space returns that element, and a smaller Space reads out only its own blades, since
    blades are orthogonal under the scalar product.
    """
    pairing = Space.scalar_product(Space)                                # [] Scalar+ <- (Space+, Space+)
    n_state_components = len(representations.output_subspace)
    # Composed with the representations and traced over the state, the map is the scalar product with
    # the element sought, a form on Space; solving against the scalar product gives the element itself.
    return pairing.solve(representations(Space, state_map).trace(0, 2) / n_state_components)   # [] Space+


# --- plumbing -------------------------------------------------------------------------
def spinor_layout(algebra: Algebra) -> tuple[GAType, tuple[int, ...]]:
    """The space of states of a nondegenerate algebra, one blade from each coset of the group that
    its chosen commuting involutions generate, and those involutions.

    Blade masks describe the layout only. Their products and signs come from the algebra; the
    projector and its action are built from them as extensors.
    """
    involutions = ()
    span = {0}
    split = algebra.dimension % 2 == 1 and algebra.pseudoscalar_squared == 1
    for blade in algebra.blade_masks:
        if blade in span or algebra.geometric_product(blade, blade).coefficient != 1:
            continue
        if any(algebra.geometric_product(blade, other).coefficient
               != algebra.geometric_product(other, blade).coefficient for other in involutions):
            continue
        # Fixing the central volume's sign would discard one of the two simple blocks.
        if split and (blade ^ algebra.pseudoscalar_mask) in span:
            continue
        involutions += (blade,)
        span |= {blade ^ other for other in span}

    # Multiplying a chart blade by the projector fills its coset. Distinct cosets
    # have disjoint support, so keeping one blade from each gives independent states.
    representatives = {min(blade ^ other for other in span) for blade in algebra.blade_masks}
    return algebra.gatype(algebra.subspace.from_masks(representatives), orientation=Orientation.extrinsic), involutions
