"""Blade charts for faithful Clifford actions on their smallest real state spaces."""

from numga import Algebra, Extensor, GAType


# --- math -----------------------------------------------------------------------------
def readout(representations: Extensor, state_map: Extensor, Space: GAType) -> Extensor:
    """The element of Space whose representation is the orthogonal projection of a map on states
    onto the representations of Space, under the trace pairing.

    Each blade's coefficient is the map composed with the representation of that blade's inverse,
    traced over the state and divided by the number of state components; solving against the
    scalar product supplies the inverse blades. The representation of an element of Space returns
    that element, and a smaller Space reads out only its own blades, since blades are orthogonal
    under the scalar product.
    """
    pairing = (1 * Space).scalar_product(Space)                 # [] Scalar <- (Space, Space)
    state_dimension = len(representations.output_subspace)
    return pairing.solve(representations(1 * Space, state_map).trace(0, 2) / state_dimension)   # [] Space


# --- plumbing -------------------------------------------------------------------------
def spinor_layout(algebra: Algebra) -> tuple[GAType, tuple[int, ...]]:
    """Choose commuting involutions and one blade per coset for a nondegenerate algebra.

    Blade masks describe the layout only. Their products and signs come from the
    algebra; the projector and its action are constructed as extensors in main.
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
    return algebra.gatype(algebra.subspace.from_masks(representatives)), involutions
