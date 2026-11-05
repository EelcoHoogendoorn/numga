"""The adjugate reads both of its pairings with the extrinsic factor first, an order that matters in even dimensions.

Swapping the factors of `a & b`, with `a` of grade r and `b` of the complementary grade, costs (-1) ** (r * (n - r)) in
an algebra of n directions: nothing in an odd dimension, the grade involute of either factor in an even one, where the
complement of an odd grade is odd. It is the sign of moving `a` across the pseudoscalar, which in an even dimension
moves across `a` as a vector does, the sign the twisted sandwich pays in every dimension. The adjugate takes the
extrinsic factor first in both pairings that define it, `reading(A.adjugate()(c), x) == reading(c, A(x))`, and so in
every dimension it is its own inverse, composes backwards, makes a form symmetric under the reading its own adjugate,
and pulls a positive form back to a positive one. A versor's sandwich keeps the pairing in either order, so its adjugate
is the reverse sandwich; the scalar product commutes, so the metric adjoint has no sides at all.
"""

import numpy as np
import pytest

from numga import Algebra, NumpyContext, Orientation

# Algebras of both parities, degenerate ones among them: the swap sign is the wedge's, and needs no metric.
ODD = ["x+y+t-", "x+y+w0", "x+y+z+w+t-", "x+y+t-n0p0q0l0"]
EVEN = ["x+y+", "x+y+z+t-", "x+y+z+w0", "x+t-n0p0q0l0"]
ORIENTATIONS = [Orientation.extrinsic, Orientation.intrinsic]
n_values = 3
tolerance = 1e-9


def algebra_and_context(signature):
    ga = Algebra(signature)
    return ga, NumpyContext(ga)


def homogeneous(ga, grade, orientation):
    return ga.gatype.oriented((ga.subspace.k_vector(grade),), (orientation,))


def spaces(ga, orientation):
    """Every grade, and the even and the odd part, each mixed in grade."""
    subspaces = [ga.subspace.k_vector(grade) for grade in range(ga.n_dimensions + 1)]
    return [ga.gatype.oriented((subspace,), (orientation,)) for subspace in subspaces + [ga.subspace.even(), ga.subspace.odd()]]


def values(context, gatype, rng):
    return context.extensor(gatype, rng.normal(size=(n_values, len(gatype.output_subspace))))


def linear_map(context, output, input, rng):
    gatype = context.algebra.gatype((output, input))
    return context.extensor(gatype, rng.normal(size=(len(output.output_subspace), len(input.output_subspace))))


def swapped(value):
    """What moving a factor of complementary grade from one side of `value` to the other costs, applied to `value`:
    its grade involute in an even dimension, nothing in an odd one. On a map it acts on the output."""
    return value.involute() if value.algebra.n_dimensions % 2 == 0 else value


def reading(first, second):
    """The pairing of complementary spaces read with the extrinsic factor first, whichever is passed first: the scalar
    part of their regressive product, all of it between two grades and one part of it between mixed ones."""
    product = first & second if first.gatype.orientation is Orientation.extrinsic else second & first
    return product.select[0]


def assert_equal(first, second):
    assert np.all(np.isfinite(first.kernel)) and np.all(np.isfinite(second.kernel))
    np.testing.assert_allclose(first.kernel, second.kernel, atol=tolerance)


# --- the swap ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("signature", ODD + EVEN)
@pytest.mark.parametrize("left_orientation", ORIENTATIONS)
@pytest.mark.parametrize("right_orientation", ORIENTATIONS)
def test_swapping_complementary_factors_costs_the_involute_in_even_dimensions(signature, left_orientation,
                                                                              right_orientation):
    """`b & a == a.involute() & b` in an even dimension and `b & a == a & b` in an odd one, for every grade of `a`
    and every orientation of either factor; the sign is (-1) ** (r * (n - r)), the same as the wedge's."""
    ga, context = algebra_and_context(signature)
    n = ga.n_dimensions
    rng = np.random.default_rng(1)
    for grade in range(n + 1):
        first = values(context, homogeneous(ga, grade, left_orientation), rng)
        second = values(context, homogeneous(ga, n - grade, right_orientation), rng)
        assert_equal(second & first, swapped(first) & second)
        assert_equal(second & first, (-1) ** (grade * (n - grade)) * (first & second))
        assert_equal(second ^ first, (-1) ** (grade * (n - grade)) * (first ^ second))


@pytest.mark.parametrize("signature", ODD + EVEN)
def test_the_involute_is_the_sign_of_moving_a_vector_across(signature):
    """`v ^ a == a.involute() ^ v` for a vector `v` in every dimension, the sign the twisted sandwich pays; the
    complement of an odd grade moves across it the same way only in an even dimension, where it is odd itself."""
    ga, context = algebra_and_context(signature)
    rng = np.random.default_rng(2)
    vectors = values(context, ga.gatype.vector(), rng)
    for grade in range(ga.n_dimensions):
        blades = values(context, homogeneous(ga, grade, Orientation.extrinsic), rng)
        assert_equal(vectors ^ blades, blades.involute() ^ vectors)


@pytest.mark.parametrize("signature", ["x+y+", "x+y+z+t-", "x+y+t-", "x+y+z+w+t-"])
def test_swapping_complementary_factors_is_moving_one_across_the_pseudoscalar(signature):
    """`b & a == (pseudoscalar * a * pseudoscalar.inverse()) & b` in every nondegenerate dimension: the swap sign is
    the pseudoscalar's commutation with `a`, which is the involute in an even dimension, where the pseudoscalar moves
    across `a` as a vector does, and nothing in an odd one, where it is central."""
    ga, context = algebra_and_context(signature)
    n = ga.n_dimensions
    rng = np.random.default_rng(3)
    pseudoscalar = context.multivector.blade(ga.pseudoscalar_mask)
    for grade in range(n + 1):
        first = values(context, homogeneous(ga, grade, Orientation.extrinsic), rng)
        second = values(context, homogeneous(ga, n - grade, Orientation.intrinsic), rng)
        conjugate = pseudoscalar * first * pseudoscalar.inverse()
        assert_equal(conjugate, swapped(first))
        assert_equal(second & first, conjugate & second)


# --- the adjugate ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("signature", ODD + EVEN)
@pytest.mark.parametrize("output_orientation", ORIENTATIONS)
@pytest.mark.parametrize("input_orientation", ORIENTATIONS)
def test_the_adjugate_reads_both_pairings_extrinsic_first_and_is_its_own_inverse(signature, output_orientation,
                                                                                 input_orientation):
    """`reading(A.adjugate()(c), x) == reading(c, A(x))` for maps between every pair of grades and of mixed-grade
    spaces, and `A.adjugate().adjugate() == A`."""
    ga, context = algebra_and_context(signature)
    rng = np.random.default_rng(5)
    for Output in spaces(ga, output_orientation):
        for Input in spaces(ga, input_orientation):
            operator = linear_map(context, Output, Input, rng)
            adjugate = operator.adjugate()
            complements = values(context, adjugate.gatype.derive.axis(1), rng)
            inputs = values(context, Input, rng)
            assert_equal(reading(adjugate(complements), inputs), reading(complements, operator(inputs)))
            assert_equal(adjugate.adjugate(), operator)


@pytest.mark.parametrize("signature", ODD + EVEN)
@pytest.mark.parametrize("orientations", [(Orientation.extrinsic,) * 3, (Orientation.intrinsic,) * 3,
                                          (Orientation.extrinsic, Orientation.intrinsic, Orientation.extrinsic),
                                          (Orientation.intrinsic, Orientation.extrinsic, Orientation.intrinsic)])
def test_the_adjugate_composes_backwards(signature, orientations):
    """`A(B).adjugate() == B.adjugate()(A.adjugate())`: the adjugate is the transpose for a pairing used the same way
    on every space, so it composes; the grades run over every parity, through a middle space of either orientation."""
    ga, context = algebra_and_context(signature)
    n = ga.n_dimensions
    rng = np.random.default_rng(7)
    grades = [(1, 2, 1), (0, 1, n - 1), (2, 1, 0), (n - 1, n - 2, 1)]
    for outer, middle, inner in grades:
        Outer, Middle, Inner = (homogeneous(ga, grade, orientation)
                                for grade, orientation in zip((outer, middle, inner), orientations))
        first, second = linear_map(context, Middle, Inner, rng), linear_map(context, Outer, Middle, rng)
        assert_equal(second(first).adjugate(), first.adjugate()(second.adjugate()))


# --- forms and their pullbacks -----------------------------------------------------------------------------------
@pytest.mark.parametrize("signature", ["x+y+w0", "x+y+z+w0", "x+y+t-", "x+y+z+t-"])
def test_a_form_symmetric_under_the_reading_is_its_own_adjugate(signature):
    """A sum of dyads read in the open, `plane * (plane & Point)` from points to planes and `point * (Plane & point)`
    from planes to points, reads the same either way round, `form(x) & y == form(y) & x` with the plane first, and is
    its own adjugate in every dimension."""
    ga, context = algebra_and_context(signature)
    rng = np.random.default_rng(8)
    Plane, Point = ga.gatype.vector(), ga.gatype.antivector()
    planes, points = values(context, Plane, rng), values(context, Point, rng)
    weights = rng.normal(size=n_values)
    for form, Space in [((weights * planes * (planes & Point)).sum(axis=0), Point),
                        ((weights * points * (Plane & points)).sum(axis=0), Plane)]:
        first, second = values(context, Space, rng), values(context, Space, rng)
        assert_equal(reading(form(first), second), reading(form(second), first))
        assert_equal(form.adjugate(), form)


@pytest.mark.parametrize("signature", ["x+y+", "x+y+z+", "x+y+z+w+", "x+y+z+w+u+"])
def test_a_metric_is_its_own_adjugate(signature):
    """The Euclidean metric on extrinsic vectors, `Vector.dual()`, read `y & metric(x) == y.dot(x)`, is symmetric
    under the reading, and so its own adjugate in every dimension."""
    ga, context = algebra_and_context(signature)
    rng = np.random.default_rng(9)
    metric = ga.gatype.vector().dual()
    first, second = values(context, ga.gatype.vector(), rng), values(context, ga.gatype.vector(), rng)
    assert_equal(second & metric(first), second.dot(first))
    assert_equal(metric.adjugate()(first), metric(first))


@pytest.mark.parametrize("signature", ["x+y+", "x+y+z+", "x+y+z+w+", "x+y+z+w+u+"])
@pytest.mark.parametrize("input_orientation", ORIENTATIONS)
def test_a_positive_material_pulled_back_through_a_gradient_is_positive(signature, input_orientation):
    """`G.adjugate()(metric(G))` for a gradient G from scalars to extrinsic vectors and the Euclidean metric on those
    is symmetric under the reading and positive in every dimension: read on a potential, it is the energy of its
    gradient, `G(potential) & metric(G(potential))`, a Laplacian of the sign the metric has."""
    ga, context = algebra_and_context(signature)
    rng = np.random.default_rng(11)
    Potential = homogeneous(ga, 0, input_orientation)
    gradient = linear_map(context, ga.gatype.vector(), Potential, rng)
    metric = ga.gatype.vector().dual()
    laplacian = gradient.adjugate()(metric(gradient))
    potentials, others = values(context, Potential, rng), values(context, Potential, rng)
    assert_equal(reading(laplacian(potentials), others), reading(laplacian(others), potentials))
    energies = reading(laplacian(potentials), potentials)
    assert np.all(energies.kernel > 0)
    assert_equal(energies, gradient(potentials) & metric(gradient(potentials)))


# --- the adjoint -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("signature", ["x+y+", "x+y+t-", "x+y+z+t-", "x+y+z+w+t-"])
def test_the_scalar_product_commutes_so_the_adjoint_has_one_side(signature):
    """`a.scalar_product(b) == b.scalar_product(a)` for every grade in every dimension, so the metric adjoint has no
    left and right: applied twice it is the map, whatever the grades and the parity of the dimension."""
    ga, context = algebra_and_context(signature)
    rng = np.random.default_rng(12)
    for output_grade in range(ga.n_dimensions + 1):
        for input_grade in range(ga.n_dimensions + 1):
            Output = homogeneous(ga, output_grade, Orientation.extrinsic)
            Input = homogeneous(ga, input_grade, Orientation.intrinsic)
            first, second = values(context, Input, rng), values(context, Input, rng)
            assert_equal(first.scalar_product(second), second.scalar_product(first))
            operator = linear_map(context, Output, Input, rng)
            assert_equal(operator.adjoint().adjoint(), operator)


# --- the sandwich ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("signature", ["x+y+", "x+y+z+", "x+y+z+t-", "x+y+z+w+t-"])
def test_a_sandwich_keeps_the_pairing_so_its_adjugate_is_the_reverse_sandwich(signature):
    """`(v >> a) & (v >> b) == a & b` for a versor `v` and complementary `a` and `b`, so the adjugate of `v >> Space`
    is `v << Complement`: a sandwich keeps grade, so never meets the swap."""
    ga, context = algebra_and_context(signature)
    n = ga.n_dimensions
    rng = np.random.default_rng(13)
    mv = context.multivector
    mirror = mv.vector(rng.normal(size=n)).normalized()
    rotor = (mirror * mv.vector(rng.normal(size=n))).normalized()
    for versor in (mirror, rotor):
        for grade in range(n + 1):
            Space = homogeneous(ga, grade, Orientation.extrinsic)
            blades = values(context, Space, rng)
            complements = values(context, Space.derive.dual, rng)
            assert_equal((versor >> blades) & (versor >> complements), blades & complements)
            assert_equal((versor >> Space).adjugate()(complements), versor << complements)
