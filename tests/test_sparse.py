"""Sparse extensors: products, reverses, applications and solves match the dense sums they stand for."""

import numpy as np
import pytest

from numga import Algebra, NumpyContext
from numga.sparse import SparseExtensor


@pytest.fixture(params=["numpy", "jax"])
def context(request):
    algebra = Algebra("x+y+z+")
    if request.param == "jax":
        pytest.importorskip("jax")
        from numga.backend.jax import JaxContext
        return JaxContext(algebra)
    return NumpyContext(algebra)


def random_sparse(context, shape, count, seed):
    rng = np.random.default_rng(seed)
    cells = context.multivector.vector(rng.normal(size=(count, 3)))
    return SparseExtensor(cells, rng.integers(0, shape[0], count), rng.integers(0, shape[1], count), shape)


def dense_product(sparse, field):
    """Element r of the product: the sum of each cell in row r times the element its column names."""
    return [sum((sparse.cells[k] * field[sparse.columns[k]] for k in np.flatnonzero(sparse.rows == r)),
                sparse.cells[0] * field[0] * 0.0) for r in range(sparse.shape[0])]


def test_products_and_reverse_match_dense_sums(context):
    mv = context.multivector
    rng = np.random.default_rng(0)
    first, second = random_sparse(context, (4, 5), 9, 1), random_sparse(context, (5, 6), 11, 2)
    field = mv.even(rng.normal(size=(6, 4)))
    product = second * field                                                  # [5] Odd
    for row, expected in enumerate(dense_product(second, field)):
        np.testing.assert_allclose(np.asarray(product[row].kernel), np.asarray(expected.kernel), atol=1e-5)
    np.testing.assert_allclose(np.asarray(((first * second) * field).kernel), np.asarray((first * (second * field)).kernel), atol=1e-4)
    # Reversing a sparse extensor reverses its products with fields: the scalar part of the reverse of
    # ~S * other times a field is that of the reverse of other times S * field.
    other = mv.odd(rng.normal(size=(5, 4)))
    left = ((~second * other).reverse() * field).sum(axis=0).select.scalar()
    right = (other.reverse() * (second * field)).sum(axis=0).select.scalar()
    np.testing.assert_allclose(np.asarray(left.kernel), np.asarray(right.kernel), atol=1e-4)


def test_open_type_cells_apply_solve_and_take_least_squares():
    pytest.importorskip("scipy")
    context = NumpyContext(Algebra("x+y+z+"))
    mv, Vector, Even = context.multivector, context.algebra.gatype.vector(), context.algebra.gatype.even()
    rng = np.random.default_rng(3)
    # A path graph's Laplacian, singular along the constant field.
    count = 6
    boundary = SparseExtensor(
        mv.scalar(np.tile([[-1.0], [1.0]], (count - 1, 1))),
        np.repeat(np.arange(count - 1), 2), np.stack([np.arange(count - 1), np.arange(1, count)], -1).ravel(), (count - 1, count),
    )
    laplacian = ~boundary * boundary
    field = mv.vector(rng.normal(size=(count, 3)))
    np.testing.assert_allclose((laplacian * Vector)(field).kernel, (laplacian * field).kernel, atol=1e-12)
    solved = (laplacian * Vector).lstsq(laplacian * field)
    np.testing.assert_allclose(solved.kernel, (field - field.mean(axis=0)).kernel, atol=1e-10)
    # Regular once a diagonal is added; the solve inverts the product.
    stiff = laplacian + SparseExtensor.from_diagonal(mv.scalar(np.ones((count, 1))))
    np.testing.assert_allclose((stiff * Vector).solve(stiff * field).kernel, field.kernel, atol=1e-12)
    # The least eigenpair of the Laplacian against the identity: the constant field, at zero.
    values, modes = (laplacian * Even).eigh(SparseExtensor.from_diagonal(mv.scalar(np.ones((count, 1)))) * Even, 1)
    np.testing.assert_allclose(values.to_array(), 0.0, atol=1e-10)
    np.testing.assert_allclose((laplacian * modes[0]).kernel, 0.0, atol=1e-10)


def test_couplings_of_one_pair_sum_exactly():
    # The boundary of a triangle's boundary: the six paths from the face through its edges to its
    # corners meet two at each corner, and cancel there exactly.
    mv = NumpyContext(Algebra("x+y+z+")).multivector
    edges = SparseExtensor.from_columns(np.array([[0, 1], [1, 2], [0, 2]]), mv.scalar(np.tile([[-1.0], [1.0]], (3, 1, 1))), 3)
    face = SparseExtensor.from_columns(np.array([[0, 1, 2]]), mv.scalar([[[1.0], [1.0], [-1.0]]]), 3)
    boundary = face * edges                                                   # [1, 3] Scalar
    assert len(boundary.rows) == 3
    np.testing.assert_array_equal(np.asarray(boundary.cells.kernel), 0.0)
    # Each corner lies on two edges.
    np.testing.assert_array_equal(np.asarray((~edges * edges).diagonal().kernel), 2.0)


def test_square_maps_between_other_blades_solve_batched_fields():
    pytest.importorskip("scipy")
    # Cells from twists to lines in the plane: square, between different blades. Fields with a leading
    # batch axis are applied, pulled back and solved one by one in a single call.
    from numga.algebras import PGA2D

    context = NumpyContext(PGA2D)
    mv, Twist, Line = context.multivector, PGA2D.gatype.bivector(), PGA2D.gatype.antibivector()
    rng = np.random.default_rng(4)
    count, fields = 5, 3
    motors = (mv.bivector(rng.normal(size=(count, 2, 3))) * 0.5).exp()          # [count, 2] Motor
    neighbours = np.stack([np.arange(count), (np.arange(count) + 1) % count], -1)   # [count, 2]
    coupling = SparseExtensor.from_columns(neighbours, (motors >> Twist).dual() * np.array([1.0, 0.3]), count)   # [count, count] Line <- Twist
    twists = mv.bivector(rng.normal(size=(fields, count, 3)))                  # [fields, count] Twist
    lines = coupling(twists)                                                  # [fields, count] Line
    for field in range(fields):
        np.testing.assert_allclose(lines[field].kernel, coupling(twists[field]).kernel, atol=1e-12)
    np.testing.assert_allclose(coupling.solve(lines).kernel, twists.kernel, atol=1e-10)
    # The adjugate pulls a field on the outputs' complements back: summed, its pairing with any input
    # field is that of the field with the coupling's output.
    probes = mv.bivector(rng.normal(size=(count, 3)))                          # [count] Twist
    pulled = coupling.adjugate()                                              # [count, count] Line <- Twist
    np.testing.assert_allclose(
        (pulled(probes) & twists[0]).sum(axis=-1).kernel, (probes & lines[0]).sum(axis=-1).kernel, atol=1e-10,
    )


def test_the_adjoint_carries_the_scalar_product_summed_over_the_elements():
    context = NumpyContext(Algebra("x+y+z+"))
    mv, Vector = context.multivector, context.algebra.gatype.vector()
    rng = np.random.default_rng(5)
    count = 4
    rotors = (mv.bivector(rng.normal(size=(count, 2, 3))) * 0.5).exp()
    neighbours = np.stack([np.arange(count), (np.arange(count) + 1) % count], -1)
    coupling = SparseExtensor.from_columns(neighbours, (rotors >> Vector) * np.array([1.0, 0.4]), count)
    values, covectors = mv.vector(rng.normal(size=(count, 3))), mv.vector(rng.normal(size=(count, 3)))
    np.testing.assert_allclose(coupling.adjoint()(covectors).scalar_product(values).sum(axis=0).kernel,
                               covectors.scalar_product(coupling(values)).sum(axis=0).kernel, atol=1e-12)


def test_leading_axes_hold_separate_maps_of_one_pattern():
    # Each case of a batched sparse extensor acts as that case alone: on fields, in compositions,
    # in solves, also against right sides batched beyond it, and in eigenproblems.
    pytest.importorskip("scipy")
    context = NumpyContext(Algebra("x+y+z+"))
    mv, Vector = context.multivector, context.algebra.gatype.vector()
    rng = np.random.default_rng(3)
    cases, size, count = 3, 5, 9
    rows, columns = rng.integers(0, size, count), rng.integers(0, size, count)
    cells = mv.vector(rng.normal(size=(cases, count, 3)))                      # [cases, count] Vector
    batched = SparseExtensor(cells, rows, columns, (size, size))
    one = [SparseExtensor(cells[case], rows, columns, (size, size)) for case in range(cases)]
    field = mv.vector(rng.normal(size=(size, 3)))                               # [size] Vector
    for case in range(cases):
        np.testing.assert_allclose((batched * field)[case].kernel, (one[case] * field).kernel, atol=1e-12)
        np.testing.assert_allclose(((batched * batched) * field)[case].kernel, ((one[case] * one[case]) * field).kernel, atol=1e-12)
    # A diagonally dominant map on vectors per case, solved against two sides for each case.
    diagonal = SparseExtensor.from_diagonal(mv.scalar(np.full((size, 1), 20.0))) * Vector
    system = (batched * Vector).adjoint()(batched * Vector) + diagonal           # [cases] [size, size] Vector <- Vector
    sides = mv.vector(rng.normal(size=(2, cases, size, 3)))                      # [sides, cases, size] Vector
    solved = system.solve(sides)                                                 # [sides, cases, size] Vector
    np.testing.assert_allclose((system(solved) - sides).kernel, 0.0, atol=1e-10)
    values, modes = system.eigh(diagonal, 2)                                     # [cases, 2] Scalar, [cases, 2, size] Vector
    for case in range(cases):
        alone, _ = SparseExtensor(system.cells[case], system.rows, system.columns, system.shape).eigh(diagonal, 2)
        np.testing.assert_allclose(values[case].kernel, alone.kernel, rtol=1e-8)
