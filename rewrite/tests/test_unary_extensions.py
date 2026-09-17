"""Typed linear algebra: reconstruction, layouts, batch broadcasting and backends."""

import numpy as np
import pytest

from numga import Algebra, Extensor, GAType, NumpyContext, SubSpace


@pytest.fixture(params=["numpy", "sparse", "jax"])
def context(request):
    algebra = Algebra("x+y+z0")
    if request.param == "jax":
        pytest.importorskip("jax")
        from numga.backend.jax import JaxContext
        return JaxContext(algebra, dtype=np.float32)
    return NumpyContext(algebra, execution="sparse" if request.param == "sparse" else "dense")


def make_map(context, output, domain, coefficients):
    return context.extensor(context.algebra.gatype((output, domain)), coefficients)


def test_cholesky_aligns_layout_and_preserves_batches(context):
    ga = context.algebra
    slot = ga.subspace.vector()
    matrix = np.array([[3, 1, 0], [1, 2, 0], [0, 0, 4]])
    operator = make_map(context, slot, slot, np.broadcast_to(matrix, (2, 1, 3, 3)))
    factor = operator.cast(ga.subspace("z -x y")).cholesky()
    assert factor.shape == (2, 1)
    assert factor.axes == (slot, slot)
    np.testing.assert_allclose(factor(factor.transpose()).kernel, operator.kernel, atol=2e-6)
    np.testing.assert_allclose(np.triu(factor.kernel, 1), 0)


def test_complex_cholesky_reconstructs_hermitian_map():
    ga = Algebra("x+y+")
    ctx = NumpyContext(ga, dtype=np.complex128)
    slot = ga.subspace.vector()
    operator = make_map(ctx, slot, slot, [[3, 1j], [-1j, 2]])
    factor = operator.cholesky()
    np.testing.assert_allclose(factor.kernel @ factor.kernel.conj().T, operator.kernel, atol=1e-12)


def test_hermitian_eigenpairs_reconstruct_batched_operator(context):
    vector = context.algebra.subspace.vector()
    matrix = np.array([[[3, 1, 0], [1, 2, 0], [0, 0, 4]],
                       [[2, 0, 0], [0, 2, 0], [0, 0, 2]]])
    operator = make_map(context, vector, vector, matrix)
    values, vectors = operator.eigh()
    assert vectors.axes == (vector,)
    assert vectors.shape == values.shape == (2, 3)
    assert values.gatype.is_scalar
    np.testing.assert_allclose(operator[:, None](vectors).kernel,
                               (vectors * values).kernel, atol=2e-6)
    np.testing.assert_allclose(operator.eigvalsh().kernel, values.kernel, atol=2e-6)
    reconstructed = np.einsum("...ki,...k,...kj->...ij", vectors.kernel,
                              values.kernel[..., 0], np.conj(vectors.kernel))
    np.testing.assert_allclose(reconstructed, matrix, atol=2e-6)
    np.testing.assert_allclose(operator.det().kernel[..., 0], [20, 8], rtol=2e-6)


@pytest.mark.parametrize("method", ["eig", "eigh"])
def test_eigenmethods_align_signed_reordered_layouts(context, method):
    ga = context.algebra
    vector = ga.subspace.vector()
    reordered = ga.subspace("z -x y")
    operator = make_map(context, vector, vector, [[3, 1, 0], [1, 2, 0], [0, 0, 4]])
    relayout = operator.cast(reordered)
    values, vectors = getattr(relayout, method)()
    compatible = Extensor(vectors.context, operator.gatype, operator.kernel)
    np.testing.assert_allclose(compatible(vectors).kernel, (vectors * values).kernel, atol=2e-6)
    np.testing.assert_allclose(relayout.det().kernel, operator.det().kernel, atol=2e-6)


def test_real_rotation_produces_complex_eigenpairs(context):
    plane = context.algebra.subspace("x y")
    operator = make_map(context, plane, plane, [[0, -1], [1, 0]])
    values, vectors = operator.eig()
    assert values.dtype.kind == vectors.dtype.kind == "c"
    assert vectors.kernel.dtype == vectors.context.dtype
    promoted = Extensor(vectors.context, operator.gatype, operator.kernel)
    np.testing.assert_allclose(promoted(vectors).kernel, (vectors * values).kernel, atol=2e-6)
    np.testing.assert_allclose(np.sort_complex(operator.eigvals().kernel[..., 0]),
                               np.sort_complex(values.kernel[..., 0]), atol=2e-6)


@pytest.mark.parametrize("rows,cols", [(3, 2), (2, 3)])
def test_rectangular_svd_reconstruction_and_pseudoinverse(context, rows, cols):
    ga = context.algebra
    output = ga.subspace("x y z" if rows == 3 else "x y")
    domain = ga.subspace("xy xz yz" if cols == 3 else "xy xz")
    matrix = np.random.default_rng(3).normal(size=(2, rows, cols))
    operator = make_map(context, output, domain, matrix)
    left, singular, right = operator.svd()
    assert left.axes == (output,)
    assert right.axes == (domain,)
    assert left.shape == right.shape == singular.shape == (2, 2)
    np.testing.assert_allclose(operator[:, None](right).kernel,
                               (left * singular).kernel, atol=2e-6)
    reconstructed = np.einsum("...ki,...k,...kj->...ij", left.kernel,
                              singular.kernel[..., 0], np.conj(right.kernel))
    np.testing.assert_allclose(reconstructed, matrix, atol=2e-6)
    np.testing.assert_allclose(operator.svdvals().kernel, singular.kernel, atol=2e-6)
    inverse = operator.pinv()
    assert inverse.axes == (domain, output)
    np.testing.assert_allclose(operator(inverse(operator)).kernel, matrix, atol=3e-6)


def test_solve_broadcasts_rhs_and_preserves_map_slots(context):
    ga = context.algebra
    output, domain = ga.subspace.vector(), ga.subspace.bivector()
    matrix = np.array([[3, 1, 0], [0, 2, 1], [0, 0, 4]])
    operator = make_map(context, output, domain, matrix)
    rhs = context.multivector.vector([[1, 2, 3], [4, 5, 6]])
    solution = operator.solve(rhs)
    assert solution.axes == (domain,)
    np.testing.assert_allclose(operator(solution).kernel, rhs.kernel, atol=2e-6)
    rhs_map = make_map(context, output, ga.subspace("x y"),
                       np.arange(12).reshape(2, 3, 2))
    map_solution = operator.solve(rhs_map)
    assert map_solution.axes == (domain, ga.subspace("x y"))
    np.testing.assert_allclose(operator(map_solution).kernel, rhs_map.kernel, atol=2e-6)
    batched = make_map(context, output, domain, matrix[None, None] * np.ones((2, 1, 1, 1)))
    broadcast = batched.solve(rhs)
    assert broadcast.shape == (2, 2)
    np.testing.assert_allclose(batched[:, 0](solution).kernel, rhs.kernel, atol=2e-6)
    np.testing.assert_allclose(batched(broadcast).kernel, np.broadcast_to(rhs.kernel, (2, 2, 3)), atol=2e-6)
    reordered = rhs.cast(ga.subspace("z -x y"))
    np.testing.assert_allclose(operator.solve(reordered).kernel, solution.kernel, atol=2e-6)


def test_rank_deficient_least_squares_has_minimum_norm(context):
    ga = context.algebra
    output, domain = ga.subspace.vector(), ga.subspace("xy xz")
    operator = make_map(context, output, domain, [[1, 2], [2, 4], [0, 0]])
    rhs = context.multivector.vector([[1, 2, 3], [2, 1, 0]])
    solution = operator.lstsq(rhs, rcond=1e-5)
    expected = np.linalg.lstsq(np.asarray(operator.kernel), np.asarray(rhs.kernel).T, rcond=1e-5)[0].T
    np.testing.assert_allclose(solution.kernel, expected, atol=2e-6)
    inverse = operator.pinv(rcond=1e-5)
    np.testing.assert_allclose(inverse(operator(inverse)).kernel, inverse.kernel, atol=2e-6)


@pytest.mark.parametrize("arity", [0, 1, 2, 3])
@pytest.mark.parametrize("method", ["solve", "lstsq"])
def test_solutions_preserve_all_rhs_slots_and_broadcast_batches(context, arity, method):
    ga = context.algebra
    output, domain = ga.subspace.vector(), ga.subspace.bivector()
    matrix = np.array([[3, 1, 0], [0, 2, 1], [0, 0, 4]])
    operator = make_map(context, output, domain, matrix[None, None] * np.array([1, 2])[:, None, None, None])
    slots = (ga.subspace("xy xz"), ga.subspace("z"), ga.subspace.vector())[:arity]
    rhs_type = ga.gatype((ga.subspace("y -x"),) + slots)
    rhs = context.extensor(rhs_type, np.random.default_rng(0).normal(size=(1, 4) + rhs_type.structural_shape))
    solution = getattr(operator, method)(rhs)
    assert solution.axes == (domain,) + slots
    assert solution.shape == (2, 4)
    assert solution.arity == arity
    expected = np.broadcast_to(rhs.cast(output).kernel, (2, 4) + (len(output),) + rhs.structural_shape[1:])
    np.testing.assert_allclose(operator(solution).kernel, expected, atol=2e-6)


def test_rectangular_lstsq_with_binary_rhs(context):
    ga = context.algebra
    output, domain = ga.subspace.vector(), ga.subspace("xy xz")
    operator = make_map(context, output, domain, [[1, 0], [0, 2], [0, 0]])
    rhs_type = ga.gatype((output, ga.subspace("x y"), ga.subspace("yz")))
    rhs = context.extensor(rhs_type, np.random.default_rng(1).normal(size=(4,) + rhs_type.structural_shape))
    solution = operator.lstsq(rhs)
    assert solution.axes == (domain,) + rhs.input_subspaces
    assert solution.shape == rhs.shape
    residual = operator(solution) - rhs
    np.testing.assert_allclose(operator.transpose()(residual).kernel, 0, atol=2e-6)


def test_complex_svd_and_hermitian_eigenvectors():
    ga = Algebra("x+y-")
    context = NumpyContext(ga, dtype=np.complex128)
    vector = ga.subspace.vector()
    operator = make_map(context, vector, vector, [[2, 1j], [-1j, 3]])
    values, vectors = operator.eigh()
    np.testing.assert_allclose(operator(vectors).kernel, (vectors * values).kernel, atol=1e-14)
    left, singular, right = operator.svd()
    np.testing.assert_allclose(operator(right).kernel, (left * singular).kernel, atol=1e-14)
    reconstructed = np.einsum("ki,k,kj->ij", left.kernel, singular.kernel[..., 0], right.kernel.conj())
    np.testing.assert_allclose(reconstructed, operator.kernel, atol=1e-14)


def test_invalid_spaces_are_rejected_by_dispatch():
    ga = Algebra("x+y+z+")
    vector, bivector = ga.subspace.vector(), ga.subspace.bivector()
    polarity = ga.gatype((vector, bivector))
    for method in ("eig", "eigh", "eigvals", "eigvalsh", "det"):
        with pytest.raises(LookupError, match="no .* implementation"):
            extension = getattr(Extensor, method)
            dispatch = extension._dispatch if method == "det" else extension.overload(1)._dispatch
            dispatch.resolve(polarity)
    for method in (Extensor.solve, Extensor.lstsq):
        with pytest.raises(LookupError, match="no .* implementation"):
            method._dispatch.resolve(polarity, ga.gatype(bivector))


def test_warm_calls_do_not_repeat_static_checks(monkeypatch):
    ga = Algebra("x+y+z+")
    context = NumpyContext(ga)
    vector = ga.subspace.vector()
    operator = make_map(context, vector, vector, [[3, 1, 0], [1, 2, 0], [0, 0, 4]])
    relayout = operator.cast(ga.subspace("z -x y"))
    rhs = context.multivector.vector([1, 2, 3])
    rhs_map = make_map(context, vector, ga.subspace("x y"), [[1, 0], [0, 1], [1, 1]])
    rhs_binary = rhs_map * ga.gatype.scalar()
    calls = [getattr(relayout, name) for name in ("eig", "eigh", "eigvals", "eigvalsh", "det")]
    calls += [lambda: operator.solve(rhs), lambda: operator.solve(rhs_map),
              lambda: operator.lstsq(rhs), lambda: operator.lstsq(rhs_map),
              lambda: operator.solve(rhs_binary), lambda: operator.lstsq(rhs_binary)]
    for call in calls:
        call()

    def repeated(*args):
        raise AssertionError("static checks must run only during dispatch resolution")

    monkeypatch.setattr(SubSpace, "same_support", repeated)
    monkeypatch.setattr(SubSpace, "support_is_subset_of", repeated)
    monkeypatch.setattr(GAType, "is_square_map", property(repeated))
    for call in calls:
        call()
