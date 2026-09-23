"""Form decompositions preserve scalar output, input layouts, and singleton batches."""

import numpy as np
import pytest

from numga import Algebra, Extensor, NumpyContext, SubSpace


@pytest.fixture(params=["numpy", "sparse", "jax"])
def context(request):
    ga = Algebra("x+y+z0")
    if request.param == "jax":
        pytest.importorskip("jax")
        from numga.backend.jax import JaxContext
        return JaxContext(ga)
    return NumpyContext(ga, execution="sparse" if request.param == "sparse" else "dense")


def form(context, first, second, matrix):
    gatype = context.algebra.gatype((context.algebra.subspace.scalar(), first, second))
    return context.extensor(gatype, np.asarray(matrix)[..., None, :, :])


def test_forms_without_a_metric_use_the_slot_metric(context):
    """x+y+z0 vectors have a singular metric: eig gives an infinite mode, while eigh, det
    and trace need an invertible metric and say so; SVD pairs two covector slots, never."""
    ga = context.algebra
    slot = ga.subspace.vector()
    value = form(context, slot, slot, [[3.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 4.0]])
    if isinstance(context, NumpyContext):
        pytest.importorskip("scipy")
        values = value.eigvals().kernel[..., 0]
        assert np.isinf(values).sum() == 1
        np.testing.assert_allclose(np.sort(np.real(values[np.isfinite(values)])), np.linalg.eigvalsh([[3, 1], [1, 2]]), atol=1e-12)
    for method in ("eigh", "eigvalsh", "det", "trace"):
        with pytest.raises(TypeError, match="metric"):
            getattr(value, method)()
    for method in ("svd", "svdvals"):
        with pytest.raises((LookupError, TypeError)):
            getattr(value, method)()


def test_forms_on_a_euclidean_slot_keep_the_plain_solver():
    ga = Algebra("x+y+z+")
    context = NumpyContext(ga)
    slot = ga.subspace.vector()
    matrix = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 4.0]])
    value = form(context, slot, slot, matrix)
    np.testing.assert_allclose(value.eigvalsh().kernel[..., 0], np.linalg.eigvalsh(matrix), atol=1e-12)
    np.testing.assert_allclose(value.det().kernel, np.linalg.det(matrix), atol=1e-12)
    np.testing.assert_allclose(value.trace().kernel, np.trace(matrix), atol=1e-12)


@pytest.mark.parametrize("execution", ["dense", "sparse"])
@pytest.mark.parametrize("method", ["eig", "eigh"])
def test_form_eigenpairs_against_a_metric_align_slots_and_keep_singleton_batches(execution, method):
    pytest.importorskip("scipy")
    ga = Algebra("x+y+z0")
    context = NumpyContext(ga, execution=execution)
    vector = ga.subspace.vector()
    matrix = np.array([[3, 1, 0], [1, 2, 0], [0, 0, 4]])
    gram = np.diag([1.0, 2.0, 0.5])
    value = form(context, vector, vector, np.broadcast_to(matrix, (2, 1, 3, 3)))
    metric = form(context, vector, vector, gram)
    first, second = ga.subspace("z -x y"), ga.subspace("-y z x")
    changed = value.bind({0: ga.operator.identity(first), 1: ga.operator.identity(second)})
    values, vectors = getattr(changed, method)(metric)
    assert vectors.axes == (second,)
    assert values.shape == vectors.shape == (2, 1, 3)
    coefficients = vectors.cast(vector).kernel
    np.testing.assert_allclose(np.einsum("ij,...kj->...ki", matrix, coefficients),
                               np.einsum("ij,...kj->...ki", gram, coefficients) * values.kernel, atol=2e-6)
    np.testing.assert_allclose(changed.det(metric).kernel,
                               np.full((2, 1, 1), np.linalg.det(matrix) / np.linalg.det(gram)), atol=2e-6)
    np.testing.assert_allclose(np.sort(np.real(changed.eigvals(metric).kernel[..., 0]), axis=-1),
                               changed.eigvalsh(metric).kernel[..., 0], atol=2e-6)


def test_signed_scalar_output_is_not_treated_as_positive():
    pytest.importorskip("scipy")
    ga = Algebra("x+y+z0")
    context = NumpyContext(ga)
    slot = ga.subspace("x")
    value = context.extensor(ga.gatype((ga.subspace("-1"), slot, slot)), [[[[-3.0]]]])
    metric = form(context, slot, slot, [[1.0]])
    values, vectors = value.eigh(metric)
    assert values.shape == vectors.shape == (1, 1)
    np.testing.assert_allclose(values.kernel, [[[3.0]]])


def test_form_cholesky_aligns_slots_and_preserves_batches(context):
    ga = context.algebra
    slot = ga.subspace.vector()
    matrix = np.array([[3, 1, 0], [1, 2, 0], [0, 0, 4]])
    value = form(context, slot, slot, np.broadcast_to(matrix, (2, 1, 3, 3)))
    first, second = ga.subspace("z -x y"), ga.subspace("-y z x")
    changed = value.bind({0: ga.operator.identity(first), 1: ga.operator.identity(second)})
    factor = changed.cholesky()
    assert factor.shape == (2, 1)
    assert factor.axes == (second, second)
    product = np.einsum("...ik,...jk->...ij", factor.kernel, factor.kernel)
    rebuilt = context.extensor(ga.gatype((second, second)), product)
    rebuilt = rebuilt(ga.operator.identity(slot)).cast(slot)
    np.testing.assert_allclose(rebuilt.kernel, np.broadcast_to(matrix, (2, 1, 3, 3)), atol=2e-6)
    np.testing.assert_allclose(np.triu(factor.kernel, 1), 0)


def test_form_solve_fills_the_first_slot(context):
    """value(x, y) == rhs(y) for every y; x lives in the form's first slot."""
    ga = context.algebra
    first, second = ga.subspace.vector(), ga.subspace("xy yz")
    matrix = np.array([[2.0, 1.0], [0.5, 3.0], [1.0, -1.0]])
    value = form(context, first, second, np.broadcast_to(matrix, (2, 1, 3, 2)))
    target = context.extensor(ga.gatype((ga.subspace.scalar(), second)), np.array([[1.0, 2.0]]))
    solution = value.lstsq(target)
    assert solution.axes == (first,)
    assert solution.shape == (2, 1)
    probe = context.extensor(second, [0.3, -0.7])
    np.testing.assert_allclose(value(solution, probe).kernel, np.broadcast_to(target(probe).kernel, (2, 1, 1)), atol=2e-5)

    square = form(context, second, second, np.broadcast_to([[2.0, 1.0], [1.0, 3.0]], (2, 1, 2, 2)))
    exact = square.solve(target)
    assert exact.axes == (second,)
    np.testing.assert_allclose(square(exact, probe).kernel, np.broadcast_to(target(probe).kernel, (2, 1, 1)), atol=2e-5)


def test_form_solve_keeps_leading_rhs_slots_as_a_map(context):
    """Solving against a bilinear right-hand side yields a map on its leading slot."""
    ga = context.algebra
    slot, extra = ga.subspace.vector(), ga.subspace("xy yz")
    value = form(context, slot, slot, [[3.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 4.0]])
    rhs_matrix = np.random.default_rng(5).normal(size=(2, 3))
    rhs = context.extensor(ga.gatype((ga.subspace.scalar(), extra, slot)), rhs_matrix[None])
    solution = value.solve(rhs)
    assert solution.axes == (slot, extra)
    e = context.extensor(extra, [1.0, -2.0])
    y = context.extensor(slot, [0.2, 0.4, -0.6])
    np.testing.assert_allclose(value(solution(e), y).kernel, rhs(e, y).kernel, atol=2e-5)


def test_pairing_solve_induces_the_plane_map_of_a_point_map(context):
    """(Plane & Point).solve(Plane & T) is the pullback of planes through T: it satisfies
    induced(l) & p == l & T(p) for every line l and point p, even for a singular T."""
    ga = context.algebra
    mv = context.multivector
    Point, Plane = ga.gatype.antivector(), ga.gatype.vector()
    pinhole = mv.antivector([0.0, 0.0, 1.0])
    screen = mv.vector([0.0, 1.0, -1.0])
    projection = (pinhole & Point) ^ screen                       # Point <- Point, rank two
    induced = (Plane & Point).solve(Plane & projection)          # Plane <- Plane
    assert induced.axes == (Plane.output_subspace, Plane.output_subspace)
    rng = np.random.default_rng(8)
    line = mv.vector(rng.normal(size=3))
    point = mv.antivector(rng.normal(size=3))
    np.testing.assert_allclose((induced(line) & point).kernel, (line & projection(point)).kernel, atol=2e-5)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_generalized_hermitian_forms_broadcast_and_mass_normalize(dtype):
    pytest.importorskip("scipy")
    ga = Algebra("x+y+")
    ctx = NumpyContext(ga, dtype=dtype)
    slot = ga.subspace.vector()
    a = np.array([[4, 1], [1, 2]], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        a[0, 1], a[1, 0] = 1j, -1j
    b = np.diag([2, 3])
    elastic = form(ctx, slot, slot, a[None, None] * np.array([1, 2])[:, None, None, None])
    metric = form(ctx, slot, slot, b[None, None] * np.array([1, 2, 3])[None, :, None, None])
    elastic = elastic.bind({0: ga.operator.identity(ga.subspace("y -x"))})
    metric = metric.bind({1: ga.operator.identity(ga.subspace("-x y"))})
    values, vectors = elastic.eigh(metric)
    assert values.shape == vectors.shape == (2, 3, 2)
    assert vectors.axes == (slot,)
    v = vectors.kernel
    norm = np.einsum("...ki,ij,...lj->...kl", v.conj(), b, v) * np.array([1, 2, 3])[None, :, None, None]
    np.testing.assert_allclose(norm, np.broadcast_to(np.eye(2), (2, 3, 2, 2)), atol=1e-12)
    av = np.einsum("ij,...kj->...ki", a, v) * np.array([1, 2])[:, None, None, None]
    bv = np.einsum("ij,...kj->...ki", b, v) * np.array([1, 2, 3])[None, :, None, None]
    np.testing.assert_allclose(av, values.kernel * bv, atol=1e-12)
    np.testing.assert_allclose(elastic.eigvalsh(metric).kernel, values.kernel, atol=1e-12)


def test_generalized_form_eig_leaves_infinite_modes_to_caller():
    pytest.importorskip("scipy")
    ga = Algebra("x+y+")
    ctx = NumpyContext(ga)
    slot = ga.subspace.vector()
    value = form(ctx, slot, slot, np.diag([2, 3])[None, None])
    metric = form(ctx, slot, slot, np.diag([1, 0]))
    values, vectors = value.eig(metric)
    assert values.shape == vectors.shape == (1, 1, 2)
    assert vectors.axes == (slot,)
    assert np.isinf(values.kernel).sum() == 1
    finite = np.isfinite(values.kernel[..., 0])
    np.testing.assert_allclose(values.kernel[..., 0][finite], [2])
    np.testing.assert_allclose(value.eigvals(metric).kernel, values.kernel)


def test_warm_form_dispatch_does_not_repeat_support_checks(monkeypatch):
    pytest.importorskip("scipy")
    ga = Algebra("x+y+")
    ctx = NumpyContext(ga)
    slot = ga.subspace.vector()
    value = form(ctx, slot, slot, [[2, 1], [1, 3]])
    metric = form(ctx, slot, slot, np.eye(2))
    calls = [value.eig, value.eigh, value.det,
             lambda: value.eig(metric), lambda: value.eigh(metric), lambda: value.det(metric)]
    for call in calls:
        call()

    def repeated(*args):
        raise AssertionError("support checks belong to cached dispatch resolution")

    monkeypatch.setattr(SubSpace, "same_support", repeated)
    for call in calls:
        call()


def test_form_signatures_are_selected_without_values():
    ga = Algebra("x+y+")
    slot = ga.subspace.vector()
    valid = ga.gatype((ga.subspace.scalar(), slot, slot))
    wrong_output = ga.gatype((slot, slot, slot))
    for name in ("eig", "eigh", "eigvals", "eigvalsh", "det"):
        method = getattr(Extensor, name)
        method.overload(1)._dispatch.resolve(valid)
        method.overload(2)._dispatch.resolve(valid, valid)
        with pytest.raises(LookupError):
            method.overload(1)._dispatch.resolve(wrong_output)
