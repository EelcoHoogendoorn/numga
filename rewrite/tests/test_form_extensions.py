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


def test_form_transpose_and_svd_preserve_all_axes(context):
    ga = context.algebra
    first, second = ga.subspace.vector(), ga.subspace("xy yz")
    matrix = np.random.default_rng(3).normal(size=(2, 1, 3, 2))
    value = form(context, first, second, matrix)
    transposed = value.transpose()
    assert transposed.shape == (2, 1)
    assert transposed.axes == (value.axes[0], second, first)
    assert transposed.kernel.shape == (2, 1, 1, 2, 3)
    np.testing.assert_array_equal(transposed.transpose().kernel, value.kernel)
    a = context.extensor(first, [1, 2, 3])
    b = context.extensor(second, [2, -1])
    np.testing.assert_allclose(value(a, b).kernel, transposed(b, a).kernel, atol=2e-6)
    left, singular, right = value.svd()
    assert left.axes == (first,)
    assert right.axes == (second,)
    assert left.shape == right.shape == singular.shape == (2, 1, 2)
    rebuilt = np.einsum("...ki,...k,...kj->...ij", left.kernel,
                        singular.kernel[..., 0], np.conj(right.kernel))
    np.testing.assert_allclose(rebuilt, matrix, atol=2e-6)
    np.testing.assert_allclose(value.svdvals().kernel, singular.kernel, atol=2e-6)


@pytest.mark.parametrize("method", ["eig", "eigh"])
def test_form_eigenpairs_align_slots_and_keep_singleton_batches(context, method):
    ga = context.algebra
    vector = ga.subspace.vector()
    matrix = np.array([[3, 1, 0], [1, 2, 0], [0, 0, 4]])
    value = form(context, vector, vector, np.broadcast_to(matrix, (2, 1, 3, 3)))
    first, second = ga.subspace("z -x y"), ga.subspace("-y z x")
    changed = value.bind({0: ga.operator.identity(first), 1: ga.operator.identity(second)})
    values, vectors = getattr(changed, method)()
    assert vectors.axes == (second,)
    assert values.shape == vectors.shape == (2, 1, 3)
    coefficients = vectors.cast(vector).kernel
    np.testing.assert_allclose(np.einsum("ij,...kj->...ki", matrix, coefficients),
                               coefficients * values.kernel, atol=2e-6)
    np.testing.assert_allclose(changed.det().kernel, np.full((2, 1, 1), 20), atol=2e-6)
    np.testing.assert_allclose(changed.trace().kernel, np.full((2, 1, 1), 9), atol=2e-6)
    np.testing.assert_allclose(np.sort(np.real(changed.eigvals().kernel[..., 0]), axis=-1),
                               changed.eigvalsh().kernel[..., 0], atol=2e-6)


def test_signed_scalar_output_is_not_treated_as_positive(context):
    ga = context.algebra
    slot = ga.subspace("x")
    value = context.extensor(ga.gatype((ga.subspace("-1"), slot, slot)), [[[[-3.0]]]])
    values, vectors = value.eigh()
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
    rebuilt = factor(factor.transpose())
    rebuilt = rebuilt(ga.operator.identity(slot)).cast(slot)
    np.testing.assert_allclose(rebuilt.kernel, np.broadcast_to(matrix, (2, 1, 3, 3)), atol=2e-6)
    np.testing.assert_allclose(np.triu(factor.kernel, 1), 0)


def test_exact_form_transpose_preserves_rationals():
    ga = Algebra("x+y+")
    value = form(ga.exact, ga.subspace.vector(), ga.subspace.scalar(), [[2], [3]])
    transposed = value.transpose()
    assert transposed.context is ga.exact
    assert transposed.kernel.to_object_array().tolist() == [[[2, 3]]]


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
    calls = [value.eig, value.eigh, value.transpose,
             lambda: value.eig(metric), lambda: value.eigh(metric)]
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
    for name in ("eig", "eigh", "eigvals", "eigvalsh"):
        method = getattr(Extensor, name)
        method.overload(1)._dispatch.resolve(valid)
        method.overload(2)._dispatch.resolve(valid, valid)
        with pytest.raises(LookupError):
            method.overload(1)._dispatch.resolve(wrong_output)
