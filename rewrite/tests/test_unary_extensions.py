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
    product = np.einsum("...ik,...jk->...ij", factor.kernel, factor.kernel)
    np.testing.assert_allclose(product, operator.kernel, atol=2e-6)
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


def test_nullary_lstsq_exact_and_minimum_norm(context):
    ga = context.algebra
    # Underdetermined: 2 equations (subspace 'x y'), 3 columns
    basis = context.extensor(ga.gatype(ga.subspace("x y")), [[1, 0], [0, 1], [1, 1]])
    target = context.extensor(ga.gatype(ga.subspace("x y")), [2, 3])
    solution = basis.lstsq(target)
    assert solution.shape == (3,)
    assert solution.arity == 0
    assert solution.axes == (ga.subspace.scalar(),)
    # Expected minimum norm solution
    expected = np.linalg.pinv([[1, 0, 1], [0, 1, 1]]) @ [2, 3]
    np.testing.assert_allclose(solution.kernel.squeeze(-1), expected, atol=2e-6)
    # Reconstruct target via (solution * basis).sum(axis=-1)
    reconstructed = (solution * basis).sum(axis=-1)
    np.testing.assert_allclose(reconstructed.kernel, target.kernel, atol=2e-6)


def test_nullary_lstsq_batched_and_broadcasting(context):
    ga = context.algebra
    rng = np.random.default_rng(12)
    # Single basis of 4 vectors in 3D
    basis_single = context.multivector.vector(rng.normal(size=(4, 3)))
    # Batch of 5 targets
    targets = context.multivector.vector(rng.normal(size=(5, 3)))
    solution = basis_single.lstsq(targets)
    assert solution.shape == (5, 4)
    reconstructed = (solution * basis_single).sum(axis=-1)
    np.testing.assert_allclose(reconstructed.kernel, targets.kernel, atol=2e-6)

    # Batched basis (2, 4) against batched targets (2,)
    basis_batched = context.multivector.vector(rng.normal(size=(2, 4, 3)))
    targets_batched = context.multivector.vector(rng.normal(size=(2, 3)))
    solution_batched = basis_batched.lstsq(targets_batched)
    assert solution_batched.shape == (2, 4)
    reconstructed_batched = (solution_batched * basis_batched).sum(axis=-1)
    np.testing.assert_allclose(reconstructed_batched.kernel, targets_batched.kernel, atol=2e-6)


def test_nullary_lstsq_user_transposed_axis(context):
    ga = context.algebra
    rng = np.random.default_rng(24)
    data = rng.normal(size=(4, 5, 3))
    basis = context.multivector.vector(data)

    # To solve along axis 0 (size 4), user transposes axis 0 to the trailing position:
    basis_t = basis.map_kernel(lambda k: context.xp.swapaxes(k, 0, 1))  # shape (5, 4)
    target = context.multivector.vector(rng.normal(size=(5, 3)))
    solution = basis_t.lstsq(target)
    assert solution.shape == (5, 4)
    recon = (solution * basis_t).sum(axis=-1)
    np.testing.assert_allclose(recon.kernel, target.kernel, atol=2e-6)


def test_nullary_lstsq_subspace_projection(context):
    ga = context.algebra
    # Basis in 2D subspace 'x y'
    basis = context.extensor(ga.gatype(ga.subspace("x y")), [[1, 0], [0, 1]])
    # Target in full 3D vector space 'x y z'
    target = context.multivector.vector([3, 4, 5])
    solution = basis.lstsq(target)
    # The 'z' component cannot be matched and should be dropped (cast onto 'x y')
    expected = [3, 4]
    np.testing.assert_allclose(solution.kernel.squeeze(-1), expected, atol=2e-6)


def test_nullary_lstsq_rejects_unbatched(context):
    single = context.multivector.vector([1, 2, 3])
    with pytest.raises(ValueError, match="nullary lstsq requires at least one batch axis"):
        single.lstsq(single)


def test_grouped_lstsq_preserves_slot_order_layouts_batches_and_minimum_norm(context):
    ga = context.algebra
    output, first, retained, last = (
        ga.subspace.vector(), ga.subspace("x y"), ga.subspace.bivector(), ga.subspace("-z x"),
    )
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(2, 1, 9, 4))
    matrix[..., 3] = 2 * matrix[..., 0]
    # Equations are (output, retained); unknowns are (first, last).
    coefficients = matrix.reshape(2, 1, 3, 3, 2, 2).transpose(0, 1, 2, 4, 3, 5)
    operator = context.extensor(ga.gatype((output, first, retained, last)), coefficients)
    rhs_coefficients = rng.normal(size=(1, 4, 3, 3))
    rhs = context.extensor(ga.gatype((output, retained)), rhs_coefficients)
    rhs = rhs.cast(ga.subspace("z -x y"))(ga.operator.identity(ga.subspace("yz -xy xz")))
    solution = operator.lstsq(rhs, rcond=1e-5)
    expected = np.linalg.pinv(matrix, rcond=1e-5) @ rhs_coefficients.reshape(1, 4, 9, 1)
    assert solution.axes == (first, last)
    assert solution.shape == (2, 4)
    np.testing.assert_allclose(solution.kernel, expected.reshape(2, 4, 2, 2), atol=2e-6)


@pytest.mark.parametrize("selected", [(2,), (1, 2, 3)])
def test_grouped_lstsq_infers_nullary_and_binary_results(context, selected):
    ga = context.algebra
    axes = (ga.subspace.vector(), ga.subspace("x y"), ga.subspace("-z y"), ga.subspace("xy xz"))
    rng = np.random.default_rng(9)
    coefficients = rng.normal(size=(3, 2, 2, 2))
    operator = context.extensor(ga.gatype(axes), coefficients)
    retained = tuple(axis for axis in range(4) if axis not in selected)
    rhs_type = ga.gatype(tuple(axes[axis] for axis in retained))
    rhs = context.extensor(rhs_type, rng.normal(size=rhs_type.structural_shape))
    solution = operator.lstsq(rhs, rcond=1e-5)
    matrix = coefficients.transpose(retained + selected).reshape(-1, 2 ** len(selected))
    expected = np.linalg.lstsq(matrix, np.asarray(rhs.kernel).reshape(-1), rcond=1e-5)[0]
    assert solution.axes == tuple(axes[axis] for axis in selected)
    assert solution.arity == len(selected) - 1
    np.testing.assert_allclose(solution.kernel, expected.reshape((2,) * len(selected)), atol=2e-6)


@pytest.mark.parametrize("input_slots,rhs_slots", [
    (("x y", "x y", "xy xz"), ("x y",)),    # Repeated-type ambiguity.
    (("x y", "-y x", "xy xz"), ("x",)),    # Ambiguous lossless embedding.
    (("x y", "xy xz"), ("1 z",)),           # Same size, incompatible support.
    (("x y", "xy xz"), ("x y z",)),         # Would require lossy projection.
    (("x y", "z", "xy xz"), ("xy xz", "x y")),  # Wrong relative order.
    (("x y", "xy xz"), ("x y", "xy xz", "1")),
])
def test_grouped_lstsq_rejects_ambiguous_or_incompatible_signatures(input_slots, rhs_slots):
    ga = Algebra("x+y+z0")
    context = NumpyContext(ga)
    scalar = ga.subspace.scalar()
    operator_type = ga.gatype((scalar,) + tuple(ga.subspace(slot) for slot in input_slots))
    operator = context.extensor(operator_type, np.ones(operator_type.structural_shape))
    rhs_type = ga.gatype((scalar,) + tuple(ga.subspace(slot) for slot in rhs_slots))
    rhs = context.extensor(rhs_type, np.ones(rhs_type.structural_shape))
    with pytest.raises(TypeError):
        operator.lstsq(rhs)


def test_grouped_lstsq_infers_lossless_rhs_embedding(context):
    ga = context.algebra
    scalar, plane, bivector = ga.subspace.scalar(), ga.subspace("x y"), ga.subspace("xy xz")
    operator = context.extensor(ga.gatype((scalar, bivector, plane)), np.eye(2)[None])
    rhs = context.extensor(ga.gatype((scalar, ga.subspace("-y"))), [[3]])
    solution = operator.lstsq(rhs)
    assert solution.axes == (bivector,)
    np.testing.assert_allclose(solution.kernel, [0, -3], atol=2e-6)


@pytest.mark.parametrize("method", ["solve", "lstsq"])
def test_symbolic_linear_operator_uses_rhs_context(context, method):
    identity = context.algebra.operator.identity(context.algebra.subspace.vector())
    rhs = context.multivector.vector([[1, 2, 3], [4, 5, 6]])
    solution = getattr(identity, method)(rhs)
    assert solution.context is context
    np.testing.assert_allclose(solution.kernel, rhs.kernel, atol=2e-6)


def test_symbolic_grouped_lstsq_uses_rhs_context(context):
    vector = context.algebra.gatype.vector()
    rhs = context.multivector.scalar([[2], [4]])
    solution = (vector | vector).lstsq(rhs)
    expected = np.array([1, 2])[:, None, None] * np.diag([1, 1, 0])[None]
    assert solution.axes == (vector.output_subspace, vector.output_subspace)
    assert solution.context is context
    np.testing.assert_allclose(solution.kernel, expected, atol=2e-6)


@pytest.mark.parametrize("method", ["eig", "eigh", "eigvals", "eigvalsh"])
def test_symbolic_generalized_eigenproblem_uses_metric_context(method):
    pytest.importorskip("scipy")
    ga = Algebra("x+y+z0")
    context = NumpyContext(ga)
    vector = ga.gatype.vector()
    form = vector | vector
    metric = context.extensor(form.gatype, np.diag([2, 4, 3])[None])
    result = getattr(form, method)(metric)
    values = result[0] if method in ("eig", "eigh") else result
    assert isinstance(values.context, NumpyContext)
    np.testing.assert_allclose(np.sort_complex(values.to_array()), [0, 0.25, 0.5], atol=1e-12)


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
    normal_equations = np.einsum("oi,bo...->bi...", operator.kernel, residual.kernel)
    np.testing.assert_allclose(normal_equations, 0, atol=2e-6)


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
            extension.overload(1)._dispatch.resolve(polarity)
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


def test_real_drops_the_imaginary_part_into_a_real_context():
    ga = Algebra("x+y+")
    value = NumpyContext(ga, dtype=np.complex128).multivector.vector([1 + 2j, 3 - 1j])
    real = value.real()
    assert real.gatype == value.gatype
    assert real.context.dtype == np.float64
    np.testing.assert_allclose(real.kernel, [1.0, 3.0])
