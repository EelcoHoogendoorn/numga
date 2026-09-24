"""Original nonzero-term execution, with unified binding and signed layouts."""

import numpy as np
import pytest

from numga import NumpyContext
from numga.algebras import PGA3D


@pytest.fixture(params=("numpy", "jax"))
def backend(request):
    if request.param == "numpy":
        return NumpyContext, lambda f: f
    jax = pytest.importorskip("jax")
    from numga.backend.jax import JaxContext
    return JaxContext, jax.jit


def test_sparse_products_broadcast_and_convert_signed_input_layouts(backend):
    context_type, compile = backend
    spaces = PGA3D.subspace
    product = spaces.bivector() * spaces.bivector()
    outputs = []
    for execution in ("dense", "sparse"):
        mv = context_type(PGA3D, execution=execution).multivector
        left = mv(spaces("xy xz yz"), [[[1, 2, 3]], [[4, 5, 6]]])
        right = mv(spaces("xy xz yz"), [[[3, 2, 1], [6, 5, 4]]])
        result = compile(lambda a, b: product(a, b))(left, right)
        assert result.shape == (2, 2)
        assert result.context.execution == execution
        outputs.append(result)
    assert outputs[0].gatype is outputs[1].gatype
    np.testing.assert_allclose(outputs[0].kernel, outputs[1].kernel, atol=1e-6)


def test_sparse_inertia_construction_and_dense_numeric_map_application(backend):
    context_type, compile = backend
    spaces = PGA3D.subspace
    outputs = []
    for execution in ("dense", "sparse"):
        mv = context_type(PGA3D, execution=execution).multivector
        points = mv.antivector([[1, 0, 0, 1], [-1, 0, 0, 1]])
        rate = mv.bivector([1, 2, 3, 4, 5, 6])

        def evaluate(points, rate):
            inertia = points.regressive(points.commutator(spaces.bivector())).sum(axis=0)
            return inertia, inertia(rate)

        outputs.append(compile(evaluate)(points, rate))
    for dense, sparse in zip(*outputs):
        assert dense.gatype is sparse.gatype
        np.testing.assert_allclose(dense.kernel, sparse.kernel, atol=1e-6)


def test_sparse_binding_splices_open_inputs_and_preserves_empty_batches(backend):
    context_type, compile = backend
    spaces = PGA3D.subspace
    vector = spaces.vector()
    outer = vector * vector
    outputs = []
    for execution in ("dense", "sparse"):
        context = context_type(PGA3D, execution=execution)
        maps = context.extensor(PGA3D.gatype((vector, vector)), [np.eye(4), 2 * np.eye(4)])
        # Replace one slot by a unary map, leaving the other slot open.
        bound = compile(lambda m: outer.bind({1: m}))(maps)
        assert bound.arity == 2
        outputs.append(bound)
        empty = context.multivector.vector(np.empty((0, 4)))
        zero = compile(lambda x: x.wedge(x))(empty)
        assert zero.shape == (0,)
    assert outputs[0].axes == outputs[1].axes
    np.testing.assert_allclose(outputs[0].kernel, outputs[1].kernel, atol=1e-6)


def test_warm_sparse_execution_does_not_rescan_or_materialize_the_kernel(monkeypatch):
    from numga.operator.kernel import SymbolicKernel

    context = NumpyContext(PGA3D, execution="sparse")
    points = context.multivector.vector([[1, 2, 3, 0], [4, 5, 6, 0]])
    product = PGA3D.subspace.vector() * PGA3D.subspace.vector()
    expected = product(points, points)

    def fail(*args, **kwargs):
        raise AssertionError("a warm sparse call inspected the exact kernel again")

    monkeypatch.setattr(SymbolicKernel, "values", property(fail))
    monkeypatch.setattr(SymbolicKernel, "materialize", fail)
    np.testing.assert_allclose(product(points, points).kernel, expected.kernel)
