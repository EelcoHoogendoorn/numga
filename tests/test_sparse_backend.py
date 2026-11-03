"""Original nonzero-term execution, with unified binding and signed layouts."""

import numpy as np
import pytest

from numga import Orientation
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
    Bivector = PGA3D.gatype.bivector()
    product = Bivector * Bivector
    outputs = []
    for execution in ("dense", "sparse"):
        mv = context_type(PGA3D, execution=execution).multivector
        left = mv(PGA3D.gatype(spaces("xy xz yz"), orientation=Orientation.extrinsic), [[[1, 2, 3]], [[4, 5, 6]]])
        right = mv(PGA3D.gatype(spaces("xy xz yz"), orientation=Orientation.extrinsic), [[[3, 2, 1], [6, 5, 4]]])
        result = compile(lambda a, b: product(a, b))(left, right)
        assert result.shape == (2, 2)
        assert result.context.execution == execution
        outputs.append(result)
    assert outputs[0].gatype is outputs[1].gatype
    np.testing.assert_allclose(outputs[0].kernel, outputs[1].kernel, atol=1e-6)


def test_sparse_inertia_construction_and_dense_numeric_map_application(backend):
    context_type, compile = backend
    outputs = []
    for execution in ("dense", "sparse"):
        mv = context_type(PGA3D, execution=execution).multivector
        points = mv.antivector([[1, 0, 0, 1], [-1, 0, 0, 1]])
        rate = mv.bivector([1, 2, 3, 4, 5, 6])

        def evaluate(points, rate):
            inertia = points.regressive(points.commutator(PGA3D.gatype.bivector())).sum(axis=0)
            return inertia, inertia(rate)

        outputs.append(compile(evaluate)(points, rate))
    for dense, sparse in zip(*outputs):
        assert dense.gatype is sparse.gatype
        np.testing.assert_allclose(dense.kernel, sparse.kernel, atol=1e-6)


def test_a_sparse_map_densifies_inside_a_compiled_function(backend):
    # Densifying sets each coupling at its pair of sites on the backend's own arrays, so that it runs traced too; a
    # coupling given twice adds at construction, so that every pair is set once.
    from numga.sparse import SparseExtensor
    context_type, compile = backend
    mv, Vector = context_type(PGA3D).multivector, PGA3D.gatype.vector()
    rows, columns = np.array([0, 1, 2, 2, 0]), np.array([1, 0, 2, 2, 1])
    cells = mv.scalar(np.arange(1.0, 6.0)[:, None]) * Vector                 # [couplings] Vector+ <- Vector+
    field = mv.vector(np.random.default_rng(3).normal(size=(3, 4))).field()   # Vector+[3]
    applied = compile(lambda cells, field: SparseExtensor(cells, rows, columns, (3, 3)).to_dense()(field))(cells, field)
    expected = SparseExtensor(cells, rows, columns, (3, 3))(field)
    np.testing.assert_allclose(applied.kernel, expected.kernel, atol=1e-12)


def test_sparse_binding_splices_open_inputs_and_preserves_empty_batches(backend):
    context_type, compile = backend
    vector = PGA3D.gatype.vector()
    outer = vector * vector
    outputs = []
    for execution in ("dense", "sparse"):
        context = context_type(PGA3D, execution=execution)
        maps = context.extensor(PGA3D.gatype((vector, vector)), [np.eye(4), 2 * np.eye(4)])
        # Composing one argument with a unary map gives a bilinear map.
        bound = compile(lambda m: outer.bind({1: m}))(maps)
        assert bound.arity == 2
        outputs.append(bound)
        empty = context.multivector.vector(np.empty((0, 4)))
        zero = compile(lambda x: x.wedge(x))(empty)
        assert zero.shape == (0,)
    assert outputs[0].axes == outputs[1].axes
    np.testing.assert_allclose(outputs[0].kernel, outputs[1].kernel, atol=1e-6)


