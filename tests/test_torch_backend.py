import numpy as np
import pytest

torch = pytest.importorskip("torch")

from numga import Algebra, Extensor, NumpyContext
from numga.algebras import PGA3D, STA
from numga.backend.context import context_from_key
from numga.backend.torch import TorchContext


def operations(ga: Algebra, rng: np.random.Generator) -> dict:
    """The library surface, each as a function of a context."""
    bivectors = rng.normal(size=(5, len(ga.subspace.bivector()))) * 0.4
    evens = rng.normal(size=(5, len(ga.subspace.even())))
    vectors = rng.normal(size=(5, len(ga.subspace.vector())))
    Vector = ga.gatype.vector()
    Map = ga.gatype((Vector, Vector))
    matrices = rng.normal(size=(5, len(Vector.output_subspace), len(Vector.output_subspace)))
    symmetric = matrices @ np.swapaxes(matrices, -1, -2) + np.eye(matrices.shape[-1])
    return {
        "product": lambda c: c.multivector.even(evens) * c.multivector.vector(vectors),
        "wedge": lambda c: c.multivector.vector(vectors) ^ c.multivector.even(evens),
        "regressive": lambda c: c.multivector.vector(vectors) & c.multivector.even(evens),
        "inner": lambda c: c.multivector.vector(vectors) | c.multivector.even(evens),
        "sum": lambda c: c.multivector.vector(vectors) + c.multivector.even(evens) * 2.0,
        "reverse dual": lambda c: (~c.multivector.even(evens)).dual(),
        "exp": lambda c: c.multivector.bivector(bivectors).exp(),
        "log": lambda c: c.multivector.bivector(bivectors).exp().log(),
        "normalized": lambda c: c.multivector.even(evens).normalized(),
        "inverse": lambda c: c.multivector.even(evens).inverse(),
        "square root": lambda c: c.multivector.bivector(bivectors).exp().square_root(),
        "decompose": lambda c: c.multivector.bivector(bivectors).decompose_invariant()[0],
        "sandwich": lambda c: c.multivector.bivector(bivectors).exp() >> c.multivector.vector(vectors),
        "sandwich map": lambda c: (c.multivector.bivector(bivectors).exp() >> Vector)(c.multivector.vector(vectors)),
        "stack reduce": lambda c: Extensor.stack([c.multivector.vector(vectors)] * 2).sum(axis=0),
        "reduce unbatched": lambda c: c.multivector.vector(vectors)[0].sum(),
        "index reshape": lambda c: c.multivector.vector(vectors)[1:3].reshape(2, 1),
        "det": lambda c: c.extensor(Map, matrices).det(),
        "trace": lambda c: c.extensor(Map, matrices).trace(),
        "map inverse": lambda c: c.extensor(Map, matrices).inverse(),
        "solve": lambda c: c.extensor(Map, matrices).solve(c.multivector.vector(vectors)),
        "pinv": lambda c: c.extensor(Map, matrices).pinv(),
        "eigh": lambda c: c.extensor(Map, symmetric).eigh()[0],
        "svdvals": lambda c: c.extensor(Map, matrices).svdvals(),
        "compose": lambda c: c.extensor(Map, matrices) * c.extensor(Map, matrices),
        "outermorphism": lambda c: c.extensor(Map, matrices).outermorphism(ga.gatype.bivector()),
    }


@pytest.mark.parametrize("execution", ["dense", "sparse"])
@pytest.mark.parametrize("ga", [PGA3D, STA, Algebra("x+y+z+w+e-")], ids=str)
def test_torch_agrees_with_numpy_across_the_library_surface(ga, execution):
    reference = NumpyContext(ga, execution=execution)
    context = TorchContext(ga, torch.float64, execution=execution)
    for name, operation in operations(ga, np.random.default_rng(0)).items():
        result = operation(context).kernel
        assert isinstance(result, torch.Tensor), name
        np.testing.assert_allclose(
            result.numpy(), operation(reference).kernel, atol=1e-10, err_msg=name,
        )


def test_gradients_flow_through_extensor_expressions():
    context = TorchContext(PGA3D, torch.float64)
    bivector = torch.randn(7, 6, dtype=torch.float64, requires_grad=True)
    points = context.multivector.vector(torch.randn(7, 4, dtype=torch.float64))

    def loss(coefficients):
        motor = context.multivector.bivector(coefficients).exp()
        return ((motor >> points) & context.multivector.antivector([1.0, 2.0, 3.0, 4.0])).kernel.sum()

    assert torch.autograd.gradcheck(loss, (bivector,))


def test_vmap_maps_extensor_expressions_over_a_batch():
    context = TorchContext(PGA3D)
    points = context.multivector.vector([1.0, 2.0, 3.0, 1.0])
    mapped = torch.vmap(lambda b: (context.multivector.bivector(b).exp() >> points).kernel)
    bivectors = torch.randn(5, 6)
    direct = context.multivector.bivector(bivectors).exp() >> points
    torch.testing.assert_close(mapped(bivectors), direct.kernel)


def test_context_key_round_trips_dtype_and_device():
    context = TorchContext(PGA3D, np.float64, "cpu", execution="sparse")
    assert context.dtype == np.dtype(np.float64)
    assert context.torch_dtype is torch.float64
    assert context_from_key(PGA3D, context.key).is_compatible_with(context)
    value = context.multivector.vector([1, 2, 3, 4])
    assert value.kernel.dtype is torch.float64 and value.kernel.device == context.device


def test_numpy_conventions_hold_on_torch_storage():
    context = TorchContext(Algebra("x+y+"), torch.float64)
    Vector = context.algebra.gatype.vector()
    singular = context.extensor(context.algebra.gatype((Vector, Vector)), [[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(np.linalg.LinAlgError):
        singular.inverse()
    scalars = context.multivector.scalar(torch.tensor([[0.1], [2.0], [5.0]], dtype=torch.float64))
    torch.testing.assert_close(
        scalars.clip(0, np.array([0.5, 1.0, 3.0])).to_array(),
        torch.tensor([0.1, 1.0, 3.0], dtype=torch.float64),
    )
    with pytest.raises(TypeError, match="floating"):
        TorchContext(PGA3D, torch.int32)
