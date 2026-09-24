import numpy as np
import pytest

torch = pytest.importorskip("torch")

from numga import Algebra, NumpyContext
from numga.algebras import PGA3D, STA
from numga.backend.torch import TorchContext
from tests.backend_surface import operations


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
    torch.testing.assert_close(mapped(bivectors), direct.kernel, rtol=1e-6, atol=1e-6)


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
