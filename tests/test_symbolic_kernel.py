import pytest

from numga.operator import SymbolicKernel


def test_symbolic_kernel_exposes_no_mutable_storage():
    kernel = SymbolicKernel([[1, -1]])
    original_hash = hash(kernel)

    exported = kernel.to_array()
    exported[0, 0] = 99

    assert kernel.values.tolist() == [[1, -1]]
    assert hash(kernel) == original_hash
    with pytest.raises(ValueError):
        kernel.values[0, 0] = 99
    with pytest.raises(AttributeError):
        kernel.shape = (2,)
