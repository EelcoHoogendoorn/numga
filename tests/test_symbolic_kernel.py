from fractions import Fraction

import numpy as np
import pytest

from numga.operator import SymbolicKernel


def test_fraction_arithmetic_remains_exact_through_contraction():
    left = SymbolicKernel([[Fraction(1, 3), Fraction(1, 6)]])
    right = SymbolicKernel([[3], [6]])

    result = left.tensordot(right, axes=(1, 0))

    assert result.to_object_array().tolist() == [[Fraction(2)]]
    assert ((left * 3) / 2).to_object_array().tolist() == [
        [Fraction(1, 2), Fraction(1, 4)]
    ]


@pytest.mark.parametrize(
    "operation",
    [
        lambda: SymbolicKernel([0.5]),
        lambda: SymbolicKernel([1]) * 0.5,
        lambda: SymbolicKernel([1]) / np.float64(2),
    ],
)
def test_symbolic_kernel_rejects_floating_point_coefficients(operation):
    with pytest.raises(TypeError, match="exact integers or rational"):
        operation()


def test_symbolic_kernel_exposes_no_mutable_storage():
    kernel = SymbolicKernel([[1, Fraction(1, 2)]])
    original_hash = hash(kernel)

    exported = kernel.to_object_array()
    exported[0, 0] = Fraction(99)

    assert kernel.to_object_array().tolist() == [[Fraction(1), Fraction(1, 2)]]
    assert hash(kernel) == original_hash
    with pytest.raises(AttributeError):
        kernel.shape = (2,)
