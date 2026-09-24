"""Square roots dispatch on scalar support and whole-value rotor facts."""

import numpy as np

from numga import Algebra, NumpyContext


def test_unit_even_versor_root_retains_its_certificate_in_five_dimensions():
    algebra = Algebra((5, 0, 0))
    context = NumpyContext(algebra)
    coefficients = np.zeros(len(algebra.subspace.even()))
    coefficients[0] = np.cos(0.6)
    coefficients[algebra.subspace.even().masks.index(3)] = np.sin(0.6)
    rotor = context.multivector.rotor(coefficients)

    root = rotor.square_root()

    assert root.gatype <= algebra.gatype.rotor()
    np.testing.assert_allclose((root * root - rotor).kernel, 0, atol=1e-13, rtol=1e-13)
