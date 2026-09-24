"""The opt-in closed-form exp and log by invariant decomposition, below six dimensions."""

import numpy as np
import pytest

from numga import Algebra, NumpyContext
from numga.extensions import invariant_decomposition
from numga.extensions.logexp import bivector_exp


def random_bivectors(algebra, scale, seed=0):
    mv = NumpyContext(algebra).multivector
    shape = (200, len(algebra.subspace.bivector()))
    return mv.bivector(np.random.default_rng(seed).normal(size=shape) * scale)


def assert_exact_rotors(b, exp, log):
    rotor = exp(b)
    size = max(1.0, np.abs(rotor.kernel).max())
    np.testing.assert_allclose((rotor * rotor.reverse() - 1).kernel / size**2, 0, atol=1e-12)
    reference = bivector_exp(b, n=40)
    np.testing.assert_allclose((rotor - reference).kernel / size, 0, atol=1e-5)
    np.testing.assert_allclose((exp(log(rotor)) - rotor).kernel / size, 0, atol=1e-8)


# --- invariant decomposition, below six dimensions --------------------------------------------

@pytest.mark.parametrize("signature", ["t+x-y-z-", "x+y+z+w+", "x+y+z+w+e-", "x+y+z+p+n-", "x+y+z+w+u+", "x+y+z+w0u+"])
@pytest.mark.parametrize("scale", [1e-6, 0.3, 1.0])
def test_decomposition_is_exact_in_four_and_five_dimensions(signature, scale):
    assert_exact_rotors(random_bivectors(Algebra(signature), scale), invariant_decomposition.exp_decomposed, invariant_decomposition.log_decomposed)


@pytest.mark.parametrize("scale", [1e-6, 0.3, 1.0])
def test_null_wedge_form_is_exact_in_pga(scale):
    assert_exact_rotors(
        random_bivectors(Algebra("x+y+z+w0"), scale),
        invariant_decomposition.exp_null_wedge, invariant_decomposition.log_null_wedge,
    )


def test_null_wedge_form_needs_no_branch_for_pure_translations():
    mv = NumpyContext(Algebra("x+y+z+w0")).multivector
    translation = mv.bivector([0, 0, 0, 0.3, -0.2, 0.5])
    rotor = invariant_decomposition.exp_null_wedge(translation)
    np.testing.assert_allclose((rotor - (1 + translation)).kernel, 0, atol=1e-15)
    np.testing.assert_allclose((invariant_decomposition.log_null_wedge(rotor) - translation).kernel, 0, atol=1e-15)


def test_isoclinic_rotations_use_the_equal_parts_form():
    """Equal angles in two orthogonal planes: the parts square alike and cannot be separated."""
    mv = NumpyContext(Algebra("x+y+z+w+")).multivector
    angle = 0.4
    rotor = invariant_decomposition.exp_decomposed((mv.xy + mv.zw) * angle)
    exact = (np.cos(angle) + mv.xy * np.sin(angle)) * (np.cos(angle) + mv.zw * np.sin(angle))
    np.testing.assert_allclose((rotor - exact).kernel, 0, atol=1e-15)
