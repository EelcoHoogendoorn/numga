"""The linked Spin(8) actions and their common octonion automorphisms."""

import numpy as np

from examples.math.triality import core


def test_every_rotation_has_two_skew_companions_preserving_the_product():
    left_component, right_component = core.multiplication_components()
    _, rates = core.action_modes(left_component, right_component)
    output, left, right = core.linked_actions(rates, left_component, right_component)
    norm = core.Octonion.involute().scalar_product(core.Octonion)
    defect = (output(core.multiply) - core.multiply(left, core.Octonion)
              - core.multiply(core.Octonion, right))
    tolerance = 1e-11

    # checks
    assert len(rates) == 28
    np.testing.assert_allclose(rates.kernel @ rates.kernel.T, np.eye(28), atol=tolerance)
    for action in (output, left, right):
        assert np.linalg.matrix_rank(action.kernel.reshape(28, -1), tol=tolerance) == 28
        skew_defect = norm(action, core.Octonion) + norm(core.Octonion, action)
        np.testing.assert_allclose(skew_defect.kernel, 0, atol=tolerance)
    np.testing.assert_allclose(defect.kernel, 0, atol=tolerance)


def test_multiplication_components_leave_fourteen_derivations():
    left_component, right_component = core.multiplication_components()
    _, rates = core.action_modes(left_component, right_component)
    output, _, _ = core.linked_actions(rates, left_component, right_component)
    left_offset = left_component(rates)
    right_offset = right_component(rates)
    derivations = (output - core.multiply(left_offset, core.Octonion)
                   - core.multiply(core.Octonion, right_offset))
    defect = (derivations(core.multiply) - core.multiply(derivations, core.Octonion)
              - core.multiply(core.Octonion, derivations))
    tolerance = 1e-11

    # checks
    assert np.linalg.matrix_rank(left_offset.kernel, tol=tolerance) == 7
    assert np.linalg.matrix_rank(right_offset.kernel, tol=tolerance) == 7
    assert np.linalg.matrix_rank(np.concatenate([left_offset.kernel, right_offset.kernel], axis=1),
                                 tol=tolerance) == 14
    assert np.linalg.matrix_rank(derivations.kernel.reshape(28, -1), tol=tolerance) == 14
    np.testing.assert_allclose(defect.kernel, 0, atol=tolerance)
    np.testing.assert_allclose(derivations(core.mv.scalar()).kernel, 0, atol=tolerance)


def test_common_actions_are_exactly_the_fourteen_automorphism_generators():
    left_component, right_component = core.multiplication_components()
    penalties, rates = core.action_modes(left_component, right_component)
    tolerance = 1e-11
    common = rates[penalties < tolerance]
    output, left, right = core.linked_actions(common, left_component, right_component)
    defect = (output(core.multiply) - core.multiply(output, core.Octonion)
              - core.multiply(core.Octonion, output))

    # checks
    assert len(common) == 14
    assert np.count_nonzero(penalties.to_array() > tolerance) == 14
    np.testing.assert_allclose((output - left).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((output - right).kernel, 0, atol=tolerance)
    np.testing.assert_allclose(defect.kernel, 0, atol=tolerance)
    np.testing.assert_allclose(common.restrict.vector().kernel, 0, atol=tolerance)
    np.testing.assert_allclose(common.restrict.bivector().commutator(core.fano).kernel,
                               0, atol=tolerance)


def test_finite_actions_preserve_the_full_product_and_return_after_two_turns():
    direction = core.mv.a + 2 * core.mv.b - 3 * core.mv.c + 5 * core.mv.d
    direction = direction / (-direction.squared()).square_root()
    angles = np.array([0, 0.37, 1.1, np.pi, 2 * np.pi, 4 * np.pi])
    output, left, right = core.finite_actions(direction, angles)
    norm = core.Octonion.involute().scalar_product(core.Octonion)
    product_defect = output(core.multiply) - core.multiply(left, right)
    tolerance = 1e-11

    # checks
    np.testing.assert_allclose(product_defect.kernel, 0, atol=tolerance)
    for action in (output, left, right):
        np.testing.assert_allclose((norm(action, action) - norm).kernel, 0, atol=tolerance)
        np.testing.assert_allclose((action[0] - core.Octonion).kernel, 0, atol=tolerance)
        np.testing.assert_allclose((action[-1] - core.Octonion).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((output[-2] - core.Octonion).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((left[-2] + core.Octonion).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((right[-2] + core.Octonion).kernel, 0, atol=tolerance)
