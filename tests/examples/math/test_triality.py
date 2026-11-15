"""The linked Spin(8) actions and their common octonion automorphisms."""

import numpy as np

from examples.math.triality import core, scenarios


def _map_norm_squared(operator):
    """Squared Frobenius norm (`[...] Scalar+`) of a multilinear octonion map via metric contractions."""
    form = operator.symmetric_conjugate_product()
    while form.arity > 0:
        form = form.contract(1, 1 + form.arity // 2)
    return form


def test_every_rotation_has_two_skew_companions_preserving_the_product():
    penalties, generators, output, left, right = scenarios.triality()
    metric = core.Generator.clifford_conjugate().scalar_product(core.Generator)
    generator_gram = metric(generators[:, None], generators[None, :])
    norm = core.Octonion.involute().scalar_product(core.Octonion)
    defect = (output(core.multiply) - core.multiply(left, core.Octonion)
              - core.multiply(core.Octonion, right))
    tolerance = 1e-8

    # checks
    assert len(generators) == 28 and len(penalties) == 28
    assert ((generator_gram - np.eye(28)).abs() < tolerance).all()
    for action in (output, left, right):
        action_gram = -action[:, None](action[None, :]).trace()
        assert ((action_gram - 2 * np.eye(28)).abs() < tolerance).all()
        skew_defect = norm(action, core.Octonion) + norm(core.Octonion, action)
        assert (_map_norm_squared(skew_defect).abs() < tolerance).all()
    assert (_map_norm_squared(defect).abs() < tolerance).all()


def test_multiplication_components_leave_fourteen_derivations():
    left_component, right_component = core.multiplication_components()
    both_components = core.stack([left_component, right_component]).field()
    _, generators = core.action_modes(left_component, right_component)
    output, _, _ = core.linked_actions(generators, left_component, right_component)
    left_offset = left_component(generators)
    right_offset = right_component(generators)
    derivations = (output - core.multiply(left_offset, core.Octonion)
                   - core.multiply(core.Octonion, right_offset))
    derivation_gram = -derivations[:14, None](derivations[None, :14]).trace()
    tail_projections = -derivations[14:, None](derivations[None, :14]).trace() / 2
    tail_norms = -derivations[14:](derivations[14:]).trace() / 2
    defect = (derivations(core.multiply) - core.multiply(derivations, core.Octonion)
              - core.multiply(core.Octonion, derivations))
    tolerance = 1e-8

    # checks
    assert (left_component.svdvals() > tolerance).sum() == 7
    assert (right_component.svdvals() > tolerance).sum() == 7
    assert (both_components.svdvals() > tolerance).sum() == 14
    assert ((derivation_gram - 2 * np.eye(14)).abs() < tolerance).all()
    assert ((tail_projections.squared().sum(axis=1) - tail_norms).abs() < tolerance).all()
    assert (_map_norm_squared(defect).abs() < tolerance).all()
    assert (derivations(core.mv.scalar()).symmetric_conjugate_product() < tolerance).all()


def test_common_actions_are_exactly_the_fourteen_automorphism_generators():
    left_component, right_component = core.multiplication_components()
    penalties, generators = core.action_modes(left_component, right_component)
    tolerance = 1e-8
    common_generators = generators[penalties < tolerance]
    output, left, right = core.linked_actions(common_generators, left_component, right_component)
    defect = (output(core.multiply) - core.multiply(output, core.Octonion)
              - core.multiply(core.Octonion, output))

    # checks
    assert len(common_generators) == 14
    assert (penalties > tolerance).sum() == 14
    assert (_map_norm_squared(output - left).abs() < tolerance).all()
    assert (_map_norm_squared(output - right).abs() < tolerance).all()
    assert (_map_norm_squared(defect).abs() < tolerance).all()
    assert (common_generators.restrict.vector().symmetric_conjugate_product() < tolerance).all()
    assert (common_generators.restrict.bivector().commutator(core.fano).scalar_norm_squared().abs() < tolerance).all()


def test_finite_actions_preserve_the_full_product_and_return_after_two_turns():
    direction = core.mv.a + 2 * core.mv.b - 3 * core.mv.c + 5 * core.mv.d
    direction = direction / (-direction.squared()).square_root()
    angles = np.array([0, 0.37, 1.1, np.pi, 2 * np.pi, 4 * np.pi])
    output, left, right = core.finite_actions(direction, angles)
    norm = core.Octonion.involute().scalar_product(core.Octonion)
    product_defect = output(core.multiply) - core.multiply(left, right)
    tolerance = 1e-8

    # checks
    assert (_map_norm_squared(product_defect).abs() < tolerance).all()
    for action in (output, left, right):
        assert (_map_norm_squared(norm(action, action) - norm).abs() < tolerance).all()
        assert (_map_norm_squared(action[0] - core.Octonion).abs() < tolerance).all()
        assert (_map_norm_squared(action[-1] - core.Octonion).abs() < tolerance).all()
    assert (_map_norm_squared(output[-2] - core.Octonion).abs() < tolerance).all()
    assert (_map_norm_squared(left[-2] + core.Octonion).abs() < tolerance).all()
    assert (_map_norm_squared(right[-2] + core.Octonion).abs() < tolerance).all()
