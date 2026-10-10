"""The derivative of a function of extensors is an extensor: the linear map from a step to the change,
typed with the step's slot after the result's own slots."""

from contextlib import contextmanager

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from numga.algebras import PGA3D, VGA3D
from numga.backend.jax import JaxContext, derivative


@contextmanager
def enable_x64():
    """Double precision for one test, restoring the global setting after."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_a_linear_map_is_its_own_derivative():
    with enable_x64():
        mv = JaxContext(VGA3D, np.float64).multivector
        rotor = (0.3 * mv.xy - 0.2 * mv.yz + 0.5 * mv.zx).exp()
        turned = derivative(lambda v: rotor >> v)(mv.vector([1.0, 2.0, 0.5]))
        np.testing.assert_allclose((turned - (rotor >> VGA3D.gatype.vector())).kernel, 0.0, atol=1e-12)


def test_the_gradient_is_a_form_that_pairs_a_step_to_the_change():
    with enable_x64():
        mv = JaxContext(PGA3D, np.float64).multivector
        twist = mv.bivector([0.2, -0.4, 0.0, 0.1, 0.0, 0.3])
        step = mv.bivector([0.5, -1.0, 0.2, 0.7, 0.1, -0.3])

        def misfit(generator):
            moved = generator.exp() >> mv.zyx
            return (moved - mv.zyx).dual().norm_squared()

        gradient = derivative(misfit)(twist)
        assert gradient.gatype == PGA3D.gatype((PGA3D.gatype.scalar(), PGA3D.gatype.bivector()))
        _, change = jax.jvp(lambda kernel: misfit(twist.context.extensor(twist.gatype, kernel)).kernel,
                            (twist.kernel,), (step.kernel,))
        np.testing.assert_allclose(gradient(step).kernel, change, atol=1e-12)


def test_the_second_derivative_is_a_symmetric_bilinear_form():
    with enable_x64():
        mv = JaxContext(VGA3D, np.float64).multivector
        generator = 0.3 * mv.xy - 0.2 * mv.yz + 0.5 * mv.zx
        first, second = mv.bivector([1.0, 0.0, 2.0]), mv.bivector([0.0, -1.0, 0.5])

        def spread(b):
            return (b.exp() >> mv.x).scalar_product(mv.z) ** 2

        curvature = derivative(derivative(spread))(generator)
        assert curvature.gatype.arity == 2
        np.testing.assert_allclose(curvature(first, second).kernel, curvature(second, first).kernel, atol=1e-12)
        # Against the change of the gradient along one step, paired with the other:
        gradient = lambda kernel: derivative(spread)(generator.context.extensor(generator.gatype, kernel))(second).kernel
        _, change = jax.jvp(gradient, (generator.kernel,), (first.kernel,))
        np.testing.assert_allclose(curvature(second, first).kernel, change, atol=1e-12)


def test_the_derivative_of_the_exponential():
    """At zero it is the identity on bivectors; elsewhere, carried back by the rotor, it stays a bivector."""
    with enable_x64():
        mv = JaxContext(VGA3D, np.float64).multivector
        steps = mv.bivector(np.eye(3))                                        # [3] Bivector
        at_zero = derivative(lambda b: b.exp())(mv.bivector([0.0, 0.0, 0.0]))
        np.testing.assert_allclose((at_zero(steps) - steps).kernel, 0.0, atol=1e-12)
        generator = 0.3 * mv.xy - 0.2 * mv.yz + 0.5 * mv.zx
        carried = generator.exp().reverse() * derivative(lambda b: b.exp())(generator)(steps)
        np.testing.assert_allclose(carried.select_grade(0).kernel, 0.0, atol=1e-12)


def test_leading_batch_axes_are_copies_and_the_rest_couple():
    """An axis the result shares with the value, leading, is differentiated element by element; an axis
    of the value the result lacks couples every element of the result to every element along it."""
    with enable_x64():
        mv = JaxContext(VGA3D, np.float64).multivector
        twists = mv.bivector(np.random.default_rng(0).normal(size=(2, 3, 3)))   # [2, 3] Bivector
        copies = derivative(lambda b: b * b)(twists)                         # [2, 3] Even <- Bivector
        assert copies.shape == (2, 3)
        # Summed over its trailing axis, each case has a form for every element along it.
        totals = derivative(lambda b: (b * b).select_grade(0).sum(axis=-1))(twists)   # [2, 3] Scalar <- Bivector
        reference = jax.jacobian(lambda kernel: (twists.context.extensor(twists.gatype, kernel) * twists.context.extensor(twists.gatype, kernel)).select_grade(0).sum(axis=-1).kernel)(twists.kernel)
        np.testing.assert_allclose(totals.kernel[..., 0, :], jnp.stack([reference[case, 0, case] for case in range(2)]), atol=1e-12)


def test_the_derivative_of_a_record_is_the_record_of_derivatives():
    with enable_x64():
        mv = JaxContext(VGA3D, np.float64).multivector
        record = (mv.bivector([0.3, -0.2, 0.5]), mv.vector([1.0, 2.0, 0.5]))

        def turned(pair):
            generator, vector = pair
            return generator.exp() >> vector

        by_generator, by_vector = derivative(turned)(record)
        assert by_generator.gatype.input_subspaces[-1].same_support(VGA3D.subspace.bivector())
        np.testing.assert_allclose((by_vector - (record[0].exp() >> VGA3D.gatype.vector())).kernel, 0.0, atol=1e-12)
        np.testing.assert_allclose(by_generator.kernel, derivative(lambda generator: generator.exp() >> record[1])(record[0]).kernel, atol=1e-12)
