"""Vortices leapfrogging, a shear layer rolling up, and a gas of vortices."""

import numpy as np

from examples.mechanics.vortices import core, scenarios


def run(vortices: core.Vortices, dt: float, steps: int) -> core.Vortices:
    for _ in range(steps):
        vortices = core.step(vortices, dt)
    return vortices


def test_the_derivative_of_the_flow_and_the_leapfrog():
    scene = scenarios.leapfrog()
    vortices = scene.vortices
    points = core.grid(2.0, 1.5, 9, 7)
    gradients = core.gradient(vortices, points)

    # The gradient is the change in velocity over a small displacement.
    step = core.mv.vector([0.3, -0.2]) * 1e-5
    change = (core.velocity(vortices, points + step) - core.velocity(vortices, points - step)) / 2
    np.testing.assert_allclose((gradients(step) - change).kernel, 0.0, atol=1e-12)
    # The derivative has no divergence, and its bivector is the blobs' vorticity.
    separations = points[..., None, None] - (vortices.centres[..., None] + vortices.copies)
    blob = (separations | separations) + vortices.core_radius**2
    vorticity = (vortices.circulations[:, None] * vortices.core_radius**2 / np.pi / (blob * blob)).sum(axis=-1).sum(axis=-1)
    np.testing.assert_allclose((core.derivative(gradients) - vorticity * core.PLANE).kernel, 0.0, atol=1e-12)
    # Without divergence, swirl is the gradient's determinant.
    np.testing.assert_allclose((core.swirl(gradients) - gradients.det()).kernel, 0.0, atol=1e-12)

    # The two pairs take turns in the lead.
    leads = []
    for _ in range(scene.frames):
        vortices = run(vortices, scene.dt, scene.substeps)
        leads.append(((vortices.centres[2] - vortices.centres[0]) | core.mv.x).kernel[0])
    assert np.count_nonzero(np.diff(np.sign(leads))) >= 6


def test_the_shear_layer_rolls_up():
    scene = scenarios.shear_layer()

    def across(vortices: core.Vortices) -> np.ndarray:
        return (vortices.centres | core.mv.y).kernel[..., 0]

    rolled = run(scene.vortices, scene.dt, scene.frames * scene.substeps)
    assert np.ptp(across(rolled)) > 10 * np.ptp(across(scene.vortices))


def test_the_gas_keeps_its_impulses():
    scene = scenarios.gas()
    weights = scene.vortices.circulations

    def impulses(vortices: core.Vortices) -> tuple[core.Vector, core.Scalar]:
        linear = (weights * vortices.centres).sum(axis=-1)                   # [] Vector
        angular = (weights * (vortices.centres | vortices.centres)).sum(axis=-1)   # [] Scalar
        return linear, angular

    (linear, angular), (linear_after, angular_after) = impulses(scene.vortices), impulses(run(scene.vortices, scene.dt, 300))
    # Runge–Kutta keeps the linear impulse to round-off; the angular one to the method's accuracy.
    np.testing.assert_allclose((linear_after - linear).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((angular_after - angular).kernel, 0.0, atol=1e-5)
