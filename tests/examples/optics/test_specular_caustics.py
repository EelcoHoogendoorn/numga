import numpy as np
import pytest

pytest.importorskip("jax")

from numga.backend.jax import derivative
from examples.optics.specular_caustics import core


ROUND_OFF = 1e-11
mv = core.mv
x, y = mv.x.dual(), mv.y.dual()


def test_refraction_obeys_snell_and_reaches_the_receiver():
    wave_angles = np.array([0, 1.1, 2.2])
    wavenumbers = np.array([3.1, 4.4, 6])
    amplitudes = np.array([0.11, 0.065, 0.035])
    phases = np.array([0, 1.2, -0.7])
    depth = 2.5
    index_ratio = 0.75
    horizontal = np.array([-0.6, -0.1, 0.2, 0.7])
    vertical = np.array([0.3, -0.5, 0.4, -0.2])
    wavevectors = ((mv.xy * (-wave_angles / 2)).exp() >> mv.x) * wavenumbers
    positions = x * horizontal + y * vertical
    incoming = -core.up

    surface, plane, outgoing = core.rays(positions, wavevectors, amplitudes, phases, index_ratio)
    landing = core.receiver(positions, wavevectors, amplitudes, phases, depth, index_ratio) - depth * core.up
    normal = plane.dual()

    # checks
    np.testing.assert_allclose(plane.squared().kernel, 1, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(outgoing.scalar_norm_squared().kernel, 1, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((outgoing - normal * (plane & outgoing)
                                - index_ratio * (incoming - normal * (plane & incoming))).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(((landing - surface) & outgoing).kernel, 0, atol=ROUND_OFF, rtol=0)
    assert np.all((mv.z & outgoing).kernel < 0)


def test_flat_surface_preserves_positions_and_area():
    wavenumbers = np.array([4])
    amplitudes = np.array([0])
    phases = np.array([0])
    depth = 2.5
    index_ratio = 0.75
    horizontal = np.array([-0.8, -0.2, 0.3, 0.9])
    vertical = np.array([0.4, -0.7, 0.2, -0.5])
    wavevectors = mv.x * wavenumbers
    positions = x * horizontal + y * vertical

    def receiver(position: core.Horizontal) -> core.Horizontal:
        return core.receiver(position, wavevectors, amplitudes, phases, depth, index_ratio)

    landing = receiver(positions)
    local = derivative(receiver)(positions)

    # checks
    np.testing.assert_allclose((landing - positions).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((local(positions) - positions).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(local.det().kernel, 1, atol=ROUND_OFF, rtol=0)


def test_local_map_matches_changes_in_ray_landing():
    wave_angles = np.array([0, 1.1, 2.2])
    wavenumbers = np.array([3.1, 4.4, 6])
    amplitudes = np.array([0.11, 0.065, 0.035])
    phases = np.array([0, 1.2, -0.7])
    depth = 2.5
    index_ratio = 0.75
    horizontal = np.array([-0.6, -0.1, 0.2, 0.7])
    vertical = np.array([0.3, -0.5, 0.4, -0.2])
    direction_angles = np.array([0.2, 0.9, 1.8])
    step = 1e-4
    wavevectors = ((mv.xy * (-wave_angles / 2)).exp() >> mv.x) * wavenumbers
    positions = x * horizontal + y * vertical
    directions = (mv.xy * (-direction_angles / 2)).exp() >> x

    def receiver(position: core.Horizontal) -> core.Horizontal:
        return core.receiver(position, wavevectors, amplitudes, phases, depth, index_ratio)

    local = derivative(receiver)(positions)
    forward = receiver(positions[:, None] + step * directions)
    backward = receiver(positions[:, None] - step * directions)
    measured = (forward - backward) / (2 * step)

    # checks
    np.testing.assert_allclose((local[:, None](directions) - measured).kernel, 0, atol=1e-6, rtol=0)


def test_three_refracted_paths_reach_one_receiver_point():
    wavenumbers = np.array([4])
    amplitudes = np.array([0.12])
    phases = np.array([0])
    depth = 2.5
    index_ratio = 0.75
    seed_offsets = np.array([-0.3, 0, 0.3])
    n_steps = 3
    wavevectors = mv.x * wavenumbers
    positions = (x * seed_offsets).cast(core.Horizontal)

    def receiver(position: core.Horizontal) -> core.Horizontal:
        return core.receiver(position, wavevectors, amplitudes, phases, depth, index_ratio)

    for _ in range(n_steps):
        local = derivative(receiver)(positions)
        positions = positions - local.solve(receiver(positions))
    area = derivative(receiver)(positions).det()

    # checks
    np.testing.assert_allclose(receiver(positions).kernel, 0, atol=1e-6, rtol=0)
    np.testing.assert_allclose((positions + positions[::-1]).kernel, 0, atol=ROUND_OFF, rtol=0)
    assert (mv.x & positions[0]).kernel.item() < -0.2
    assert (mv.x & positions[2]).kernel.item() > 0.2
    assert area.kernel[1, 0] < 0
    assert np.all(area.kernel[[0, 2], 0] > 0)
