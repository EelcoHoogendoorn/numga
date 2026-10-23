"""The camera's shader computes the Kerr rays and their screens as `core` does."""

import numpy as np
import pytest

pytest.importorskip("wgpu")

from examples.relativity.kerr import core, gpu, scenarios
from examples.util.wgsl import Shader

mv = core.mv
# Single precision against double, relative to each rate's size.
SINGLE_PRECISION = 3e-5

probe = """
var<storage, read> positions: array<Antivector>;
var<storage, read> momenta: array<Vector>;
var<storage, read> screens: array<ScreenFrame>;
var<storage, read_write> position_rates: array<Antivector>;
var<storage, read_write> momentum_rates: array<Vector>;
var<storage, read_write> screen_rates: array<ScreenFrame>;

@compute @workgroup_size(8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&positions)) {
        return;
    }
    let rate = rates(Ray(positions[id.x], momenta[id.x], screens[id.x]));
    position_rates[id.x] = rate.position;
    momentum_rates[id.x] = rate.momentum;
    screen_rates[id.x] = rate.screen;
}
"""


def test_shader_rates_match_core():
    spin = mv.xy * scenarios.camera_spin + mv.yz * 0.2
    random = np.random.default_rng(1)
    n_rays = 64
    # Events between the photon orbits and the camera, and null momenta in random directions there.
    positions = (mv.vector(random.normal(size=(n_rays, 4))).dual() * random.uniform(3, 20, (n_rays,)))
    directions = core.space(mv.vector(random.normal(size=(n_rays, 4))).dual())
    directions = directions / directions.squared().square_root()
    momenta = -core.initial_momentum(positions, directions, spin, scenarios.mass)
    orientation = (mv.zx * (-scenarios.camera_inclination / 2)).exp()
    screens = core.camera_screen(positions, momenta, orientation, spin, scenarios.mass)

    shader = Shader(gpu.geometry + gpu.rays + probe, core.ga, gpu.types)
    traced = shader((1, n_rays), spin=spin, mass=scenarios.mass,
                    positions=positions, momenta=momenta, screens=screens)
    expected = core.polarized_rates(positions, momenta, screens, spin, scenarios.mass)
    for name, rate in zip(("position_rates", "momentum_rates", "screen_rates"), expected):
        error = (traced[name][0] - rate).kernel
        np.testing.assert_allclose(error, 0, atol=SINGLE_PRECISION * np.abs(rate.kernel).max(), rtol=0)
