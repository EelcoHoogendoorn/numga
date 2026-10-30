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
    spin = (mv.xy * scenarios.camera_spin + mv.yz * 0.2).dual()
    random = np.random.default_rng(1)
    n_rays = 64
    # Events between the photon orbits and the camera, and null momenta in random directions there.
    positions = core.space(mv.vector(random.normal(size=(n_rays, 4))).dual())
    positions = positions / positions.squared().square_root() * random.uniform(3, 20, (n_rays,))
    directions = core.space(mv.vector(random.normal(size=(n_rays, 4))).dual())
    directions = directions / directions.squared().square_root()
    momenta = -core.initial_momentum(positions, directions, spin, scenarios.mass)
    orientation = (mv.zx * (-scenarios.camera_inclination / 2)).exp()
    screens = core.camera_screen(positions, momenta, orientation, spin, scenarios.mass)

    shader = Shader(gpu.geometry + gpu.rays + probe, core.ga, gpu.types, {})
    traced = shader((1, n_rays), spin=spin, mass=scenarios.mass,
                    positions=positions, momenta=momenta, screens=screens)
    expected = core.polarized_rates(positions, momenta, screens, spin, scenarios.mass)
    for name, rate in zip(("position_rates", "momentum_rates", "screen_rates"), expected):
        error = (traced[name][0] - rate).kernel
        np.testing.assert_allclose(error, 0, atol=SINGLE_PRECISION * np.abs(rate.kernel).max(), rtol=0)


def test_shader_camera_screen_uses_the_same_static_observer_as_core():
    spin, mass = .8 * mv.xy.dual(), 1.
    orientation = (mv.zx * (-1.2 / 2)).exp()
    eye = (orientation >> mv.z.dual()) * 28
    positions, momenta = core.camera_rays(eye, orientation, 5, 3, .35, spin, mass)
    camera_functions = gpu.camera.split("// The static camera's rest frame")[1].split("// The starlight")[0]
    orientation_function = gpu.camera.split("// The camera's orientation")[1].split("// The camera's eye")[0]
    camera_functions = ("// The camera's orientation" + orientation_function
                        + "// The static camera's rest frame" + camera_functions)
    camera_probe = """
var<uniform> azimuth: f32;
var<uniform> inclination: f32;
var<storage, read> positions: array<Antivector>;
var<storage, read> momenta: array<Vector>;
var<storage, read_write> screens: array<ScreenFrame>;

@compute @workgroup_size(8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&positions)) {
        return;
    }
    screens[id.x] = camera_screen(positions[id.x], momenta[id.x]);
}
"""
    shader = Shader(gpu.geometry + camera_functions + camera_probe, core.ga, gpu.types, gpu.constants)
    result = shader((1, 15), spin=spin, mass=mass, azimuth=0.0, inclination=1.2, positions=positions, momenta=momenta)
    expected = core.camera_screen(positions, momenta, orientation, spin, mass)
    np.testing.assert_allclose((result["screens"][0] - expected).kernel, 0, atol=SINGLE_PRECISION, rtol=0)
