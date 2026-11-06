"""The camera's shader computes the Kerr rays' screens as `core` does."""

import numpy as np
import pytest

pytest.importorskip("wgpu")

from examples.relativity.kerr import core, gpu
from examples.util.wgsl import Shader

mv = core.mv
# Single precision against double.
SINGLE_PRECISION = 3e-5


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
