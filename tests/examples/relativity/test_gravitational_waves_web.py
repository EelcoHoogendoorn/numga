"""Tests that the gravitational-waves WebGPU page stays within guaranteed storage-buffer limits and agrees with the
notebook shaders and CPU geometry on camera motion, relativistic sightline aberration, and retarded crystal blending."""

import json

import numpy as np
import pytest

pytest.importorskip("wgpu")

from examples.relativity.gravitational_waves import core, gpu, render, scenarios, web
from numga.wgsl import Shader

mv = core.mv
n_guaranteed_buffers = 8
single_precision = 1e-4


def test_page_contents_build_within_the_guaranteed_storage_buffers():
    contents = json.loads(json.dumps(web.contents()))
    assert all(len(shader["bindings"]) <= n_guaranteed_buffers for shader in contents["shaders"].values())


def test_camera_steps_along_its_forward_axis_through_the_round_sphere():
    speed, dt = 0.6, 0.05
    controls = [dict(dt=dt, thrust=speed, roll=0.0, yaw=0.0, pitch=0.0, braking=0.0)]
    _, _, camera = next(web.flight(iter(controls), (4, 6), render.ghost_share, scenarios.eye, 0.0, web.start_leg,
                                   core.n_rolls))
    step = dt / np.sqrt(1 - speed ** 2)
    moved = scenarios.eye * (core.moving(mv.z.dual() * speed) * (-step / 2)).exp()   # [] Motor+
    assert ((camera["motor"] - moved).scalar_norm_squared() < single_precision ** 2).all()
    assert abs(camera["clock"][0] - step) < single_precision * step


def test_page_camera_sees_as_the_notebooks_at_rest_and_aberrated_moving():
    rows, columns = shape = (6, 9)
    half_height, speed = 0.8, 0.7
    across, up = np.meshgrid(np.arange(columns) + 0.5, np.arange(rows) + 0.5)
    at_rest = np.stack([(across * 2 - columns) / rows * half_height, (1 - up * 2 / rows) * half_height,
                        np.ones(shape)], axis=-1)
    at_rest /= np.linalg.norm(at_rest, axis=-1, keepdims=True)
    cosines = at_rest[..., 2]
    sight = gpu.sight_shader()
    for traced_speed, expected in ((0.0, cosines), (speed, (cosines - speed) / (1 - speed * cosines))):
        lines = sight(shape, half_height=half_height, speed=traced_speed)["sightlines"]
        directions = lines / np.linalg.norm(lines, axis=-1, keepdims=True)
        bearings = directions[..., :2] / np.linalg.norm(directions[..., :2], axis=-1, keepdims=True)
        rest_bearings = at_rest[..., :2] / np.linalg.norm(at_rest[..., :2], axis=-1, keepdims=True)
        assert np.abs(directions[..., 2] - expected).max() < single_precision
        assert np.abs(bearings - rest_bearings).max() < single_precision


seen_source = '''
var<storage, read> probes: array<f32>;
var<storage, read_write> seen: array<Point>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    seen[id.y * n_particles + id.x] = atom_at(id.x, probes[id.y]);
}
'''

crystals_stub = '''
struct BondSeen {
    tail: Point,
    head: Point,
    ink: vec3<f32>,
}

var<uniform> n_particles: u32;
var<uniform> n_bonds: u32;
'''


def test_notebooks_and_page_see_each_crystal_when_its_light_passed():
    bonds, atoms = scenarios.crystals(scenarios.crystal_places())
    V, E, F, T = bonds.n_elements
    n_levels = gpu.n_levels
    levels = np.arange(n_levels)                                                # [n_levels]
    turns = (mv.xw * (0.3 * levels[:, None] / n_levels)).exp()                  # [n_levels, 1] Motor+
    kept = (turns >> atoms.batch()[None]).normalized()                          # [n_levels, V] Point-
    probes = gpu.first_time + gpu.level_step * np.array([0.25, 3.5, 17.75, n_levels - 3.4])   # [4]
    shares = (probes - gpu.first_time) / gpu.level_step                         # [4]
    early = np.floor(shares).astype(int)
    blended = (kept[early] * (1 - (shares - early))[:, None] + kept[early + 1] * (shares - early)[:, None]).normalized()
    shape = (len(probes), V)
    counts = dict(n_particles=V, n_bonds=E, probes=probes,
                  history=kept.reshape(-1), inks=np.zeros((n_levels * E, 4), dtype=np.float32))
    tails, heads = bonds.elements[1].T
    notebooks = Shader(crystals_stub + render.crystals_kept_source + seen_source, core.ga, render.inside_types, {})(
        shape, bonds=dict(tail=tails, head=heads), kept_start=gpu.first_time, kept_step=gpu.level_step,
        n_kept=n_levels, **counts)["seen"]
    page = Shader(crystals_stub + "var<uniform> pixels: vec2<u32>;\nvar<uniform> eye_time: f32;\n"
                  + gpu.bond_ends_source(bonds) + gpu.moving_view_source + seen_source, core.ga, render.inside_types,
                  gpu.ring)(shape, sightlines=np.zeros((1, 3), dtype=np.float32),
                            eye_time=gpu.first_time + gpu.level_step * (n_levels - 1.5), **counts)["seen"]
    for seen in (notebooks, page):
        assert ((seen - blended).scalar_norm_squared() < single_precision ** 2).all()
