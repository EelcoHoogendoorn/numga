"""The gravitational waves' page builds within WebGPU's storage buffers, and the example's tracers keep to one
mathematics: the camera's step through the round sphere is the motion along its own forward axis; the page's camera
sees what the notebooks' camera sees at rest, and moving, as light's aberration has it; and the notebooks' view and the
page's see each crystal as it was when the light that meets it passed."""

import json

import numpy as np
import pytest

pytest.importorskip("wgpu")

from examples.relativity.gravitational_waves import core, gpu, render, scenarios, web
from examples.util.wgsl import Shader

mv = core.mv
# The storage buffers WebGPU guarantees a shader stage; the uniform block is one of them.
GUARANTEED_STORAGE_BUFFERS = 8
# Single precision on the GPU against numga's double, relative to values of about one.
SINGLE_PRECISION = 1e-4


def test_page_contents_build_within_the_guaranteed_storage_buffers():
    contents = json.loads(json.dumps(web.contents()))
    assert all(len(shader["bindings"]) <= GUARANTEED_STORAGE_BUFFERS for shader in contents["shaders"].values())


def test_camera_steps_along_its_forward_axis_through_the_round_sphere():
    # With no wave, a frame's step at a speed is the motion that moves the origin along the camera's own forward axis
    # by the speed times the lattice's time over the frame, its clock's tick dilated.
    speed, dt = 0.6, 0.05
    controls = [dict(dt=dt, thrust=speed, roll=0.0, yaw=0.0, pitch=0.0, braking=0.0)]
    _, _, camera = next(web.flight(iter(controls), (4, 6), render.ghost_share, scenarios.eye, 0.0, web.start_leg,
                                   core.n_rolls))
    step = dt / np.sqrt(1 - speed ** 2)
    moved = scenarios.eye * (core.moving(mv.z.dual() * speed) * (-step / 2)).exp()   # [] Motor+
    assert np.abs(np.asarray((camera["motor"] - moved).kernel)).max() < SINGLE_PRECISION
    assert abs(camera["clock"][0] - step) < SINGLE_PRECISION * step


def test_page_camera_sees_as_the_notebooks_at_rest_and_aberrated_moving():
    # At rest each pixel looks along the screen's direction through it, as the notebooks' camera does. Moving along its
    # forward axis, each sightline keeps its bearing about the axis, and its cosine from the axis in the lattice's frame
    # is that at rest, less the speed, over one less the speed times it.
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
        assert np.abs(directions[..., 2] - expected).max() < SINGLE_PRECISION
        assert np.abs(bearings - rest_bearings).max() < SINGLE_PRECISION


# Where an atom of a crystal is seen at times, as a view reads it from the moments its crystals keep.
seen_source = """
var<storage, read> probes: array<f32>;
var<storage, read_write> seen: array<Point>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    seen[id.y * n_particles + id.x] = atom_at(id.x, probes[id.y]);
}
"""
# What a view of the crystals declares around them: their counts, and a bond as it is seen.
crystals_stub = """
struct BondSeen {
    tail: Point,
    head: Point,
    ink: vec3<f32>,
}

var<uniform> n_particles: u32;
var<uniform> n_bonds: u32;
"""


def test_notebooks_and_page_see_each_crystal_when_its_light_passed():
    # The crystals kept at moments a level apart, the atoms turning along their great circles; each view reads an atom
    # at any time the light passed it from the two moments about that time, blended by their nearness, the notebooks'
    # from its kept moments and the page's from its ring.
    bonds, atoms = scenarios.crystals(scenarios.crystal_places())
    levels = np.arange(gpu.n_levels)                                            # [levels]
    turns = (mv.xw * (0.3 * levels[:, None] / gpu.n_levels)).exp()              # [levels, 1] Motor+
    kept = (turns >> atoms.batch()[None]).normalized()                          # [levels, atoms] Point-
    probes = gpu.first_time + gpu.level_step * np.array([0.25, 3.5, 17.75, gpu.n_levels - 3.4])   # [probes]
    shares = (probes - gpu.first_time) / gpu.level_step                         # [probes]
    early = np.floor(shares).astype(int)
    blended = (kept[early] * (1 - (shares - early))[:, None] + kept[early + 1] * (shares - early)[:, None]).normalized()
    shape = (len(probes), bonds.n_vertices)
    counts = dict(n_particles=bonds.n_vertices, n_bonds=bonds.n_elements[1], probes=probes,
                  history=kept.reshape(-1), inks=np.zeros((gpu.n_levels * bonds.n_elements[1], 4), dtype=np.float32))
    tails, heads = bonds.elements[1].T
    notebooks = Shader(crystals_stub + render.crystals_kept_source + seen_source, core.ga, render.inside_types, {})(
        shape, bonds=dict(tail=tails, head=heads), kept_start=gpu.first_time, kept_step=gpu.level_step,
        n_kept=gpu.n_levels, **counts)["seen"]
    page = Shader(crystals_stub + "var<uniform> pixels: vec2<u32>;\nvar<uniform> eye_time: f32;\n"
                  + gpu.bond_ends_source(bonds) + gpu.moving_view_source + seen_source, core.ga, render.inside_types,
                  gpu.ring)(shape, sightlines=np.zeros((1, 3), dtype=np.float32),
                            eye_time=gpu.first_time + gpu.level_step * (gpu.n_levels - 1.5), **counts)["seen"]
    for seen in (notebooks, page):
        assert np.abs(np.asarray((seen - blended).kernel)).max() < SINGLE_PRECISION
