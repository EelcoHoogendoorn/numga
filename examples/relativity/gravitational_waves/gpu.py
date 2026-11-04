"""The page's own physics on the GPU, in numga's dialect, beside the walk and the view from inside that it shares with
the notebooks: the camera's and the crystals' step through the exact slowest wave, one invocation a frame; the camera's
sightlines, boosted into the lattice's frame, one invocation a pixel; and the crystals as the light that reaches the
eye saw them, from the moments the step keeps.

The boost keeps a null line null, so that in the lattice's frame the light's time runs back one for one with its
length, as it does for the notebooks' camera at rest: only the directions change, the sky ahead drawn in and what lies
beside swung forward as the camera speeds up.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numga import Algebra, NumpyContext, stack
from numga.algebras import STA

from examples.relativity.gravitational_waves import core, render, scenarios
from examples.util.topology import Topology
from examples.util.wgsl import Shader, sampled

# The spacetime algebra at the eye, in which the camera's sightlines are boosted.
spacetime = NumpyContext(STA).multivector
# The rates that turn the view, the planes through the eye, and the turns they make, rotors about it.
Turn = core.Rate.derive.within(Algebra("x+y+z+"))                                # Bivector+
Aim = core.Motor.derive.within(Algebra("x+y+z+"))                                # Rotor+

# The wave's period, and the seconds of the page's clock it lasts, at the pace of the animations; the camera's fastest
# speed, as a share of the light's; and the longest a frame's step runs in the page's seconds, however long the frame
# took.
wave_period = 2 * np.pi / core.quadrupole_frequency
seconds_per_period = scenarios.n_wave_frames * scenarios.duration_ms / 1000
fastest = 0.95
longest_frame_seconds = 1 / 15
# The ring of the crystals' kept moments: a level each of the wave's step as the notebooks store theirs, counted from a
# time before the earliest any ray reaches at the start, as many as cover the longest a ray runs back, half a turn of
# the sphere, with a level either side.
level_step = wave_period / scenarios.n_steps_per_swing
first_time = -(np.pi + level_step)
n_levels = int(np.ceil(np.pi / level_step)) + 3
ring = dict(level_step=level_step, first_time=first_time, n_levels=n_levels)
# The most levels a frame passes, at the longest frame and the fastest speed.
max_levels_a_frame = int(np.ceil(longest_frame_seconds * wave_period / seconds_per_period
                                 / np.sqrt(1 - fastest ** 2) / level_step)) + 1
# The colour map's entries the shader blends the bonds' colours from.
n_strain_inks = 33


# --- math -----------------------------------------------------------------------------
# The camera and the crystals, stepped once a frame by one invocation, in numga's dialect, through the waves the walk
# reads, `local_waves`, and along its connection. The camera flies as an airplane does: its motor carries the origin
# to the eye and the origin's steps to its right, up and forward, and it moves along its own forward axis at its speed
# in the lattice's frame, so that turning the view turns its motion with it. A frame is a tick of the camera's own
# clock: the lattice's time, the wave's and the crystals', runs on by the boost's time component over it, `dt` over the
# root of one less the squared speed. Each frame the eye rolls along the round step of its velocity over that time,
# and the view turns with the frames' roll along that step, as a ray's heading does, and then by the keys. In the
# round sphere the step is exactly the one-parameter motion `motor * (moving(velocity) * (-step / 2)).exp()`, since
# `frame(eye) * exp(-(moving + turning)(step) / 2)` is the frame at the rolled eye. The crystals' atoms step as
# `core.massive` steps them, all crystals in the lattice's time, and the moments the light carries back are kept: at
# each level of the wave's step, and at the frame's end, the atoms' places and the bonds' inks are written into a ring
# of levels that reaches as far back as light takes to cross the universe.
advance_source = """
var<uniform> dt: f32;
var<uniform> thrust: f32;
var<uniform> roll: f32;
var<uniform> yaw: f32;
var<uniform> pitch: f32;
var<uniform> braking: f32;
var<uniform> fastest: f32;
var<uniform> lamp: Point;
var<uniform> n_crystals: u32;
var<uniform> longest_substep: f32;
var<uniform> saturating_strain: f32;
var<storage, read_write> camera: array<Camera>;
var<storage, read_write> atoms: array<Atom>;
var<storage, read_write> springs: array<Spring>;
var<storage, read_write> history: array<Point>;
var<storage, read_write> inks: array<vec4<f32>>;

// The camera: its motor and its speed along its own forward axis in the lattice's frame, the lattice's time at it,
// the tetrahedron holding the eye, and what the tracer reads of it, the eye, its axes in the frame there, and the lamp
// carried with it.
struct Camera {
    motor: Motor,
    speed: f32,
    clock: f32,
    tet: u32,
    place: Point,
    right: Step,
    up: Step,
    forward: Step,
    light: Point,
}

// A crystal's atom: its place, a point of the sphere, and its velocity, a step of the origin in the frame there.
struct Atom {
    place: Point,
    velocity: Step,
}

// A crystal's spring: the rest length, stiffness and damping of its spring and dashpot, and its proper length last
// step; a bond's atoms are `bond_ends`.
struct Spring {
    rest: f32,
    stiffness: f32,
    damping: f32,
    length: f32,
}

fn camera_step(state: Camera) -> Camera {
    var next = state;
    let eye = state.motor >> origin;
    let here = frame(eye);
    // The view's turn in the frame at the eye, a rotor about it, which carries the camera's own steps to the frame's:
    // held unit, so that round-off never compounds through the motor's size.
    let view = (here.reverse() * state.motor).cast(Aim).normalized();
    // The speed along the forward axis, raised and lowered by thrust, stopped by the brake, and held under the fastest
    // either way; and the lattice's time over the frame's tick of the camera's clock.
    let speed = clamp((state.speed + thrust) * (1.0 - braking), -fastest, fastest);
    let step = dt / sqrt(1.0 - speed * speed);
    let waves = local_waves(state.tet, eye, state.clock);
    // The velocity, along the forward axis, in the frame at the eye; the round step it takes, and the frames' roll
    // along it.
    let velocity = (view >> (mv.z.dual() * speed)).cast(Step);
    let round = round_step(velocity, waves);
    let frames_roll = connection(round, waves);
    let moved = ((here * (moving(round) * (-0.5 * step)).exp()) >> origin).normalized();
    // The view, and the motion along it, turned with the frames' roll, then by the keys: the forward toward the right
    // by the yaw and toward the up by the pitch, and the up toward the right by the roll.
    let following = (frames_roll.cast(Turn) * (0.5 * step)).exp();
    let aiming = ((mv.zx * yaw + mv.zy * pitch + mv.yx * roll) * -0.5).exp();
    let turned = following * view * aiming;
    next.motor = frame(moved) * turned;
    next.speed = speed;
    next.clock = state.clock + step;
    next.place = moved;
    // The tetrahedron holding the eye: across the face whose corner's weight has run out most, until none has.
    var tet = state.tet;
    for (var crossing = 0u; crossing < 8u; crossing++) {
        var lowest = 0.0;
        var exit = 4u;
        for (var corner = 0u; corner < 4u; corner++) {
            let weight = tets[tet].ramps[corner] & moved;
            if (weight < lowest) {
                lowest = weight;
                exit = corner;
            }
        }
        if (exit == 4u) {
            break;
        }
        tet = tets[tet].neighbours[exit];
    }
    next.tet = tet;
    next.right = (turned >> mv.x.dual()).cast(Step);
    next.up = (turned >> mv.y.dual()).cast(Step);
    next.forward = (turned >> mv.z.dual()).cast(Step);
    next.light = next.motor >> lamp;
    return next;
}

// One step of a crystal's atoms through the waves at a time: the springs and dashpots along the bonds' proper
// separations pull their ends together, each velocity turns with the connection along its round step, is squeezed as
// the slices stretch and pushed, and each atom rolls along the round step of its velocity.
fn crystal_step(crystal: u32, now: f32, step: f32) {
    let first_atom = crystal * atoms_each;
    let first_bond = crystal * bonds_each;
    // The force on each of the crystal's atoms this step, from nothing.
    var forces: array<Step, atoms_each>;
    for (var b = first_bond; b < first_bond + bonds_each; b++) {
        let ends = bond_ends[b];
        let spring = springs[b];
        let tail = atoms[ends.x].place;
        // The exact squash is the same in every tetrahedron, so the crystals read tetrahedron 0's waves.
        let waves = local_waves(0u, tail, now);
        // The bond's round step at its tail, and its proper separation, the step the connection moves the origin by.
        let round = (frame(tail).reverse() >> atoms[ends.y].place).cast(Step);
        let proper = origin.commutator(connection(round, waves)).cast(Step);
        let length = proper.norm();
        let pull = (spring.stiffness * (length - spring.rest) + spring.damping * (length - spring.length) / step)
                   / length;
        springs[b].length = length;
        forces[ends.x - first_atom] = forces[ends.x - first_atom] + proper * pull;
        forces[ends.y - first_atom] = forces[ends.y - first_atom] - proper * pull;
    }
    for (var k = 0u; k < atoms_each; k++) {
        let i = first_atom + k;
        let place = atoms[i].place;
        let velocity = atoms[i].velocity;
        // Tetrahedron 0's waves, the same in every tetrahedron, as for the bonds.
        let waves = local_waves(0u, place, now);
        let carried = velocity - velocity.commutator(connection(round_step(velocity, waves), waves)).cast(Step) * step;
        let pushed = carried - translating(waves.strain_rate(carried)) * (0.5 * step) + forces[k] * step;
        atoms[i].velocity = pushed;
        atoms[i].place = ((frame(place) * (moving(round_step(pushed, waves)) * (-0.5 * step)).exp())
                          >> origin).normalized();
    }
}

// The crystals as they are, written into a level of the ring: the atoms' places, and each bond's ink, blue squeezed and
// red stretched.
fn record(level: u32) {
    let slot = level % n_levels;
    for (var i = 0u; i < n_crystals * atoms_each; i++) {
        history[slot * n_crystals * atoms_each + i] = atoms[i].place;
    }
    for (var b = 0u; b < n_crystals * bonds_each; b++) {
        let share = 0.5 + 0.5 * clamp((springs[b].length / springs[b].rest - 1.0) / saturating_strain, -1.0, 1.0);
        let at = share * f32(n_strain_inks - 1u);
        let below = u32(floor(at));
        let ink = mix(strain_inks[below], strain_inks[min(below + 1u, n_strain_inks - 1u)], at - floor(at));
        inks[slot * n_crystals * bonds_each + b] = vec4<f32>(ink, 1.0);
    }
}

// The crystals stepped through the lattice's time from one time to another, stopping at each level of the wave's step
// to keep that moment, in as few equal substeps between stops as keep each no longer than the longest; and the frame's
// end kept as the level after the last, which the next level's moment takes over once the clock reaches it.
fn crystals(from: f32, to: f32) {
    // The newest level the frame reaches, counted as the tracer counts it.
    let newest = u32(floor((to - first_time) / level_step));
    var level = u32(floor((from - first_time) / level_step));
    var clock = from;
    for (var stop = 0u; stop <= max_levels_a_frame; stop++) {
        let reaches = level < newest;
        let until = select(to, first_time + f32(level + 1u) * level_step, reaches);
        let n_substeps = u32(ceil(max(until - clock, 0.0) / longest_substep));
        for (var sub = 0u; sub < n_substeps; sub++) {
            for (var crystal = 0u; crystal < n_crystals; crystal++) {
                crystal_step(crystal, clock + (until - clock) * f32(sub) / f32(n_substeps), (until - clock) / f32(n_substeps));
            }
        }
        clock = until;
        if (!reaches) {
            break;
        }
        level = level + 1u;
        record(level);
    }
    record(newest + 1u);
}

@compute @workgroup_size(1)
fn advance(@builtin(global_invocation_id) id: vec3<u32>) {
    let from = camera[0].clock;
    camera[0] = camera_step(camera[0]);
    crystals(from, camera[0].clock);
}
"""


# The camera's sightlines, in the spacetime algebra at the eye, one invocation to a pixel. In its own frame the camera
# is an eye and a sensor a focal length of one behind it, across from where each pixel looks; the light that reaches
# a pixel came through the eye, along the null line from the eye's event to the pixel's. The boost by the camera's
# rapidity along its forward axis carries that line into the lattice's frame, where the walk runs: back along it, the
# scene's direction there, read along the camera's right, up and forward.
sight_source = """
var<uniform> pixels: vec2<u32>;
var<uniform> half_height: f32;
var<uniform> speed: f32;
var<storage, read_write> sightlines: array<vec3<f32>>;

@compute @workgroup_size(8, 8)
fn sight(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= pixels.x || id.y >= pixels.y) {
        return;
    }
    // Where the pixel looks across the view, in half heights of the image from its centre, at the camera's rest.
    let place = vec2<f32>(id.xy) + 0.5;
    let across = vec2<f32>((place.x * 2.0 - f32(pixels.x)) / f32(pixels.y), 1.0 - place.y * 2.0 / f32(pixels.y))
                 * half_height;
    // The pixel on the sensor behind the eye, across from where it looks, and the null line from the eye's event to
    // the pixel's, the light reaching the pixel as long after passing the eye as it is far from it.
    let pixel = -(ahead + right * across.x + up * across.y);
    let arrival = pixel + time * pixel.squared().square_root();
    let rapidity = 0.5 * log((1.0 + speed) / (1.0 - speed));
    let boost = ((ahead & time).dual() * (0.5 * rapidity)).exp();
    let seen = boost >> arrival;
    // The line's part in space in the lattice's frame.
    let in_space = seen - time * (mv.t & seen);
    sightlines[id.y * pixels.x + id.x] = -(axes & in_space);
}
"""


# What the page's tracer reads besides the shared view from inside: each pixel's sightline, as the camera's pass makes
# it; and the crystals as they were when a leg's light passed, from the ring of kept moments, the two nearest blended
# by their nearness as the wave's walk blends its stored levels, and after the newest level, between it and the
# frame's end.
moving_view_source = """
var<storage, read> sightlines: array<vec3<f32>>;
var<storage, read> history: array<Point>;
var<storage, read> inks: array<vec4<f32>>;

fn sightline(pixel: vec2<f32>) -> vec3<f32> {
    return sightlines[u32(pixel.y) * pixels.x + u32(pixel.x)];
}

// The two kept moments about a time, by their slots in the ring, and the later's share.
struct Moments {
    early: u32,
    late: u32,
    share: f32,
}

fn moments(time: f32) -> Moments {
    let newest = floor((eye_time - first_time) / level_step);
    let level = min(floor((time - first_time) / level_step), newest);
    let early_time = first_time + level * level_step;
    let span = select(level_step, eye_time - early_time, level == newest);
    let share = clamp((time - early_time) / max(span, 1e-6), 0.0, 1.0);
    return Moments(u32(level) % n_levels, (u32(level) + 1u) % n_levels, share);
}

fn atom_at(i: u32, time: f32) -> Point {
    let kept = moments(time);
    return (history[kept.early * n_particles + i] * (1.0 - kept.share)
            + history[kept.late * n_particles + i] * kept.share).normalized();
}

fn bond_at(i: u32, time: f32) -> BondSeen {
    let kept = moments(time);
    let ends = bond_ends[i];
    let ink = mix(inks[kept.early * n_bonds + i], inks[kept.late * n_bonds + i], kept.share);
    return BondSeen(atom_at(ends.x, time), atom_at(ends.y, time), ink.xyz);
}
"""


# --- plumbing -------------------------------------------------------------------------
def strain_inks_source() -> str:
    """The colour map's entries the bonds' inks blend between, evenly from most squeezed to most stretched, as a
    table in WGSL, `strain_inks`."""
    inks = plt.get_cmap(render.strain_map)(np.linspace(0, 1, n_strain_inks))[:, :3]
    entries = ", ".join(f"vec3<f32>({float(red)!r}, {float(green)!r}, {float(blue)!r})" for red, green, blue in inks)
    return f"\nconst strain_inks = array<vec3<f32>, {n_strain_inks}>({entries});\n"


def bond_ends_source(bonds: Topology) -> str:
    """The atoms at each bond's ends, as a table in WGSL, `bond_ends`."""
    entries = ", ".join(f"vec2<u32>({tail}u, {head}u)" for tail, head in bonds.elements[1])
    return f"\nconst bond_ends = array<vec2<u32>, {bonds.n_elements[1]}>({entries});\n"


def tracer_shader(bonds: Topology) -> Shader:
    """The view from inside through the exact slowest wave, each ray walked back in time from the moving camera, one
    sample a pixel: resolution, not samples, is what a live view spends."""
    return Shader(sampled(1) + core.quadrupole_walk_source + render.inside_source + bond_ends_source(bonds)
                  + moving_view_source, core.ga, render.inside_types,
                  dict(background=render.inside_background, frequency=core.quadrupole_frequency) | ring)


def advance_shader(bonds: Topology) -> Shader:
    """The camera's and the crystals' step through the same waves, for crystals of so many atoms and bonds each."""
    n_crystals = len(scenarios.tunings)
    return Shader(core.quadrupole_walk_source + strain_inks_source() + bond_ends_source(bonds) + advance_source,
                  core.ga, render.inside_types | dict(Turn=Turn, Aim=Aim),
                  dict(frequency=core.quadrupole_frequency, atoms_each=bonds.n_vertices // n_crystals,
                       bonds_each=bonds.n_elements[1] // n_crystals, n_strain_inks=n_strain_inks,
                       max_levels_a_frame=max_levels_a_frame) | ring)


def sight_shader() -> Shader:
    """The camera's sightlines through its pixels, in the spacetime algebra at the eye: the camera's own axes right,
    up and forward along `x`, `y` and `z`, the lattice's time along `t`."""
    time = -spacetime.t.dual()                                                  # [] Antivector-
    right, up, ahead = stack((spacetime.x, spacetime.y, spacetime.z)).dual()     # [] Antivector-, Antivector-, Antivector-
    return Shader(sight_source, STA, {}, dict(time=time, right=right, up=up, ahead=ahead,
                                              axes=stack((spacetime.x, spacetime.y, spacetime.z))))
