"""GPU shaders in numga's WGSL dialect step the flying camera and spring-bound crystals through the exact quadrupole
gravitational wave, boost sightlines into the lattice rest frame, and interpolate retarded crystal history.

Lorentz boosts at the eye preserve null rays so backward light time still runs one-for-one with geodesic arc length in
the lattice frame while aberration swings sightlines forward at relativistic speeds.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numga import Algebra, NumpyContext, stack
from numga.algebras import STA
from numga.topology import Topology
from numga.wgsl import Shader, sampled

from examples.relativity.gravitational_waves import core, render, scenarios

# Spacetime algebra at the eye for relativistic sightline boosts.
spacetime = NumpyContext(STA).multivector
# Spatial rotation rates and rotors about the eye.
Turn = core.Rate.derive.within(Algebra("x+y+z+"))                                # Bivector+
Aim = core.Motor.derive.within(Algebra("x+y+z+"))                                # Rotor+

wave_period = 2 * np.pi / core.quadrupole_frequency
seconds_per_period = scenarios.n_wave_frames * scenarios.duration_ms / 1000
fastest = 0.95
longest_frame_seconds = 1 / 15
# Ring buffer of retarded crystal moments spanning half a great-circle turn of backward light travel time.
level_step = wave_period / scenarios.n_steps_per_swing
first_time = -(np.pi + level_step)
n_levels = int(np.ceil(np.pi / level_step)) + 3
ring = dict(level_step=level_step, first_time=first_time, n_levels=n_levels)
max_levels_a_frame = int(np.ceil(longest_frame_seconds * wave_period / seconds_per_period
                                 / np.sqrt(1 - fastest ** 2) / level_step)) + 1
n_strain_inks = 33


# --- math -----------------------------------------------------------------------------
advance_source = '''
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
var<uniform> strain_reach: f32;
var<storage, read_write> camera: array<Camera>;
var<storage, read_write> atoms: array<Atom>;
var<storage, read_write> springs: array<Spring>;
var<storage, read_write> history: array<Point>;
var<storage, read_write> inks: array<vec4<f32>>;
var<storage, read_write> balls: array<Ball>;

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

struct Atom {
    place: Point,
    velocity: Step,
}

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
    let view = (here.reverse() * state.motor).cast(Aim).normalized();
    let speed = clamp((state.speed + thrust) * (1.0 - braking), -fastest, fastest);
    let step = dt / sqrt(1.0 - speed * speed);
    let waves = local_waves(state.tet, eye, state.clock);
    let velocity = (view >> (mv.z.dual() * speed)).cast(Step);
    let round = round_step(velocity, waves);
    let frames_roll = connection(round, waves);
    let moved = ((here * (moving(round) * (-0.5 * step)).exp()) >> origin).normalized();
    let following = (frames_roll.cast(Turn) * (0.5 * step)).exp();
    let aiming = ((mv.zx * yaw + mv.zy * pitch + mv.yx * roll) * -0.5).exp();
    let turned = following * view * aiming;
    next.motor = frame(moved) * turned;
    next.speed = speed;
    next.clock = state.clock + step;
    next.place = moved;
    var tet = state.tet;
    for (var crossing = 0u; crossing < 8u; crossing++) {
        var lowest = 0.0;
        var exit = 4u;
        for (var corner = 0u; corner < 4u; corner++) {
            let weight = tets[tet].gradients[corner] & moved;
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

fn crystal_step(crystal: u32, now: f32, step: f32) {
    let first_atom = crystal * atoms_each;
    let first_bond = crystal * bonds_each;
    var forces: array<Step, atoms_each>;
    for (var b = first_bond; b < first_bond + bonds_each; b++) {
        let ends = bond_ends[b];
        let spring = springs[b];
        let tail = atoms[ends.x].place;
        let waves = local_waves(0u, tail, now);
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
        let waves = local_waves(0u, place, now);
        let carried = velocity - velocity.commutator(connection(round_step(velocity, waves), waves)).cast(Step) * step;
        let pushed = carried - translating(waves.strain_rate(carried)) * (0.5 * step) + forces[k] * step;
        atoms[i].velocity = pushed;
        atoms[i].place = ((frame(place) * (moving(round_step(pushed, waves)) * (-0.5 * step)).exp())
                          >> origin).normalized();
    }
}

fn record(level: u32) {
    let slot = level % n_levels;
    for (var i = 0u; i < n_crystals * atoms_each; i++) {
        history[slot * n_crystals * atoms_each + i] = atoms[i].place;
    }
    for (var b = 0u; b < n_crystals * bonds_each; b++) {
        let share = 0.5 + 0.5 * clamp((springs[b].length / springs[b].rest - 1.0) / strain_reach, -1.0, 1.0);
        let at = share * f32(n_strain_inks - 1u);
        let below = u32(floor(at));
        let ink = mix(strain_inks[below], strain_inks[min(below + 1u, n_strain_inks - 1u)], at - floor(at));
        inks[slot * n_crystals * bonds_each + b] = vec4<f32>(ink, 1.0);
    }
}

fn crystals(from: f32, to: f32) {
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

fn kept_ball(crystal: u32) -> Ball {
    var middle: Point;
    for (var level = 0u; level < n_levels; level++) {
        for (var k = 0u; k < atoms_each; k++) {
            middle = middle + history[(level * n_crystals + crystal) * atoms_each + k];
        }
    }
    let centre = middle.normalized();
    var reach = 0.0;
    for (var level = 0u; level < n_levels; level++) {
        for (var k = 0u; k < atoms_each; k++) {
            let cosine = centre.dot(history[(level * n_crystals + crystal) * atoms_each + k]);
            reach = max(reach, acos(clamp(cosine, -1.0, 1.0)));
        }
    }
    return Ball(centre, reach);
}

@compute @workgroup_size(1)
fn advance(@builtin(global_invocation_id) id: vec3<u32>) {
    let from = camera[0].clock;
    camera[0] = camera_step(camera[0]);
    crystals(from, camera[0].clock);
    for (var crystal = 0u; crystal < n_crystals; crystal++) {
        balls[crystal] = kept_ball(crystal);
    }
}
'''


sight_source = '''
var<uniform> pixels: vec2<u32>;
var<uniform> half_height: f32;
var<uniform> speed: f32;
var<storage, read_write> sightlines: array<vec3<f32>>;

@compute @workgroup_size(8, 8)
fn sight(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= pixels.x || id.y >= pixels.y) {
        return;
    }
    let place = vec2<f32>(id.xy) + 0.5;
    let across = vec2<f32>((place.x * 2.0 - f32(pixels.x)) / f32(pixels.y), 1.0 - place.y * 2.0 / f32(pixels.y))
                 * half_height;
    let pixel = -(ahead + right * across.x + up * across.y);
    let arrival = pixel + time * pixel.squared().square_root();
    let rapidity = 0.5 * log((1.0 + speed) / (1.0 - speed));
    let boost = ((ahead & time).dual() * (0.5 * rapidity)).exp();
    let seen = boost >> arrival;
    let in_space = seen - time * (mv.t & seen);
    sightlines[id.y * pixels.x + id.x] = -(axes & in_space);
}
'''


moving_view_source = '''
var<storage, read> sightlines: array<vec3<f32>>;
var<storage, read> history: array<Point>;
var<storage, read> inks: array<vec4<f32>>;

fn sightline(pixel: vec2<f32>) -> vec3<f32> {
    return sightlines[u32(pixel.y) * pixels.x + u32(pixel.x)];
}

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
'''


# --- plumbing -------------------------------------------------------------------------
def strain_inks_source() -> str:
    """Evenly spaced colour-map entries from compressed blue to stretched red format as a WGSL `strain_inks` array."""
    inks = plt.get_cmap(render.strain_map)(np.linspace(0, 1, n_strain_inks))[:, :3]
    entries = ", ".join(f"vec3<f32>({float(red)!r}, {float(green)!r}, {float(blue)!r})" for red, green, blue in inks)
    return f"\nconst strain_inks = array<vec3<f32>, {n_strain_inks}>({entries});\n"


def bond_ends_source(bonds: Topology) -> str:
    """Endpoint atom indices of `bonds` format as a constant WGSL `bond_ends` array."""
    V, E, F, T = bonds.n_elements
    entries = ", ".join(f"vec2<u32>({tail}u, {head}u)" for tail, head in bonds.elements[1])
    return f"\nconst bond_ends = array<vec2<u32>, {E}>({entries});\n"


def tracer_shader(bonds: Topology) -> Shader:
    """Tracing one sample per pixel walks backward rays from the moving camera through the exact quadrupole wave."""
    return Shader(sampled(1) + core.quadrupole_walk_source + render.inside_source + bond_ends_source(bonds)
                  + moving_view_source, core.ga, render.inside_types,
                  dict(background=render.inside_background, frequency=core.quadrupole_frequency) | ring)


def advance_shader(bonds: Topology) -> Shader:
    """One compute invocation per frame steps the flying camera and crystal rings through the exact quadrupole wave."""
    V, E, F, T = bonds.n_elements
    n_crystals = len(scenarios.tunings)
    return Shader(core.quadrupole_walk_source + render.ball_source + strain_inks_source() + bond_ends_source(bonds)
                  + advance_source, core.ga, render.inside_types | dict(Turn=Turn, Aim=Aim),
                  dict(frequency=core.quadrupole_frequency, atoms_each=V // n_crystals,
                       bonds_each=E // n_crystals, n_strain_inks=n_strain_inks,
                       max_levels_a_frame=max_levels_a_frame) | ring)


def sight_shader() -> Shader:
    """Spacetime boosts at the eye transform rest-frame pixel sightlines into lattice-frame ray directions."""
    time = -spacetime.t.dual()                                                  # [] Antivector-
    right, up, ahead = stack((spacetime.x, spacetime.y, spacetime.z)).dual()    # [] Antivector-, Antivector-, Antivector-
    return Shader(sight_source, STA, {}, dict(time=time, right=right, up=up, ahead=ahead,
                                              axes=stack((spacetime.x, spacetime.y, spacetime.z))))
