"""GPU ray-traced rendering for the closed universe from inside a tetrahedral triangulation of the three-sphere and
from outside on the base two-sphere below the Hopf circles."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from numga import stack
from numga.render import meshes, simplices, solids
from numga.topology import Topology, lookup_simplices

from examples.relativity.gravitational_waves import core, numerical
from examples.util import charts


# --- the view from inside a three-sphere, traced on the GPU ------------------------------------------------------------
inside_shape = (300, 400)
fov_radians = np.radians(100.0)
max_legs = 1024
inside_background = np.array([0.035, 0.04, 0.06], dtype=np.float32)
rail_share = 0.014
bead_scale = 2.0
ghost_ink, ghost_share, wave_ghost_share = np.array([0.72, 0.75, 0.80], dtype=np.float32), 0.003, 0.0
rung_ink = np.array([0.85, 0.87, 0.90])
particle_ink, particle_radius = np.array([1.0, 0.86, 0.35], dtype=np.float32), 0.012
bond_radius = 0.005
# Bond strains are coloured linearly, blue at `-strain_reach` through grey to red at `strain_reach`, the tuned
# crystal's peak strain.
strain_map, strain_reach = "coolwarm", 0.2
lamp_placement = (core.mv.yw * 0.4 - core.mv.xw * 0.3).exp()                  # [] Motor+
inside_types = dict(Point=core.Point, Plane=core.Plane, Rate=core.Rate, Step=core.Step, Motor=core.Motor,
                    Strain=core.Strain, Turning=core.Turning, Rolling=core.Rolling, Moving=core.Moving,
                    Translating=core.Translating)

tet_edges = ", ".join(f"vec2<u32>({tail}u, {head}u)" for tail, head in Topology(4, [[0, 1, 2, 3]]).corners[3, 1].T)
visual_source = '''
struct TetVisual {
    places: array<Point, 4>,
    beads: array<vec4<f32>, 4>,
    struts: array<vec4<f32>, 6>,
}

const tet_edges = array<vec2<u32>, 6>(''' + tet_edges + ''');
'''

ball_source = '''
struct Ball {
    centre: Point,
    reach: f32,
}
'''

inside_source = visual_source + ball_source + '''
var<uniform> start_place: Point;
var<uniform> start_tet: u32;
var<uniform> right: Step;
var<uniform> up: Step;
var<uniform> forward: Step;
var<uniform> eye_time: f32;
var<uniform> max_legs: u32;
var<uniform> light: Point;
var<uniform> n_particles: u32;
var<uniform> particle_ink: vec3<f32>;
var<uniform> particle_radius_squared: f32;
var<uniform> n_bonds: u32;
var<uniform> bond_radius_squared: f32;
var<uniform> n_crystals: u32;

struct BondSeen {
    tail: Point,
    head: Point,
    ink: vec3<f32>,
}

var<storage, read> visuals: array<TetVisual>;
var<storage, read> balls: array<Ball>;

alias Sample = vec3<f32>;

fn tangent(place: Point, toward: Point) -> Point {
    return (toward - place * place.dot(toward)).normalized();
}

fn ball_share(leg: Leg, centre: Point, radius_squared: f32) -> f32 {
    // The ball is the cap of points within the angle whose squared sine is `radius_squared` of the unit `centre`,
    // `(centre . p)^2 >= (1 - radius_squared) |p|^2` along `p = chart + segment * share`: one cap in every cell's chart.
    let cosine_squared = 1.0 - radius_squared;
    let start = centre.dot(leg.chart);
    let along = centre.dot(leg.segment);
    let a = along * along - cosine_squared * leg.segment.dot(leg.segment);
    let b = start * along - cosine_squared * leg.chart.dot(leg.segment);
    let c = start * start - cosine_squared * leg.chart.dot(leg.chart);
    // A leg that starts inside the cap entered it at the cell's facet.
    if (c >= 0.0 && start > 0.0) {
        return 0.0;
    }
    let disc = b * b - a * c;
    if (disc < 0.0) {
        return 2.0;
    }
    // The root where the cap's reading rises through its edge, entering it.
    let share = (sqrt(disc) - b) / a;
    return select(2.0, share, share >= 0.0 && start + along * share > 0.0);
}

fn ball_nearest(leg: Leg, centre: Point) -> f32 {
    let reading = leg.plane & centre;
    let offset = leg.chart - centre / reading;
    let along = clamp(-offset.dot(leg.segment) / leg.segment.dot(leg.segment), 0.0, 1.0);
    return select(0.0, along, reading > 0.0);
}

fn nearest_cosine(leg: Leg, centre: Point) -> f32 {
    let start = centre.dot(leg.chart);
    let along = centre.dot(leg.segment);
    let share = clamp((along * leg.chart.dot(leg.chart) - start * leg.chart.dot(leg.segment))
                      / (start * leg.segment.dot(leg.segment) - along * leg.chart.dot(leg.segment)), 0.0, 1.0);
    return centre.dot((leg.chart + leg.segment * share).normalized());
}

fn bond_nearest(leg: Leg, bond: BondSeen) -> f32 {
    let tail_reading = leg.plane & bond.tail;
    let head_reading = leg.plane & bond.head;
    let tail = bond.tail / tail_reading;
    let bond_span = bond.head / head_reading - tail;
    let offset = leg.chart - tail;
    let a = leg.segment.dot(leg.segment);
    let b = leg.segment.dot(bond_span);
    let c = bond_span.dot(bond_span);
    let d = leg.segment.dot(offset);
    let e = bond_span.dot(offset);
    let s = clamp((a * e - b * d) / max(a * c - b * b, 1e-20), 0.0, 1.0);
    return select(0.0, clamp((b * s - d) / a, 0.0, 1.0), tail_reading > 0.0 && head_reading > 0.0);
}

fn bond_share(leg: Leg, bond: BondSeen, reach: f32) -> f32 {
    let tail_reading = leg.plane & bond.tail;
    let head_reading = leg.plane & bond.head;
    if (tail_reading <= 0.0 || head_reading <= 0.0) {
        return 2.0;
    }
    let tail = bond.tail / tail_reading;
    let bond_span = bond.head / head_reading - tail;
    let offset = leg.chart - tail;
    let a = leg.segment.dot(leg.segment);
    let b = leg.segment.dot(bond_span);
    let c = bond_span.dot(bond_span);
    let d = leg.segment.dot(offset);
    let e = bond_span.dot(offset);
    let s = clamp((a * e - b * d) / max(a * c - b * b, 1e-20), 0.0, 1.0);
    let u_b = clamp((b * s - d) / a, 0.0, reach);
    let gap = offset + leg.segment * u_b - bond_span * s;
    return select(2.0, u_b, gap.dot(gap) < bond_radius_squared && u_b < reach);
}

fn faded(lit: Sample, distance: f32) -> Sample {
    return mix(background, lit, max(1.0 - distance / 3.14159, 0.0));
}

fn sample(pixel: vec2<f32>) -> Sample {
    let sight = sightline(pixel);
    var heading = (forward * sight.z + right * sight.x + up * sight.y).normalized();
    var place = start_place;
    var tet = start_tet;
    var entry = 4u;
    var travelled = 0.0;
    for (var k = 0u; k < max_legs; k++) {
        let leg_time = eye_time - travelled;
        let leg = next_leg(place, heading, tet, entry, leg_time);
        var nearest = 1.0;
        var hit_strut = 6u;
        var strut_normal = leg.chart;
        for (var edge = 0u; edge < 6u; edge++) {
            let ends = tet_edges[edge];
            let tail = visuals[tet].places[ends.x];
            let edge_span = visuals[tet].places[ends.y] - tail;
            let offset = leg.chart - tail;
            let radial_offset = offset - edge_span * (offset.dot(edge_span) / edge_span.dot(edge_span));
            let radial_segment = leg.segment - edge_span * (leg.segment.dot(edge_span) / edge_span.dot(edge_span));
            let a = radial_segment.dot(radial_segment);
            let b = radial_offset.dot(radial_segment);
            let c = radial_offset.dot(radial_offset) - visuals[tet].struts[edge].w;
            let disc = b * b - a * c;
            if (c > 0.0 && disc >= 0.0 && a > 0.0) {
                let share = (-b - sqrt(disc)) / a;
                if (share >= 0.0 && share < nearest) {
                    nearest = share;
                    hit_strut = edge;
                    strut_normal = radial_offset + radial_segment * share;
                }
            }
        }
        var hit_bead = 4u;
        var bead_normal = leg.chart;
        for (var corner = 0u; corner < 4u; corner++) {
            let offset = leg.chart - visuals[tet].places[corner];
            let a = leg.segment.dot(leg.segment);
            let b = offset.dot(leg.segment);
            let c = offset.dot(offset) - visuals[tet].beads[corner].w;
            let disc = b * b - a * c;
            if (c > 0.0 && disc >= 0.0 && a > 0.0) {
                let share = (-b - sqrt(disc)) / a;
                if (share >= 0.0 && share < nearest) {
                    nearest = share;
                    hit_strut = 6u;
                    hit_bead = corner;
                    bead_normal = offset + leg.segment * share;
                }
            }
        }
        let widening = sqrt(max(particle_radius_squared, bond_radius_squared)) * leg.plane.norm();
        var hit_particle = n_particles;
        var particle_time = leg_time;
        var hit_bond = n_bonds;
        var bond_time = leg_time;
        for (var crystal = 0u; crystal < n_crystals; crystal++) {
            if (nearest_cosine(leg, balls[crystal].centre) < cos(balls[crystal].reach + widening)) {
                continue;
            }
            for (var i = crystal * n_particles / n_crystals; i < (crystal + 1u) * n_particles / n_crystals; i++) {
                let near_time = leg_time - ball_nearest(leg, atom_at(i, leg_time)) * leg.length;
                let first = ball_share(leg, atom_at(i, near_time), particle_radius_squared);
                let seen_time = select(near_time, leg_time - first * leg.length, first <= 1.0);
                let share = select(first, ball_share(leg, atom_at(i, seen_time), particle_radius_squared),
                                   first <= 1.0);
                if (share < nearest) {
                    nearest = share;
                    hit_particle = i;
                    particle_time = seen_time;
                    hit_bond = n_bonds;
                }
            }
            for (var i = crystal * n_bonds / n_crystals; i < (crystal + 1u) * n_bonds / n_crystals; i++) {
                let near_time = leg_time - bond_nearest(leg, bond_at(i, leg_time)) * leg.length;
                let first = bond_share(leg, bond_at(i, near_time), nearest);
                let seen_time = select(near_time, leg_time - first * leg.length, first <= 1.0);
                let share = select(first, bond_share(leg, bond_at(i, seen_time), nearest), first <= 1.0);
                if (share < nearest) {
                    nearest = share;
                    hit_bond = i;
                    bond_time = seen_time;
                }
            }
        }
        let hit = (leg.chart + leg.segment * nearest).normalized();
        let distance = travelled + nearest * leg.length;
        let to_light = tangent(hit, light);
        if (hit_bond < n_bonds) {
            return faded(bond_at(hit_bond, bond_time).ink * (0.55 + 0.45 * max(tangent(hit, leg.segment).dot(to_light), 0.0)),
                         distance);
        }
        if (hit_particle < n_particles) {
            let normal = tangent(hit, hit - atom_at(hit_particle, particle_time).normalized());
            let to_eye = -tangent(hit, leg.segment);
            let lit = particle_ink * (0.30 + 0.50 * max(normal.dot(to_light), 0.0)
                      + 0.25 * max(normal.dot(to_eye), 0.0));
            return faded(lit, distance);
        }
        if (hit_bead < 4u) {
            let to_eye = -tangent(hit, leg.segment);
            let normal = tangent(hit, bead_normal);
            let halfway = tangent(hit, to_eye + to_light);
            let glint = pow(max(normal.dot(halfway), 0.0), 36.0);
            let lit = visuals[tet].beads[hit_bead].xyz * (0.22 + 0.56 * max(normal.dot(to_light), 0.0)
                      + 0.28 * max(normal.dot(to_eye), 0.0)) + vec3<f32>(0.95, 0.97, 1.0) * (0.36 * glint);
            return faded(lit, distance);
        }
        if (hit_strut < 6u) {
            let to_eye = -tangent(hit, leg.segment);
            let normal = tangent(hit, strut_normal);
            let facing = normal * select(-1.0, 1.0, normal.dot(to_eye) >= 0.0);
            let halfway = tangent(hit, to_eye + to_light);
            let glint = pow(max(facing.dot(halfway), 0.0), 36.0);
            let lit = visuals[tet].struts[hit_strut].xyz * (0.22 + 0.56 * max(facing.dot(to_light), 0.0)
                      + 0.28 * max(facing.dot(to_eye), 0.0)) + vec3<f32>(0.95, 0.97, 1.0) * (0.36 * glint);
            return faded(lit, distance);
        }
        heading = leg.heading;
        place = (leg.chart + leg.segment).normalized();
        tet = leg.tet;
        entry = leg.entry;
        travelled = travelled + leg.length;
        if (travelled > 3.14159) {
            break;
        }
    }
    return background;
}
'''


rest_sight_source = '''
var<uniform> half_height: f32;

fn sightline(pixel: vec2<f32>) -> vec3<f32> {
    let across = screen(pixel) * half_height;
    return vec3<f32>(across, 1.0);
}
'''

crystals_kept_source = '''
struct Bond {
    tail: u32,
    head: u32,
}

var<storage, read> bonds: array<Bond>;
var<storage, read> history: array<Point>;
var<storage, read> inks: array<vec4<f32>>;
var<uniform> kept_start: f32;
var<uniform> kept_step: f32;
var<uniform> n_kept: u32;

struct Moments {
    early: u32,
    late: u32,
    share: f32,
}

fn moments(time: f32) -> Moments {
    let level = clamp((time - kept_start) / kept_step, 0.0, f32(n_kept - 1u));
    let early = u32(floor(level));
    return Moments(early, min(early + 1u, n_kept - 1u), level - floor(level));
}

fn atom_at(i: u32, time: f32) -> Point {
    let kept = moments(time);
    return (history[kept.early * n_particles + i] * (1.0 - kept.share)
            + history[kept.late * n_particles + i] * kept.share).normalized();
}

fn bond_at(i: u32, time: f32) -> BondSeen {
    let kept = moments(time);
    let bond = bonds[i];
    let ink = mix(inks[kept.early * n_bonds + i], inks[kept.late * n_bonds + i], kept.share);
    return BondSeen(atom_at(bond.tail, time), atom_at(bond.head, time), ink.xyz);
}
'''


no_struts = np.zeros((0, 2), dtype=int), np.zeros((0, 3))
no_beads = np.zeros(0, dtype=int), np.zeros((0, 3))


def hues(n_colours: int) -> np.ndarray:
    """Evenly spaced HSV hues around the colour wheel convert to RGB `[n_colours, 3]`."""
    return hsv_to_rgb(np.stack([np.arange(n_colours) / n_colours, np.full(n_colours, 0.75),
                                np.full(n_colours, 0.95)], axis=-1))


def edge_styles(topology: Topology, vertices: core.Point, struts: np.ndarray, strut_inks: np.ndarray,
                ghosts: float) -> np.ndarray:
    """Highlighted `struts` receive `strut_inks` and rail radius while remaining edges receive `ghosts` radius in `[E, 4]`."""
    mean_edge = mean_chord(topology, vertices)
    edges = topology.elements[1]                                                # [E, 2]
    styles = np.tile(np.append(ghost_ink, (ghosts * mean_edge) ** 2), (len(edges), 1)).astype(np.float32)
    styles[lookup_simplices(edges, np.sort(struts, axis=-1))] = np.concatenate(
        [strut_inks, np.full((len(struts), 1), (rail_share * mean_edge) ** 2)], axis=-1)
    return styles


def mean_chord(topology: Topology, vertices: core.Point) -> float:
    """Mean chord length across all edges of `topology` sets the strut and bead radius scale."""
    chords = topology.d[0] * vertices                                           # Point-[E]
    return float(chords.norm().to_array().mean())


def bead_styles(topology: Topology, vertices: core.Point, beads: np.ndarray, bead_inks: np.ndarray) -> np.ndarray:
    """Selected `beads` receive `bead_inks` and scaled bead radius while unmarked vertices have zero radius in `[V, 4]`."""
    V, E, F, T = topology.n_elements
    mean_edge = mean_chord(topology, vertices)
    styles = np.zeros((V, 4), dtype=np.float32)
    radii_squared = np.full((len(beads), 1), (bead_scale * rail_share * mean_edge) ** 2)
    styles[beads] = np.concatenate([bead_inks, radii_squared], axis=-1)
    return styles


def polygon_beads(polygons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed vertex polygons `[n_polygons, n_sides]` flatten to bead indices and matching polygon hues."""
    n_polygons, n_sides = polygons.shape
    return polygons.reshape(-1), np.repeat(hues(n_polygons), n_sides, axis=0)


def polygon_struts(polygons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed vertex polygons `[n_polygons, n_sides]` expand to loop edge pairs `[n_polygons * n_sides, 2]` and hues."""
    n_polygons, n_sides = polygons.shape
    loops = Topology.loops([n_sides] * n_polygons).elements[1]                  # [n_polygons * n_sides, 2]
    return polygons.reshape(-1)[loops], hues(n_polygons)[loops[:, 0] // n_sides]


def round_values(topology: Topology, vertices: core.Point) -> dict:
    """Simplex gradients, origin, translating rate map, and leg reach assemble the unperturbed walk uniforms."""
    return dict(gradients=core.simplex_gradients(topology, vertices), origin=core.origin, moving=core.moving,
                reach=core.round_reach)


def wave_values(topology: Topology, waves: core.Waves) -> dict:
    """Stored wave tables, simplex gradients, and step constants assemble the wave-perturbed walk bindings."""
    V, E, F, T = topology.n_elements
    return dict(gradients=waves.gradients, strains=waves.strains, strain_rates=waves.strain_rates,
                traces=waves.traces, turns=waves.turns, n_vertices=V, n_tets=T, first_time=waves.first_time,
                level_step=waves.level_step, origin=core.origin, translating=core.translating, moving=core.moving,
                turning=core.turning, step=core.wave_step, n_rolls=core.n_rolls)


def inside(walk: str, walk_values: dict, topology: Topology, vertices: core.Point, cameras: core.Motor,
           eye_times: np.ndarray, styles: np.ndarray, beads: np.ndarray,
           crystals: Iterable[dict]) -> Iterator[np.ndarray]:
    """Yields supersampled RGB frames ray-traced from `cameras` at `eye_times` through `topology` using `walk`."""
    from numga.wgsl import Shader, supersampled

    mv = core.mv
    shader = Shader(supersampled() + walk + inside_source + rest_sight_source + crystals_kept_source, core.ga,
                    inside_types, dict(background=inside_background))
    axes = stack((mv.x, mv.y, mv.z)).dual()                                     # [3] Point-
    lamp = lamp_placement >> core.origin                                        # [] Point-
    pick = numerical.picking(topology, vertices)
    half_height = float(np.tan(fov_radians / 2) * inside_shape[0] / inside_shape[1])
    walk_bindings = {name: value for name, value in walk_values.items() if name != "gradients"}
    bindings = walk_bindings | dict(tets=tet_table(topology, walk_values["gradients"]),
                                    visuals=visual_table(topology, vertices, styles, beads),
                                    half_height=half_height, max_legs=max_legs, particle_ink=particle_ink,
                                    particle_radius_squared=particle_radius ** 2,
                                    bond_radius_squared=bond_radius ** 2)
    for frame_index, (eye_time, crystal) in enumerate(zip(eye_times, crystals)):
        camera = cameras[frame_index]                                           # [] Motor+
        place = camera >> core.origin                                           # [] Point-
        view = (core.frame(place).inverse() * camera) >> axes                   # [3] Point-
        image = shader(inside_shape, eye_time=float(eye_time), start_place=place,
                       start_tet=int(pick(place[None])[0]), right=view[0], up=view[1], forward=view[2],
                       light=camera >> lamp, **crystal, **bindings)["image"]
        yield np.round(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def tet_table(topology: Topology, gradients: core.Plane) -> dict:
    """Corner indices, facet neighbours, entry facets, and simplex gradients `[4] Plane+[T]` pack into `Tet` buffers."""
    neighbours, entries = topology.neighbours
    return dict(corners=topology.incidence_indices[3, 0], neighbours=neighbours, entries=entries,
                gradients=stack(tuple(gradients.batch()), axis=-1))


def visual_table(topology: Topology, vertices: core.Point, styles: np.ndarray, beads: np.ndarray) -> dict:
    """Corner positions, vertex bead styles `[V, 4]`, and edge strut styles `[E, 4]` pack into `TetVisual` buffers."""
    corners = topology.incidence[3, 0] * vertices                               # [4] Point-[T]
    return dict(places=stack(tuple(corners.batch()), axis=-1), beads=beads[topology.incidence_indices[3, 0]],
                struts=styles[topology.incidence_indices[3, 1]])


def plain_view(walk: str, walk_values: dict, topology: Topology, vertices: core.Point,
               cameras: core.Motor) -> Iterator[np.ndarray]:
    """Yields interior frames along `cameras` with every edge drawn as a faint ghost strut."""
    return inside(walk, walk_values, topology, vertices, cameras, np.zeros(cameras.shape[0]),
                  edge_styles(topology, vertices, *no_struts, ghost_share), bead_styles(topology, vertices, *no_beads),
                  repeat(no_crystals))


def polygon_view(walk: str, walk_values: dict, topology: Topology, vertices: core.Point, cameras: core.Motor,
                 polygons: np.ndarray) -> Iterator[np.ndarray]:
    """Yields interior frames along `cameras` colouring each closed polygon in `polygons` around the hue wheel."""
    return inside(walk, walk_values, topology, vertices, cameras, np.zeros(cameras.shape[0]),
                  edge_styles(topology, vertices, *polygon_struts(polygons), ghost_share),
                  bead_styles(topology, vertices, *no_beads), repeat(no_crystals))


def helix_view(walk: str, walk_values: dict, topology: Topology, vertices: core.Point, cameras: core.Motor,
               rails: np.ndarray) -> Iterator[np.ndarray]:
    """Yields interior frames colouring the Boerdijk–Coxeter helix `rails` and pale rungs against ghost background struts."""
    sides, side_inks = polygon_struts(rails)
    edges = topology.elements[1]                                                # [E, 2]
    on_rails = np.isin(edges, rails).all(axis=-1)                               # [E]
    is_side = np.isin(np.arange(len(edges)), lookup_simplices(edges, np.sort(sides, axis=-1)))   # [E]
    rungs = edges[on_rails & ~is_side]                                          # [..., 2]
    styles = edge_styles(topology, vertices, np.concatenate([sides, rungs]),
                         np.concatenate([side_inks, np.broadcast_to(rung_ink, (len(rungs), 3))]), ghost_share)
    return inside(walk, walk_values, topology, vertices, cameras, np.zeros(cameras.shape[0]), styles,
                  bead_styles(topology, vertices, *no_beads), repeat(no_crystals))


def waves_view(walk: str, walk_values: dict, topology: Topology, vertices: core.Point, eye: core.Motor,
               eye_times: np.ndarray, rails: np.ndarray, bonds: Topology, kept_times: np.ndarray,
               states: Iterable[tuple[core.Point, core.Scalar]], ghosts: float = ghost_share) -> Iterator[np.ndarray]:
    """Yields interior frames from `eye` across `eye_times` showing retarded crystal `states` and guardrail `rails`."""
    return inside(walk, walk_values, topology, vertices, eye.broadcast_to(eye_times.shape), eye_times,
                  edge_styles(topology, vertices, *polygon_struts(rails), ghosts),
                  bead_styles(topology, vertices, *polygon_beads(rails)),
                  crystal_frames(bonds, kept_times, states, eye_times))


def crystal_frames(bonds: Topology, kept_times: np.ndarray, states: Iterable[tuple[core.Point, core.Scalar]],
                   eye_times: np.ndarray) -> Iterator[dict]:
    """Streams `states` on demand up to each `eye_time`, packing retarded atom history, bond colours, and bounding balls."""
    V, E, F, T = bonds.n_elements
    tails, heads = bonds.elements[1].T
    states = iter(states)
    places, bond_inks = [], []
    for eye_time in eye_times:
        while len(places) < len(kept_times) and (not places or kept_times[len(places) - 1] < eye_time):
            positions, strains = next(states)
            places.append(positions.batch().normalized())
            bond_inks.append(inks(strains, strain_reach))
        n_kept = len(places)
        kept = stack(places)                                                    # [n_kept, V] Point-
        yield dict(bonds=dict(tail=tails, head=heads), history=kept.reshape(-1),
                   inks=np.concatenate(bond_inks).reshape(-1, 4).astype(np.float32), kept_start=float(kept_times[0]),
                   kept_step=float(kept_times[1] - kept_times[0]), n_kept=n_kept, n_particles=V,
                   n_bonds=E, n_crystals=T, balls=crystal_balls(bonds, kept.field()))


def crystal_balls(bonds: Topology, kept: core.Point) -> dict:
    """Bounding-ball centres and angular reaches enclose every stored moment of each tetrahedral crystal in `bonds`."""
    corners = bonds.incidence[3, 0][:, None] * kept                             # [4, ...] Point-[T]
    centres = corners.sum(axis=0).sum(axis=0).normalized()                      # Point-[T]
    arcs = centres.dot(corners).arccos()                                        # [4, ...] Scalar+[T]
    return dict(centre=centres.batch(), reach=arcs.batch().to_array().max(axis=(0, 1)))


no_crystals = dict(bonds=dict(tail=np.zeros(0, dtype=np.uint32), head=np.zeros(0, dtype=np.uint32)),
                   history=core.mv.vector(np.zeros((0, 4))).dual(), inks=np.zeros((0, 4), dtype=np.float32),
                   kept_start=0.0, kept_step=1.0, n_kept=1, n_particles=0, n_bonds=0, n_crystals=0,
                   balls=dict(centre=core.mv.vector(np.zeros((0, 4))).dual(), reach=np.zeros(0, dtype=np.float32)))


volume_chart_px = (480, 220)
crystal_inks = ["#4c72b0", "#dd8452", "#55a868"]


@charts.style()
def crystal_volumes(times: np.ndarray, volumes: core.Amount, names: list[str]) -> plt.Figure:
    """Signed crystal volumes normalized by their initial rest volume plot across `times` for each crystal in `names`."""
    n_times = len(times)
    shares = (volumes / volumes[0]).batch().to_array().reshape(n_times, -1)     # [n_times, ...]
    figure, ax = charts.subplots(*volume_chart_px)
    ax.axhline(0, color="0.6", linewidth=charts.FRAME_WIDTH)
    for share, name, ink in zip(shares.T, names, crystal_inks):
        ax.plot(times, share, color=ink, label=name)
    ax.set(xlabel="time", ylabel="volume / rest", xlim=(times[0], times[-1]))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    figure.tight_layout(pad=0.3)
    return figure


bounds_chart_px = (420, 130)
bound_kinds = ["with the bound", "without the bound", "no crystals"]
bound_inks = ["#dd8452", "#4c72b0", "#999999"]


@charts.style()
def bounded_crystals(milliseconds: np.ndarray, pixels: tuple[int, int]) -> plt.Figure:
    """Horizontal bars compare per-frame ray-tracing times in milliseconds with and without crystal bounding balls."""
    figure, ax = charts.subplots(*bounds_chart_px)
    n_kinds = len(bound_kinds)
    bars = np.arange(n_kinds)                                                   # [n_kinds]
    ax.barh(bars, milliseconds, height=0.6, color=bound_inks)
    for bar, ms in zip(bars, milliseconds):
        ax.annotate(f"{ms:.0f} ms", (ms, bar), textcoords="offset points", xytext=(4, 0), va="center")
    rows, columns = pixels
    ax.set(yticks=bars, yticklabels=bound_kinds, xlim=(0, 1.25 * milliseconds.max()),
           xlabel=f"ms a frame at {columns} × {rows}")
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(pad=0.3)
    return figure


# --- the sphere below the Hopf circles, seen from outside ---------------------------------------------------------------
base_pixels = 300
base_half = 1.15
base_elevation, base_azimuth = 0.45, 2.35
outside = solids.View(solids.aim(solids.origin, base_elevation, base_azimuth), np.full(2, base_half), base_pixels)
base_background = np.ones(3)
dot_radius, atom_radius = 4.5, 2.5
bond_width, light_width = 3.0, 1.4
atom_ink = np.array([0.25, 0.25, 0.25])
light_ink = np.array([0.1, 0.1, 0.1])


def lying(steps: core.Point, height: float) -> solids.Point:
    """Embedding unit origin `steps` into 3D PGA places points on the base sphere raised by `height` pixels."""
    return solids.origin + solids.context.include_from_algebra(steps) * (1 + height * outside.pixel)


def inks(stretches: core.Scalar, reach: float) -> np.ndarray:
    """Diverging colour-map RGBA values, linear in `stretches`: blue at `-reach`, red at `reach`."""
    return plt.get_cmap(strain_map)(0.5 + 0.5 * stretches.to_array().reshape(-1) / reach)


def base_view(sphere_topology: Topology, sphere_points: core.Point, decagon_bases: core.Point, stretch_reach: float,
              stretches: core.Scalar, atoms: core.Point, arcs: core.Point, strains: core.Scalar,
              light: core.Point) -> np.ndarray:
    """Renders one exterior frame of the base sphere shaded by fibre `stretches` with triangle gauges and `light` paths."""
    pixel = outside.pixel

    points = lying(sphere_points.batch(), 0.0)                                  # [V] Point-
    faces = sphere_topology.elements[2]
    normals = meshes.smooth(points, sphere_topology)
    surface = meshes.samples(points, faces, inks(stretches, stretch_reach)[faces, :3], normals, outside, np.zeros(3),
                             meshes.plain, solids.gloss)
    n_bonds, n_arc_points = arcs.shape
    bond_inks = np.repeat(inks(strains, strain_reach)[:, :3], n_arc_points - 1, axis=0)   # [n_bonds (n_arc_points - 1), 3]
    bonds = simplices.tubes(lying(arcs.reshape(-1), bond_width / 2), Topology.chains([n_arc_points] * n_bonds),
                            bond_inks, bond_width, outside, solids.gloss)
    n_rays, n_path_points = light.shape
    rays = simplices.tubes(lying(light.reshape(-1), light_width / 2), Topology.chains([n_path_points] * n_rays),
                           light_ink, light_width, outside, solids.gloss)
    dot_inks = hues(decagon_bases.shape[0])
    parts = [(solids.ball(lying(decagon_bases, 0.0), dot_radius * pixel)[:, None], dot_inks),
             (solids.ball(lying(atoms, bond_width / 2), atom_radius * pixel)[:, None],
              np.broadcast_to(atom_ink, (atoms.shape[0], 3)))]
    return solids.draw(parts, outside, base_background, inks=[], translucent=[],
                       meshes=[(surface, 1.0), (bonds, 1.0), (rays, 1.0)])


def base_views(sphere_topology: Topology, sphere_points: core.Point, decagon_bases: core.Point,
               states: Iterable[tuple]) -> Iterator[np.ndarray]:
    """Yields exterior base-sphere frames across `states`, the fibre stretch coloured linearly with the run's largest
    stretch at either end of the colour map."""
    states = list(states)
    stretch_reach = max(np.abs(stretches.to_array()).max() for stretches, *_ in states)
    for state in states:
        yield base_view(sphere_topology, sphere_points, decagon_bases, stretch_reach, *state)
