"""Drawing for the closed universe: the view from inside a three-sphere meshed by tetrahedra, each ray walked back from
the eye on the GPU leg by leg, and the sphere below the Hopf circles seen from outside, ray traced on the GPU, its
waves, gauges and light lying on it."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from numga import stack

from examples.relativity.gravitational_waves import core
from examples.util import charts
from examples.util.topology import Topology, lookup_simplices
from examples.util.wgsl import meshes, simplices, solids


# --- the view from inside a three-sphere, traced on the GPU ------------------------------------------------------------
# The picture's rows and columns, and the view's opening; the legs a ray may take, past any its half turn needs.
inside_shape = (300, 400)
fov_radians = np.radians(100.0)
max_legs = 1024
inside_background = np.array([0.035, 0.04, 0.06], dtype=np.float32)
# The struts' radii, as shares of the mesh's mean edge, in lengths on the tetrahedra's charts: a coloured strut, and
# every other edge a thin light-grey ghost, none in the pinned picture of the waves, for its file's size. A helix's
# rungs, between its rails.
rail_share = 0.014
# A bead's radius as a multiple of a coloured strut's.
bead_scale = 2.0
ghost_ink, ghost_share, wave_ghost_share = np.array([0.72, 0.75, 0.80], dtype=np.float32), 0.003, 0.0
rung_ink = np.array([0.85, 0.87, 0.90])
# The atoms' ink and radius, the bonds' radius, and the strain that saturates their colour, blue squeezed, red
# stretched, from the colour map that colours every strain and stretch.
particle_ink, particle_radius = np.array([1.0, 0.86, 0.35], dtype=np.float32), 0.012
bond_radius, saturating_strain = 0.005, 0.15
strain_map = "coolwarm"
# The lamp, up and to the left of each eye: the motor that carries the origin to it, before the camera.
lamp_placement = (core.mv.yw * 0.4 - core.mv.xw * 0.3).exp()                  # [] Motor+
# The algebra's types as the shader of the view from inside names them.
inside_types = dict(Point=core.Point, Plane=core.Plane, Rate=core.Rate, Step=core.Step, Motor=core.Motor,
                    Strain=core.Strain, Turning=core.Turning, Rolling=core.Rolling, Moving=core.Moving,
                    Translating=core.Translating)

# A tetrahedron as the view from inside draws it: the place and the bead of each of its corners, an ink and a squared
# radius, and the strut along each of its edges, the same, in its own order of them; and the corners at the ends of
# each edge, in that order, as the topology gives them.
cell_edges = ", ".join(f"vec2<u32>({tail}u, {head}u)" for tail, head in Topology(4, [[0, 1, 2, 3]]).corners[3, 1].T)
cell_source = """
struct Cell {
    places: array<Point, 4>,
    beads: array<vec4<f32>, 4>,
    struts: array<vec4<f32>, 6>,
}

const cell_edges = array<vec2<u32>, 6>(""" + cell_edges + """);
"""

inside_source = cell_source + """
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

// A bond as the view sees it at a time: its atoms' places and its ink. The crystals' atoms and bonds at a time,
// `atom_at` and `bond_at`, and the direction a pixel looks along, `sightline`, come with the view.
struct BondSeen {
    tail: Point,
    head: Point,
    ink: vec3<f32>,
}

var<storage, read> cells: array<Cell>;

alias Sample = vec3<f32>;

// The direction at a place on the sphere toward a point or along a direction, square to the place.
fn tangent(place: Point, toward: Point) -> Point {
    return (toward - place * place.dot(toward)).normalized();
}

// Where along a leg it first meets a ball about a point of the sphere, as a share of the leg, the point scaled onto the
// leg's chart from the chart's side of the sphere; two, more than all of it, where it misses.
fn ball_share(leg: Leg, centre: Point, radius_squared: f32) -> f32 {
    let reading = leg.plane & centre;
    if (reading <= 0.0) {
        return 2.0;
    }
    let offset = leg.chart - centre / reading;
    let a = leg.segment.dot(leg.segment);
    let b = offset.dot(leg.segment);
    let c = offset.dot(offset) - radius_squared;
    let disc = b * b - a * c;
    if (c <= 0.0 || disc < 0.0) {
        return 2.0;
    }
    let share = (-b - sqrt(disc)) / a;
    return select(2.0, share, share >= 0.0);
}

// Where along a leg it passes nearest a point of the sphere, as a share of the leg, the point scaled onto the leg's
// chart from the chart's side of the sphere; the leg's start for a point on the far side.
fn ball_nearest(leg: Leg, centre: Point) -> f32 {
    let reading = leg.plane & centre;
    let offset = leg.chart - centre / reading;
    let along = clamp(-offset.dot(leg.segment) / leg.segment.dot(leg.segment), 0.0, 1.0);
    return select(0.0, along, reading > 0.0);
}

// Where along a leg it passes nearest a bond's segment between its atoms, as a share of the leg, the atoms scaled onto
// the leg's chart from the chart's side of the sphere; the leg's start for a bond with an atom on the far side.
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

// Where along a leg, short of the share `reach`, it comes within a bond's radius of the bond's segment between its
// atoms scaled onto the leg's chart from the chart's side of the sphere, as a share of the leg; two where it does not.
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

// A lit colour seen from a distance along the ray, its brightness falling linearly to nothing at the antipode.
fn faded(lit: Sample, distance: f32) -> Sample {
    return mix(background, lit, max(1.0 - distance / 3.14159, 0.0));
}

fn sample(pixel: vec2<f32>) -> Sample {
    // The direction the pixel looks along, read along the eye's right, up and forward.
    let sight = sightline(pixel);
    var heading = (forward * sight.z + right * sight.x + up * sight.y).normalized();
    var place = start_place;
    var tet = start_tet;
    var entry = 4u;
    var travelled = 0.0;
    for (var k = 0u; k < max_legs; k++) {
        // The ray runs back in time: here it is the light that passed the eye's time less the distance travelled.
        let leg_time = eye_time - travelled;
        let leg = next_leg(place, heading, tet, entry, leg_time);
        // The struts about the tetrahedron's edges, tubes about each edge's line on the chart: where the leg first
        // comes within a strut's radius of the line, from the parts of its offset and its segment square to the edge.
        var nearest = 1.0;
        var hit_strut = 6u;
        var strut_normal = leg.chart;
        for (var edge = 0u; edge < 6u; edge++) {
            let ends = cell_edges[edge];
            let tail = cells[tet].places[ends.x];
            let edge_span = cells[tet].places[ends.y] - tail;
            let offset = leg.chart - tail;
            let radial_offset = offset - edge_span * (offset.dot(edge_span) / edge_span.dot(edge_span));
            let radial_segment = leg.segment - edge_span * (leg.segment.dot(edge_span) / edge_span.dot(edge_span));
            let a = radial_segment.dot(radial_segment);
            let b = radial_offset.dot(radial_segment);
            let c = radial_offset.dot(radial_offset) - cells[tet].struts[edge].w;
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
        // The beads at the tetrahedron's corners, balls about its corners on its chart, each in its colour and squared
        // radius.
        var hit_bead = 4u;
        var bead_normal = leg.chart;
        for (var corner = 0u; corner < 4u; corner++) {
            let offset = leg.chart - cells[tet].places[corner];
            let a = leg.segment.dot(leg.segment);
            let b = offset.dot(leg.segment);
            let c = offset.dot(offset) - cells[tet].beads[corner].w;
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
        // The atoms and the bonds as they were when the light that reaches the eye passed them, the time back along
        // the leg running one for one with its length: each met as it was when the light passed the leg's point
        // nearest it, and where it is met, met again as it was when the light passed that meeting, which for crystals
        // this slow settles at once.
        var hit_particle = n_particles;
        var particle_time = leg_time;
        for (var i = 0u; i < n_particles; i++) {
            let near_time = leg_time - ball_nearest(leg, atom_at(i, leg_time)) * leg.length;
            let first = ball_share(leg, atom_at(i, near_time), particle_radius_squared);
            let seen_time = select(near_time, leg_time - first * leg.length, first <= 1.0);
            let share = select(first, ball_share(leg, atom_at(i, seen_time), particle_radius_squared), first <= 1.0);
            if (share < nearest) {
                nearest = share;
                hit_particle = i;
                particle_time = seen_time;
            }
        }
        var hit_bond = n_bonds;
        var bond_time = leg_time;
        for (var i = 0u; i < n_bonds; i++) {
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
            let lit = cells[tet].beads[hit_bead].xyz * (0.22 + 0.56 * max(normal.dot(to_light), 0.0)
                      + 0.28 * max(normal.dot(to_eye), 0.0)) + vec3<f32>(0.95, 0.97, 1.0) * (0.36 * glint);
            return faded(lit, distance);
        }
        if (hit_strut < 6u) {
            let to_eye = -tangent(hit, leg.segment);
            let normal = tangent(hit, strut_normal);
            let facing = normal * select(-1.0, 1.0, normal.dot(to_eye) >= 0.0);
            let halfway = tangent(hit, to_eye + to_light);
            let glint = pow(max(facing.dot(halfway), 0.0), 36.0);
            let lit = cells[tet].struts[hit_strut].xyz * (0.22 + 0.56 * max(facing.dot(to_light), 0.0)
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
"""


# A camera at rest: each pixel looks along the screen's direction through it.
rest_sight_source = """
var<uniform> half_height: f32;

fn sightline(pixel: vec2<f32>) -> vec3<f32> {
    let across = screen(pixel) * half_height;
    return vec3<f32>(across, 1.0);
}
"""

# The crystals as they were when a leg's light passed, as the page sees them: their atoms' places and their bonds'
# inks kept at `n_kept` moments `kept_step` apart from `kept_start`, the two kept about a time blended by their nearness.
crystals_kept_source = """
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

// The two kept moments about a time, and the later's share.
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
"""


# No struts and no beads: the vertex pairs or vertices, and their inks, of none.
no_struts = np.zeros((0, 2), dtype=int), np.zeros((0, 3))
no_beads = np.zeros(0, dtype=int), np.zeros((0, 3))


def hues(n: int) -> np.ndarray:
    """So many colours evenly round the colour wheel, `[n, 3]`."""
    return hsv_to_rgb(np.stack([np.arange(n) / n, np.full(n, 0.75), np.full(n, 0.95)], axis=-1))


def edge_styles(mesh: Topology, vertices: core.Point, struts: np.ndarray, inks: np.ndarray, ghosts: float
                ) -> np.ndarray:
    """Each edge's colour and squared strut radius, `[edges, 4]`: the edges `struts`, vertex pairs `[struts, 2]`, thick
    struts in their inks `[struts, 3]`, and every other edge a thin ghost, `ghosts` of the mesh's mean edge thick, and
    so not drawn where that is nothing."""
    mean_edge = mean_chord(mesh, vertices)
    edges = mesh.elements[1]                                                    # [edges, 2]
    styles = np.tile(np.append(ghost_ink, (ghosts * mean_edge) ** 2), (len(edges), 1)).astype(np.float32)
    # Each strut's edge, found by its ends.
    styles[lookup_simplices(edges, np.sort(struts, axis=-1))] = np.concatenate(
        [inks, np.full((len(struts), 1), (rail_share * mean_edge) ** 2)], axis=-1)
    return styles


def mean_chord(mesh: Topology, vertices: core.Point) -> float:
    """The mesh's mean edge, a chord's length, the scale of the struts' and beads' radii."""
    chords = mesh.d[0] * vertices                                               # Point-[edges]
    return float(chords.dot(chords).square_root().to_array().mean())


def bead_styles(mesh: Topology, vertices: core.Point, beads: np.ndarray, inks: np.ndarray) -> np.ndarray:
    """Each vertex's bead, its colour and squared radius, `[vertices, 4]`: the vertices `beads`, `[beads]`, balls in
    their inks `[beads, 3]`, `bead_scale` times as thick as a coloured strut, and no bead at any other vertex."""
    mean_edge = mean_chord(mesh, vertices)
    styles = np.zeros((mesh.n_vertices, 4), dtype=np.float32)
    radii_squared = np.full((len(beads), 1), (bead_scale * rail_share * mean_edge) ** 2)
    styles[beads] = np.concatenate([inks, radii_squared], axis=-1)
    return styles


def polygon_beads(polygons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The corners of closed polygons of vertices `[polygons, sides]` as beads `[polygons * sides]`, each polygon in its
    own colour round the colour wheel, `[polygons * sides, 3]`, as `polygon_struts` colours its sides."""
    n_polygons, n_sides = polygons.shape
    return polygons.reshape(-1), np.repeat(hues(n_polygons), n_sides, axis=0)


def polygon_struts(polygons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The sides of closed polygons of vertices `[polygons, sides]` as struts `[polygons * sides, 2]`, each polygon in
    its own colour round the colour wheel, `[polygons * sides, 3]`: the edges of loops through the polygons' vertices,
    one loop to each."""
    n_polygons, n_sides = polygons.shape
    loops = Topology.loops([n_sides] * n_polygons).elements[1]                  # [polygons * sides, 2]
    return polygons.reshape(-1)[loops], hues(n_polygons)[loops[:, 0] // n_sides]


def round_values(mesh: Topology, vertices: core.Point) -> dict:
    """What the core's walk through the round sphere reads besides the topology: the tetrahedra's ramps, the origin,
    the rates that move it, and a leg's reach."""
    return dict(ramps=core.coframe(mesh, vertices), origin=core.origin, moving=core.moving, reach=core.round_reach)


def wave_values(mesh: Topology, waves: core.Waves) -> dict:
    """What the core's walk through the waves reads besides the topology: the waves' stored moments and their counts
    of vertices and tetrahedra, the tetrahedra's ramps, the origin, and the maps of its steps."""
    _, n_vertices = waves.traces.shape
    return dict(ramps=waves.ramps, strains=waves.strains, strain_rates=waves.strain_rates, traces=waves.traces,
                turns=waves.turns, n_vertices=n_vertices, n_tets=mesh.n_elements[3], first_time=waves.first_time,
                level_step=waves.level_step, origin=core.origin, translating=core.translating, moving=core.moving,
                turning=core.turning, step=core.wave_step)


def inside(walk: str, walk_values: dict, mesh: Topology, vertices: core.Point, cameras: core.Motor,
           eye_times: np.ndarray, styles: np.ndarray, beads: np.ndarray, crystals: Iterable[dict]
           ) -> Iterator[np.ndarray]:
    """Frames of the view from inside a round three-sphere meshed by tetrahedra, one for each camera and time: each
    camera, a motor `[frames] Motor+`, carries the origin to the eye and the origin's steps along `x`, `y` and `z` to
    its right, up and forward, read in the frame at the eye; each ray is walked back from the eye leg by leg
    by the WGSL `walk`, which defines `next_leg` and its `Leg` and the `Tet` of the tetrahedra `tets`, of the members
    `tet_table` gives, with the ramps of the corners `walk_values["ramps"]`, `[corners] Plane+[tets]`, and reads the
    rest of `walk_values`; the edges are lit struts, each in its colour and squared radius `styles`, `[edges, 4]`; the
    vertices are lit beads, each in its colour and squared radius `beads`, `[vertices, 4]`, none where that is nothing;
    and at each frame the atoms and bonds of `crystals`, each as it was when a leg's light passed it."""
    from examples.util.wgsl import Shader, supersampled

    mv = core.mv
    shader = Shader(supersampled() + walk + inside_source + rest_sight_source + crystals_kept_source, core.ga,
                    inside_types, dict(background=inside_background))
    axes = stack((mv.x, mv.y, mv.z)).dual()                                     # [3] Point-
    lamp = lamp_placement >> core.origin                                        # [] Point-
    pick = core.picking(mesh, vertices)
    half_height = float(np.tan(fov_radians / 2) * inside_shape[0] / inside_shape[1])
    walk_bindings = {name: value for name, value in walk_values.items() if name != "ramps"}
    bindings = walk_bindings | dict(tets=tet_table(mesh, walk_values["ramps"]),
                                    cells=cell_table(mesh, vertices, styles, beads),
                                    half_height=half_height, max_legs=max_legs, particle_ink=particle_ink,
                                    particle_radius_squared=particle_radius ** 2,
                                    bond_radius_squared=bond_radius ** 2)
    for frame, (eye_time, crystal) in enumerate(zip(eye_times, crystals)):
        camera = cameras[frame]                                                 # [] Motor+
        place = camera >> core.origin                                           # [] Point-
        view = (core.frame(place).inverse() * camera) >> axes                  # [3] Point-
        image = shader(inside_shape, eye_time=float(eye_time), start_place=place,
                       start_tet=int(pick(place[None])[0]), right=view[0], up=view[1], forward=view[2],
                       light=camera >> lamp, **crystal, **bindings)["image"]
        yield np.round(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def tet_table(mesh: Topology, ramps: core.Plane) -> dict:
    """The tetrahedra as a walk reads them, by member of its `Tet`, which takes those it declares: each one's corners
    in its own order, across the face facing each corner the neighbour and the face it is entered by there, and the
    ramps of its corners, `[corners] Plane+[tets]`."""
    neighbours, entries = mesh.neighbours
    return dict(corners=mesh.incidence_indices[3, 0], neighbours=neighbours, entries=entries,
                ramps=stack(tuple(ramps.batch()), axis=-1))


def cell_table(mesh: Topology, vertices: core.Point, styles: np.ndarray, beads: np.ndarray) -> dict:
    """The tetrahedra as the view from inside draws them, by member of its `Cell`: the places of each one's corners and
    their beads, of the vertices' `beads`, `[vertices, 4]`, and the struts along its edges, of the edges' `styles`,
    `[edges, 4]`, in its own order of them."""
    corners = mesh.incidence[3, 0] * vertices                                   # [corners] Point-[tets]
    return dict(places=stack(tuple(corners.batch()), axis=-1), beads=beads[mesh.incidence_indices[3, 0]],
                struts=styles[mesh.incidence_indices[3, 1]])


def plain_view(walk: str, walk_values: dict, mesh: Topology, vertices: core.Point, cameras: core.Motor
               ) -> Iterator[np.ndarray]:
    """Frames of the view from inside the round three-sphere, as `inside` draws them, along the cameras `[frames]
    Motor+`, every edge a thin ghost."""
    return inside(walk, walk_values, mesh, vertices, cameras, np.zeros(cameras.shape[0]),
                  edge_styles(mesh, vertices, *no_struts, ghost_share), bead_styles(mesh, vertices, *no_beads),
                  repeat(no_crystals))


def polygon_view(walk: str, walk_values: dict, mesh: Topology, vertices: core.Point, cameras: core.Motor,
                 polygons: np.ndarray) -> Iterator[np.ndarray]:
    """Frames of the view from inside the round three-sphere, as `inside` draws them, along the cameras `[frames]
    Motor+`: the closed polygons of edges, vertex indices `[polygons, sides]`, as struts each in its own colour round
    the colour wheel, every other edge a thin ghost."""
    return inside(walk, walk_values, mesh, vertices, cameras, np.zeros(cameras.shape[0]),
                  edge_styles(mesh, vertices, *polygon_struts(polygons), ghost_share),
                  bead_styles(mesh, vertices, *no_beads), repeat(no_crystals))


def helix_view(walk: str, walk_values: dict, mesh: Topology, vertices: core.Point, cameras: core.Motor,
               rails: np.ndarray) -> Iterator[np.ndarray]:
    """Frames of the view from inside the round three-sphere, as `inside` draws them, along the cameras `[frames]
    Motor+`: a Boerdijk–Coxeter helix's rails, closed polygons of vertices `[rails, sides]`, as struts each in its own
    colour, the rungs between them, every other edge with both ends on the rails, light, and every edge beyond a thin
    ghost."""
    sides, side_inks = polygon_struts(rails)
    edges = mesh.elements[1]                                                    # [edges, 2]
    on_rails = np.isin(edges, rails).all(axis=-1)                               # [edges]
    is_side = np.isin(np.arange(len(edges)), lookup_simplices(edges, np.sort(sides, axis=-1)))   # [edges]
    rungs = edges[on_rails & ~is_side]                                          # [rungs, 2]
    styles = edge_styles(mesh, vertices, np.concatenate([sides, rungs]),
                         np.concatenate([side_inks, np.broadcast_to(rung_ink, (len(rungs), 3))]), ghost_share)
    return inside(walk, walk_values, mesh, vertices, cameras, np.zeros(cameras.shape[0]), styles,
                  bead_styles(mesh, vertices, *no_beads), repeat(no_crystals))


def waves_view(walk: str, walk_values: dict, mesh: Topology, vertices: core.Point, eye: core.Motor,
               eye_times: np.ndarray, rails: np.ndarray, bonds: Topology, kept_times: np.ndarray,
               states: Iterable[tuple[core.Point, core.Scalar]], ghosts: float = ghost_share) -> Iterator[np.ndarray]:
    """Frames of the view from inside the round three-sphere carrying gravitational waves, as `inside` draws them
    through a walk through the waves, from the camera `eye` at each of the `eye_times`: the closed polygons of edges
    `rails`, vertex indices `[rails, sides]`, as struts each in its own colour, and every other edge a ghost `ghosts`
    of the mesh's mean edge thick, none where that is nothing; and the crystals, each as it was when a leg's light
    passed it, from their atoms and their bonds' strains of `states`, `(Point-[atoms], Scalar+[bonds])` at each of the
    `kept_times`, the bonds the edges of `bonds`, coloured by their strains."""
    return inside(walk, walk_values, mesh, vertices, eye.broadcast_to(eye_times.shape), eye_times,
                  edge_styles(mesh, vertices, *polygon_struts(rails), ghosts),
                  bead_styles(mesh, vertices, *polygon_beads(rails)),
                  crystal_frames(bonds, kept_times, states, eye_times))


def crystal_frames(bonds: Topology, kept_times: np.ndarray, states: Iterable[tuple[core.Point, core.Scalar]],
                   eye_times: np.ndarray) -> Iterator[dict]:
    """The crystals at each of the `eye_times` as the view from inside draws them, each as it was when a leg's light
    passed it: every moment of `states`, `(Point-[atoms], Scalar+[bonds])` at each of the `kept_times`, evenly spaced,
    up to the first at or past the frame's time, drawn from the states as the frames need them; their atoms balls, and
    the edges of `bonds` segments between them coloured by their strains, blue squeezed and red stretched."""
    tails, heads = bonds.elements[1].T
    states = iter(states)
    places, inks = [], []
    for eye_time in eye_times:
        while len(places) < len(kept_times) and (not places or kept_times[len(places) - 1] < eye_time):
            positions, strains = next(states)
            places.append(positions.batch().normalized())
            inks.append(plt.get_cmap(strain_map)(0.5 + 0.5 * np.clip(strains.to_array() / saturating_strain, -1, 1)))
        yield dict(bonds=dict(tail=tails, head=heads), history=stack(places).reshape(-1),
                   inks=np.concatenate(inks).reshape(-1, 4).astype(np.float32), kept_start=float(kept_times[0]),
                   kept_step=float(kept_times[1] - kept_times[0]), n_kept=len(places), n_particles=bonds.n_vertices,
                   n_bonds=bonds.n_elements[1])


# No crystals: no atoms and no bonds, and one moment of nothing kept.
no_crystals = dict(bonds=dict(tail=np.zeros(0, dtype=np.uint32), head=np.zeros(0, dtype=np.uint32)),
                   history=core.mv.vector(np.zeros((0, 4))).dual(), inks=np.zeros((0, 4), dtype=np.float32),
                   kept_start=0.0, kept_step=1.0, n_kept=1, n_particles=0, n_bonds=0)


# The chart of the crystals' signed volumes: its size, and each crystal's colour, the soft, the tuned and the stiff one.
volume_chart_px = (480, 220)
crystal_inks = ["#4c72b0", "#dd8452", "#55a868"]


@charts.style()
def crystal_volumes(times: np.ndarray, volumes: core.Scalar, names: list[str]) -> plt.Figure:
    """Each crystal's signed volume, the join of its corners, against its volume at rest, at each of the `times`:
    `[times] Scalar-[crystals]`, one curve for each crystal of `names`; below nothing, the crystal is inside out."""
    shares = (volumes / volumes[0]).batch().to_array().reshape(len(times), -1)   # [times, crystals]
    figure, ax = charts.subplots(*volume_chart_px)
    ax.axhline(0, color="0.6", linewidth=charts.FRAME_WIDTH)
    for share, name, ink in zip(shares.T, names, crystal_inks):
        ax.plot(times, share, color=ink, label=name)
    ax.set(xlabel="time", ylabel="volume / rest", xlim=(times[0], times[-1]))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    figure.tight_layout(pad=0.3)
    return figure


# --- the sphere below the Hopf circles, seen from outside ---------------------------------------------------------------
# The picture's pixels across, half its width in the scene, and the camera's elevation and azimuth; the view, and its
# background; the stretch that saturates the colours of the circles' stretches and the gauges' strains.
base_pixels = 300
base_half = 1.15
base_elevation, base_azimuth = 0.45, 2.75
outside = solids.View(solids.aim(solids.origin, base_elevation, base_azimuth), np.full(2, base_half), base_pixels)
base_background = np.ones(3)
saturating_stretch = 0.12
# The balls at the decagons' base points and the gauges' atoms, their radii in pixels, and the widths of the gauges'
# bonds and of the light's paths, in pixels; atoms, bonds and light lie on the sphere, raised by half the bonds' or
# the light's width.
dot_radius, atom_radius = 4.5, 2.5
bond_width, light_width = 3.0, 1.4
atom_ink = np.array([0.25, 0.25, 0.25])
light_ink = np.array([0.1, 0.1, 0.1])


def lying(steps: core.Point, height: float) -> solids.Point:
    """Points of the base, unit steps of the three-sphere's origin `[...] Point-`, in the scene, raised above the
    sphere by `height` pixels: each step the offset of its point from the scene's origin, as the algebra of the
    three-sphere includes in the scene's by the steps' complements, the planes through the origin."""
    return solids.origin + solids.context.include_from_algebra(steps) * (1 + height * outside.pixel)


def inks(stretches: core.Scalar) -> np.ndarray:
    """The colours of stretches `[n] Scalar+`, blue shortened and red lengthened."""
    return plt.get_cmap(strain_map)(0.5 + 0.5 * np.clip(stretches.to_array().reshape(-1) / saturating_stretch,
                                                        -1, 1))[:, :3]


def base_view(sphere_mesh: Topology, sphere_points: core.Point, decagon_bases: core.Point, stretches: core.Scalar,
              atoms: core.Point, arcs: core.Point, strains: core.Scalar, light: core.Point) -> np.ndarray:
    """The base from outside: the icosphere on the base's points `Point-[points]`, the decagons' base points
    `[decagons] Point-` in hues around the decagons; and at a frame the icosphere coloured by how much the circles
    above its points stretch, `Scalar+[points]`, the gauges' atoms `[atoms] Point-` and their bonds' arcs `[bonds, arc
    points] Point-` coloured by their strains `[bonds] Scalar+`, and the light's paths `[rays, steps + 1] Point-`, all
    on the base."""
    pixel = outside.pixel

    points = lying(sphere_points.batch(), 0.0)                                  # [points] Point-
    faces = sphere_mesh.elements[2]
    normals = meshes.smooth(points, sphere_mesh)
    surface = meshes.samples(points, faces, inks(stretches)[faces], normals, outside, np.zeros(3), meshes.plain,
                             solids.gloss)
    n_bonds, n_arc_points = arcs.shape
    bonds = simplices.tubes(lying(arcs.reshape(-1), bond_width / 2), Topology.chains([n_arc_points] * n_bonds),
                            np.repeat(inks(strains), n_arc_points - 1, axis=0), bond_width, outside, solids.gloss)
    n_rays, n_path_points = light.shape
    rays = simplices.tubes(lying(light.reshape(-1), light_width / 2), Topology.chains([n_path_points] * n_rays),
                           light_ink, light_width, outside, solids.gloss)
    dot_inks = hues(decagon_bases.shape[0])
    parts = [(solids.ball(lying(decagon_bases, 0.0), dot_radius * pixel)[:, None], dot_inks),
             (solids.ball(lying(atoms, bond_width / 2), atom_radius * pixel)[:, None],
              np.broadcast_to(atom_ink, (atoms.shape[0], 3)))]
    return solids.draw(parts, outside, base_background, inks=[], translucent=[],
                       meshes=[(surface, 1.0), (bonds, 1.0), (rays, 1.0)])
