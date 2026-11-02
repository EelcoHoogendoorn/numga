"""Drawing for the closed universe: the view from inside a three-sphere meshed by tetrahedra, its rays rolled on the
GPU through the waves' field of strain, and the sphere below the Hopf circles seen from outside, ray traced on the
GPU, its waves, gauges and light lying on it."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from numga import Extensor, stack

from examples.relativity.closed_universe import core
from examples.util.topology import Topology, lookup_simplices
from examples.util.wgsl import meshes, simplices, solids


# --- the view from inside a three-sphere, traced on the GPU ------------------------------------------------------------
# The picture's rows and columns, the view's opening, the step of length along each ray, the background, the struts'
# ink and radius.
inside_shape = (300, 400)
fov_radians = np.radians(100.0)
# The length of a ray's steps through the waves, each read at its midpoint, short enough that the picture no longer
# changes with it; through the round sphere a ray runs straight in each tetrahedron's chart, and a step as long as the
# 600-cell's edges takes it to the next face.
wave_step = 0.02
round_step = 0.6
# The faces a ray may cross on its way, besides its steps.
max_crossings = 256
inside_background = np.array([0.035, 0.04, 0.06], dtype=np.float32)
# The guardrails' struts, each decagon in its own colour, and every other edge a thin light-grey ghost.
rail_radius = 0.028
ghost_ink, ghost_radius = np.array([0.72, 0.75, 0.80], dtype=np.float32), 0.006
# A helix's rungs, between its rails.
rung_ink = np.array([0.85, 0.87, 0.90])
# The atoms' ink and radius, the bonds' radius, and the strain that saturates their colour in the stepped squash,
# blue squeezed, red stretched.
particle_ink, particle_radius = np.array([1.0, 0.86, 0.35], dtype=np.float32), 0.012
bond_radius, saturating_strain = 0.005, 0.15
# How far an atom may move from where it starts and still be drawn in the tetrahedra listed for it.
motion_room = 0.03

inside_source = """
var<uniform> start_place: Point;
var<uniform> start_tet: u32;
var<uniform> right: Step;
var<uniform> up: Step;
var<uniform> forward: Step;
var<uniform> eye_time: f32;
var<uniform> half_height: f32;
var<uniform> step: f32;
var<uniform> max_steps: u32;
var<uniform> light: Point;
var<storage, read> edge_style: array<vec4<f32>>;
var<storage, read> particles: array<Point>;
var<uniform> n_particles: u32;
var<storage, read> particle_starts: array<u32>;
var<storage, read> tet_particles: array<u32>;
var<storage, read> bond_starts: array<u32>;
var<storage, read> tet_bonds: array<u32>;
var<uniform> particle_ink: vec3<f32>;
var<uniform> particle_radius_squared: f32;
var<storage, read> bond_tails: array<Point>;
var<storage, read> bond_heads: array<Point>;
var<storage, read> bond_inks: array<vec4<f32>>;
var<uniform> n_bonds: u32;
var<uniform> bond_radius_squared: f32;

alias Sample = vec3<f32>;

fn along(place: Point, toward: Point) -> Point {
    return (toward - place * place.dot(toward)).normalized();
}

// Where along a step, as a share of it, a ray first comes within a strut's radius of the edge whose two opposite
// corners' weights are `lc` and `ld`, changing by `dlc` and `dld` over the step; past `u_max`, or from inside, none.
fn edge_test(lc: f32, ld: f32, dlc: f32, dld: f32, r2: f32, u_max: f32) -> f32 {
    if (r2 <= 1e-8) {
        return 1e30;
    }
    let A = 3.0 * lc * lc + 2.0 * lc * ld + 3.0 * ld * ld - r2;
    let B = 3.0 * lc * dlc + (lc * dld + ld * dlc) + 3.0 * ld * dld;
    let C = 3.0 * dlc * dlc + 2.0 * dlc * dld + 3.0 * dld * dld;
    if (A <= 0.0) {
        return 1e30;
    }
    let disc = B * B - A * C;
    if (disc < 0.0 || abs(C) < 1e-12) {
        return 1e30;
    }
    let u = (-B - sqrt(disc)) / C;
    if (u >= 0.0 && u <= u_max) {
        return u;
    }
    return 1e30;
}

fn sample(place: vec2<f32>) -> Sample {
    let on_screen = screen(place) * half_height;
    var heading = (forward + right * on_screen.x + up * on_screen.y).normalized();
    let opp_c = array<u32, 6>(2u, 1u, 1u, 0u, 0u, 0u);
    let opp_d = array<u32, 6>(3u, 3u, 2u, 3u, 2u, 1u);
    var position = start_place;
    var tet = start_tet;
    var entry_face = 4u;
    var travelled = 0.0;
    for (var k = 0u; k < max_steps; k++) {
        let base = tet * 4u;
        let unit = unit_plane(tet);
        // The place on the tetrahedron's chart, its plane through the corners, on which great circles run straight and
        // the barycentric weights change evenly.
        let chart = position / (unit & position);
        // The ray runs back in time: here it is the light that passed the eye's time less the distance travelled.
        let time = eye_time - travelled;
        let l = array<f32, 4>(ramps[base] & chart, ramps[base + 1u] & chart, ramps[base + 2u] & chart,
                              ramps[base + 3u] & chart);
        let start_waves = local_waves(tet, l, time);
        // The ray's round step at the start, its heading less half the waves' stretch of it, and how far along it the
        // step goes: a step's length at most, cut at the face it leaves by, never the one it came in by.
        let start_steps = heading - translating(start_waves.strain(heading)) * 0.5;
        let full = (frame(position) * ((equator ^ polar(start_steps)) * (-0.5 * step)).exp()) >> origin;
        let full_segment = full / (unit & full) - chart;
        var u_exit = 1.0;
        var exit_face = 4u;
        for (var f = 0u; f < 4u; f++) {
            let dl_f = ramps[base + f] & full_segment;
            if (f != entry_face && dl_f < -1e-9) {
                let u_f = max(0.0, -l[f] / dl_f);
                if (u_f < u_exit) {
                    u_exit = u_f;
                    exit_face = f;
                }
            }
        }
        // The step's length, the arc from the place to where it ends.
        let length = acos(clamp(position.dot((chart + full_segment * u_exit).normalized()), -1.0, 1.0));
        // The step reads the waves at its midpoint: the heading half a step on, turned by the waves at the start, and the
        // place there, half the step's length along the start's round step, at the light's time there.
        let start_change = heading_change(heading, connection(start_steps, start_waves), start_waves);
        let half_heading = (heading + start_change * (0.5 * length)).normalized();
        let half_place = (frame(position) * ((equator ^ polar(start_steps)) * (-0.25 * length)).exp()) >> origin;
        let half_chart = half_place / (unit & half_place);
        let half_l = array<f32, 4>(ramps[base] & half_chart, ramps[base + 1u] & half_chart,
                                   ramps[base + 2u] & half_chart, ramps[base + 3u] & half_chart);
        let half_waves = local_waves(tet, half_l, time - 0.5 * length);
        let steps = half_heading - translating(half_waves.strain(half_heading)) * 0.5;
        let change = heading_change(half_heading, connection(steps, half_waves), half_waves);
        // The place the step ends, rolled the step's length along its round step on its great circle by the frame at the
        // start, and the straight line to it on the chart.
        let ahead = (frame(position) * ((equator ^ polar(steps)) * (-0.5 * length)).exp()) >> origin;
        let segment = ahead / (unit & ahead) - chart;
        let dl = array<f32, 4>(ramps[base] & segment, ramps[base + 1u] & segment, ramps[base + 2u] & segment,
                               ramps[base + 3u] & segment);
        var best_u = 1.0;
        var hit_edge = 6u;
        for (var e = 0u; e < 6u; e++) {
            let u_e = edge_test(l[opp_c[e]], l[opp_d[e]], dl[opp_c[e]], dl[opp_d[e]],
                                edge_style[tet * 6u + e].w, best_u);
            if (u_e < best_u) {
                best_u = u_e;
                hit_edge = e;
            }
        }
        // The test particles near the tetrahedron, balls about their places scaled onto its plane, where the step comes
        // nearest.
        var hit_particle = n_particles;
        for (var listed = particle_starts[tet]; listed < particle_starts[tet + 1u]; listed++) {
            let i = tet_particles[listed];
            let reading = unit & particles[i];
            if (reading > 0.0) {
                let offset = chart - particles[i] / reading;
                let a = segment.dot(segment);
                let b = offset.dot(segment);
                let c = offset.dot(offset) - particle_radius_squared;
                let disc = b * b - a * c;
                if (c > 0.0 && disc >= 0.0) {
                    let u_p = (-b - sqrt(disc)) / a;
                    if (u_p >= 0.0 && u_p < best_u) {
                        best_u = u_p;
                        hit_particle = i;
                    }
                }
            }
        }
        // The bonds near the tetrahedron, segments between their ends scaled onto its plane, where the step comes nearest.
        var hit_bond = n_bonds;
        var bond_share = 0.0;
        for (var listed = bond_starts[tet]; listed < bond_starts[tet + 1u]; listed++) {
            let i = tet_bonds[listed];
            let tail_reading = unit & bond_tails[i];
            let head_reading = unit & bond_heads[i];
            if (tail_reading > 0.0 && head_reading > 0.0) {
                let tail = bond_tails[i] / tail_reading;
                let along_bond = bond_heads[i] / head_reading - tail;
                let offset = chart - tail;
                let a = segment.dot(segment);
                let b = segment.dot(along_bond);
                let c = along_bond.dot(along_bond);
                let d = segment.dot(offset);
                let e = along_bond.dot(offset);
                let s = clamp((a * e - b * d) / max(a * c - b * b, 1e-20), 0.0, 1.0);
                let u_b = clamp((b * s - d) / a, 0.0, best_u);
                let gap = offset + segment * u_b - along_bond * s;
                if (gap.dot(gap) < bond_radius_squared && u_b < best_u) {
                    best_u = u_b;
                    hit_bond = i;
                    bond_share = s;
                }
            }
        }
        if (hit_bond < n_bonds) {
            let hit = (chart + segment * best_u).normalized();
            let distance = travelled + best_u * length;
            let fog = max(1.0 - distance / 3.14159, 0.0);
            let to_light = along(hit, light);
            let lit = bond_inks[hit_bond].xyz * (0.55 + 0.45 * max(along(hit, segment).dot(to_light), 0.0));
            return mix(background, lit, fog);
        }
        if (hit_particle < n_particles) {
            let hit = (chart + segment * best_u).normalized();
            let reading = unit & particles[hit_particle];
            let n_tan = along(hit, hit - particles[hit_particle].normalized());
            let to_eye = -along(hit, segment);
            let to_light = along(hit, light);
            let diff1 = max(n_tan.dot(to_light), 0.0);
            let diff2 = max(n_tan.dot(to_eye), 0.0);
            let lit = particle_ink * (0.30 + 0.50 * diff1 + 0.25 * diff2);
            let distance = travelled + best_u * length;
            // Brightness falls linearly with distance, to nothing at the antipode.
            let fog = max(1.0 - distance / 3.14159, 0.0);
            return mix(background, lit, fog);
        }
        if (hit_edge < 6u) {
            let hit = (chart + segment * best_u).normalized();
            let ray = along(hit, segment);
            let ic = opp_c[hit_edge];
            let id = opp_d[hit_edge];
            let lc = l[ic] + best_u * dl[ic];
            let ld = l[id] + best_u * dl[id];
            let grad_q = ramps[base + ic] * (3.0 * lc + ld) + ramps[base + id] * (lc + 3.0 * ld);
            let n_tan = along(hit, grad_q.dual());
            let to_eye = -ray;
            let facing = n_tan * select(-1.0, 1.0, n_tan.dot(to_eye) >= 0.0);
            let to_light = along(hit, light);
            let diff1 = max(facing.dot(to_light), 0.0);
            let diff2 = max(facing.dot(to_eye), 0.0);
            let halfway = along(hit, to_eye + to_light);
            let spec = pow(max(facing.dot(halfway), 0.0), 36.0);
            let lit = edge_style[tet * 6u + hit_edge].xyz * (0.22 + 0.56 * diff1 + 0.28 * diff2) + vec3<f32>(0.95, 0.97, 1.0) * (0.36 * spec);
            let distance = travelled + best_u * length;
            // Brightness falls linearly with distance, to nothing at the antipode.
            let fog = max(1.0 - distance / 3.14159, 0.0);
            return mix(background, lit, fog);
        }
        heading = (heading + change * length).normalized();
        position = (chart + segment).normalized();
        travelled = travelled + length;
        if (travelled > 3.14159) {
            break;
        }
        if (exit_face < 4u) {
            tet = neighbours[base + exit_face];
            entry_face = entries[base + exit_face];
        } else {
            entry_face = 4u;
        }
    }
    return background;
}
"""


def edge_styles(mesh: Topology, struts: np.ndarray, inks: np.ndarray) -> np.ndarray:
    """Each edge of each tetrahedron's colour and squared strut radius, `[tets * 6, 4]`, in the tetrahedron's own order
    of its edges: the edges `struts`, vertex pairs `[struts, 2]`, thick struts in their inks `[struts, 3]`, and every
    other edge a thin ghost."""
    edges = mesh.elements[1]                                                    # [edges, 2]
    styles = np.tile(np.append(ghost_ink, ghost_radius ** 2), (len(edges), 1)).astype(np.float32)
    # Each strut's edge, and each tetrahedron's six, found by their ends.
    chosen = lookup_simplices(edges, np.sort(struts, axis=-1))                  # [struts]
    styles[chosen] = np.concatenate([inks, np.full((len(struts), 1), rail_radius ** 2)], axis=-1)
    tet_edges = lookup_simplices(edges, np.sort(mesh.elements[3][:, mesh.corners[3, 1].T], axis=-1))   # [tets, 6]
    return styles[tet_edges.reshape(-1)]


def hues(n: int) -> np.ndarray:
    """So many colours evenly round the colour wheel, `[n, 3]`."""
    return hsv_to_rgb(np.stack([np.arange(n) / n, np.full(n, 0.75), np.full(n, 0.95)], axis=-1))


def polygon_struts(polygons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The sides of closed polygons of vertices `[polygons, sides]` as struts `[polygons * sides, 2]`, each polygon in
    its own colour round the colour wheel, `[polygons * sides, 3]`."""
    sides = np.stack([polygons, np.roll(polygons, -1, axis=-1)], axis=-1).reshape(-1, 2)
    return sides, np.repeat(hues(len(polygons)), polygons.shape[1], axis=0)


def helix_struts(mesh: Topology, rails: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A Boerdijk–Coxeter helix's edges as struts, `[struts, 2]`, with their inks, `[struts, 3]`: its rails, closed
    polygons of vertices `[rails, sides]`, each in its own colour, and the rungs between them, every other edge with
    both ends on the rails, light."""
    sides, side_inks = polygon_struts(rails)
    edges = mesh.elements[1]                                                    # [edges, 2]
    on_rails = np.isin(edges, rails).all(axis=-1)                               # [edges]
    is_side = np.isin(np.arange(len(edges)), lookup_simplices(edges, np.sort(sides, axis=-1)))   # [edges]
    rungs = edges[on_rails & ~is_side]                                          # [rungs, 2]
    return (np.concatenate([sides, rungs]),
            np.concatenate([side_inks, np.broadcast_to(rung_ink, (len(rungs), 3))]))


def great_polygon_struts(mesh: Topology, vertices: Extensor) -> tuple[np.ndarray, np.ndarray]:
    """Every edge of a mesh as a strut, `[edges, 2]`, coloured by the great circle it runs along, the colours round the
    colour wheel, `[edges, 3]`: an edge's great circle is the join of its ends, read by the planes across each pair of
    directions, up to its sense."""
    mv = core.mv
    tails, heads = mesh.incidence[1, 0] * vertices                              # Point-[edges], Point-[edges]
    circles = (tails & heads).normalized().batch()                             # [edges] Antibivector-
    readings = (stack((mv.xy, mv.xz, mv.xw, mv.yz, mv.yw, mv.zw)) & circles[:, None]).to_array()   # [edges, 6]
    signs = np.sign(readings[np.arange(len(readings)), np.abs(readings).round(9).argmax(axis=-1)])
    _, circle_of = np.unique(np.round(readings * signs[:, None], 6), axis=0, return_inverse=True)
    return mesh.elements[1], hues(circle_of.max() + 1)[circle_of.reshape(-1)]


def inside(mesh: Topology, vertices: Extensor, waves, cameras: Extensor, eye_times: np.ndarray, styles: np.ndarray, step: float,
           crystals: Iterable[dict]) -> Iterator[np.ndarray]:
    """Frames of the view from inside a round three-sphere meshed by tetrahedra and carrying the stepped gravitational
    waves `waves`, one for each camera and time: each camera, a motor `[frames] Motor+`, carries the origin to the eye
    and the origin's steps across `x`, `y` and `z` to its right, up and forward, read in the isoclinic frame at the
    eye; the tetrahedra's edges are lit struts, each in its colour and squared radius `styles`, `[tets * 6, 4]` in
    each tetrahedron's own order of its edges, as `edge_styles` gives them; each ray stepped at most `step` at a time,
    and never past a face; and at each frame the atoms and bonds of `crystals`, as `crystal_frames` or `no_crystals`
    gives them."""
    from examples.util.wgsl import Shader, supersampled

    mv = core.mv
    shader = Shader(supersampled() + core.geodesic_step_source + inside_source, core.ga,
                    dict(Point=core.Point, Plane=core.Plane, Rate=core.Rate, Step=core.Step,
                         Motor=core.Motor, Strain=core.Strain, Turning=core.Turning, Polarity=core.Polarity,
                         Translating=core.Translating),
                    dict(background=inside_background))
    ramps = waves.ramps                                                         # [corners] Plane+[tets]
    neighbours, entries = mesh.neighbours
    _, steps = core.invariant_generators()                                      # [3] Antivector-
    steps = steps.cast(core.Step)
    # The lamp, up and to the left of each eye.
    lamp = (mv.yw * 0.4 + mv.xw * -0.3).exp() >> core.origin                  # [] Point-
    pick = core.picking(mesh, vertices)
    half_height = float(np.tan(fov_radians / 2) * inside_shape[0] / inside_shape[1])
    n_levels, n_vertices = waves.traces.shape
    fixed = dict(
        ramps=stack(tuple(ramps.batch()), axis=-1).reshape(-1), neighbours=neighbours.astype(np.uint32).reshape(-1),
        entries=entries.astype(np.uint32).reshape(-1),
        tet_vertices=mesh.elements[3].astype(np.uint32).reshape(-1), strains=waves.strains.reshape(-1),
        strain_rates=waves.strain_rates.reshape(-1), traces=waves.traces.to_array().reshape(-1).astype(np.float32),
        turns=waves.turns.reshape(-1), n_vertices=n_vertices, n_tets=mesh.n_elements[3], n_levels=n_levels,
        first_time=waves.first_time, level_step=waves.level_step, origin=core.origin, equator=core.equator,
        polar=core.polar, translating=core.translating, half_height=half_height, step=step,
        max_steps=int(np.pi / step) + max_crossings, edge_style=styles, particle_ink=particle_ink, particle_radius_squared=particle_radius ** 2,
        bond_radius_squared=bond_radius ** 2)
    for frame, (eye_time, crystal) in enumerate(zip(eye_times, crystals)):
        camera = cameras[frame]                                                 # [] Motor+
        place = camera >> core.origin                                           # [] Point-
        view = (core.isoclinic(place).inverse() * camera) >> steps             # [3] Antivector-
        image = shader(inside_shape, eye_time=float(eye_time), start_place=place,
                       start_tet=int(pick(place[None])[0]), right=view[0], up=view[1], forward=view[2],
                       light=camera >> lamp, **crystal, **fixed)["image"]
        yield np.round(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def crystal_frames(mesh: Topology, vertices: Extensor, bonds: Topology, states: list, saturating: float
                   ) -> Iterator[dict]:
    """The crystals at each frame as the view from inside draws them: the atoms of `states`, `(Point-[atoms],
    Scalar+[bonds])` per frame, balls, and the edges of `bonds`, segments coloured by their strains, blue
    squeezed and red stretched, saturating at the strain `saturating`; each listed in the tetrahedra near it."""
    first_atoms, _ = states[0]
    lists = nearby(mesh, vertices, first_atoms, bonds)
    strain_inks = plt.get_cmap("coolwarm")
    for atoms, strains in states:
        tails, heads = bonds.incidence[1, 0] * atoms                           # Point-[bonds], Point-[bonds]
        inks = strain_inks(0.5 + 0.5 * np.clip(strains.to_array() / saturating, -1, 1)).astype(np.float32)
        yield dict(particles=atoms.batch().normalized(), n_particles=atoms.batch().shape[0],
                   bond_tails=tails.batch().normalized(), bond_heads=heads.batch().normalized(), bond_inks=inks,
                   n_bonds=bonds.n_elements[1], **lists)


def no_crystals(mesh: Topology) -> Iterator[dict]:
    """No crystals at any frame: no atoms, no bonds, and none listed in any tetrahedron."""
    nothing = core.mv.vector(np.zeros((0, 4))).dual()                           # [0] Point-
    unlisted, starts = np.zeros(0, dtype=np.uint32), np.zeros(mesh.n_elements[3] + 1, dtype=np.uint32)
    return repeat(dict(particles=nothing, n_particles=0, bond_tails=nothing, bond_heads=nothing,
                       bond_inks=np.zeros((0, 4), dtype=np.float32), n_bonds=0, particle_starts=starts,
                       tet_particles=unlisted, bond_starts=starts, tet_bonds=unlisted))


def nearby(mesh: Topology, vertices: Extensor, atoms: Extensor, bonds: Topology) -> dict[str, np.ndarray]:
    """For each tetrahedron, the atoms and the bonds near enough to it to be drawn in it, as lists one after another
    with each tetrahedron's start, `[tets + 1]` and `[listed]` for each: an atom within the tetrahedron's reach of its
    middle, the farthest of its corners, plus the atom's radius and room for its motion, and a bond whose middle is
    that near plus half its length."""
    from scipy.spatial import cKDTree

    mv = core.mv
    spanning = stack((mv.x, mv.y, mv.z, mv.w))                                  # [4] Plane+

    def directions(points: Extensor) -> np.ndarray:
        """The points' directions as unit coordinates `[n, 4]`, read by pairing with the planes across each
        direction."""
        coordinates = (spanning & points[..., None]).to_array()
        return coordinates / np.linalg.norm(coordinates, axis=-1, keepdims=True)

    corners = directions(vertices.batch())[mesh.elements[3]]                    # [tets, 4, 4]
    middles = directions((mesh.incidence[3, 0].mean(axis=0) * vertices).batch())   # [tets, 4]
    reach = np.arccos(np.clip(np.einsum("tk,tck->tc", middles, corners), -1, 1)).max(axis=-1)   # [tets]
    places = directions(atoms.batch())                                          # [atoms, 4]
    tails, heads = places[bonds.elements[1][:, 0]], places[bonds.elements[1][:, 1]]   # [bonds, 4], [bonds, 4]
    bond_middles = (tails + heads) / np.linalg.norm(tails + heads, axis=-1, keepdims=True)   # [bonds, 4]
    half_lengths = np.arccos(np.clip((tails * heads).sum(axis=-1), -1, 1)) / 2   # [bonds]
    tree = cKDTree(middles)

    def listed(centres: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each tetrahedron's start and the listed indices of the centres `[n, 4]` within its reach plus their radii
        `[n]`."""
        widest = reach.max() + radii.max()
        candidates = tree.query_ball_point(centres, 2 * np.sin(widest / 2))
        tets, indices = np.array([(tet, index) for index, near in enumerate(candidates) for tet in near],
                                 dtype=np.int64).reshape(-1, 2).T
        angles = np.arccos(np.clip((middles[tets] * centres[indices]).sum(axis=-1), -1, 1))
        kept = angles <= reach[tets] + radii[indices]
        tets, indices = tets[kept], indices[kept]
        order = np.lexsort((indices, tets))
        starts = np.searchsorted(tets[order], np.arange(mesh.n_elements[3] + 1))
        return starts.astype(np.uint32), indices[order].astype(np.uint32)

    particle_starts, tet_particles = listed(places, np.full(len(places), particle_radius + motion_room))
    bond_starts, tet_bonds = listed(bond_middles, half_lengths + bond_radius + motion_room)
    return dict(particle_starts=particle_starts, tet_particles=tet_particles, bond_starts=bond_starts,
                tet_bonds=tet_bonds)


# --- the sphere below the Hopf circles, seen from outside ---------------------------------------------------------------
# The picture's pixels across, half its width in the scene, and the camera's elevation and azimuth; the view, and its
# background.
base_pixels = 300
base_half = 1.15
base_elevation, base_azimuth = 0.45, 2.75
outside = solids.View(solids.aim(solids.origin, base_elevation, base_azimuth), np.full(2, base_half), base_pixels)
base_background = np.ones(3)
strain_map = "coolwarm"
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


def inks(stretches: core.Scalar, saturating: float) -> np.ndarray:
    """The colours of stretches `[n] Scalar+`, blue shortened and red lengthened, saturating at `saturating`."""
    return plt.get_cmap(strain_map)(0.5 + 0.5 * np.clip(stretches.to_array().reshape(-1) / saturating, -1, 1))[:, :3]


def base_view(sphere_mesh: Topology, sphere_points: core.Point, stretches: core.Scalar, decagon_bases: core.Point,
              atoms: core.Point, arcs: core.Point, strains: core.Scalar, light: core.Point, saturating: float
              ) -> np.ndarray:
    """The base from outside: the icosphere on the base's points `[points] Point-`, coloured by how much the circles
    above them stretch, `Scalar+[points]`, the decagons' base points `[decagons] Point-` in hues around the decagons,
    the gauges' atoms `[atoms] Point-` and their bonds' arcs `[bonds, arc points] Point-` coloured by their strains
    `[bonds] Scalar+`, and the light's paths `[rays, steps + 1] Point-`, all on the base."""
    pixel = outside.pixel

    points = lying(sphere_points, 0.0)                                          # [points] Point-
    faces = sphere_mesh.elements[2]
    normals = meshes.smooth(points, sphere_mesh)
    surface = meshes.samples(points, faces, inks(stretches, saturating)[faces], normals, outside, np.zeros(3),
                             meshes.plain, solids.gloss)
    n_bonds, n_arc_points = arcs.shape
    bonds = simplices.tubes(lying(arcs.reshape(-1), bond_width / 2), Topology.chains([n_arc_points] * n_bonds),
                            np.repeat(inks(strains, saturating), n_arc_points - 1, axis=0), bond_width, outside,
                            solids.gloss)
    n_rays, n_path_points = light.shape
    rays = simplices.tubes(lying(light.reshape(-1), light_width / 2), Topology.chains([n_path_points] * n_rays),
                           light_ink, light_width, outside, solids.gloss)
    dot_inks = hues(decagon_bases.shape[0])
    parts = [(solids.ball(lying(decagon_bases, 0.0), dot_radius * pixel)[:, None], dot_inks),
             (solids.ball(lying(atoms, bond_width / 2), atom_radius * pixel)[:, None],
              np.broadcast_to(atom_ink, (atoms.shape[0], 3)))]
    return solids.draw(parts, outside, base_background, inks=[], translucent=[],
                       meshes=[(surface, 1.0), (bonds, 1.0), (rays, 1.0)])

