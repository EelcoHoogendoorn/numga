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

from examples.relativity.closed_universe import core
from examples.util.topology import Topology, lookup_simplices
from examples.util.wgsl import meshes, simplices, solids


# --- the view from inside a three-sphere, traced on the GPU ------------------------------------------------------------
# The picture's rows and columns, and the view's opening; the legs a ray may take, past any its half turn needs.
inside_shape = (300, 400)
fov_radians = np.radians(100.0)
max_legs = 1024
inside_background = np.array([0.035, 0.04, 0.06], dtype=np.float32)
# The struts' radii, as shares of the mesh's mean edge, in lengths on the tetrahedra's charts: a coloured strut, and
# every other edge a thin light-grey ghost. A helix's rungs, between its rails.
rail_share = 0.014
ghost_ink, ghost_share = np.array([0.72, 0.75, 0.80], dtype=np.float32), 0.003
rung_ink = np.array([0.85, 0.87, 0.90])
# The atoms' ink and radius, the bonds' radius, and the strain that saturates their colour, blue squeezed, red
# stretched, from the colour map that colours every strain and stretch.
particle_ink, particle_radius = np.array([1.0, 0.86, 0.35], dtype=np.float32), 0.012
bond_radius, saturating_strain = 0.005, 0.15
strain_map = "coolwarm"
# How far an atom may move from where it starts and still be drawn in the tetrahedra listed for it.
motion_room = 0.03
# The lamp, up and to the left of each eye: the motor that carries the origin to it, before the camera.
lamp_placement = (core.mv.yw * 0.4 - core.mv.xw * 0.3).exp()                  # [] Motor+

inside_source = """
var<uniform> start_place: Point;
var<uniform> start_tet: u32;
var<uniform> right: Step;
var<uniform> up: Step;
var<uniform> forward: Step;
var<uniform> eye_time: f32;
var<uniform> half_height: f32;
var<uniform> max_legs: u32;
var<uniform> light: Point;
var<storage, read> struts: array<vec4<f32>>;
var<storage, read> strut_tails: array<Point>;
var<storage, read> strut_heads: array<Point>;
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

// The direction at a place on the sphere toward a point or along a direction, square to the place.
fn tangent(place: Point, toward: Point) -> Point {
    return (toward - place * place.dot(toward)).normalized();
}

// A lit colour seen from a distance along the ray, its brightness falling linearly to nothing at the antipode.
fn faded(lit: Sample, distance: f32) -> Sample {
    return mix(background, lit, max(1.0 - distance / 3.14159, 0.0));
}

fn sample(pixel: vec2<f32>) -> Sample {
    let screen_place = screen(pixel) * half_height;
    var heading = (forward + right * screen_place.x + up * screen_place.y).normalized();
    var place = start_place;
    var tet = start_tet;
    var entry = 4u;
    var travelled = 0.0;
    for (var k = 0u; k < max_legs; k++) {
        // The ray runs back in time: here it is the light that passed the eye's time less the distance travelled.
        let leg = next_leg(place, heading, tet, entry, eye_time - travelled);
        // The struts about the tetrahedron's edges, tubes about each edge's line on the chart: where the leg first
        // comes within a strut's radius of the line, from the parts of its offset and its segment square to the edge.
        var nearest = 1.0;
        var hit_strut = 6u;
        var strut_normal = leg.chart;
        for (var edge = 0u; edge < 6u; edge++) {
            let index = tet * 6u + edge;
            let tail = strut_tails[index];
            let edge_span = strut_heads[index] - tail;
            let offset = leg.chart - tail;
            let radial_offset = offset - edge_span * (offset.dot(edge_span) / edge_span.dot(edge_span));
            let radial_segment = leg.segment - edge_span * (leg.segment.dot(edge_span) / edge_span.dot(edge_span));
            let a = radial_segment.dot(radial_segment);
            let b = radial_offset.dot(radial_segment);
            let c = radial_offset.dot(radial_offset) - struts[index].w;
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
        // The atoms near the tetrahedron, balls about their places scaled onto its chart, where the leg comes nearest.
        var hit_particle = n_particles;
        for (var listed = particle_starts[tet]; listed < particle_starts[tet + 1u]; listed++) {
            let i = tet_particles[listed];
            let reading = leg.plane & particles[i];
            if (reading > 0.0) {
                let offset = leg.chart - particles[i] / reading;
                let a = leg.segment.dot(leg.segment);
                let b = offset.dot(leg.segment);
                let c = offset.dot(offset) - particle_radius_squared;
                let disc = b * b - a * c;
                if (c > 0.0 && disc >= 0.0) {
                    let share = (-b - sqrt(disc)) / a;
                    if (share >= 0.0 && share < nearest) {
                        nearest = share;
                        hit_particle = i;
                    }
                }
            }
        }
        // The bonds near the tetrahedron, segments between their ends scaled onto its chart, where the leg comes
        // nearest.
        var hit_bond = n_bonds;
        for (var listed = bond_starts[tet]; listed < bond_starts[tet + 1u]; listed++) {
            let i = tet_bonds[listed];
            let tail_reading = leg.plane & bond_tails[i];
            let head_reading = leg.plane & bond_heads[i];
            if (tail_reading > 0.0 && head_reading > 0.0) {
                let tail = bond_tails[i] / tail_reading;
                let bond_span = bond_heads[i] / head_reading - tail;
                let offset = leg.chart - tail;
                let a = leg.segment.dot(leg.segment);
                let b = leg.segment.dot(bond_span);
                let c = bond_span.dot(bond_span);
                let d = leg.segment.dot(offset);
                let e = bond_span.dot(offset);
                let s = clamp((a * e - b * d) / max(a * c - b * b, 1e-20), 0.0, 1.0);
                let u_b = clamp((b * s - d) / a, 0.0, nearest);
                let gap = offset + leg.segment * u_b - bond_span * s;
                if (gap.dot(gap) < bond_radius_squared && u_b < nearest) {
                    nearest = u_b;
                    hit_bond = i;
                }
            }
        }
        let hit = (leg.chart + leg.segment * nearest).normalized();
        let distance = travelled + nearest * leg.length;
        let to_light = tangent(hit, light);
        if (hit_bond < n_bonds) {
            return faded(bond_inks[hit_bond].xyz * (0.55 + 0.45 * max(tangent(hit, leg.segment).dot(to_light), 0.0)),
                         distance);
        }
        if (hit_particle < n_particles) {
            let normal = tangent(hit, hit - particles[hit_particle].normalized());
            let to_eye = -tangent(hit, leg.segment);
            let lit = particle_ink * (0.30 + 0.50 * max(normal.dot(to_light), 0.0)
                      + 0.25 * max(normal.dot(to_eye), 0.0));
            return faded(lit, distance);
        }
        if (hit_strut < 6u) {
            let to_eye = -tangent(hit, leg.segment);
            let normal = tangent(hit, strut_normal);
            let facing = normal * select(-1.0, 1.0, normal.dot(to_eye) >= 0.0);
            let halfway = tangent(hit, to_eye + to_light);
            let glint = pow(max(facing.dot(halfway), 0.0), 36.0);
            let lit = struts[tet * 6u + hit_strut].xyz * (0.22 + 0.56 * max(facing.dot(to_light), 0.0)
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


def hues(n: int) -> np.ndarray:
    """So many colours evenly round the colour wheel, `[n, 3]`."""
    return hsv_to_rgb(np.stack([np.arange(n) / n, np.full(n, 0.75), np.full(n, 0.95)], axis=-1))


def edge_styles(mesh: Topology, vertices: core.Point, struts: np.ndarray, inks: np.ndarray) -> np.ndarray:
    """Each edge of each tetrahedron's colour and squared strut radius, `[tets * 6, 4]`, in the tetrahedron's own order
    of its edges: the edges `struts`, vertex pairs `[struts, 2]`, thick struts in their inks `[struts, 3]`, and every
    other edge a thin ghost, each as thick as its share of the mesh's mean edge."""
    chords = mesh.d[0] * vertices                                               # Point-[edges]
    mean_edge = chords.dot(chords).square_root().to_array().mean()
    edges = mesh.elements[1]                                                    # [edges, 2]
    styles = np.tile(np.append(ghost_ink, (ghost_share * mean_edge) ** 2), (len(edges), 1)).astype(np.float32)
    # Each strut's edge, and each tetrahedron's six, found by their ends.
    styles[lookup_simplices(edges, np.sort(struts, axis=-1))] = np.concatenate(
        [inks, np.full((len(struts), 1), (rail_share * mean_edge) ** 2)], axis=-1)
    tet_edges = lookup_simplices(edges, np.sort(mesh.elements[3][:, mesh.corners[3, 1].T], axis=-1))   # [tets, 6]
    return styles[tet_edges.reshape(-1)]


def polygon_struts(polygons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The sides of closed polygons of vertices `[polygons, sides]` as struts `[polygons * sides, 2]`, each polygon in
    its own colour round the colour wheel, `[polygons * sides, 3]`."""
    sides = np.stack([polygons, np.roll(polygons, -1, axis=-1)], axis=-1).reshape(-1, 2)
    return sides, np.repeat(hues(len(polygons)), polygons.shape[1], axis=0)


def round_values(mesh: Topology, vertices: core.Point) -> dict:
    """What the core's walk through the round sphere reads: the tetrahedra's ramps and neighbours, the origin, the
    rates that move it, and a leg's reach."""
    neighbours, entries = mesh.neighbours
    return dict(ramps=core.coframe(mesh, vertices), neighbours=neighbours, entries=entries, n_tets=mesh.n_elements[3],
                origin=core.origin, moving=core.moving, reach=core.round_reach)


def wave_values(mesh: Topology, waves: core.Waves) -> dict:
    """What the core's walk through the waves reads: the waves' stored moments, the tetrahedra's ramps, corners and
    neighbours, the origin, and the maps of its steps."""
    neighbours, entries = mesh.neighbours
    _, n_vertices = waves.traces.shape
    return dict(ramps=waves.ramps, neighbours=neighbours, entries=entries, tet_vertices=mesh.elements[3],
                strains=waves.strains, strain_rates=waves.strain_rates, traces=waves.traces, turns=waves.turns,
                n_vertices=n_vertices, n_tets=mesh.n_elements[3], first_time=waves.first_time,
                level_step=waves.level_step, origin=core.origin, translating=core.translating, moving=core.moving,
                turning=core.turning, step=core.wave_step)


def inside(walk: str, walk_values: dict, mesh: Topology, vertices: core.Point, cameras: core.Motor,
           eye_times: np.ndarray, styles: np.ndarray, crystals: Iterable[dict]) -> Iterator[np.ndarray]:
    """Frames of the view from inside a round three-sphere meshed by tetrahedra, one for each camera and time: each
    camera, a motor `[frames] Motor+`, carries the origin to the eye and the origin's steps along `x`, `y` and `z` to
    its right, up and forward, read in the isoclinic frame at the eye; each ray is walked back from the eye leg by leg
    by the WGSL `walk`, which defines `next_leg` and its `Leg` and reads `walk_values`; the tetrahedra's edges are lit
    struts, each in its colour and squared radius `styles`, `[tets * 6, 4]` in each tetrahedron's own order of its
    edges; and at each frame the atoms and bonds of `crystals`."""
    from examples.util.wgsl import Shader, supersampled

    mv = core.mv
    shader = Shader(supersampled() + walk + inside_source, core.ga,
                    dict(Point=core.Point, Plane=core.Plane, Rate=core.Rate, Step=core.Step, Motor=core.Motor,
                         Strain=core.Strain, Turning=core.Turning, Rolling=core.Rolling, Moving=core.Moving,
                         Translating=core.Translating),
                    dict(background=inside_background))
    # The struts' ends: each tetrahedron's edges' tails and heads, in its own order of its edges.
    tails, heads = mesh.incidence[1, 0] * vertices                              # Point-[edges], Point-[edges]
    strut_tails = stack(tuple((mesh.incidence[3, 1] * tails).batch()), axis=-1)   # [tets, 6] Point-
    strut_heads = stack(tuple((mesh.incidence[3, 1] * heads).batch()), axis=-1)   # [tets, 6] Point-
    axes = stack((mv.x, mv.y, mv.z)).dual()                                     # [3] Point-
    lamp = lamp_placement >> core.origin                                        # [] Point-
    pick = core.picking(mesh, vertices)
    half_height = float(np.tan(fov_radians / 2) * inside_shape[0] / inside_shape[1])
    fixed = walk_values | dict(struts=styles, strut_tails=strut_tails, strut_heads=strut_heads,
                               half_height=half_height, max_legs=max_legs, particle_ink=particle_ink,
                               particle_radius_squared=particle_radius ** 2, bond_radius_squared=bond_radius ** 2)
    for frame, (eye_time, crystal) in enumerate(zip(eye_times, crystals)):
        camera = cameras[frame]                                                 # [] Motor+
        place = camera >> core.origin                                           # [] Point-
        view = (core.isoclinic(place).inverse() * camera) >> axes              # [3] Point-
        image = shader(inside_shape, eye_time=float(eye_time), start_place=place,
                       start_tet=int(pick(place[None])[0]), right=view[0], up=view[1], forward=view[2],
                       light=camera >> lamp, **crystal, **fixed)["image"]
        yield np.round(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def plain_view(walk: str, walk_values: dict, mesh: Topology, vertices: core.Point, cameras: core.Motor
               ) -> Iterator[np.ndarray]:
    """Frames of the view from inside the round three-sphere, as `inside` draws them, along the cameras `[frames]
    Motor+`, every edge a thin ghost."""
    no_struts = np.zeros((0, 2), dtype=int), np.zeros((0, 3))
    return inside(walk, walk_values, mesh, vertices, cameras, np.zeros(cameras.shape[0]),
                  edge_styles(mesh, vertices, *no_struts), repeat(no_crystals(mesh)))


def polygon_view(walk: str, walk_values: dict, mesh: Topology, vertices: core.Point, cameras: core.Motor,
                 polygons: np.ndarray) -> Iterator[np.ndarray]:
    """Frames of the view from inside the round three-sphere, as `inside` draws them, along the cameras `[frames]
    Motor+`: the closed polygons of edges, vertex indices `[polygons, sides]`, as struts each in its own colour round
    the colour wheel, every other edge a thin ghost."""
    return inside(walk, walk_values, mesh, vertices, cameras, np.zeros(cameras.shape[0]),
                  edge_styles(mesh, vertices, *polygon_struts(polygons)), repeat(no_crystals(mesh)))


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
                         np.concatenate([side_inks, np.broadcast_to(rung_ink, (len(rungs), 3))]))
    return inside(walk, walk_values, mesh, vertices, cameras, np.zeros(cameras.shape[0]), styles,
                  repeat(no_crystals(mesh)))


def waves_view(mesh: Topology, vertices: core.Point, waves: core.Waves, eye: core.Motor, eye_times: np.ndarray,
               rails: np.ndarray, bonds: Topology, atoms: core.Point, states: Iterable[tuple[core.Point, core.Scalar]]
               ) -> Iterator[np.ndarray]:
    """Frames of the view from inside the round three-sphere carrying the gravitational waves `waves`, as `inside`
    draws them through the core's walk through the waves, from the camera `eye` at each of the `eye_times`: the closed
    polygons of edges `rails`, vertex indices `[rails, sides]`, as struts each in its own colour, every other edge a
    thin ghost; and the crystals, their atoms at rest `atoms` and at each frame the atoms and their bonds' strains of
    `states`, `(Point-[atoms], Scalar+[bonds])`, the bonds the edges of `bonds`, coloured by their strains."""
    return inside(core.wave_walk_source, wave_values(mesh, waves), mesh, vertices,
                  eye.broadcast_to(eye_times.shape), eye_times, edge_styles(mesh, vertices, *polygon_struts(rails)),
                  crystal_frames(mesh, vertices, bonds, atoms, states))


def crystal_frames(mesh: Topology, vertices: core.Point, bonds: Topology, atoms: core.Point,
                   states: Iterable[tuple[core.Point, core.Scalar]]) -> Iterator[dict]:
    """The crystals at each frame as the view from inside draws them: the atoms of `states`, `(Point-[atoms],
    Scalar+[bonds])` per frame, balls, and the edges of `bonds`, segments coloured by their strains, blue squeezed and
    red stretched; each listed in the tetrahedra near the atoms at rest, `atoms`."""
    lists = nearby(mesh, vertices, atoms, bonds)
    for positions, strains in states:
        tails, heads = bonds.incidence[1, 0] * positions                       # Point-[bonds], Point-[bonds]
        inks = plt.get_cmap(strain_map)(0.5 + 0.5 * np.clip(strains.to_array() / saturating_strain, -1, 1))
        yield dict(particles=positions.batch().normalized(), n_particles=positions.batch().shape[0],
                   bond_tails=tails.batch().normalized(), bond_heads=heads.batch().normalized(),
                   bond_inks=inks.astype(np.float32), n_bonds=bonds.n_elements[1], **lists)


def no_crystals(mesh: Topology) -> dict:
    """No crystals: no atoms, no bonds, and none listed in any tetrahedron."""
    nothing = core.mv.vector(np.zeros((0, 4))).dual()                           # [0] Point-
    unlisted, starts = np.zeros(0, dtype=np.uint32), np.zeros(mesh.n_elements[3] + 1, dtype=np.uint32)
    return dict(particles=nothing, n_particles=0, bond_tails=nothing, bond_heads=nothing,
                bond_inks=np.zeros((0, 4), dtype=np.float32), n_bonds=0, particle_starts=starts,
                tet_particles=unlisted, bond_starts=starts, tet_bonds=unlisted)


def nearby(mesh: Topology, vertices: core.Point, atoms: core.Point, bonds: Topology) -> dict[str, np.ndarray]:
    """For each tetrahedron, the atoms and the bonds near enough to it to be drawn in it, as lists one after another
    with each tetrahedron's start, `[tets + 1]` and `[listed]` for each: an atom within the tetrahedron's reach of its
    middle, the farthest of its corners, plus the atom's radius and room for its motion, and a bond whose middle is
    that near plus half its length. The angles are the algebra's; the lists are gathered outside it, by a k-d tree
    over the middles' coordinates, read by pairing with the planes across the four directions."""
    from scipy.spatial import cKDTree

    coordinate_planes = core.mv.vector(np.eye(4))                               # [4] Plane+
    middles = (mesh.incidence[3, 0].mean(axis=0) * vertices).normalized().batch()   # [tets] Point-
    corners = (mesh.incidence[3, 0] * vertices).batch()                         # [corners, tets] Point-
    reach = middles[None].dot(corners).arccos().to_array().max(axis=0)          # [tets]
    tails, heads = bonds.incidence[1, 0] * atoms                                # Point-[bonds], Point-[bonds]
    places = atoms.batch()                                                      # [atoms] Point-
    bond_middles = (tails + heads).normalized().batch()                        # [bonds] Point-
    half_lengths = (tails.dot(heads).arccos() / 2).to_array()                  # [bonds]
    tree = cKDTree((coordinate_planes & middles[:, None]).to_array())

    def listed(centres: core.Point, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each tetrahedron's start and the listed indices of the centres `[n] Point-` within its reach plus their
        radii `[n]`."""
        widest = reach.max() + radii.max()
        candidates = tree.query_ball_point((coordinate_planes & centres[:, None]).to_array(), 2 * np.sin(widest / 2))
        tets, indices = np.array([(tet, index) for index, near in enumerate(candidates) for tet in near],
                                 dtype=np.int64).reshape(-1, 2).T
        kept = middles[tets].dot(centres[indices]).arccos() <= reach[tets] + radii[indices]
        tets, indices = tets[kept], indices[kept]
        order = np.lexsort((indices, tets))
        starts = np.searchsorted(tets[order], np.arange(mesh.n_elements[3] + 1))
        return starts.astype(np.uint32), indices[order].astype(np.uint32)

    particle_starts, tet_particles = listed(places, np.full(places.shape[0], particle_radius + motion_room))
    bond_starts, tet_bonds = listed(bond_middles, half_lengths + bond_radius + motion_room)
    return dict(particle_starts=particle_starts, tet_particles=tet_particles, bond_starts=bond_starts,
                tet_bonds=tet_bonds)


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
    """The base from outside: the icosphere on the base's points `[points] Point-`, the decagons' base points
    `[decagons] Point-` in hues around the decagons; and at a frame the icosphere coloured by how much the circles
    above its points stretch, `Scalar+[points]`, the gauges' atoms `[atoms] Point-` and their bonds' arcs `[bonds, arc
    points] Point-` coloured by their strains `[bonds] Scalar+`, and the light's paths `[rays, steps + 1] Point-`, all
    on the base."""
    pixel = outside.pixel

    points = lying(sphere_points, 0.0)                                          # [points] Point-
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
