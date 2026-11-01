"""Drawing for geodesics through a connection of motors: the chart's triangles faint, and the paths through them as
they grow, in the plane or in spacetime alike, read by the planes of the algebra's first two generators."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor, stack
from examples.util import charts
from examples.util.animation import capture
from examples.util.wgsl import simplices
from examples.util.wgsl.camera import panel, plane

# The picture's longer side in pixels, and the margin about the chart as a share of its larger half-width.
n_pixels = 300
margin = 0.06
# The triangles' ink and width, and the paths' width, in pixels.
edge_ink, edge_width = np.full(3, 0.82), 1.0
path_width = 2.0
# The steps between frames.
n_frame_steps = 2


def animate(mesh: simplices.Mesh, vertices: Extensor, paths: Extensor, extent: tuple[float, float, float, float]
            ) -> Iterator[np.ndarray]:
    """One frame every few steps of the paths `[paths, steps + 1] Point-`: each path from its start to where it has
    reached, over the chart's triangles, within the extent `(left, right, bottom, top)` of the readings by the planes
    of the algebra's first two generators, across and up."""
    algebra = vertices.algebra
    across, up = (plane(algebra, name) for name, square in zip(algebra.basis_names, algebra.signature) if square != 0)
    n_paths, n_steps = paths.shape
    inks = plt.get_cmap("plasma")(np.linspace(0.1, 0.8, n_paths))[:, :3]
    left, right, bottom, top = extent
    reach = margin * max(right - left, top - bottom) / 2
    scale = n_pixels / max(right - left, top - bottom)
    figure, ax = charts.subplots(round(scale * (right - left + 2 * reach)), round(scale * (top - bottom + 2 * reach)))
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(left - reach, right + reach), ylim=(bottom - reach, top + reach), aspect="equal")
    ax.set_axis_off()
    camera = panel(ax, across, up)
    triangles = simplices.strokes(vertices, mesh, edge_ink, edge_width, camera)

    def layers(reached: int) -> list:
        """The triangles, and each path's points up to the step reached, path by path, as chains of strokes."""
        trail = paths[:, : reached + 1].reshape(-1)                             # [paths * (reached + 1)] Point-
        chains = simplices.Mesh.chains([reached + 1] * n_paths)
        return [triangles, simplices.strokes(trail, chains, np.repeat(inks, reached, axis=0), path_width, camera)]

    frames = range(1, n_steps, n_frame_steps)
    image = simplices.overlay(ax, layers(frames[0]))
    yield capture(figure)
    for reached in frames[1:]:
        image.set_data(simplices.composited(layers(reached)))
        yield capture(figure)
    plt.close(figure)


# --- the view from inside a three-sphere, traced on the GPU ------------------------------------------------------------
# The picture's rows and columns, the view's opening, the step of length along each ray, the background, the struts'
# ink and radius.
inside_shape = (300, 400)
fov_radians = np.radians(100.0)
inside_step = 0.02
inside_background = np.array([0.035, 0.04, 0.06], dtype=np.float32)
# The guardrails' struts, each decagon in its own colour, and every other edge a thin light-grey ghost.
rail_radius = 0.028
ghost_ink, ghost_radius = np.array([0.72, 0.75, 0.80], dtype=np.float32), 0.006
# The atoms' ink and radius, the bonds' radius, and the strain that saturates their colour, blue squeezed, red stretched.
particle_ink, particle_radius = np.array([1.0, 0.86, 0.35], dtype=np.float32), 0.012
bond_radius, saturating_strain = 0.005, 0.15

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
    let across = screen(place) * half_height;
    var heading = (forward + right * across.x + up * across.y).normalized();
    let opp_c = array<u32, 6>(2u, 1u, 1u, 0u, 0u, 0u);
    let opp_d = array<u32, 6>(3u, 3u, 2u, 3u, 2u, 1u);
    var position = start_place;
    var tet = start_tet;
    var entry_face = 4u;
    var travelled = 0.0;
    for (var k = 0u; k < max_steps; k++) {
        let base = tet * 4u;
        let unit = unit_plane(tet);
        position = position / (unit & position);
        // The ray runs back in time: here it is the light that passed the eye's time less the distance travelled.
        let time = eye_time - travelled;
        let l = array<f32, 4>(ramps[base] & position, ramps[base + 1u] & position, ramps[base + 2u] & position,
                              ramps[base + 3u] & position);
        let waves = local_waves(tet, l, time);
        let v = velocity(tet, position, unit, heading, waves);
        let dl = array<f32, 4>((ramps[base] & v) * step, (ramps[base + 1u] & v) * step, (ramps[base + 2u] & v) * step,
                               (ramps[base + 3u] & v) * step);
        // The share of the step up to the facet the ray leaves by, never the one it came in by.
        var u_exit = 1.0;
        var exit_face = 4u;
        for (var f = 0u; f < 4u; f++) {
            if (f != entry_face && dl[f] < -1e-9) {
                let u_f = max(0.0, -l[f] / dl[f]);
                if (u_f < u_exit) {
                    u_exit = u_f;
                    exit_face = f;
                }
            }
        }
        var best_u = u_exit;
        var hit_edge = 6u;
        for (var e = 0u; e < 6u; e++) {
            let u_e = edge_test(l[opp_c[e]], l[opp_d[e]], dl[opp_c[e]], dl[opp_d[e]],
                                edge_style[tet * 6u + e].w, best_u);
            if (u_e < best_u) {
                best_u = u_e;
                hit_edge = e;
            }
        }
        // The test particles, balls about their places scaled onto the tetrahedron's plane, where the step comes nearest.
        let segment = v * step;
        var hit_particle = n_particles;
        for (var i = 0u; i < n_particles; i++) {
            let reading = unit & particles[i];
            if (reading > 0.0) {
                let offset = position - particles[i] / reading;
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
        // The bonds, segments between their ends scaled onto the tetrahedron's plane, where the step comes nearest.
        var hit_bond = n_bonds;
        var bond_share = 0.0;
        for (var i = 0u; i < n_bonds; i++) {
            let tail_reading = unit & bond_tails[i];
            let head_reading = unit & bond_heads[i];
            if (tail_reading > 0.0 && head_reading > 0.0) {
                let tail = bond_tails[i] / tail_reading;
                let along_bond = bond_heads[i] / head_reading - tail;
                let offset = position - tail;
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
            let hit = (position + segment * best_u).normalized();
            let distance = travelled + best_u * step;
            let fog = max(1.0 - distance / 3.14159, 0.0);
            let to_light = along(hit, light);
            let lit = bond_inks[hit_bond].xyz * (0.55 + 0.45 * max(along(hit, v).dot(to_light), 0.0));
            return mix(background, lit, fog);
        }
        if (hit_particle < n_particles) {
            let hit = (position + segment * best_u).normalized();
            let reading = unit & particles[hit_particle];
            let n_tan = along(hit, hit - particles[hit_particle].normalized());
            let to_eye = -along(hit, v);
            let to_light = along(hit, light);
            let diff1 = max(n_tan.dot(to_light), 0.0);
            let diff2 = max(n_tan.dot(to_eye), 0.0);
            let lit = particle_ink * (0.30 + 0.50 * diff1 + 0.25 * diff2);
            let distance = travelled + best_u * step;
            // Brightness falls linearly with distance, to nothing at the antipode.
            let fog = max(1.0 - distance / 3.14159, 0.0);
            return mix(background, lit, fog);
        }
        if (hit_edge < 6u) {
            let hit = (position + v * (step * best_u)).normalized();
            let ray = along(hit, v);
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
            let distance = travelled + best_u * step;
            // Brightness falls linearly with distance, to nothing at the antipode.
            let fog = max(1.0 - distance / 3.14159, 0.0);
            return mix(background, lit, fog);
        }
        let h = step * u_exit;
        heading = headed(heading, connection(tet, position & v, waves), waves, h);
        position = position + v * h;
        travelled = travelled + h;
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


def edge_styles(core, mesh, guardrails: np.ndarray) -> np.ndarray:
    """Each edge of each tetrahedron's colour and squared strut radius, `[tets * 6, 4]`, in the tetrahedron's own order
    of its edges: along each guardrail `[decagons, 10]`, a thick strut in the guardrail's colour, evenly spaced in hue;
    every other edge a thin ghost."""
    from matplotlib.colors import hsv_to_rgb

    styles = np.tile(np.append(ghost_ink, ghost_radius ** 2), (len(mesh.edges), 1)).astype(np.float32)
    inks = hsv_to_rgb(np.stack([np.arange(len(guardrails)) / len(guardrails), np.full(len(guardrails), 0.75),
                                np.full(len(guardrails), 0.95)], axis=-1))
    edges = np.sort(mesh.edges, axis=-1)
    for rail, ink in zip(guardrails, inks):
        steps = np.sort(np.stack((rail, np.roll(rail, -1)), axis=-1), axis=-1)       # [10, 2]
        styles[(edges[:, None, :] == steps[None]).all(axis=-1).any(axis=-1)] = np.append(ink, rail_radius ** 2)
    edge_ids = core.mesh_module.as_scalar(np.arange(len(mesh.edges))).field()
    tet_edges = np.rint((mesh.incidence[3, 1] * edge_ids).to_array()).astype(int).T   # [tets, 6]
    return styles[tet_edges.reshape(-1)]


def inside(core, mesh, vertices: Extensor, camera: Extensor, gauge, waves, eye_times: np.ndarray, guardrails: np.ndarray,
           bonds, states: list) -> Iterator[np.ndarray]:
    """Frames of the view from inside a three-sphere meshed by tetrahedra and carrying the stepped gravitational waves
    `waves`, one at each of the eye's times: the camera, a motor, carries the origin to the eye and the origin's steps
    across `x`, `y` and `z` to its right, up and forward, read in the frame the vertices' frames give the eye,
    `gauge(place)`; the tetrahedra's edges are lit struts, the guardrails `[decagons, 10]` in colour; at each time the
    atoms of `states`, `(Point-[atoms], Scalar+[bonds])` per frame, are balls and their `bonds`, a graph `Mesh`, segments
    coloured by their strains."""
    from examples.util.wgsl import Shader, supersampled

    mv = core.mv
    shader = Shader(supersampled() + core.geodesic_step_source + inside_source, core.ga,
                    dict(Point=core.Point, Plane=core.Plane, Rate=core.Rate, Line=core.Line, Step=core.Step,
                         Strain=core.Strain, Turning=core.Turning, Across=core.Across, Translating=core.Translating),
                    dict(background=inside_background))
    ramps = waves.ramps                                                         # [corners] Plane+[tets]
    neighbours, entries = mesh.neighbours
    spanning = stack((mv.x, mv.y, mv.z, mv.w)).dual()                         # [4] Point-
    _, _, steps = core.invariant_generators()                                   # [3] Antivector-
    steps = steps.cast(core.Step)
    place = camera >> core.origin                                               # [] Point-
    view = (gauge(place).inverse() * camera) >> steps                          # [3] Antivector-
    half_height = float(np.tan(fov_radians / 2) * inside_shape[0] / inside_shape[1])
    n_levels, n_vertices = waves.traces.shape
    fixed = dict(
        ramps=stack(tuple(ramps.batch()), axis=-1).reshape(-1), neighbours=neighbours.astype(np.uint32).reshape(-1),
        entries=entries.astype(np.uint32).reshape(-1),
        coframes=stack(tuple(waves.coframes.batch()), axis=-1).reshape(-1),
        tet_vertices=mesh.tets.astype(np.uint32).reshape(-1), strains=waves.strains.reshape(-1),
        strain_rates=waves.strain_rates.reshape(-1), traces=waves.traces.to_array().reshape(-1).astype(np.float32),
        turns=waves.turns.reshape(-1), n_vertices=n_vertices, n_tets=len(mesh.tets), n_levels=n_levels,
        first_time=waves.first_time, level_step=waves.level_step, origin=core.origin, infinity=core.infinity,
        basis_0=spanning[0], basis_1=spanning[1], basis_2=spanning[2], basis_3=spanning[3], step_x=steps[0],
        step_y=steps[1], step_z=steps[2], across=core.across, translating=core.translating, start_place=place,
        start_tet=int(mesh.picking(vertices)(place[None])[0]), right=view[0], up=view[1], forward=view[2],
        half_height=half_height, step=inside_step, max_steps=int(4 * np.pi / inside_step),
        light=camera >> ((mv.yw * 0.4 + mv.xw * -0.3).exp() >> core.origin),
        edge_style=edge_styles(core, mesh, guardrails), particle_ink=particle_ink,
        particle_radius_squared=particle_radius ** 2, bond_radius_squared=bond_radius ** 2)
    strain_inks = plt.get_cmap("coolwarm")
    for eye_time, (atoms, strains) in zip(eye_times, states):
        tails, heads = bonds.incidence[1, 0] * atoms                           # Point-[bonds], Point-[bonds]
        inks = strain_inks(0.5 + 0.5 * np.clip(strains.to_array() / saturating_strain, -1, 1)).astype(np.float32)
        image = shader(inside_shape, eye_time=float(eye_time), particles=atoms.batch().normalized(),
                       n_particles=atoms.batch().shape[0], bond_tails=tails.batch().normalized(),
                       bond_heads=heads.batch().normalized(), bond_inks=inks, n_bonds=len(bonds.edges), **fixed)["image"]
        yield np.round(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


# --- the gate for stepping the motors: spectra of forms ---------------------------------------------------------------
gate_width, gate_height = 560, 230


def spectrum(correlation: Extensor, Input) -> np.ndarray:
    """The eigenvalues of the form `x & correlation(y)` of a dense field correlation over inputs of type `Input`, in
    the blades of the input, read by the regressive product and symmetrized: by Sylvester's law their signs do not
    depend on the basis."""
    blocks = np.asarray(correlation.kernel)                                     # [sites, sites, out blades, in blades]
    context = correlation.context
    inputs = context.extensor(Input, np.eye(blocks.shape[-1]))                  # [in blades] Input
    outputs = context.extensor(correlation.gatype.derive.plain.derive.output, np.eye(blocks.shape[-2]))
    pairing = (inputs[:, None] & outputs[None, :]).to_array()                   # [in blades, out blades]
    form = np.einsum("ik,abkj->aibj", pairing, blocks).reshape(blocks.shape[0] * blocks.shape[-1], -1)
    return np.linalg.eigvalsh((form + form.T) / 2)


@charts.style()
def gate(panels: list[tuple[np.ndarray, int]]) -> plt.Figure:
    """Each form's eigenvalues, sorted, against its expected count of null directions `[(values, expected)]`: the
    null ones in red where there are more than expected."""
    figure, axes = charts.subplots(gate_width, gate_height, ncols=len(panels))
    for ax, (values, expected) in zip(axes, panels):
        small = 1e-9 * np.abs(values).max()
        null = np.abs(values) <= small
        order = np.argsort(values)
        ax.axhline(0, color="0.6", linewidth=charts.FRAME_WIDTH)
        ax.scatter(np.arange(len(values)), values[order], s=2, c=np.where(null[order], "#d0453b", "#3b6fb6"))
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xticks([0, len(values)])
        ax.set_yticks([-1, 0, 1])
        ax.text(0.03, 0.95, f"null {null.sum()}, expected {expected}", transform=ax.transAxes, va="top")
    figure.tight_layout()
    return figure


# --- the stepped waves: their spectrum, and the squash ringing ---------------------------------------------------------
waves_width, waves_height = 560, 220


@charts.style()
def waves(spectrum: np.ndarray, squash_squared: float, times: np.ndarray, shares: np.ndarray, swing: float
          ) -> plt.Figure:
    """The waves' frequencies squared, sorted, against the continuum's, `n * n - 1` for n from three; and the squash's
    share of the stepped strains against the cosine at its own frequency."""
    figure, (levels, ringing) = charts.subplots(waves_width, waves_height, ncols=2)
    stiff = np.sort(spectrum[spectrum > 1e-6 * spectrum.max()])
    for n in range(3, 6):
        levels.axhline(n * n - 1, color="0.8", linewidth=charts.FRAME_WIDTH)
    levels.scatter(np.arange(len(stiff)), stiff, s=3, color="#3b6fb6")
    levels.set_yticks([0, 8, 15, 24])
    levels.set_xticks([0, len(stiff)])
    levels.set_ylabel("frequency squared")
    ringing.axhline(0, color="0.6", linewidth=charts.FRAME_WIDTH)
    ringing.plot(times, swing * np.cos(np.sqrt(squash_squared) * times), color="0.75")
    ringing.plot(times, shares, color="#d0453b", linewidth=charts.LINE_WIDTH * 0.8, linestyle=(0, (2, 2)))
    ringing.set_yticks([-swing, 0, swing])
    ringing.set_xticks([0, times[-1]])
    ringing.set_xlabel("time")
    figure.tight_layout()
    return figure
