"""The 600-cell's gravitational waves seen on the sphere below its Hopf circles, the guardrails' decagons: a quadrupole
travelling round the base, the squash and the squash turned an eighth of a turn a quarter swing behind it. The base is
shaded by how much the circles above it stretch, the decagons stand on it as their base points, triangle gauges ring
on it, tuned to the waves, and a bundle of light crosses it."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numga import stack
from numga.algebras import VGA3D

from examples import instantiate
from examples.relativity.hopf_projection import core
from examples.relativity.motor_connection import scenarios as rolling

three_sphere = core.three_sphere
flat = instantiate("examples.util.mesh", VGA3D)

# The base drawn as an icosphere, the icosahedron's faces split in four so many times; the points sampled round each
# circle for its average, so many that they fall between the tenth turns that carry the 600-cell onto itself; the
# frames over one swing of the waves, as the view from inside has them; the swing of space.
n_icosphere_levels = 3
n_circle_samples = 7
n_frames = rolling.n_wave_frames
swing = 0.085
# The squash stretches space along x and squeezes it along y in the frames; turned an eighth of a turn about z, it
# stretches and squeezes along the diagonals between them, and a quarter swing behind the squash, the two make a
# quadrupole that turns round about z, travelling round the base.
eighth = (three_sphere.mv.xy * (-np.pi / 8)).exp()                              # [] Motor+
# The triangle gauges, one in each face of the icosahedron the decagons' base points make, its corners this share of
# the way from the face's middle to its corners. Each atom is held to its place, a free particle at rest, by a spring
# swinging at this share of the waves' frequency, damped critically, so that a gauge keeps its place and its turn
# while it rings, rather than taking up the travelling waves' turning; its bonds put the triangle's shear swing at the
# waves' frequency, damped to this share of critical. The points sampled along each bond for its arc.
gauge_share = 0.4
holding_share = 1 / 4
gauge_damping = 0.4
n_arc_points = 12
# Light: a narrow bundle of so many rays set out across the circles from the circle above the base point along z,
# aimed about the direction toward the base point along y, which faces the viewer, and spread over this angle, as
# each frame begins; traced in steps of this length for a quarter turn, the base's half from pole to pole.
n_rays = 7
light_spread = np.radians(18)
light_step = 0.02
n_light_steps = int(np.pi / 2 / light_step)
# What the gauges and the light are tuned to show: a bond's strain from one end of its swing to the other, and a ray's
# largest sway on the screen from its undisturbed path, as shares of the picture.
target_strain = 0.2
target_sways = (0.1, 0.2)


def base_sphere() -> tuple[flat.Mesh, flat.Point, three_sphere.Point]:
    """The icosphere drawn as the base and its points `Point-[points]` in the algebra of space; and the same points as
    unit steps of the three-sphere's origin, `[points] Point-`."""
    mesh, points = flat.Mesh.icosphere(n_icosphere_levels)
    coordinates = (flat.mv.vector(np.eye(3)) & points.batch()[..., None]).to_array()   # [points, 3]
    bases = three_sphere.mv.vector(np.pad(coordinates, ((0, 0), (0, 1)))).dual().cast(three_sphere.Step)
    return mesh, points, bases


def triangle_gauges(decagon_bases: three_sphere.Point, circling: three_sphere.Rate
                    ) -> tuple[three_sphere.Mesh, three_sphere.Point]:
    """The triangle gauges' bonds, a graph over their atoms, and the atoms, `Point-[atoms]` on the three-sphere: in
    each face of the icosahedron the decagons' base points make, the corners drawn toward the face's middle, lifted
    from the circle above the middle across the circles, so that the bonds run across them too."""
    from scipy.spatial import ConvexHull

    coordinates = (three_sphere.mv.vector(np.eye(4))[:3] & decagon_bases[..., None]).to_array()   # [decagons, 3]
    faces = ConvexHull(coordinates).simplices                                   # [faces, 3]
    corners = decagon_bases[faces]                                              # [faces, 3] Point-
    middles = corners.sum(axis=-1).normalized()                                 # [faces] Point-
    atoms = (middles[:, None] + gauge_share * (corners - middles[:, None])).normalized()   # [faces, 3] Point-
    # Each atom's direction along the base from its face's middle, and the cosine of the angle between them: the step
    # across the circles that the direction is the image of carries the middle's lift half that angle, which the Hopf
    # map doubles.
    cosines = three_sphere.polar(middles[:, None]) & atoms                     # [faces, 3] Scalar+
    directions = (atoms - middles[:, None] * cosines).cast(three_sphere.Step).normalized()   # [faces, 3] Point-
    moves = (three_sphere.moving(core.across(middles[:, None], directions)) * (-cosines.arccos() / 4)).exp()   # [faces, 3] Motor+
    lifted = (three_sphere.isoclinic(core.lifts(middles, circling))[:, None] * moves) >> three_sphere.origin   # [faces, 3] Point-
    # Each triangle's three sides, by its atoms' places, three to a face.
    firsts = 3 * np.arange(len(faces))[:, None] + np.arange(3)[None]           # [faces, 3]
    bonds = np.stack([firsts, firsts - firsts % 3 + (firsts + 1) % 3], axis=-1).reshape(-1, 2)
    return three_sphere.Mesh(3 * len(faces), bonds), lifted.reshape(-1).field()


def bond_arcs(bonds: three_sphere.Mesh, positions: three_sphere.Point, circling: three_sphere.Rate
              ) -> three_sphere.Point:
    """Each bond's great arc between its atoms on the three-sphere, `Point-[atoms]`, seen on the base: so many points
    along it, its chord's points moved out onto the sphere, and their base points, `[bonds, arc points] Point-`."""
    tails, heads = bonds.incidence[1, 0] * positions                            # Point-[bonds], Point-[bonds]
    shares = np.linspace(0, 1, n_arc_points)                                    # [arc points]
    along = (tails.batch()[:, None] * (1 - shares) + heads.batch()[:, None] * shares).normalized()   # [bonds, arc points] Point-
    return core.base_points(along, circling)


def light_fan(circling: three_sphere.Rate, source: three_sphere.Point, toward: three_sphere.Point
              ) -> tuple[three_sphere.Point, three_sphere.Point]:
    """The light's start on the circle above the base point `source`, `[rays] Point-`, and its rays' headings, unit
    steps across the circles, `[rays] Antivector-`: the steps whose images on the base fan out about the direction
    toward the base point `toward`, turned about the source."""
    start = core.lifts(source[None], circling)                                  # [1] Point-
    aim = (toward - source * (three_sphere.polar(source) & toward)).cast(three_sphere.Step).normalized()   # [] Point-
    angles = np.linspace(-light_spread / 2, light_spread / 2, n_rays)           # [rays]
    fan = (three_sphere.turning(source) * (-angles / 2)).exp()                  # [rays] Motor+
    return start.broadcast_to((n_rays,)), core.across(source, fan >> aim).normalized()


def light_paths(mesh: three_sphere.Mesh, vertices: three_sphere.Point, waves: three_sphere.Waves,
                circling: three_sphere.Rate, starts: three_sphere.Point, headings: three_sphere.Point, time: float
                ) -> three_sphere.Point:
    """The light's paths set out at a time through the waves, seen on the base, `[rays, steps + 1] Point-`."""
    places = stack([starts, *three_sphere.light(mesh, vertices, waves, starts, headings, time, light_step,
                                                n_light_steps)], axis=-1)       # [rays, steps + 1] Point-
    return core.base_points(places, circling)


def undisturbed_paths(circling: three_sphere.Rate, starts: three_sphere.Point, headings: three_sphere.Point
                      ) -> three_sphere.Point:
    """The light's paths through the round sphere, seen on the base, `[rays, steps + 1] Point-`: each start rolled
    along its heading on its great circle, in its isoclinic frame."""
    lengths = light_step * np.arange(n_light_steps + 1)                         # [steps + 1]
    rolls = (three_sphere.moving(headings)[:, None] * (-lengths / 2)).exp()     # [rays, steps + 1] Motor+
    places = (three_sphere.isoclinic(starts)[:, None] * rolls) >> three_sphere.origin   # [rays, steps + 1] Point-
    return core.base_points(places, circling)


def gauge_states(cell_mesh: three_sphere.Mesh, cell_vertices: three_sphere.Point, waves: three_sphere.Waves,
                 frequency: float, bonds: three_sphere.Mesh, atoms: three_sphere.Point, times: np.ndarray) -> list:
    """The gauges' atoms and their bonds' strains, `(Point-[atoms], Scalar+[bonds])`, at each of the times, after their
    motion has settled into the waves' swing at the frequency."""
    holding = (holding_share * frequency) ** 2
    # A triangle of unit masses swings in shear at the root of three halves of its bonds' stiffness and its atoms'
    # holding; its bonds' dashpots, three halves of them, and the holding's damp the swing.
    stiffness = 2 / 3 * (frequency ** 2 - holding)
    damping = 2 / 3 * (2 * gauge_damping * frequency - 2 * np.sqrt(holding))
    round_steps = three_sphere.bond_steps(bonds, atoms)                         # Antivector-[bonds]
    rest = round_steps.dot(round_steps).square_root()                           # Scalar+[bonds]
    n_bonds = len(bonds.edges)
    stiffnesses = three_sphere.mesh_module.as_scalar(np.full(n_bonds, stiffness)).field()   # Scalar+[bonds]
    dampings = three_sphere.mesh_module.as_scalar(np.full(n_bonds, damping)).field()       # Scalar+[bonds]
    _, _, steps = three_sphere.invariant_generators()                           # [3] Antivector-
    at_rest = (steps[0] * np.zeros(bonds.n_vertices)).field()                  # Antivector-[atoms]
    start = times[0] - rolling.n_settling_swings * 2 * np.pi / frequency
    return list(three_sphere.massive(cell_mesh, cell_vertices, waves, bonds, rest, stiffnesses, dampings, holding,
                                     atoms, at_rest, start, rolling.crystal_step, times))


def quadrupole_history(cell_mesh: three_sphere.Mesh, cell_vertices: three_sphere.Point
                       ) -> tuple[three_sphere.Waves, float]:
    """The 600-cell's travelling quadrupole as a smooth field over time, and its frequency: the squash, stretching
    space along x and squeezing it along y in the frames, and the squash turned an eighth of a turn about z, a quarter
    swing behind it."""
    mv = three_sphere.mv
    cell_rates = three_sphere.round_rates(cell_mesh, three_sphere.isoclinic(cell_vertices))   # Rate+[edges]
    patterns = rolling.squash(cell_mesh, cell_vertices, cell_rates, stack((mv.x, eighth >> mv.x)),
                              stack((mv.y, eighth >> mv.y)))                     # [patterns] Scalar+[edges]
    return rolling.stepped_history(cell_mesh, cell_vertices, patterns, np.array([0, np.pi / 2]), swing)


def quadrupole_on_base() -> tuple[three_sphere.Point, three_sphere.Point, list]:
    """One swing of the 600-cell's travelling quadrupole seen on the base: the decagons' base points `[decagons]
    Point-`, the light's undisturbed paths `[rays, steps + 1] Point-`, and at each frame the circles' stretch at the
    icosphere's points `Scalar+[points]`, the gauges' atoms `[atoms] Point-`, their bonds' arcs `[bonds, arc points]
    Point-` and strains `[bonds] Scalar+`, and the light's paths `[rays, steps + 1] Point-`."""
    cell_mesh, cell_vertices = rolling.six_hundred_cell()
    waves, frequency = quadrupole_history(cell_mesh, cell_vertices)
    times = np.arange(n_frames) * 2 * np.pi / frequency / n_frames             # [frames]
    turn = rolling.circle_turn(cell_vertices, rolling.guardrail_sides)          # [] Motor+
    circling = turn.log()                                                       # [] Rate+
    decagons = rolling.guardrails(cell_vertices, rolling.guardrail_sides)        # [decagons, sides]
    decagon_bases = core.base_points(cell_vertices.batch()[decagons[:, 0]], circling)   # [decagons] Point-
    _, _, bases = base_sphere()
    bonds, atoms = triangle_gauges(decagon_bases, circling)
    states = gauge_states(cell_mesh, cell_vertices, waves, frequency, bonds, atoms, times)
    mv = three_sphere.mv
    starts, headings = light_fan(circling, mv.z.dual().cast(three_sphere.Step), mv.y.dual().cast(three_sphere.Step))
    frames = [(core.stretches(waves, cell_mesh, cell_vertices, bases, bases, turn, rolling.guardrail_sides,
                              n_circle_samples, time),
               core.base_points(positions.batch(), circling), bond_arcs(bonds, positions, circling), strains.batch(),
               light_paths(cell_mesh, cell_vertices, waves, circling, starts, headings, time))
              for time, (positions, strains) in zip(times, states)]
    return decagon_bases, undisturbed_paths(circling, starts, headings), frames


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.relativity.hopf_projection import render

    sphere_mesh, sphere_points, _ = base_sphere()
    decagon_bases, undisturbed, frames = quadrupole_on_base()
    save_animation((render.base_view(sphere_mesh, sphere_points, stretches, decagon_bases, atoms, arcs, strains, light,
                                     rolling.wave_swing)
                    for stretches, atoms, arcs, strains, light in frames), "hopf_projection_quadrupole",
                   rolling.duration_ms)
    save_figure(render.response(stack([strains for *_, strains, _ in frames]), stack([light for *_, light in frames]),
                                undisturbed, target_strain, target_sways), "hopf_projection_quadrupole_response")

    # checks
    # The quadrupole is the same all along each circle, between the tenth turns that carry the mesh onto itself too: a
    # circle's stretch read at one point of it is its average round it.
    cell_mesh, cell_vertices = rolling.six_hundred_cell()
    waves, _ = quadrupole_history(cell_mesh, cell_vertices)
    turn = rolling.circle_turn(cell_vertices, rolling.guardrail_sides)          # [] Motor+
    _, _, bases = base_sphere()
    at_one = core.stretches(waves, cell_mesh, cell_vertices, bases, bases, turn, rolling.guardrail_sides, 1, 0.0)
    averaged = core.stretches(waves, cell_mesh, cell_vertices, bases, bases, turn, rolling.guardrail_sides,
                              n_circle_samples, 0.0)
    assert np.abs((at_one - averaged).to_array()).max() < 1e-6 * np.abs(averaged.to_array()).max()


if __name__ == "__main__":
    main()
