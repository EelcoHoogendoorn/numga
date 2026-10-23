"""Scenes for connections on curved surfaces: holonomy around latitude circles on the unit sphere, and
vector heat transport and harmonic tangent fields on a torus; parallel fields of trivial connections on a
sphere with two singularities of index one at opposite poles, one singularity of index two, and three of
index one with one of index minus one, on a torus with none at all and with a pair of opposite indices,
and a field turned through a whole turn about every normal; stripes along those fields and across cross
fields, and stripes sliding along their field as the stripe field turns."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from numga import stack
from examples.surfaces.connections import core

mv = core.mv
# The sphere and its three latitude circles: 30, 60 and 90 degrees from the north pole.
SPHERE_LEVELS = 3
COLATITUDES = np.array([np.pi / 6, np.pi / 3, np.pi / 2])
N_RING_STEPS = 120
# The steps between frames of the transport around the circles.
RING_STRIDE = 2
# The torus of every scene.
MAJOR_RADIUS = 2.0
MINOR_RADIUS = 0.75
# The torus of the vector heat transport and the harmonic modes, the localized tangent bump on its outer
# equator, the vector heat steps, and the harmonic modes.
N_HEAT_MAJOR = 28
N_HEAT_MINOR = 14
SOURCE_WIDTH = 0.55
STEP_TIME = 0.18
N_HEAT_STEPS = 20
N_MODES = 4
# For each scene on the sphere, the icosahedron's vertices whose faces carry a singularity and the index
# of each; vertices 9 and 10 are opposite each other.
SPHERE_CORNERS = [[9, 10], [9], [9, 4, 8, 3]]
SPHERE_INDICES = [[1, 1], [2], [1, 1, 1, -1]]
# The torus of the designed fields, and for each of its scenes the faces, by their place around the axis
# and around the tube, that carry a singularity, and the index of each.
N_FIELD_MAJOR = 40
N_FIELD_MINOR = 20
FIELD_TORUS_PLACES = [np.zeros((0, 2), dtype=int), np.array([[32, 2], [0, 2]])]
TORUS_INDICES = [[], [1, -1]]
# The sphere of the stripes and their frequency on it; the torus of the stripes, the places of its
# singular faces, and the stripes' frequency on it.
STRIPE_SPHERE_LEVELS = 4
SPHERE_FREQUENCY = 20.0
N_STRIPE_MAJOR = 80
N_STRIPE_MINOR = 40
STRIPE_TORUS_PLACES = [np.zeros((0, 2), dtype=int), np.array([[64, 4], [0, 4]])]
TORUS_FREQUENCY = 12.0
# The cross fields' singularities: at the cube's eight corners, each turning by a quarter, and at the
# four corners of a tetrahedron inscribed in it, each turning by a half; the crosses' frequency.
CUBE = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]]
CROSS_CORNERS = [CUBE, [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]]
CROSS_INDICES = [0.25, 0.5]
CROSS_FREQUENCY = 20.0
# The weight keeping fields out of the normal bundle.
NORMAL_STIFFNESS = 100.0
# The frames of a whole turn of the field, of the stripes sliding by one stripe, and the duration of a frame.
N_TURNING_FRAMES = 48
N_SLIDING_FRAMES = 36
DURATION_MS = 70


# --- math -----------------------------------------------------------------------------
Pattern = tuple[core.Vector, core.Vector, core.Scalar]


def holonomy() -> tuple[core.Mesh, core.Vector, core.Vector]:
    """The unit sphere, three latitude circles on it `[steps + 1, rings] Vector`, and a meridian tangent
    vector at the start of each `[rings] Vector`."""
    sphere = core.Mesh.icosphere(SPHERE_LEVELS)
    rings, starts = core.latitude_rings(COLATITUDES, N_RING_STEPS)              # [steps + 1, rings] Vector, [rings] Vector
    return sphere, rings, starts


def carried_around(rings: core.Vector, starts: core.Vector, stride: int) -> Iterator[tuple[core.Vector, core.Vector]]:
    """Every `stride`-th step along the circles: the points, and the vectors carried there."""
    yield from islice(zip(rings, core.transported(rings, starts)), None, None, stride)


def torus_fields() -> tuple[core.Mesh, core.Vector, core.Vector, core.Vector, core.Scalar, core.SparseExtensor]:
    """Vector heat parallel transport and the smoothest global tangent fields on a torus: the torus, the
    diffused waves `[steps] Vector[V]`, the unit transported field `Vector[V]`, the harmonic modes
    `[modes] Vector[V]` with their energies, and the covariant derivative."""
    torus = core.Mesh.torus(MAJOR_RADIUS, MINOR_RADIUS, N_HEAT_MAJOR, N_HEAT_MINOR)
    normals = torus.vertex_normals.batch()                                      # [V] Vector
    centre = (mv.x * (MAJOR_RADIUS + MINOR_RADIUS)).dual()                 # [] Point
    weight = ((torus.vertices.batch() - centre).norm_squared() * (-0.5 / SOURCE_WIDTH**2)).exp()   # [V] Scalar
    direction = (mv.y + mv.z).normalized()                                      # [] Vector
    tangent_direction = direction - normals * (normals | direction)             # [V] Vector
    source = (weight * tangent_direction).field()                               # Vector[V]

    # The surface's own transport: no extra turn on any edge.
    no_turns = core.as_scalar(np.zeros(len(torus.edges))).field()               # Scalar[E]
    covariant, laplacian = core.connection(torus, core.adjusted_transport(torus, no_turns), torus.edge_ratio)
    waves, units = zip(*core.heat_transport(laplacian, torus.vertex_areas, source, STEP_TIME, N_HEAT_STEPS))
    energies, modes = core.low_modes(laplacian, torus.vertex_normals, torus.vertex_areas, NORMAL_STIFFNESS, N_MODES)
    return torus, stack(waves), units[-1], modes, energies, covariant


def design(mesh: core.Mesh, index: np.ndarray, generators: list[np.ndarray]
           ) -> tuple[core.Scalar, core.Vector, core.Scalar, core.Scalar]:
    """The trivial connection with every face turning by its index `[F]` and its unit parallel field: the
    extra turns on the edges, the field, the energies of the two parallel fields, and the circulation's
    residual."""
    angles, circulation, target = core.trivial_connection(mesh, index, generators)
    _, laplacian = core.connection(mesh, core.adjusted_transport(mesh, angles), mesh.edge_ratio)
    energies, fields = core.low_modes(laplacian, mesh.vertex_normals, mesh.vertex_areas, NORMAL_STIFFNESS, 2)
    field = fields.batch()[0]                                                    # [V] Vector
    return angles, (field / field.norm()).field(), energies, circulation(angles) - target


def field_spheres() -> tuple[core.Mesh, list[tuple[core.Vector, np.ndarray, list[int]]], list[core.Scalar], list[core.Scalar]]:
    """The sphere; for each scene its unit parallel field, singular faces and their indices; and each
    design's circulation residual and the energies of its two parallel fields."""
    sphere = core.Mesh.icosphere(SPHERE_LEVELS)
    singulars = [np.array([core.faces_around(sphere, corner) for corner in corners]) for corners in SPHERE_CORNERS]
    designs = [design(sphere, face_indices(sphere, singular, indices), [])
               for singular, indices in zip(singulars, SPHERE_INDICES)]
    scenes = [(field, singular, indices) for (_, field, _, _), singular, indices in zip(designs, singulars, SPHERE_INDICES)]
    return sphere, scenes, [residual for *_, residual in designs], [energies for _, _, energies, _ in designs]


def field_tori() -> tuple[core.Mesh, list[tuple[core.Vector, np.ndarray, list[int]]], list[core.Scalar], list[core.Scalar]]:
    """The torus; for each scene its unit parallel field, singular faces and their indices; and each
    design's circulation residual and the energies of its two parallel fields."""
    torus = core.Mesh.torus(MAJOR_RADIUS, MINOR_RADIUS, N_FIELD_MAJOR, N_FIELD_MINOR)
    # Its two generators: once around the tube, and once around the axis along the outer equator.
    generators = [np.arange(N_FIELD_MINOR), np.arange(N_FIELD_MAJOR) * N_FIELD_MINOR]
    singulars = [places[:, 0] * N_FIELD_MINOR + places[:, 1] for places in FIELD_TORUS_PLACES]
    designs = [design(torus, face_indices(torus, singular, indices), generators)
               for singular, indices in zip(singulars, TORUS_INDICES)]
    scenes = [(field, singular, indices) for (_, field, _, _), singular, indices in zip(designs, singulars, TORUS_INDICES)]
    return torus, scenes, [residual for *_, residual in designs], [energies for _, _, energies, _ in designs]


def turning(mesh: core.Mesh, field: core.Vector) -> core.Vector:
    """The parallel field turned about every normal through a whole turn `[frames] Vector[V]`: every
    turned field is parallel too, the null space being a field and its quarter turn."""
    angles = np.linspace(0, 2 * np.pi, N_TURNING_FRAMES, endpoint=False)         # [frames]
    planes = mesh.vertex_normals.batch().dual()                                  # [V] Bivector
    return ((planes * (-angles[:, None] / 2)).exp() >> field.batch()).field()   # [frames] Vector[V]


def pattern(mesh: core.Mesh, index: np.ndarray, generators: list[np.ndarray], frequency: float
            ) -> tuple[Pattern, core.Scalar]:
    """The stripes along the unit direction field of the trivial connection with every face turning by its
    index `[F]`, and along the field turned a quarter turn: the two lines `[V, 2]`, the unit stripe field
    of each `[V, 2]`, and the edges' targets for each `[E, 2]`; and the energies of each line's two
    smoothest stripe fields `[2, 2]`."""
    angles, field, _, _ = design(mesh, index, generators)
    # Every face is regular except the singular ones.
    regular = core.as_scalar(np.where(index == 0, 1.0, 0.0)).field()           # Scalar[F]
    layers = [along(mesh, angles, direction, regular, frequency) for direction in (field, core.quarter_turn(mesh, field))]
    *families, energies = zip(*layers)
    return tuple(stack([layer.batch() for layer in family], axis=-1) for family in families), stack(energies)


def along(mesh: core.Mesh, angles: core.Scalar, field: core.Vector, regular: core.Scalar, frequency: float
          ) -> tuple[core.Vector, core.Vector, core.Scalar, core.Scalar]:
    """The unit stripe field along a direction field parallel for the connection of the given turns, at
    the given frequency, with the field, the edges' targets and the two smallest energies: the smoothest
    tangent field of the connection of the turns `angles` with every edge's phase advance added. Each
    face adds its half-cotangents to its edges' weights only where it is regular `Scalar[F]`, one, and
    not where the direction field turns inside it, zero: there the field's average across an edge means
    nothing."""
    advance = core.targets(mesh, field, frequency)                               # Scalar[E]
    _, laplacian = core.connection(mesh, core.adjusted_transport(mesh, angles + advance), mesh.weighted_edge_ratio(regular))
    energies, stripe_fields = core.low_modes(laplacian, mesh.vertex_normals, mesh.vertex_areas, NORMAL_STIFFNESS, 2)
    stripe = stripe_fields.batch()[0]                                            # [V] Vector
    return field, (stripe / stripe.norm()).field(), advance, energies


def cross(mesh: core.Mesh, index: np.ndarray, frequency: float) -> tuple[Pattern, core.Scalar]:
    """The two families of stripes across the cross field of the trivial connection with every face
    turning by its quarter-turn index `[F]`, and the connection's extra turns on the edges."""
    angles, _, _ = core.trivial_connection(mesh, index, [])
    regular = core.as_scalar(np.where(index == 0, 1.0, 0.0)).field()           # Scalar[F]
    return core.cross_stripes(mesh, angles, core.carried_out(mesh, angles), regular, frequency, NORMAL_STIFFNESS), angles


def stripe_spheres() -> tuple[core.Mesh, list[Pattern], list[core.Scalar]]:
    """The sphere, and for each scene the stripes along its field and its quarter turn, with their energies."""
    sphere = core.Mesh.icosphere(STRIPE_SPHERE_LEVELS)
    singulars = [np.array([core.faces_around(sphere, corner) for corner in corners]) for corners in SPHERE_CORNERS]
    scenes = [pattern(sphere, face_indices(sphere, singular, indices), [], SPHERE_FREQUENCY)
              for singular, indices in zip(singulars, SPHERE_INDICES)]
    return sphere, *map(list, zip(*scenes))


def stripe_tori() -> tuple[core.Mesh, list[Pattern], list[core.Scalar]]:
    """The torus, and for each scene the stripes along its field and its quarter turn, with their energies."""
    torus = core.Mesh.torus(MAJOR_RADIUS, MINOR_RADIUS, N_STRIPE_MAJOR, N_STRIPE_MINOR)
    # Its two generators: once around the tube, and once around the axis along the outer equator.
    generators = [np.arange(N_STRIPE_MINOR), np.arange(N_STRIPE_MAJOR) * N_STRIPE_MINOR]
    scenes = [pattern(torus, face_indices(torus, places[:, 0] * N_STRIPE_MINOR + places[:, 1], indices), generators, TORUS_FREQUENCY)
              for places, indices in zip(STRIPE_TORUS_PLACES, TORUS_INDICES)]
    return torus, *map(list, zip(*scenes))


def crosses() -> tuple[core.Mesh, list[Pattern], list[core.Scalar]]:
    """The sphere, and for each set of corners the two families of stripes across its cross field, with
    the connection's extra turns on the edges."""
    sphere = core.Mesh.icosphere(STRIPE_SPHERE_LEVELS)
    scenes = [cross(sphere, face_indices(sphere, nearest_faces(sphere, mv.vector(corners)), index), CROSS_FREQUENCY)
              for corners, index in zip(CROSS_CORNERS, CROSS_INDICES)]
    return sphere, *map(list, zip(*scenes))


def sliding(mesh: core.Mesh, stripes: core.Vector) -> core.Vector:
    """The stripe fields `[V, n]` turned about every normal through a whole turn `[frames, V, n]`: every
    turn of one is a stripe field too, its stripes slid along its line by the turn's share of one
    stripe."""
    angles = np.linspace(0, 2 * np.pi, N_SLIDING_FRAMES, endpoint=False)         # [frames]
    planes = mesh.vertex_normals.batch()[:, None].dual()                         # [V, 1] Bivector
    return (planes * (-angles[:, None, None] / 2)).exp() >> stripes              # [frames, V, n] Vector


# --- plumbing -------------------------------------------------------------------------
def face_indices(mesh: core.Mesh, singular: np.ndarray, indices: list[float]) -> np.ndarray:
    """The index of every face `[F]`: the given indices on the singular faces `[S]`, and none elsewhere."""
    index = np.zeros(len(mesh.faces))
    index[singular] = indices
    return index


def nearest_faces(mesh: core.Mesh, directions: core.Vector) -> np.ndarray:
    """The face whose centre lies furthest along each direction `[n]`."""
    return np.argmax((mesh.face_centers.batch()[:, None] & directions.batch()[None]).to_array(), axis=0)


def stripe_mismatch(mesh: core.Mesh, lines: core.Vector, stripes: core.Vector, advance: core.Scalar) -> np.ndarray:
    """How far, in angle, the stripe phase across every edge misses its target `[E, n]`: the phase is the
    angle from the stripe field to its line, about each vertex normal."""
    normals = mesh.vertex_normals.batch()[:, None]                               # [V, 1] Vector
    sines = ((stripes ^ lines) | normals.dual()).to_array()                      # [V, n]
    phase = np.arctan2(sines, (stripes | lines).to_array())
    steps = phase[mesh.edges[:, 1]] - phase[mesh.edges[:, 0]] - advance.to_array()
    return np.abs(np.angle(np.exp(1j * steps)))


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.surfaces.connections import render

    sphere, rings, starts = holonomy()
    save_animation(render.animate_holonomy(sphere, rings, carried_around(rings, starts, RING_STRIDE)),
                   "connections_holonomy", DURATION_MS)
    torus, diffused, transported, modes, mode_energies, covariant = torus_fields()
    save_figure(render.draw_torus_fields(torus, [transported, modes[0], modes[2]]), "connections_modes")
    save_animation(render.animate_diffusion(torus, diffused), "connections_diffusion", DURATION_MS)

    sphere, sphere_scenes, sphere_residuals, sphere_energies = field_spheres()
    save_figure(render.draw_fields(sphere, sphere_scenes), "connections_fields_spheres")
    ring, torus_scenes, torus_residuals, torus_energies = field_tori()
    save_figure(render.draw_fields(ring, torus_scenes), "connections_fields_tori")
    field, singular, indices = sphere_scenes[0]
    save_animation(render.animate_turning(sphere, turning(sphere, field), singular, indices), "connections_turning",
                   DURATION_MS)

    fine, sphere_patterns, stripe_energies = stripe_spheres()
    bands = [tuple(family[..., :1] for family in pattern) for pattern in sphere_patterns]
    save_figure(render.draw_bands(fine, bands), "connections_stripes_spheres")
    save_figure(render.draw_lines(fine, sphere_patterns), "connections_stripes_sphere_grids")
    stripe_torus, torus_patterns, torus_stripe_energies = stripe_tori()
    bands = [tuple(family[..., :1] for family in pattern) for pattern in torus_patterns]
    save_figure(render.draw_bands(stripe_torus, bands), "connections_stripes_tori")
    save_figure(render.draw_lines(stripe_torus, torus_patterns), "connections_stripes_torus_grids")
    _, cross_patterns, cross_angles = crosses()
    save_figure(render.draw_lines(fine, cross_patterns), "connections_stripes_crosses")
    lines, stripe_fields, advance = (family[..., :1] for family in sphere_patterns[2])
    save_animation(render.animate_flow(fine, lines, sliding(fine, stripe_fields), advance),
                   "connections_stripes_sliding", DURATION_MS)

    # checks
    # Every transported vector stays tangent to the sphere and keeps unit length, and on closing the
    # loop it has turned by the enclosed cap's solid angle.
    carried = stack(tuple(core.transported(rings, starts)))                      # [steps + 1, rings] Vector
    np.testing.assert_allclose((carried | rings).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose(carried.norm().to_array(), 1.0, atol=1e-12)
    np.testing.assert_allclose((carried[-1] | starts).to_array(), np.cos(2 * np.pi * (1 - np.cos(COLATITUDES))), atol=2e-3)
    # Each edge rotor carries the tail's normal to the head's, so neither heat diffusion nor the low
    # modes leak into the normal bundle.
    np.testing.assert_allclose((diffused | torus.vertex_normals).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose((modes | torus.vertex_normals).to_array(), 0.0, atol=1e-11)
    np.testing.assert_allclose(transported.norm().to_array(), 1.0, atol=1e-12)
    # A quarter turn in every tangent plane commutes with the connection: the energies come in equal
    # pairs, and a quarter-turned mode keeps its energy.
    values = mode_energies.to_array()
    np.testing.assert_allclose(values[0::2], values[1::2], rtol=1e-10)
    edge_weights = core.spdiag(torus.edge_ratio)                                 # [E, E] Scalar
    quarter_turned = (modes.batch() | torus.vertex_normals.batch().dual()).field()   # [modes] Vector[V]
    np.testing.assert_allclose((edge_weights * covariant(quarter_turned).norm_squared()).sites.sum().to_array(),
                               (edge_weights * covariant(modes).norm_squared()).sites.sum().to_array(), rtol=1e-10)
    # Every design's indices add up to the Euler characteristic: the connection is trivial, and its
    # field and the field's quarter turn carry no energy.
    np.testing.assert_allclose(np.concatenate([residual.to_array() for residual in sphere_residuals + torus_residuals]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.concatenate([energies.to_array() for energies in sphere_energies + torus_energies]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.concatenate([(field | sphere.vertex_normals).to_array() for field, _, _ in sphere_scenes]), 0.0, atol=1e-12)
    # The face curvatures add up to two whole turns, and indices one short of the Euler characteristic
    # leave the missing turn spread over all faces.
    curvature = core.loop_turn(sphere.vertex_normals.batch()[sphere.faces])      # [F] Scalar
    np.testing.assert_allclose(curvature.sum(axis=0).to_array(), 4 * np.pi, rtol=1e-12)
    angles, circulation, target = core.trivial_connection(sphere, face_indices(sphere, np.array([0]), [1]), [])
    np.testing.assert_allclose((circulation(angles) - target).to_array(), 2 * np.pi / len(sphere.faces), rtol=1e-9)
    # Every stripe field comes with its quarter turn, of equal energy, and across most edges its phase
    # advances by the edge's target; the edges around the forks miss it, many on a torus, whose stripes
    # must fork to space themselves evenly around its longer outer equator and shorter inner one.
    for mesh, patterns, energies in ((fine, sphere_patterns, stripe_energies), (stripe_torus, torus_patterns, torus_stripe_energies)):
        for (lines, stripes, advance), pair in zip(patterns, energies):
            np.testing.assert_allclose(pair.to_array()[:, 1], pair.to_array()[:, 0], rtol=1e-6)
            assert np.median(stripe_mismatch(mesh, lines, stripes, advance), axis=0).max() < 0.25
    # Carried across every edge, each of the tail's lines lands along or against one of the head's
    # lines, and some edges trade the two lines or reverse them: around a singularity the cross turns
    # by a quarter or a half, not by a whole turn.
    for (lines, _, _), angles in zip(cross_patterns, cross_angles):
        forward = core.adjusted_transport(fine, angles)[:, 0]                    # [E] Rotor
        tails, heads = lines[fine.edges[:, 0]], lines[fine.edges[:, 1]]          # [E, 2] Vector each
        match = (heads[:, :, None] | (forward[:, None] >> tails)[:, None, :]).to_array()   # [E, 2, 2]
        np.testing.assert_allclose(np.abs(match) * (1 - np.abs(match)), 0, atol=1e-8)
        assert (match[:, 0, 0] < 0.5).any()


if __name__ == "__main__":
    main()
