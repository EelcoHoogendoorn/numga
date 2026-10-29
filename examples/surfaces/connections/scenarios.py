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
N_RING_STEPS = 210
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
STEP_TIME = 0.1
N_HEAT_STEPS = 36
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
# The cross fields' figure at a share of the other patterns' panel side.
CROSS_SCALE = 0.5
# The weight keeping fields out of the normal bundle.
NORMAL_STIFFNESS = 100.0
# The frames of a whole turn of the field, of the stripes sliding by one stripe, and the duration of a frame.
N_TURNING_FRAMES = 84
N_SLIDING_FRAMES = 63
DURATION_MS = 40


# --- math -----------------------------------------------------------------------------
Pattern = tuple[core.Point, core.Point, core.Scalar]


def holonomy() -> tuple[core.Mesh, core.Point, core.Vector, core.Point]:
    """The unit sphere and its vertices come with three latitude circles on it, as their normals
    `[steps + 1, rings] Vector-`, and a meridian tangent vector at the start of each, `[rings] Point-`."""
    mesh, vertices = core.Mesh.icosphere(SPHERE_LEVELS)
    rings, starts = core.latitude_rings(COLATITUDES, N_RING_STEPS)              # [steps + 1, rings] Vector-, [rings] Point-
    return mesh, vertices, rings, starts


def carried_around(rings: core.Vector, starts: core.Point, stride: int) -> Iterator[tuple[core.Vector, core.Point]]:
    """Every `stride`-th step along the circles: the points, and the vectors carried there."""
    yield from islice(zip(rings, core.transported(rings, starts)), None, None, stride)


def torus_fields() -> tuple[core.Mesh, core.Point, core.Point, core.Point, core.Point, core.Scalar, core.SparseExtensor]:
    """Vector heat parallel transport and the smoothest global tangent fields on a torus: the torus and its
    vertices, the diffused waves `[steps] Point-[vertices]`, the unit transported field `Point-[vertices]`, the
    harmonic modes `[modes] Point-[vertices]` with their energies, and the covariant derivative."""
    mesh, vertices = core.Mesh.torus(MAJOR_RADIUS, MINOR_RADIUS, N_HEAT_MAJOR, N_HEAT_MINOR)
    unit_normals = core.vertex_normals(mesh, vertices)                         # Vector-[vertices]
    areas = core.vertex_areas(mesh, vertices)                                  # Scalar+[vertices]
    weights = core.cotangent_weights(mesh, vertices, core.as_scalar(np.ones(len(mesh.faces))).field())   # Scalar+[edges]
    normals = unit_normals.batch()                                              # [vertices] Vector-
    centre = (mv.x * (MAJOR_RADIUS + MINOR_RADIUS)).dual()                      # [] Point-
    weight = ((vertices.batch() - centre).norm_squared() * (-0.5 / SOURCE_WIDTH**2)).exp()   # [vertices] Scalar+
    direction = (mv.y + mv.z).dual().normalized()                               # [] Point-
    # The direction less its part along each normal: the normal's direction times its pairing with the patch.
    tangent_direction = direction - normals.dual() * (normals & direction)      # [vertices] Point-
    source = (weight * tangent_direction).field()                               # Point-[vertices]

    # The surface's own transport: no extra turn on any edge.
    no_turns = core.as_scalar(np.zeros(len(mesh.edges))).field()               # Scalar+[edges]
    covariant, laplacian = core.connection(mesh, core.adjusted_transport(mesh, unit_normals, no_turns), weights)
    waves, units = zip(*core.heat_transport(laplacian, areas, source, STEP_TIME, N_HEAT_STEPS))
    energies, modes = core.low_modes(laplacian, unit_normals, areas, NORMAL_STIFFNESS, N_MODES)
    return mesh, vertices, stack(waves), units[-1], modes, energies, covariant


def design(mesh: core.Mesh, normals: core.Vector, areas: core.Scalar, weights: core.Scalar,
           index: np.ndarray, generators: list[np.ndarray]) -> tuple[core.Scalar, core.Point, core.Scalar, core.Scalar]:
    """The trivial connection with every face turning by its index `[faces]` has a unit parallel field: the extra
    turns on the edges, the field, the energies of the two parallel fields, and the circulation's residual."""
    angles, circulation, target = core.trivial_connection(mesh, normals, index, generators)
    _, laplacian = core.connection(mesh, core.adjusted_transport(mesh, normals, angles), weights)
    energies, fields = core.low_modes(laplacian, normals, areas, NORMAL_STIFFNESS, 2)
    field = fields.batch()[0]                                                    # [vertices] Point-
    return angles, (field / field.norm()).field(), energies, circulation(angles) - target


def field_spheres() -> tuple[core.Mesh, core.Point, list[tuple[core.Point, np.ndarray, list[int]]], list[core.Scalar],
                             list[core.Scalar]]:
    """The sphere and its vertices; for each scene its unit parallel field, singular faces and their indices; and each
    design's circulation residual and the energies of its two parallel fields."""
    mesh, vertices = core.Mesh.icosphere(SPHERE_LEVELS)
    normals, areas = core.vertex_normals(mesh, vertices), core.vertex_areas(mesh, vertices)
    weights = core.cotangent_weights(mesh, vertices, core.as_scalar(np.ones(len(mesh.faces))).field())
    singulars = [mesh.vertex_faces[corners] for corners in SPHERE_CORNERS]
    designs = [design(mesh, normals, areas, weights, face_indices(mesh, singular, indices), [])
               for singular, indices in zip(singulars, SPHERE_INDICES)]
    scenes = [(field, singular, indices) for (_, field, _, _), singular, indices in zip(designs, singulars, SPHERE_INDICES)]
    return mesh, vertices, scenes, [residual for *_, residual in designs], [energies for _, _, energies, _ in designs]


def field_tori() -> tuple[core.Mesh, core.Point, list[tuple[core.Point, np.ndarray, list[int]]], list[core.Scalar],
                          list[core.Scalar]]:
    """The torus and its vertices; for each scene its unit parallel field, singular faces and their indices; and each
    design's circulation residual and the energies of its two parallel fields."""
    mesh, vertices = core.Mesh.torus(MAJOR_RADIUS, MINOR_RADIUS, N_FIELD_MAJOR, N_FIELD_MINOR)
    normals, areas = core.vertex_normals(mesh, vertices), core.vertex_areas(mesh, vertices)
    weights = core.cotangent_weights(mesh, vertices, core.as_scalar(np.ones(len(mesh.faces))).field())
    # Its two generators: once around the tube, and once around the axis along the outer equator.
    generators = [np.arange(N_FIELD_MINOR), np.arange(N_FIELD_MAJOR) * N_FIELD_MINOR]
    singulars = [places[:, 0] * N_FIELD_MINOR + places[:, 1] for places in FIELD_TORUS_PLACES]
    designs = [design(mesh, normals, areas, weights, face_indices(mesh, singular, indices), generators)
               for singular, indices in zip(singulars, TORUS_INDICES)]
    scenes = [(field, singular, indices) for (_, field, _, _), singular, indices in zip(designs, singulars, TORUS_INDICES)]
    return mesh, vertices, scenes, [residual for *_, residual in designs], [energies for _, _, energies, _ in designs]


def turning(mesh: core.Mesh, vertices: core.Point, field: core.Point) -> core.Point:
    """The parallel field turned about every normal through a whole turn, `[frames] Point-[vertices]`, is parallel
    at every turn, the null space being a field and its quarter turn."""
    angles = np.linspace(0, 2 * np.pi, N_TURNING_FRAMES, endpoint=False)         # [frames]
    planes = core.vertex_normals(mesh, vertices).batch().dual()                  # [vertices] Bivector+
    return ((planes * (-angles[:, None] / 2)).exp() >> field.batch()).field()   # [frames] Point-[vertices]


def pattern(mesh: core.Mesh, vertices: core.Point, normals: core.Vector, areas: core.Scalar, weights: core.Scalar,
            index: np.ndarray, generators: list[np.ndarray], frequency: float) -> tuple[Pattern, core.Scalar]:
    """The stripes run along the unit direction field of the trivial connection with every face turning by its
    index `[faces]`, and along the field turned a quarter turn: the two lines `[2] Point-[vertices]`, the unit
    stripe field of each `[2] Point-[vertices]`, and the edges' targets for each `[2] Scalar+[edges]`; with the
    energies of each line's two smoothest stripe fields `[2, 2]`."""
    angles, field, _, _ = design(mesh, normals, areas, weights, index, generators)
    # Every face is regular except the singular ones.
    regular = core.as_scalar(np.where(index == 0, 1.0, 0.0)).field()           # Scalar+[faces]
    # A quarter turn in every tangent plane: the commutator with the plane.
    quarter_turned = field.batch().commutator(normals.batch().dual()).field()   # Point-[vertices]
    layers = [along(mesh, vertices, normals, areas, angles, direction, regular, frequency)
              for direction in (field, quarter_turned)]
    *families, energies = zip(*layers)
    return tuple(stack(family) for family in families), stack(energies)


def along(mesh: core.Mesh, vertices: core.Point, normals: core.Vector, areas: core.Scalar, angles: core.Scalar,
          field: core.Point, regular: core.Scalar, frequency: float) -> tuple[core.Point, core.Point, core.Scalar, core.Scalar]:
    """The unit stripe field along a direction field parallel for the connection of the given turns, at the given
    frequency, is the smoothest tangent field of the connection of the turns `angles` with every edge's phase
    advance added; it comes with the field, the edges' targets and the two smallest energies. Each face adds its
    half-cotangents to its edges' weights only where it is regular, `Scalar+[faces]`, one, and not where the
    direction field turns inside it, zero: there the field's average across an edge means nothing."""
    # The field read along each edge, by the metric: the paper's inner product of the field with the edge.
    edge_means = mesh.incidence[1, 0].mean(axis=0)                               # Scalar+[edges <- vertices]
    advance = (edge_means * field).dot(mesh.d[0] * vertices) * frequency         # Scalar+[edges]
    _, laplacian = core.connection(mesh, core.adjusted_transport(mesh, normals, angles + advance),
                                   core.cotangent_weights(mesh, vertices, regular))
    energies, stripe_fields = core.low_modes(laplacian, normals, areas, NORMAL_STIFFNESS, 2)
    stripe = stripe_fields.batch()[0]                                            # [vertices] Point-
    return field, (stripe / stripe.norm()).field(), advance, energies


def stripe_spheres() -> tuple[core.Mesh, core.Point, list[Pattern], list[core.Scalar]]:
    """The sphere and its vertices, and for each scene the stripes along its field and its quarter turn, with their energies."""
    mesh, vertices = core.Mesh.icosphere(STRIPE_SPHERE_LEVELS)
    normals, areas = core.vertex_normals(mesh, vertices), core.vertex_areas(mesh, vertices)
    weights = core.cotangent_weights(mesh, vertices, core.as_scalar(np.ones(len(mesh.faces))).field())
    singulars = [mesh.vertex_faces[corners] for corners in SPHERE_CORNERS]
    scenes = [pattern(mesh, vertices, normals, areas, weights, face_indices(mesh, singular, indices), [], SPHERE_FREQUENCY)
              for singular, indices in zip(singulars, SPHERE_INDICES)]
    return mesh, vertices, *map(list, zip(*scenes))


def stripe_tori() -> tuple[core.Mesh, core.Point, list[Pattern], list[core.Scalar]]:
    """The torus and its vertices, and for each scene the stripes along its field and its quarter turn, with their energies."""
    mesh, vertices = core.Mesh.torus(MAJOR_RADIUS, MINOR_RADIUS, N_STRIPE_MAJOR, N_STRIPE_MINOR)
    normals, areas = core.vertex_normals(mesh, vertices), core.vertex_areas(mesh, vertices)
    weights = core.cotangent_weights(mesh, vertices, core.as_scalar(np.ones(len(mesh.faces))).field())
    # Its two generators: once around the tube, and once around the axis along the outer equator.
    generators = [np.arange(N_STRIPE_MINOR), np.arange(N_STRIPE_MAJOR) * N_STRIPE_MINOR]
    scenes = [pattern(mesh, vertices, normals, areas, weights,
                      face_indices(mesh, places[:, 0] * N_STRIPE_MINOR + places[:, 1], indices), generators, TORUS_FREQUENCY)
              for places, indices in zip(STRIPE_TORUS_PLACES, TORUS_INDICES)]
    return mesh, vertices, *map(list, zip(*scenes))


def crosses() -> tuple[core.Mesh, core.Point, list[Pattern], list[core.Scalar]]:
    """The sphere and its vertices, and for each set of corners the two families of stripes across its cross field, with
    the connection's extra turns on the edges."""
    mesh, vertices = core.Mesh.icosphere(STRIPE_SPHERE_LEVELS)
    normals, areas = core.vertex_normals(mesh, vertices), core.vertex_areas(mesh, vertices)
    indices = [face_indices(mesh, nearest_faces(mesh, vertices, mv.vector(corners)), index)
               for corners, index in zip(CROSS_CORNERS, CROSS_INDICES)]
    angles = [core.trivial_connection(mesh, normals, index, [])[0] for index in indices]
    patterns = [core.cross_stripes(mesh, vertices, normals, areas, turns, core.carried_out(mesh, vertices, normals, turns),
                                   core.as_scalar(np.where(index == 0, 1.0, 0.0)).field(), CROSS_FREQUENCY, NORMAL_STIFFNESS)
                for index, turns in zip(indices, angles)]
    return mesh, vertices, patterns, angles


def sliding(mesh: core.Mesh, vertices: core.Point, stripes: core.Point) -> core.Point:
    """The stripe fields `[n] Point-[vertices]` turned about every normal through a whole turn,
    `[frames, n] Point-[vertices]`, are stripe fields at every turn, their stripes slid along their lines by the
    turn's share of one stripe."""
    angles = np.linspace(0, 2 * np.pi, N_SLIDING_FRAMES, endpoint=False)         # [frames]
    planes = core.vertex_normals(mesh, vertices).dual()                          # Bivector+[vertices]
    return (planes * (-angles[:, None] / 2)).exp() >> stripes                    # [frames, n] Point-[vertices]


# --- plumbing -------------------------------------------------------------------------
def face_indices(mesh: core.Mesh, singular: np.ndarray, indices: list[float]) -> np.ndarray:
    """The index of every face, `[faces]`, is the given index on the singular faces `[singular]`, and none elsewhere."""
    index = np.zeros(len(mesh.faces))
    index[singular] = indices
    return index


def nearest_faces(mesh: core.Mesh, vertices: core.Point, directions: core.Vector) -> np.ndarray:
    """The face whose centre, the mean of its corners, lies furthest along each direction `[n]`, read by the planes
    through the centre across them."""
    centres = (mesh.incidence[2, 0].mean(axis=0) * vertices).batch()             # [faces] Point-
    return np.argmax((centres[:, None] & directions.batch()[None]).to_array(), axis=0)


def stripe_mismatch(mesh: core.Mesh, vertices: core.Point, lines: core.Point, stripes: core.Point, advance: core.Scalar) -> np.ndarray:
    """The stripe phase across every edge misses its target by an angle `[n, edges]`; the phase is the angle from
    the stripe field to its line, about each vertex normal."""
    normals = core.vertex_normals(mesh, vertices)                                # Vector-[vertices]
    # The sine of the angle from the stripe field to its line: the plane their commutator turns in, read across the
    # tangent plane; the cosine is their inner product.
    sines = (lines.commutator(stripes) | normals.dual()).batch().to_array()      # [n, vertices]
    phase = core.as_scalar(np.arctan2(sines, stripes.dot(lines).batch().to_array())).field()   # [n] Scalar+[vertices]
    steps = (mesh.d[0] * phase - advance).batch().to_array()                     # [n, edges]
    return np.abs(np.angle(np.exp(1j * steps)))


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.surfaces.connections import render

    sphere_mesh, sphere_vertices, rings, starts = holonomy()
    save_animation(render.animate_holonomy(sphere_mesh, sphere_vertices, rings, carried_around(rings, starts, RING_STRIDE)),
                   "connections_holonomy", DURATION_MS)
    torus_mesh, torus_vertices, diffused, transported, modes, mode_energies, covariant = torus_fields()
    save_figure(render.draw_torus_fields(torus_mesh, torus_vertices, [transported, modes[0], modes[2]]), "connections_modes")
    save_animation(render.animate_diffusion(torus_mesh, torus_vertices, diffused), "connections_diffusion", DURATION_MS)

    sphere_mesh, sphere_vertices, sphere_scenes, sphere_residuals, sphere_energies = field_spheres()
    save_figure(render.draw_fields(sphere_mesh, sphere_vertices, sphere_scenes), "connections_fields_spheres")
    ring_mesh, ring_vertices, torus_scenes, torus_residuals, torus_energies = field_tori()
    save_figure(render.draw_fields(ring_mesh, ring_vertices, torus_scenes), "connections_fields_tori")
    field, singular, indices = sphere_scenes[0]
    save_animation(render.animate_turning(sphere_mesh, sphere_vertices, turning(sphere_mesh, sphere_vertices, field), singular,
                                          indices), "connections_turning", DURATION_MS)

    fine_mesh, fine_vertices, sphere_patterns, stripe_energies = stripe_spheres()
    bands = [tuple(family[:1] for family in pattern) for pattern in sphere_patterns]
    save_figure(render.draw_bands(fine_mesh, fine_vertices, bands), "connections_stripes_spheres")
    save_figure(render.draw_lines(fine_mesh, fine_vertices, sphere_patterns, 1.0), "connections_stripes_sphere_grids")
    stripe_torus, stripe_torus_vertices, torus_patterns, torus_stripe_energies = stripe_tori()
    bands = [tuple(family[:1] for family in pattern) for pattern in torus_patterns]
    save_figure(render.draw_bands(stripe_torus, stripe_torus_vertices, bands), "connections_stripes_tori")
    save_figure(render.draw_lines(stripe_torus, stripe_torus_vertices, torus_patterns, 1.0), "connections_stripes_torus_grids")
    _, _, cross_patterns, cross_angles = crosses()
    save_figure(render.draw_lines(fine_mesh, fine_vertices, cross_patterns, CROSS_SCALE), "connections_stripes_crosses")
    lines, stripe_fields, advance = (family[:1] for family in sphere_patterns[2])
    save_animation(render.animate_flow(fine_mesh, fine_vertices, lines, sliding(fine_mesh, fine_vertices, stripe_fields), advance),
                   "connections_stripes_sliding", DURATION_MS)

    # checks
    # Every transported vector stays tangent to the sphere and keeps unit length, and on closing the
    # loop it has turned by the enclosed cap's solid angle.
    carried = stack(tuple(core.transported(rings, starts)))                      # [steps + 1, rings] Point-
    # A direction's pairing with a normal, the tangent plane's patch, is its part along the normal.
    np.testing.assert_allclose((carried & rings).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose(carried.norm().to_array(), 1.0, atol=1e-12)
    np.testing.assert_allclose(carried[-1].dot(starts).to_array(), np.cos(2 * np.pi * (1 - np.cos(COLATITUDES))), atol=2e-3)
    # Each edge rotor carries the tail's normal to the head's, so neither heat diffusion nor the low
    # modes leak into the normal bundle.
    torus_normals = core.vertex_normals(torus_mesh, torus_vertices)                   # Vector-[vertices]
    np.testing.assert_allclose((diffused & torus_normals).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose((modes & torus_normals).to_array(), 0.0, atol=1e-11)
    np.testing.assert_allclose(transported.norm().to_array(), 1.0, atol=1e-12)
    # A quarter turn in every tangent plane commutes with the connection: the energies come in equal
    # pairs, and a quarter-turned mode keeps its energy.
    values = mode_energies.to_array()
    np.testing.assert_allclose(values[0::2], values[1::2], rtol=1e-10)
    edge_weights = core.spdiag(core.cotangent_weights(torus_mesh, torus_vertices, core.as_scalar(np.ones(len(torus_mesh.faces))).field()))   # Scalar+[edges <- edges]
    quarter_turned = modes.batch().commutator(torus_normals.batch().dual()).field()   # [modes] Point-[vertices]
    np.testing.assert_allclose((edge_weights * covariant(quarter_turned).norm_squared()).sites.sum().to_array(),
                               (edge_weights * covariant(modes).norm_squared()).sites.sum().to_array(), rtol=1e-10)
    # Every design's indices add up to the Euler characteristic: the connection is trivial, and its
    # field and the field's quarter turn carry no energy.
    np.testing.assert_allclose(np.concatenate([residual.to_array() for residual in sphere_residuals + torus_residuals]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.concatenate([energies.to_array() for energies in sphere_energies + torus_energies]), 0.0, atol=1e-12)
    sphere_normals = core.vertex_normals(sphere_mesh, sphere_vertices)                # Vector-[vertices]
    np.testing.assert_allclose(np.concatenate([(field & sphere_normals).to_array() for field, _, _ in sphere_scenes]), 0.0, atol=1e-12)
    # The face curvatures add up to two whole turns, and indices one short of the Euler characteristic
    # leave the missing turn spread over all faces.
    corner_normals = (sphere_mesh.incidence[2, 0] * sphere_normals).batch()   # [corners, faces] Vector-
    curvature = core.loop_turn(corner_normals)                                        # [faces] Scalar+
    np.testing.assert_allclose(curvature.sum(axis=0).to_array(), 4 * np.pi, rtol=1e-12)
    angles, circulation, target = core.trivial_connection(sphere_mesh, sphere_normals, face_indices(sphere_mesh, np.array([0]), [1]), [])
    np.testing.assert_allclose((circulation(angles) - target).to_array(), 2 * np.pi / len(sphere_mesh.faces), rtol=1e-9)
    # Every stripe field comes with its quarter turn, of equal energy, and across most edges its phase
    # advances by the edge's target; the edges around the forks miss it, many on a torus, whose stripes
    # must fork to space themselves evenly around its longer outer equator and shorter inner one.
    for mesh, vertices, patterns, energies in ((fine_mesh, fine_vertices, sphere_patterns, stripe_energies),
                                               (stripe_torus, stripe_torus_vertices, torus_patterns, torus_stripe_energies)):
        for (lines, stripes, advance), pair in zip(patterns, energies):
            np.testing.assert_allclose(pair.to_array()[:, 1], pair.to_array()[:, 0], rtol=1e-6)
            assert np.median(stripe_mismatch(mesh, vertices, lines, stripes, advance), axis=-1).max() < 0.25
    # Carried across every edge, each of the tail's lines lands along or against one of the head's
    # lines, and some edges trade the two lines or reverse them: around a singularity the cross turns
    # by a quarter or a half, not by a whole turn.
    fine_normals = core.vertex_normals(fine_mesh, fine_vertices)                      # Vector-[vertices]
    for (lines, _, _), angles in zip(cross_patterns, cross_angles):
        forward = core.adjusted_transport(fine_mesh, fine_normals, angles)[0]         # Rotor+[edges]
        tails, heads = fine_mesh.incidence[1, 0][:, None] * lines                     # [2] Point-[edges], [2] Point-[edges]
        match = heads[:, None].dot((forward >> tails)[None, :]).batch().to_array()    # [2, 2, edges]
        np.testing.assert_allclose(np.abs(match) * (1 - np.abs(match)), 0, atol=1e-8)
        assert (match[0, 0] < 0.5).any()


if __name__ == "__main__":
    main()
