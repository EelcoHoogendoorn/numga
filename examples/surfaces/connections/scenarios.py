"""Scenes for connections on curved surfaces: holonomy around latitude circles on the unit sphere, and
vector heat transport and harmonic tangent fields on a torus; parallel fields of trivial connections on a
sphere with two singularities of index one at opposite poles, one singularity of index two, and three of
index one with one of index minus one, on a torus with none at all and with a pair of opposite indices,
and a field turned through a whole turn about every normal; stripes along those fields and across cross
fields, and stripes sliding along their field as the stripe field turns."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.surfaces.connections import core

mv = core.mv
# The sphere and its three latitude circles: 30, 60 and 90 degrees from the north pole.
SPHERE_LEVELS = 3
COLATITUDES = np.array([np.pi / 6, np.pi / 3, np.pi / 2])
RING_STEPS = 120
# The torus of every scene.
MAJOR_RADIUS = 2.0
MINOR_RADIUS = 0.75
# The torus of the vector heat transport and the harmonic modes, the localized tangent bump on its outer
# equator, the vector heat steps, and the harmonic modes.
HEAT_MAJOR_COUNT = 28
HEAT_MINOR_COUNT = 14
SOURCE_WIDTH = 0.55
STEP_TIME = 0.18
HEAT_STEPS = 20
MODE_COUNT = 4
# For each scene on the sphere, the icosahedron's vertices whose faces carry a singularity and the index
# of each; vertices 9 and 10 are opposite each other.
SPHERE_CORNERS = [[9, 10], [9], [9, 4, 8, 3]]
SPHERE_INDICES = [[1, 1], [2], [1, 1, 1, -1]]
# The torus of the designed fields, and for each of its scenes the faces, by their place around the axis
# and around the tube, that carry a singularity, and the index of each.
FIELD_MAJOR_COUNT = 40
FIELD_MINOR_COUNT = 20
FIELD_TORUS_PLACES = [np.zeros((0, 2), dtype=int), np.array([[32, 2], [0, 2]])]
TORUS_INDICES = [[], [1, -1]]
# The sphere of the stripes and their frequency on it; the torus of the stripes, the places of its
# singular faces, and the stripes' frequency on it.
STRIPE_SPHERE_LEVELS = 4
SPHERE_FREQUENCY = 20.0
STRIPE_MAJOR_COUNT = 80
STRIPE_MINOR_COUNT = 40
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
# How far around the centre the views of the striped torus reach.
TORUS_REACH = 2.9
# The frames of a whole turn of the field, of the stripes sliding by one stripe, and the duration of a frame.
TURNING_FRAMES = 48
SLIDING_FRAMES = 36
DURATION_MS = 70


# --- math -----------------------------------------------------------------------------
Pattern = tuple[core.Vector, core.Vector, core.Scalar]


def holonomy():
    """A meridian tangent vector carried around three latitude circles on the unit sphere: the sphere,
    the circles `[steps + 1, rings] Vector`, and the transported vectors `[steps + 1, rings] Vector`."""
    sphere = core.Mesh.icosphere(SPHERE_LEVELS)
    rings, starts = core.latitude_rings(COLATITUDES, RING_STEPS)                # [steps + 1, rings] Vector, [rings] Vector
    carried = core.loop_transport(rings, starts)                                # [steps + 1, rings] Vector

    # --- checks
    # Every transported vector stays tangent to the sphere and keeps unit length.
    np.testing.assert_allclose((carried | rings).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose(carried.norm().to_array(), 1.0, atol=1e-12)
    # On closing the loop, the vector has turned by the enclosed cap's solid angle.
    cos_turn = (carried[-1] | starts).to_array()
    expected = np.cos(2 * np.pi * (1 - np.cos(COLATITUDES)))
    np.testing.assert_allclose(cos_turn, expected, atol=2e-3)
    return sphere, rings, carried


def torus_fields():
    """Vector heat parallel transport and the smoothest global tangent fields on a torus: the torus, the
    diffused waves `[steps] Vector[V]`, the unit transported field `Vector[V]`, and the harmonic modes
    `[modes] Vector[V]`."""
    torus = core.Mesh.torus(MAJOR_RADIUS, MINOR_RADIUS, HEAT_MAJOR_COUNT, HEAT_MINOR_COUNT)
    normals = torus.vertex_normals.batch()                                      # [V] Vector
    centre = mv.x * (MAJOR_RADIUS + MINOR_RADIUS)                               # [] Vector
    weight = ((torus.vertices.batch() - centre).norm_squared() * (-0.5 / SOURCE_WIDTH**2)).exp()   # [V] Scalar
    direction = (mv.y + mv.z).normalized()                                      # [] Vector
    tangent_direction = direction - normals * (normals | direction)             # [V] Vector
    source = (weight * tangent_direction).field()                               # Vector[V]

    # The surface's own transport: no extra turn on any edge.
    no_turns = core.as_scalar(np.zeros(len(torus.edges))).field()               # Scalar[E]
    covariant, laplacian = core.connection(torus, core.adjusted_transport(torus, no_turns), torus.edge_ratio)
    steps = list(core.heat_transport(laplacian, torus.vertex_areas, source, STEP_TIME, HEAT_STEPS))
    diffused = stack([wave for wave, _ in steps])                               # [steps] Vector[V]
    transported = steps[-1][1]                                                  # Vector[V]
    energies, modes = core.low_modes(laplacian, torus.vertex_normals, torus.vertex_areas, NORMAL_STIFFNESS, MODE_COUNT)

    # --- checks
    # Because each edge rotor maps tail normal to head normal, neither heat diffusion nor the
    # connection Laplacian's low modes leak into the normal bundle.
    np.testing.assert_allclose((diffused | torus.vertex_normals).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose((modes | torus.vertex_normals).to_array(), 0.0, atol=1e-11)
    np.testing.assert_allclose(transported.norm().to_array(), 1.0, atol=1e-12)
    # Turning every tangent vector a quarter turn in its tangent plane commutes with the connection,
    # so eigenvalues come in equal pairs and quarter-turned fields keep the same Dirichlet energy.
    values = energies.to_array()
    np.testing.assert_allclose(values[0], values[1], rtol=1e-10)
    np.testing.assert_allclose(values[2], values[3], rtol=1e-10)
    H1 = core.spdiag(torus.edge_ratio)
    quarter_turned = (modes.batch() | normals.dual()).field()                   # [modes] Vector[V]
    mode_energy = (H1 * covariant(modes).norm_squared()).sites.sum().to_array()
    turned_energy = (H1 * covariant(quarter_turned).norm_squared()).sites.sum().to_array()
    np.testing.assert_allclose(mode_energy, turned_energy, rtol=1e-10)
    return torus, diffused, transported, modes


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


def field_spheres() -> tuple[core.Mesh, list[tuple[core.Vector, np.ndarray, list[int]]]]:
    """The sphere, and for each scene its unit parallel field, singular faces and their indices."""
    sphere = core.Mesh.icosphere(SPHERE_LEVELS)
    scenes = []
    for corners, indices in zip(SPHERE_CORNERS, SPHERE_INDICES):
        singular = np.array([core.faces_around(sphere, corner) for corner in corners])
        _, field, energies, residual = design(sphere, face_indices(sphere, singular, indices), [])
        scenes.append((field, singular, indices))

        # --- checks
        # The indices add up to the sphere's Euler characteristic, the connection is trivial, and the
        # field and its quarter turn carry no energy.
        np.testing.assert_allclose(residual.to_array(), 0.0, atol=1e-12)
        np.testing.assert_allclose(energies.to_array(), 0.0, atol=1e-12)
        np.testing.assert_allclose((field | sphere.vertex_normals).to_array(), 0.0, atol=1e-12)

    # --- checks
    # The face curvatures add up to two whole turns, the sphere's Euler characteristic.
    curvature = core.loop_turn(sphere.vertex_normals.batch()[sphere.faces])      # [F] Scalar
    np.testing.assert_allclose(curvature.sum(axis=0).to_array(), 4 * np.pi, rtol=1e-12)
    # Indices that miss the Euler characteristic by one leave the missing turn spread over all faces.
    lone = np.zeros(len(sphere.faces))
    lone[0] = 1
    angles, circulation, target = core.trivial_connection(sphere, lone, [])
    residual = (circulation(angles) - target).to_array()
    np.testing.assert_allclose(residual, 2 * np.pi / len(sphere.faces), rtol=1e-9)
    return sphere, scenes


def field_tori() -> tuple[core.Mesh, list[tuple[core.Vector, np.ndarray, list[int]]]]:
    """The torus, and for each scene its unit parallel field, singular faces and their indices."""
    torus = core.Mesh.torus(MAJOR_RADIUS, MINOR_RADIUS, FIELD_MAJOR_COUNT, FIELD_MINOR_COUNT)
    # Its two generators: once around the tube, and once around the axis along the outer equator.
    generators = [np.arange(FIELD_MINOR_COUNT), np.arange(FIELD_MAJOR_COUNT) * FIELD_MINOR_COUNT]
    scenes = []
    for places, indices in zip(FIELD_TORUS_PLACES, TORUS_INDICES):
        singular = places[:, 0] * FIELD_MINOR_COUNT + places[:, 1]
        _, field, energies, residual = design(torus, face_indices(torus, singular, indices), generators)
        scenes.append((field, singular, indices))

        # --- checks
        np.testing.assert_allclose(residual.to_array(), 0.0, atol=1e-12)
        np.testing.assert_allclose(energies.to_array(), 0.0, atol=1e-12)
    return torus, scenes


def turning(mesh: core.Mesh, field: core.Vector) -> core.Vector:
    """The parallel field turned about every normal through a whole turn `[frames] Vector[V]`: every
    turned field is parallel too, the null space being a field and its quarter turn."""
    angles = np.linspace(0, 2 * np.pi, TURNING_FRAMES, endpoint=False)           # [frames]
    planes = mesh.vertex_normals.batch().dual()                                  # [V] Bivector
    return ((planes * (-angles[:, None] / 2)).exp() >> field.batch()).field()   # [frames] Vector[V]


def pattern(mesh: core.Mesh, index: np.ndarray, generators: list[np.ndarray], frequency: float) -> Pattern:
    """The stripes along the unit direction field of the trivial connection with every face turning by its
    index `[F]`, and along the field turned a quarter turn: the two lines `[V, 2]`, the unit stripe field
    of each `[V, 2]`, and the edges' targets for each `[E, 2]`."""
    angles, field, _, _ = design(mesh, index, generators)
    # Every face is regular except the singular ones.
    regular = core.as_scalar(np.where(index == 0, 1.0, 0.0)).field()           # Scalar[F]
    layers = [along(mesh, angles, direction, regular, frequency) for direction in (field, core.quarter_turn(mesh, field))]
    return tuple(stack([layer.batch() for layer in family], axis=-1) for family in zip(*layers))


def along(mesh: core.Mesh, angles: core.Scalar, field: core.Vector, regular: core.Scalar, frequency: float
          ) -> tuple[core.Vector, core.Vector, core.Scalar]:
    """The unit stripe field along a direction field parallel for the connection of the given turns, at
    the given frequency, with the field and the edges' targets: the smoothest tangent field of the
    connection of the turns `angles` with every edge's phase advance added. Each face adds its
    half-cotangents to its edges' weights only where it is regular `Scalar[F]`, one, and not where the
    direction field turns inside it, zero: there the field's average across an edge means nothing."""
    advance = core.targets(mesh, field, frequency)                               # Scalar[E]
    _, laplacian = core.connection(mesh, core.adjusted_transport(mesh, angles + advance), mesh.weighted_edge_ratio(regular))
    energies, stripe_fields = core.low_modes(laplacian, mesh.vertex_normals, mesh.vertex_areas, NORMAL_STIFFNESS, 2)
    stripe = stripe_fields.batch()[0]                                            # [V] Vector
    stripe_field = (stripe / stripe.norm()).field()

    # --- checks
    # The stripe field comes with its quarter turn, of equal energy.
    np.testing.assert_allclose(energies.to_array()[1], energies.to_array()[0], rtol=1e-6)
    # Across most edges the phase, the angle from the stripe field to the direction field, advances by
    # the edge's target; the edges around the forks miss it, many on a torus, whose stripes must fork to
    # space themselves evenly around its longer outer equator and shorter inner one.
    normals = mesh.vertex_normals.batch()                                        # [V] Vector
    cosines = (stripe_field.batch() | field.batch()).to_array()                  # [V]
    sines = ((stripe_field.batch() ^ field.batch()) | normals.dual()).to_array()
    phase = np.arctan2(sines, cosines)
    steps = phase[mesh.edges[:, 1]] - phase[mesh.edges[:, 0]] - advance.batch().to_array()
    mismatch = np.abs(np.angle(np.exp(1j * steps)))                              # [E]
    assert np.median(mismatch) < 0.25
    return field, stripe_field, advance


def cross(mesh: core.Mesh, index: np.ndarray, frequency: float) -> Pattern:
    """The two families of stripes across the cross field of the trivial connection with every face
    turning by its quarter-turn index `[F]`."""
    angles, _, _ = core.trivial_connection(mesh, index, [])
    regular = core.as_scalar(np.where(index == 0, 1.0, 0.0)).field()           # Scalar[F]
    lines, stripes, advance = core.cross_stripes(mesh, angles, core.carried_out(mesh, angles), regular, frequency, NORMAL_STIFFNESS)

    # --- checks
    # Carried across every edge, each of the tail's lines lands along or against one of the head's lines.
    forward = core.adjusted_transport(mesh, angles)[:, 0]                        # [E] Rotor
    tails, heads = lines[mesh.edges[:, 0]], lines[mesh.edges[:, 1]]              # [E, 2] Vector each
    match = (heads[:, :, None] | (forward[:, None] >> tails)[:, None, :]).to_array()   # [E, 2, 2]
    np.testing.assert_allclose(np.abs(match) * (1 - np.abs(match)), 0, atol=1e-8)
    # Some edges trade the two lines or reverse them: around a singularity the cross turns by a quarter
    # or a half, not by a whole turn.
    assert (match[:, 0, 0] < 0.5).any()
    return lines, stripes, advance


def stripe_spheres() -> tuple[core.Mesh, list[Pattern]]:
    """The sphere, and for each scene the stripes along its field and its quarter turn."""
    sphere = core.Mesh.icosphere(STRIPE_SPHERE_LEVELS)
    scenes = []
    for corners, indices in zip(SPHERE_CORNERS, SPHERE_INDICES):
        singular = np.array([core.faces_around(sphere, corner) for corner in corners])
        scenes.append(pattern(sphere, face_indices(sphere, singular, indices), [], SPHERE_FREQUENCY))
    return sphere, scenes


def stripe_tori() -> tuple[core.Mesh, list[Pattern]]:
    """The torus, and for each scene the stripes along its field and its quarter turn."""
    torus = core.Mesh.torus(MAJOR_RADIUS, MINOR_RADIUS, STRIPE_MAJOR_COUNT, STRIPE_MINOR_COUNT)
    # Its two generators: once around the tube, and once around the axis along the outer equator.
    generators = [np.arange(STRIPE_MINOR_COUNT), np.arange(STRIPE_MAJOR_COUNT) * STRIPE_MINOR_COUNT]
    scenes = []
    for places, indices in zip(STRIPE_TORUS_PLACES, TORUS_INDICES):
        singular = places[:, 0] * STRIPE_MINOR_COUNT + places[:, 1]
        scenes.append(pattern(torus, face_indices(torus, singular, indices), generators, TORUS_FREQUENCY))
    return torus, scenes


def crosses() -> tuple[core.Mesh, list[Pattern]]:
    """The sphere, and for each set of corners the two families of stripes across its cross field."""
    sphere = core.Mesh.icosphere(STRIPE_SPHERE_LEVELS)
    return sphere, [cross(sphere, face_indices(sphere, nearest_faces(sphere, mv.vector(corners)), index), CROSS_FREQUENCY)
                    for corners, index in zip(CROSS_CORNERS, CROSS_INDICES)]


def sliding(mesh: core.Mesh, stripes: core.Vector) -> core.Vector:
    """The stripe fields `[V, n]` turned about every normal through a whole turn `[frames, V, n]`: every
    turn of one is a stripe field too, its stripes slid along its line by the turn's share of one
    stripe."""
    angles = np.linspace(0, 2 * np.pi, SLIDING_FRAMES, endpoint=False)           # [frames]
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
    return np.argmax((mesh.face_centers.batch()[:, None] | directions.batch()[None]).to_array(), axis=0)


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.surfaces.connections import render

    sphere, rings, carried = holonomy()
    save_figure(render.draw_holonomy(sphere, rings, carried), "connections_holonomy")
    torus, diffused, transported, modes = torus_fields()
    save_figure(render.draw_torus_fields(torus, [transported, modes[0], modes[2]]), "connections_modes")
    save_animation(render.animate_diffusion(torus, diffused), "connections_diffusion", DURATION_MS)

    sphere, sphere_scenes = field_spheres()
    save_figure(render.draw_fields(sphere, sphere_scenes, render.SPHERE_LIMITS, render.SPHERE_NEEDLE), "connections_fields_spheres")
    torus, torus_scenes = field_tori()
    save_figure(render.draw_fields(torus, torus_scenes, render.TORUS_LIMITS, render.TORUS_NEEDLE), "connections_fields_tori")
    field, singular, indices = sphere_scenes[0]
    save_animation(render.animate_turning(sphere, turning(sphere, field), singular, indices), "connections_turning",
                   DURATION_MS)

    views = np.broadcast_to(render.VIEW, (len(SPHERE_CORNERS), 2))               # every panel seen alike
    sphere, sphere_patterns = stripe_spheres()
    bands = [tuple(family[..., :1] for family in pattern) for pattern in sphere_patterns]
    save_figure(render.draw_patterns(sphere, bands, render.draw_bands, render.SPHERE_REACH, views), "connections_stripes_spheres")
    save_figure(render.draw_patterns(sphere, sphere_patterns, render.draw_lines, render.SPHERE_REACH, views),
                "connections_stripes_sphere_grids")
    torus, torus_patterns = stripe_tori()
    bands = [tuple(family[..., :1] for family in pattern) for pattern in torus_patterns]
    save_figure(render.draw_patterns(torus, bands, render.draw_bands, TORUS_REACH, views), "connections_stripes_tori")
    save_figure(render.draw_patterns(torus, torus_patterns, render.draw_lines, TORUS_REACH, views), "connections_stripes_torus_grids")
    sphere, cross_patterns = crosses()
    save_figure(render.draw_patterns(sphere, cross_patterns, render.draw_lines, render.SPHERE_REACH, views), "connections_stripes_crosses")
    lines, stripe_fields, advance = (family[..., :1] for family in sphere_patterns[2])
    save_animation(render.animate_flow(sphere, lines, sliding(sphere, stripe_fields), advance, render.SPHERE_REACH),
                   "connections_stripes_sliding", DURATION_MS)
