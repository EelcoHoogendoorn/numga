"""Scene configurations and animation entry point for connections on curved surfaces."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import concatenate, stack
from numga.sparse import SparseExtensor, spdiag
from numga.topology import Topology
from examples.surfaces.connections import core
from examples.util import shapes

mv = core.mv
# Subdivision level of the sphere and colatitudes (30, 60, and 90 degrees) of its latitude rings.
sphere_levels = 3
colatitudes = np.array([np.pi / 6, np.pi / 3, np.pi / 2])
n_ring_steps = 210
ring_stride = 2
# Torus major and minor radii shared across heat, field, and stripe scenes.
major_radius = 2.0
minor_radius = 0.75
n_heat_major = 28
n_heat_minor = 14
source_width = 0.55
step_time = 0.1
n_heat_steps = 36
n_modes = 4
# Icosahedral corner indices carrying singularities on the sphere and their topological indices.
sphere_corners = [[9, 10], [9], [9, 4, 8, 3]]
sphere_indices = [[1, 1], [2], [1, 1, 1, -1]]
# Torus grid dimensions and singular face coordinates for designed tangent vector fields.
n_field_major = 40
n_field_minor = 20
field_torus_places = [np.zeros((0, 2), dtype=int), np.array([[32, 2], [0, 2]])]
torus_indices = [[], [1, -1]]
# Finer sphere and torus parameters for stripe and cross-stripe patterns.
stripe_sphere_levels = 4
sphere_frequency = 20.0
n_stripe_major = 80
n_stripe_minor = 40
stripe_torus_places = [np.zeros((0, 2), dtype=int), np.array([[64, 4], [0, 4]])]
torus_frequency = 12.0
# Cube and inscribed-tetrahedron singularity directions for quarter-turn and half-turn cross fields.
cube_directions = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                   [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]]
cross_corners = [cube_directions, [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]]
cross_indices = [0.25, 0.5]
cross_frequency = 20.0
cross_scale = 0.5
normal_stiffness = 100.0
n_turning_frames = 84
n_sliding_frames = 63
frame_duration_ms = 40

Pattern = tuple[core.Point, core.Point, core.Scalar]


# --- math -----------------------------------------------------------------------------
def holonomy_scene(levels: int, n_steps: int) -> tuple[Topology, core.Point, core.Patch, core.Point]:
    """Subdividing the unit icosahedron to `levels` builds the triangulated sphere (`Point-[V]`)
    alongside three closed latitude loops (`[n_steps + 1, 3] Patch-`) and their meridian start
    tangents (`[3] Point-`)."""
    topology, coordinates = shapes.sphere(levels)
    vertices = mv.vector(coordinates).dual().field()                                      # Point-[V]
    rings, starts = core.latitude_rings(colatitudes, n_steps)                             # [n_steps + 1, 3] Patch-, [3] Point-
    return topology, vertices, rings, starts


def torus_setup(n_major: int, n_minor: int
                ) -> tuple[Topology, core.Point, core.Patch, core.Scalar, core.Scalar, core.Point, SparseExtensor, SparseExtensor]:
    """Building an `n_major` by `n_minor` triangulated torus gives its vertices, unit patches,
    lumped areas, cotangent weights, localized outer-equator tangent impulse, covariant derivative,
    and connection Laplacian."""
    topology, coordinates = shapes.torus(major_radius, minor_radius, n_major, n_minor)
    V, E, F = topology.n_elements
    vertices = mv.vector(coordinates).dual().field()                                      # Point-[V]
    patches = core.vertex_patches(topology, vertices)                                     # Patch-[V]
    areas = core.vertex_areas(topology, vertices)                                         # Scalar+[V]
    unit_faces = mv.scalar(np.ones((F, 1))).field()                                       # Scalar+[F]
    weights = core.cotangent_weights(topology, vertices, unit_faces)                      # Scalar+[E]
    batch_patches = patches.batch()                                                       # [V] Patch-
    centre = (mv.x * (major_radius + minor_radius)).dual()                                # [] Point-
    bump = ((vertices.batch() - centre).norm_squared() * (-0.5 / source_width**2)).exp()  # [V] Scalar+
    direction = (mv.y + mv.z).dual().normalized()                                         # [] Point-
    tangent_direction = direction - batch_patches.dual() * (batch_patches & direction)    # [V] Point-
    source = (bump * tangent_direction).field()                                           # Point-[V]
    zero_turns = mv.scalar(np.zeros((E, 1))).field()                                      # Scalar+[E]
    covariant, laplacian = core.connection_operators(
        topology, core.adjusted_transport(topology, patches, zero_turns), weights,
    )
    return topology, vertices, patches, areas, weights, source, covariant, laplacian


def field_design(topology: Topology, patches: core.Patch, areas: core.Scalar, weights: core.Scalar,
                 index: np.ndarray, generators: list[np.ndarray]
                 ) -> tuple[core.Scalar, core.Point, core.Scalar, core.Scalar]:
    """Solving the trivial connection for triangle `index` (`[F]`) and handle `generators` returns
    its edge turns (`Scalar+[E]`), unit parallel field (`Point-[V]`), lowest two Dirichlet energies
    (`[2] Scalar+`), and holonomy circulation residual."""
    angles, circulation, target = core.trivial_connection(topology, patches, index, generators)
    energies, field = core.parallel_field(topology, patches, areas, weights, angles, normal_stiffness)
    return angles, field, energies, circulation(angles) - target


def sphere_field_scenes(levels: int
                        ) -> tuple[Topology, core.Point, list[tuple[core.Point, np.ndarray, list[int]]], list[core.Scalar], list[core.Scalar]]:
    """Three icosahedral singularity configurations on a `levels`-subdivided sphere determine
    trivial connections and their unit parallel tangent fields."""
    topology, coordinates = shapes.sphere(levels)
    V, E, F = topology.n_elements
    vertices = mv.vector(coordinates).dual().field()                                      # Point-[V]
    patches = core.vertex_patches(topology, vertices)                                     # Patch-[V]
    areas = core.vertex_areas(topology, vertices)                                         # Scalar+[V]
    weights = core.cotangent_weights(topology, vertices, mv.scalar(np.ones((F, 1))).field())  # Scalar+[E]
    singular_faces = [topology.vertex_simplices[corners] for corners in sphere_corners]
    designs = [field_design(topology, patches, areas, weights, face_indices(topology, faces, indices), [])
               for faces, indices in zip(singular_faces, sphere_indices)]
    scenes = [(field, faces, indices)
              for (_, field, _, _), faces, indices in zip(designs, singular_faces, sphere_indices)]
    return (topology, vertices, scenes,
            [residual for *_, residual in designs],
            [energies for _, _, energies, _ in designs])


def torus_field_scenes(n_major: int, n_minor: int
                       ) -> tuple[Topology, core.Point, list[tuple[core.Point, np.ndarray, list[int]]], list[core.Scalar], list[core.Scalar]]:
    """Singularity-free and opposite-index trivial connections on an `n_major` by `n_minor`
    triangulated torus determine two unit parallel tangent fields."""
    topology, coordinates = shapes.torus(major_radius, minor_radius, n_major, n_minor)
    V, E, F = topology.n_elements
    vertices = mv.vector(coordinates).dual().field()                                      # Point-[V]
    patches = core.vertex_patches(topology, vertices)                                     # Patch-[V]
    areas = core.vertex_areas(topology, vertices)                                         # Scalar+[V]
    weights = core.cotangent_weights(topology, vertices, mv.scalar(np.ones((F, 1))).field())  # Scalar+[E]
    generators = [np.arange(n_minor), np.arange(n_major) * n_minor]
    singular_faces = [places[:, 0] * n_minor + places[:, 1] for places in field_torus_places]
    designs = [field_design(topology, patches, areas, weights, face_indices(topology, faces, indices), generators)
               for faces, indices in zip(singular_faces, torus_indices)]
    scenes = [(field, faces, indices)
              for (_, field, _, _), faces, indices in zip(designs, singular_faces, torus_indices)]
    return (topology, vertices, scenes,
            [residual for *_, residual in designs],
            [energies for _, _, energies, _ in designs])


def orthogonal_pattern(topology: Topology, vertices: core.Point, patches: core.Patch,
                       areas: core.Scalar, weights: core.Scalar, index: np.ndarray,
                       generators: list[np.ndarray], frequency: float) -> tuple[Pattern, core.Scalar]:
    """A trivial-connection parallel field and its tangent-plane quarter turn drive orthogonal
    stripe families (`[2] Point-[V]`) at `frequency`, paired with their lowest mode energies
    (`[2, 2] Scalar+`)."""
    angles, field, _, _ = field_design(topology, patches, areas, weights, index, generators)
    regular_faces = mv.scalar(np.where(index == 0, 1.0, 0.0)[:, None]).field()            # Scalar+[F]
    orthogonal_field = field.batch().commutator(patches.batch().dual()).field()           # Point-[V]
    layers = [core.stripe_family(topology, vertices, patches, areas, angles, direction,
                                 regular_faces, frequency, normal_stiffness)
              for direction in (field, orthogonal_field)]
    *families, energies = zip(*layers)
    return tuple(stack(family) for family in families), stack(energies)


def sphere_stripe_scenes(levels: int) -> tuple[Topology, core.Point, list[Pattern], list[core.Scalar]]:
    """Three icosahedral singularity configurations on a `levels`-subdivided sphere produce
    orthogonal stripe patterns and mode-energy pairs."""
    topology, coordinates = shapes.sphere(levels)
    V, E, F = topology.n_elements
    vertices = mv.vector(coordinates).dual().field()                                      # Point-[V]
    patches = core.vertex_patches(topology, vertices)                                     # Patch-[V]
    areas = core.vertex_areas(topology, vertices)                                         # Scalar+[V]
    weights = core.cotangent_weights(topology, vertices, mv.scalar(np.ones((F, 1))).field())  # Scalar+[E]
    singular_faces = [topology.vertex_simplices[corners] for corners in sphere_corners]
    scenes = [orthogonal_pattern(topology, vertices, patches, areas, weights,
                                 face_indices(topology, faces, indices), [], sphere_frequency)
              for faces, indices in zip(singular_faces, sphere_indices)]
    return topology, vertices, *map(list, zip(*scenes))


def torus_stripe_scenes(n_major: int, n_minor: int
                        ) -> tuple[Topology, core.Point, list[Pattern], list[core.Scalar]]:
    """Singularity-free and opposite-index trivial connections on an `n_major` by `n_minor` torus
    produce orthogonal stripe patterns and mode-energy pairs."""
    topology, coordinates = shapes.torus(major_radius, minor_radius, n_major, n_minor)
    V, E, F = topology.n_elements
    vertices = mv.vector(coordinates).dual().field()                                      # Point-[V]
    patches = core.vertex_patches(topology, vertices)                                     # Patch-[V]
    areas = core.vertex_areas(topology, vertices)                                         # Scalar+[V]
    weights = core.cotangent_weights(topology, vertices, mv.scalar(np.ones((F, 1))).field())  # Scalar+[E]
    generators = [np.arange(n_minor), np.arange(n_major) * n_minor]
    scenes = [orthogonal_pattern(topology, vertices, patches, areas, weights,
                                 face_indices(topology, places[:, 0] * n_minor + places[:, 1], indices),
                                 generators, torus_frequency)
              for places, indices in zip(stripe_torus_places, torus_indices)]
    return topology, vertices, *map(list, zip(*scenes))


def cross_stripe_scenes(levels: int) -> tuple[Topology, core.Point, list[Pattern], list[core.Scalar]]:
    """Cube-corner (`+1/4`) and tetrahedron-corner (`+1/2`) cross fields on a `levels`-subdivided
    sphere produce coupled two-family stripe patterns."""
    topology, coordinates = shapes.sphere(levels)
    vertices = mv.vector(coordinates).dual().field()                                      # Point-[V]
    patches = core.vertex_patches(topology, vertices)                                     # Patch-[V]
    areas = core.vertex_areas(topology, vertices)                                         # Scalar+[V]
    indices = [face_indices(topology, nearest_faces(topology, vertices, mv.vector(corners)), index)
               for corners, index in zip(cross_corners, cross_indices)]
    angles = [core.trivial_connection(topology, patches, index, [])[0] for index in indices]
    patterns = [
        core.cross_stripes(
            topology, vertices, patches, areas, turns,
            core.spanning_tree_field(topology, vertices, patches, turns),
            mv.scalar(np.where(index == 0, 1.0, 0.0)[:, None]).field(),
            cross_frequency, normal_stiffness,
        )
        for index, turns in zip(indices, angles)
    ]
    return topology, vertices, patterns, angles


def stripe_phase_mismatch(topology: Topology, vertices: core.Point, lines: core.Point,
                          stripes: core.Point, advance: core.Scalar) -> core.Scalar:
    """Measuring each vertex's tangent-plane rotor angle from `stripes` to `lines` and wrapping the
    edge difference minus `advance` into `[0, pi]` gives the edge phase mismatch
    (`[..., E] Scalar+`)."""
    patches = core.vertex_patches(topology, vertices)                                     # Patch-[V]
    phase_rotors = (stripes * lines.reverse()).normalized()                               # [...] Rotor+[V]
    phases = phase_rotors.log().scalar_product(patches.dual())                            # [...] Scalar+[V]
    steps = (topology.d[0] * phases - advance).batch()                                    # [..., E] Scalar+
    return ((mv.xy * steps).exp().log() | mv.xy).abs()                                    # [..., E] Scalar+


# --- plumbing -------------------------------------------------------------------------
def face_indices(topology: Topology, singular_faces: np.ndarray, indices: list[float] | float) -> np.ndarray:
    """The triangles in `singular_faces` take topological `indices` while all remaining triangles
    take zero (`[F]`)."""
    V, E, F = topology.n_elements
    index = np.zeros(F)
    index[singular_faces] = indices
    return index


def nearest_faces(topology: Topology, vertices: core.Point, planes: core.Plane) -> np.ndarray:
    """Each normal plane in `planes` picks the triangle index whose centroid pairs maximally with
    it."""
    centres = (topology.incidence[2, 0].mean(axis=0) * vertices).batch()                  # [F] Point-
    return (planes.batch()[None] & centres[:, None]).argmax(axis=0)


def main() -> None:
    """Running the scenario suite writes all holonomy, heat-diffusion, designed-field, and stripe
    visuals and checks geometric and topological invariants on the computed states."""
    from examples.surfaces.connections import render
    from examples.util.animation import save_animation, save_figure

    sphere_topology, sphere_vertices, rings, starts = holonomy_scene(sphere_levels, n_ring_steps)
    carried_vectors = []

    def streamed_holonomy() -> Iterator[tuple[core.Patch, core.Point]]:
        for step_index, (ring_patch, vector) in enumerate(zip(rings, core.parallel_transport(rings, starts))):
            carried_vectors.append(vector)
            if step_index % ring_stride == 0:
                yield ring_patch, vector

    save_animation(render.holonomy_animation(sphere_topology, sphere_vertices, rings, streamed_holonomy()),
                   "connections_holonomy", frame_duration_ms)
    carried_tangents = stack(carried_vectors)                                             # [n_ring_steps + 1, 3] Point-

    (torus_topology, torus_vertices, torus_patches, torus_areas,
     torus_weights, torus_source, covariant, laplacian) = torus_setup(n_heat_major, n_heat_minor)
    diffusion_steps = []

    def streamed_diffusion() -> Iterator[core.Point]:
        for step_field in core.heat_diffusion(laplacian, torus_areas, torus_source, step_time, n_heat_steps):
            diffusion_steps.append(step_field)
            yield step_field

    save_animation(render.diffusion_animation(torus_topology, torus_vertices, streamed_diffusion()),
                   "connections_diffusion", frame_duration_ms)
    diffused_fields = stack(diffusion_steps)                                              # [n_heat_steps] Point-[V]
    transported_field = diffused_fields[-1] / diffused_fields[-1].norm()                  # Point-[V]
    mode_energies, modes = core.low_modes(laplacian, torus_patches, torus_areas, normal_stiffness, n_modes)
    save_figure(render.torus_fields_figure(torus_topology, torus_vertices,
                                           [transported_field, modes[0], modes[2]]), "connections_modes")

    sphere_topology, sphere_vertices, sphere_scenes, sphere_residuals, sphere_energies = sphere_field_scenes(sphere_levels)
    save_figure(render.fields_figure(sphere_topology, sphere_vertices, sphere_scenes), "connections_fields_spheres")
    ring_topology, ring_vertices, torus_scenes, torus_residuals, torus_energies = torus_field_scenes(n_field_major, n_field_minor)
    save_figure(render.fields_figure(ring_topology, ring_vertices, torus_scenes), "connections_fields_tori")

    sphere_patches = core.vertex_patches(sphere_topology, sphere_vertices)                # Patch-[V]
    first_field, _, _ = sphere_scenes[0]
    save_animation(
        render.turning_animation(sphere_topology, sphere_vertices,
                                 core.turned_fields(sphere_patches, first_field, n_turning_frames)),
        "connections_turning", frame_duration_ms,
    )

    fine_topology, fine_vertices, sphere_patterns, stripe_energies = sphere_stripe_scenes(stripe_sphere_levels)
    sphere_bands = [tuple(family[:1] for family in pattern) for pattern in sphere_patterns]
    save_figure(render.bands_figure(fine_topology, fine_vertices, sphere_bands), "connections_stripes_spheres")
    stripe_torus, stripe_torus_vertices, torus_patterns, torus_stripe_energies = torus_stripe_scenes(n_stripe_major, n_stripe_minor)
    torus_bands = [tuple(family[:1] for family in pattern) for pattern in torus_patterns]
    save_figure(render.bands_figure(stripe_torus, stripe_torus_vertices, torus_bands), "connections_stripes_tori")
    _, _, cross_patterns, cross_angles = cross_stripe_scenes(stripe_sphere_levels)
    save_figure(render.lines_figure(fine_topology, fine_vertices, cross_patterns, cross_scale),
                "connections_stripes_crosses")

    fine_patches = core.vertex_patches(fine_topology, fine_vertices)                      # Patch-[V]
    lines, stripe_fields, advance = (family[:1] for family in sphere_patterns[2])
    save_animation(
        render.sliding_animation(fine_topology, fine_vertices, lines,
                                 core.turned_fields(fine_patches, stripe_fields, n_sliding_frames), advance),
        "connections_stripes_sliding", frame_duration_ms,
    )

    # checks
    # Transported vectors stay orthogonal to the sphere patches, keep unit length, and rotate by the cap solid angle.
    assert ((carried_tangents & rings).antinorm() < 1e-10).all()
    assert ((carried_tangents.norm() - 1).abs() < 1e-10).all()
    expected_cosines = mv.scalar(np.cos(2 * np.pi * (1 - np.cos(colatitudes)))[:, None])  # [3] Scalar+
    assert ((-(carried_tangents[-1] | starts) - expected_cosines).abs() < 2e-3).all()
    # Heat diffusion and low-energy connection modes stay purely tangent, and quarter-turn modes share equal energy.
    assert ((diffused_fields & torus_patches).batch().antinorm() < 1e-10).all()
    assert ((modes & torus_patches).batch().antinorm() < 1e-10).all()
    assert ((transported_field.norm() - 1).batch().abs() < 1e-10).all()
    assert ((mode_energies[0::2] - mode_energies[1::2]).abs() < 1e-9).all()
    edge_weights = spdiag(torus_weights)                                                  # Scalar+[E <- E]
    quarter_turn_modes = modes.batch().commutator(torus_patches.batch().dual()).field()   # [n_modes] Point-[V]
    mode_dirichlet = (edge_weights * covariant(modes).norm_squared()).sites.sum()         # [n_modes] Scalar+
    turned_dirichlet = (edge_weights * covariant(quarter_turn_modes).norm_squared()).sites.sum()  # [n_modes] Scalar+
    assert ((turned_dirichlet - mode_dirichlet).abs() < 1e-9).all()
    # Trivial connections with Euler-characteristic index sum have zero circulation residual and zero parallel energy.
    all_residuals = concatenate([r.batch() for r in sphere_residuals + torus_residuals])
    all_energies = concatenate(sphere_energies + torus_energies)
    assert (all_residuals.abs() < 1e-10).all()
    assert (all_energies.abs() < 1e-9).all()
    for field, _, _ in sphere_scenes:
        assert ((field & sphere_patches).batch().antinorm() < 1e-10).all()
    # Discrete Gauss-Bonnet sums face curvatures to 4 * pi on the sphere, and an index sum short by 1 spreads 2 * pi / F.
    V_sphere, E_sphere, F_sphere = sphere_topology.n_elements
    corner_patches = (sphere_topology.incidence[2, 0] * sphere_patches).batch()           # [3, F_sphere] Patch-
    curvature = core.loop_holonomy(corner_patches)                                        # [F_sphere] Scalar+
    assert ((curvature.sum(axis=0) - 4 * np.pi).abs() < 1e-10).all()
    unmet_angles, unmet_circulation, unmet_target = core.trivial_connection(
        sphere_topology, sphere_patches, face_indices(sphere_topology, np.array([0]), [1]), [],
    )
    unmet_holonomy = (unmet_circulation(unmet_angles) - unmet_target).batch()             # [F_sphere] Scalar+
    assert ((unmet_holonomy - 2 * np.pi / F_sphere).abs() < 1e-8).all()
    # Stripe eigenmodes come in equal-energy pairs and follow their edge phase advance away from forks.
    for surface_top, surface_verts, patterns, energies in (
        (fine_topology, fine_vertices, sphere_patterns, stripe_energies),
        (stripe_torus, stripe_torus_vertices, torus_patterns, torus_stripe_energies),
    ):
        for (pat_lines, pat_stripes, pat_advance), pair in zip(patterns, energies):
            assert ((pair[:, 1] - pair[:, 0]).abs() < 1e-5 * (pair[:, 0].abs() + 1)).all()
            mismatch = stripe_phase_mismatch(surface_top, surface_verts, pat_lines, pat_stripes, pat_advance)
            n_edges = mismatch.shape[-1]
            median_index = n_edges // 2
            # At least half of the edges track the prescribed phase advance closely.
            assert ((mismatch < 0.25).sum(axis=-1) > median_index).all()
    # Cross-field lines map onto each other up to sign across every edge, with nontrivial swaps around singularities.
    for (cross_lines, _, _), angles in zip(cross_patterns, cross_angles):
        edge_rotors = core.adjusted_transport(fine_topology, fine_patches, angles)[0]     # Rotor+[E]
        tails, heads = fine_topology.incidence[1, 0][:, None] * cross_lines               # [2] Point-[E], [2] Point-[E]
        transported_lines = edge_rotors >> tails                                          # [2] Point-[E]
        alignment = (-(heads[:, None] | transported_lines[None, :])).batch()              # [2, 2, E] Scalar+
        assert ((alignment.abs() * (1 - alignment.abs())).abs() < 1e-7).all()
        assert (alignment[0, 0] < 0.5).any()


if __name__ == "__main__":
    main()
