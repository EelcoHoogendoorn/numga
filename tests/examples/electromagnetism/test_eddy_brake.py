"""Charge conservation, magnetic braking, cotangent correspondence, and covariance of the
conducting disc solver and renderer."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from numga.topology import Topology
from examples.util import shapes
from examples.electromagnetism.eddy_brake import core, render, scenarios

principal_axes = stack((core.mv.x.dual(), core.mv.y.dual()))                 # [2] Current-
equal_weights = np.array([1.0, 1.0])                                         # [2]


def disc_from_shape(shape: tuple[Topology, np.ndarray]) -> tuple:
    """Triangulated disc geometry unpacks from a `(topology, coordinates)` shape pair."""
    topology, coordinates = shape
    return scenarios.disc_geometry(topology, coordinates)


def uniform_conductivity(principal_conductances: np.ndarray) -> core.Conductivity:
    """Principal conductances along `x` and `y` combine into a spatially uniform sheet conductivity."""
    return (principal_axes * (core.Field & (principal_axes & core.time)) * np.asarray(principal_conductances)).sum(axis=-1)


def test_uniform_field_is_cancelled_by_radial_potential():
    radius, n_rings, n_sectors = 0.1, 4, 24
    strength, speed, conductance = 0.2, 1.0, 5.8e4
    topology, vertices, edge_displacements, edge_midpoints, _, _, triangle_areas, _, reconstruction = disc_from_shape(
        shapes.disk(radius, n_rings, n_sectors))
    conductivity = uniform_conductivity(conductance * np.array([2.0, 1.0]))
    potential, flux, current, _, torque = core.steady_response(
        topology, edge_displacements, edge_midpoints, triangle_areas, reconstruction,
        core.mv.xy * strength, core.mv.xy * speed, conductivity)

    # Rigid rotation in a uniform perpendicular field induces an exact quadratic radial potential.
    exact_potential = strength * speed * vertices.squared() / 2              # Scalar+[V]
    exact_potential = exact_potential - exact_potential.batch()[0]           # Scalar+[V]
    assert ((potential - exact_potential).norm() < 1e-10).all()
    assert (flux.antinorm() < 1e-8).all()
    assert (current.antinorm() < 1e-7).all()
    assert (torque.antinorm() < 1e-10).all()


def test_localized_field_conserves_charge_and_matches_power_to_heating():
    radius, n_rings, n_sectors = 0.1, 5, 30
    strength, width, offset = 0.2, 0.02, 0.055
    speed, conductance = 1.0, 5.8e4
    topology, _, edge_displacements, edge_midpoints, _, _, triangle_areas, face_centers, reconstruction = disc_from_shape(
        shapes.disk(radius, n_rings, n_sectors))
    centre = (core.mv.x * offset).dual()                                     # [] Point-
    magnetic = core.mv.xy * strength * (-(edge_midpoints - centre).antinorm().squared() / (2 * width ** 2)).exp()
    spin = core.mv.xy * speed                                                # [] Bivector+
    principal = conductance * np.array([[1.0, 1.0], [1.9, 0.1], [0.1, 1.9]])  # [3, 2]
    conductivity = core.fibre_conductivity(face_centers, principal, 0.0, radius, conductance)
    _, flux, _, heating, torque = core.steady_response(
        topology, edge_displacements, edge_midpoints, triangle_areas, reconstruction,
        magnetic, spin, conductivity)

    charge_balance = topology.d[0].adjugate() * flux                         # [3] Amount-[V]
    heat_rate = heating.sites.sum()                                          # [3] Scalar+
    mechanical_loss = -(spin & torque)                                       # [3] Scalar+
    assert (charge_balance.antinorm() < 1e-8).all()
    assert ((mechanical_loss - heat_rate).norm() < 1e-9 * heat_rate).all()
    assert (heat_rate > 0).all()
    assert ((spin & torque) < 0).all()
    assert (heat_rate[1:] < heat_rate[0] / 2).all()


def test_speed_field_and_conductance_scalings():
    radius, n_rings, n_sectors = 0.1, 4, 24
    strength, width, offset = 0.2, 0.02, 0.055
    conductance = 5.8e4
    speeds = np.array([1.0, 2.0, 1.0, -1.0])                                 # [4]
    strengths = np.array([1.0, 1.0, 2.0, 1.0])                               # [4]
    topology, _, edge_displacements, edge_midpoints, _, _, triangle_areas, _, reconstruction = disc_from_shape(
        shapes.disk(radius, n_rings, n_sectors))
    centre = (core.mv.x * offset).dual()                                     # [] Point-
    magnetic = core.mv.xy * strength * (-(edge_midpoints - centre).antinorm().squared() / (2 * width ** 2)).exp()
    potential, _, current, heating, torque = core.steady_response(
        topology, edge_displacements, edge_midpoints, triangle_areas, reconstruction,
        magnetic * strengths, core.mv.xy * speeds, uniform_conductivity(conductance * equal_weights))
    scaled_potential, _, scaled_current, scaled_heating, scaled_torque = core.steady_response(
        topology, edge_displacements, edge_midpoints, triangle_areas, reconstruction,
        magnetic, core.mv.xy, uniform_conductivity(2 * conductance * equal_weights))

    current_scales = speeds * strengths                                      # [4]
    torque_scales = speeds * strengths ** 2                                  # [4]
    heat_scales = speeds ** 2 * strengths ** 2                               # [4]
    assert ((current - current[0] * current_scales).antinorm() < 1e-7).all()
    assert ((torque - torque[0] * torque_scales).antinorm() < 1e-10).all()
    assert ((heating - heating[0] * heat_scales).norm() < 1e-10).all()
    assert ((scaled_potential - potential[0]).norm() < 1e-10).all()
    assert ((scaled_current - 2 * current[0]).antinorm() < 1e-7).all()
    assert ((scaled_torque - 2 * torque[0]).antinorm() < 1e-10).all()
    assert ((scaled_heating - 2 * heating[0]).norm() < 1e-10).all()


def test_rotation_and_reflection_covariance():
    radius, n_rings, n_sectors = 0.1, 4, 24
    strength, width, offset = 0.2, 0.02, 0.055
    conductance, angle = 5.8e4, 0.7
    topology, vertices, edge_displacements, edge_midpoints, _, _, triangle_areas, _, reconstruction = disc_from_shape(
        shapes.disk(radius, n_rings, n_sectors))
    centre = (core.mv.x * offset).dual()                                     # [] Point-
    magnetic = core.mv.xy * strength * (-(edge_midpoints - centre).antinorm().squared() / (2 * width ** 2)).exp()
    conductivity = uniform_conductivity(conductance * np.array([2.0, 1.0]))
    potential, flux, current, heating, torque = core.steady_response(
        topology, edge_displacements, edge_midpoints, triangle_areas, reconstruction,
        magnetic, core.mv.xy, conductivity)

    for versor in ((core.mv.xy * (-angle / 2)).exp(), core.mv.x):
        image_vertices = versor >> vertices                                  # Point-[V]
        image_edges = topology.d[0] * image_vertices                         # Point-[E]
        image_midpoints = topology.incidence[1, 0].mean(axis=0) * image_vertices  # Point-[E]
        image_reconstruction = core.disc_reconstruction(topology, image_vertices)
        image_centre = versor >> centre                                      # [] Point-
        image_plane = versor >> core.mv.xy                                   # [] Bivector+
        image_magnetic = image_plane * strength * (-(image_midpoints - image_centre).antinorm().squared() / (2 * width ** 2)).exp()
        image_conductivity = versor >> conductivity(versor << core.Field)    # Current- <- Field+
        image_potential, image_flux, image_current, image_heating, image_torque = core.steady_response(
            topology, image_edges, image_midpoints, triangle_areas, image_reconstruction,
            image_magnetic, versor >> core.mv.xy, image_conductivity)

        assert ((image_potential - potential).norm() < 1e-10).all()
        assert ((image_flux - (versor >> flux)).antinorm() < 1e-8).all()
        assert ((image_current - (versor >> current)).antinorm() < 1e-7).all()
        assert ((image_torque - (versor >> torque)).antinorm() < 1e-10).all()
        assert ((image_heating - heating).norm() < 1e-10).all()


def test_isotropic_conductance_recovers_cotangent_laplacian():
    radius, n_rings, n_sectors, conductance = 0.1, 4, 24, 2.5
    topology, _, _, _, _, triangle_edges, triangle_areas, _, reconstruction = disc_from_shape(
        shapes.disk(radius, n_rings, n_sectors))
    V, E, F = topology.n_elements
    gradient = reconstruction(topology.d[0] * core.Scalar)                   # (Field+ <- Scalar+)[F <- V]
    excitation_map = triangle_areas * (core.Current & core.time)(uniform_conductivity(conductance * equal_weights))
    balance = gradient.adjugate()(excitation_map(gradient))                  # (Amount- <- Scalar+)[V <- V]

    tails, heads = topology.corners[2, 1]                                    # [3], [3]
    after, before = triangle_edges[tails], triangle_edges[heads]             # [3] Point-[F], [3] Point-[F]
    cotangents = -(after.dual_inverse() & before) / after.commutator(before).norm()  # [3] Scalar+[F]
    weights = (topology.incidence[2, 1].adjugate() * (cotangents / 2)).sum(axis=0)  # Scalar+[E]
    cotangent = (topology.d[0].adjugate() * core.spdiag(conductance * weights) * topology.d[0]) * core.Scalar
    potentials = core.mv.scalar(np.random.default_rng(3).normal(size=(V, 1))).field()  # Scalar+[V]
    assert (((potentials & balance(potentials)).sites.sum() - (potentials * cotangent(potentials)).sites.sum()).abs() < 1e-9).all()


def test_material_frame_braking_matches_world_frame_solve():
    radius, n_rings, n_sectors = 0.1, 3, 18
    strength, width, offset, conductance = 0.2, 0.02, 0.055, 5.8e4
    inertia, speed, angle, step = 0.005, 2.0, 0.4, 0.1
    topology, vertices, edge_displacements, edge_midpoints, _, triangle_edges, triangle_areas, face_centers, reconstruction = disc_from_shape(
        shapes.disk(radius, n_rings, n_sectors))
    tilt = (core.mv.xy * (-0.2)).exp()                                       # [] Rotor+
    plane = tilt >> core.mv.xy                                               # [] Bivector+
    centre = tilt >> (core.mv.x * offset).dual()                             # [] Point-
    half_angle_turn = (plane * (-angle / 2)).exp()                           # [] Rotor+
    orientation = half_angle_turn * tilt                                     # [] Rotor+
    spin = plane * speed                                                     # [] Bivector+
    conductivity = core.fibre_conductivity(face_centers, conductance * np.array([1.9, 0.1]),
                                           0.0, radius, conductance)         # (Current- <- Field+)[F]
    heat = triangle_areas * 0                                                # Scalar+[F]
    states = list(core.braking(topology, vertices, edge_displacements, edge_midpoints,
                               triangle_edges, triangle_areas, reconstruction, centre, plane,
                               strength, width, conductivity[None], inertia, orientation[None],
                               spin[None], heat[None], step, 1))

    # Compare against an independent world-frame solve on the rotated midpoint disc.
    quarter_step_turn = (spin * (-step / 4)).exp()                           # [] Rotor+
    midpoint_rotor = quarter_step_turn * orientation                         # [] Rotor+
    world_vertices = midpoint_rotor >> vertices                              # Point-[V]
    world_edges = topology.d[0] * world_vertices                             # Point-[E]
    world_midpoints = topology.incidence[1, 0].mean(axis=0) * world_vertices  # Point-[E]
    world_reconstruction = core.disc_reconstruction(topology, world_vertices)
    world_magnetic = plane * strength * (-(world_midpoints - centre).antinorm().squared() / (2 * width ** 2)).exp()
    world_conductivity = midpoint_rotor >> conductivity(midpoint_rotor << core.Field)
    _, _, _, unit_heating, unit_torque = core.steady_response(
        topology, world_edges, world_midpoints, triangle_areas, world_reconstruction,
        world_magnetic, plane, world_conductivity)
    drag = -(plane & unit_torque)                                            # [] Scalar+
    midpoint_spin = spin / (1 + drag * step / (2 * inertia))                 # [] Bivector+
    gained_heat = step * midpoint_spin.scalar_norm_squared() * unit_heating  # Scalar+[F]
    assert ((states[-1][3][0] - gained_heat).norm() < 1e-10).all()
    assert ((states[-1][1][0] - (2 * midpoint_spin - spin)).norm() < 1e-10).all()


def test_inductive_braking_and_render_entrypoints():
    geometry, conductivity = scenarios.materials_setup(3)
    topology, vertices, _, _, _, triangle_edges, triangle_areas, _, _ = geometry
    _, _, steady_current, _, _ = scenarios.steady_materials(geometry, conductivity)
    steady_streams = core.streamfunction_projector(topology, triangle_edges, triangle_areas,
                                                   conductivity)(steady_current)
    steady_fig = render.steady_figure(topology, vertices, triangle_areas, steady_current,
                                      steady_streams, scenarios.centre, scenarios.field_width,
                                      scenarios.magnet_field)
    plt.close(steady_fig)

    mode_shapes, embedder, stream = scenarios.inductive_spin_up(
        geometry, conductivity, n_kept_modes=2, n_drive_harmonics=2, n_frames=1, n_frame_substeps=1)
    modes_fig = render.modes_figure(topology, vertices, embedder(mode_shapes))
    plt.close(modes_fig)

    states = list(stream)
    fibres = (render.radial_fibres(scenarios.fibre_inner_radius, scenarios.fibre_outer_radius, 2, 4),
              render.circular_fibres(scenarios.fibre_inner_radius, scenarios.fibre_outer_radius, 2, 4))
    shown = np.ones((len(scenarios.radial_conductivity), 2), dtype=bool)
    list(render.animate_currents(topology, vertices, triangle_areas, scenarios.centre,
                                 scenarios.field_width, scenarios.magnet_field, fibres, shown, states))
    list(render.animate_heat(topology, vertices, triangle_areas, scenarios.centre,
                             scenarios.field_width, scenarios.magnet_field, fibres, shown, states))
    list(render.animate_induction(topology, vertices, triangle_areas, scenarios.centre,
                                  scenarios.field_width, scenarios.inductive_magnet_field, fibres,
                                  shown, states))
    list(render.animate_induction_heat(topology, vertices, triangle_areas, scenarios.centre,
                                       scenarios.field_width, scenarios.inductive_magnet_field,
                                       fibres, shown, states))
