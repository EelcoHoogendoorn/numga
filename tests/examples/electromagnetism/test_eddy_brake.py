"""Charge conservation, magnetic braking, and field covariance of the conducting disc solve."""

from __future__ import annotations

import numpy as np

from numga import stack
from numga.sparse import SparseExtensor
from examples.util import shapes
from examples.electromagnetism.eddy_brake import core, render, scenarios

# The planes of x and of y, whose duals are the directions across the disc.
Spatial = core.ga.gatype.from_blades("x y")             # Vector+
# The conductivity's two principal directions, along x and along y.
axes = stack((core.mv.x.dual(), core.mv.y.dual()))      # [axes] Current-
equal_principal = np.array([1.0, 1.0])


def placed(shape) -> core.Disc:
    """A stock disk's topology with its vertices placed across the plane of x and y."""
    topology, coordinates = shape
    return core.disc(topology, core.mv(Spatial, coordinates).dual().field())


def conducting(along: np.ndarray) -> core.Conductivity:
    """A uniform sheet conductance along x and along y."""
    return (axes * (core.Field & (axes & core.time)) * np.asarray(along)).sum(axis=-1)


def radial_conductivity(disc: core.Disc, along: np.ndarray) -> core.Conductivity:
    """A sheet conductance along each face's radius and around it."""
    distance = disc.face_centers.squared().square_root()
    radial = (disc.face_centers / distance).cast(core.Current)
    directions = stack((radial, core.quarter_turn >> radial), axis=-1)
    return (directions * (core.Field & (directions & core.time)) * np.asarray(along)).sum(axis=-1)


def test_a_uniform_field_is_cancelled_by_the_charge_distribution():
    radius, rings, sectors = 0.1, 6, 36
    strength, speed, conductance = 0.2, 1.0, 5.8e4
    disc = placed(shapes.disk(radius, rings, sectors))
    conductivity = conducting(conductance * np.array([2.0, 1.0]))
    response = core.solve(disc, core.mv.xy * strength, core.mv.xy * speed, conductivity)

    # The midpoint edge integral is exact for rigid rotation in a uniform field.
    potential = strength * speed * disc.vertices.squared() / 2
    potential = potential - potential.batch()[0]
    np.testing.assert_allclose(response.potential.kernel, potential.kernel, atol=1e-12)
    np.testing.assert_allclose(response.flux.kernel, 0, atol=1e-9)
    np.testing.assert_allclose(response.current.kernel, 0, atol=1e-8)
    np.testing.assert_allclose(response.torque.kernel, 0, atol=1e-12)


def test_localized_field_conserves_charge_and_turns_mechanical_power_into_heat():
    radius, rings, sectors = 0.1, 8, 48
    strength, width, offset = 0.2, 0.02, 0.055
    speed, conductance = 1.0, 5.8e4
    disc = placed(shapes.disk(radius, rings, sectors))
    magnetic = (core.mv.xy * strength * (-(disc.edge_midpoints - (core.mv.x * offset).dual()).squared() / (2 * width ** 2)).exp())
    spin = core.mv.xy * speed
    principal = conductance * np.array([[1.0, 1.0], [1.9, 0.1], [0.1, 1.9]])
    conductivity = core.fibre_conductivity(disc, principal, 0, radius, conductance)
    response = core.solve(disc, magnetic, spin, conductivity)

    charge_balance = ~disc.topology.d[0] * response.flux
    heat = response.heating.sites.sum()
    # The moment paired with the spin: minus the power it takes out of the turning.
    mechanical_loss = -(spin & response.torque)
    np.testing.assert_allclose(charge_balance.kernel, 0, atol=1e-9)
    np.testing.assert_allclose(mechanical_loss.kernel, heat.kernel, atol=1e-12)
    assert np.all(heat.kernel > 0)
    assert np.all((spin & response.torque).kernel < 0)
    # With the same mean conductance, resistance across the fibres reduces both current loops.
    assert np.all(heat[1:].kernel < heat[0].kernel / 2)


def test_speed_field_and_conductance_have_their_physical_scalings():
    radius, rings, sectors = 0.1, 6, 36
    strength, width, offset = 0.2, 0.02, 0.055
    conductance = 5.8e4
    speeds = np.array([1.0, 2.0, 1.0, -1.0])
    strengths = np.array([1.0, 1.0, 2.0, 1.0])
    disc = placed(shapes.disk(radius, rings, sectors))
    magnetic = (core.mv.xy * strength * (-(disc.edge_midpoints - (core.mv.x * offset).dual()).squared() / (2 * width ** 2)).exp())
    response = core.solve(disc, magnetic * strengths,
                          core.mv.xy * speeds, conducting(conductance * equal_principal))
    doubled_conductance = core.solve(disc, magnetic, core.mv.xy, conducting(2 * conductance * equal_principal))

    current_factors = speeds * strengths
    torque_factors = speeds * strengths ** 2
    heat_factors = speeds ** 2 * strengths ** 2
    np.testing.assert_allclose(response.current.kernel,
                               (response.current[0] * current_factors).kernel, atol=1e-8)
    np.testing.assert_allclose(response.torque.kernel,
                               (response.torque[0] * torque_factors).kernel, atol=1e-12)
    np.testing.assert_allclose(response.heating.kernel,
                               (response.heating[0] * heat_factors).kernel, atol=1e-12)
    np.testing.assert_allclose(doubled_conductance.potential.kernel, response.potential[0].kernel, atol=1e-12)
    np.testing.assert_allclose(doubled_conductance.current.kernel, (2 * response.current[0]).kernel, atol=1e-8)
    np.testing.assert_allclose(doubled_conductance.torque.kernel, (2 * response.torque[0]).kernel, atol=1e-12)
    np.testing.assert_allclose(doubled_conductance.heating.kernel, (2 * response.heating[0]).kernel, atol=1e-12)


def test_turning_the_disc_and_magnet_turns_the_solution():
    radius, rings, sectors = 0.1, 6, 36
    strength, width, offset = 0.2, 0.02, 0.055
    conductance, angle = 5.8e4, 0.7
    disc = placed(shapes.disk(radius, rings, sectors))
    centre, plane, spin = (core.mv.x * offset).dual(), core.mv.xy, core.mv.xy
    turn = (core.mv.xy * (-angle / 2)).exp()
    turned_disc = core.disc(disc.topology, turn >> disc.vertices)
    magnetic = (plane * strength * (-(disc.edge_midpoints - centre).squared() / (2 * width ** 2)).exp())
    turned_magnetic = (turn >> plane * strength * (-(turned_disc.edge_midpoints - (turn >> centre)).squared() / (2 * width ** 2)).exp())
    conductivity = conducting(conductance * np.array([2.0, 1.0]))
    turned_conductivity = turn >> conductivity(turn << core.Field)
    response = core.solve(disc, magnetic, spin, conductivity)
    turned_response = core.solve(turned_disc, turned_magnetic, turn >> spin, turned_conductivity)

    np.testing.assert_allclose(turned_response.potential.kernel, response.potential.kernel, atol=1e-12)
    np.testing.assert_allclose(turned_response.flux.kernel, response.flux.kernel, atol=1e-9)
    np.testing.assert_allclose(turned_response.current.kernel, (turn >> response.current).kernel, atol=1e-8)
    np.testing.assert_allclose((turned_response.torque - (turn >> response.torque)).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(turned_response.heating.kernel, response.heating.kernel, atol=1e-12)


def test_braking_power_converges_under_mesh_refinement():
    radius, strength, width, offset = 0.1, 0.2, 0.02, 0.055
    conductance, rings, sectors = 5.8e4, 4, 24
    powers = []
    for refinement in (1, 2, 4):
        disc = placed(shapes.disk(radius, rings * refinement, sectors * refinement))
        magnetic = (core.mv.xy * strength * (-(disc.edge_midpoints - (core.mv.x * offset).dual()).squared() / (2 * width ** 2)).exp())
        principal = conductance * np.array([[1.0, 1.0], [1.9, 0.1], [0.1, 1.9]])
        conductivity = core.fibre_conductivity(disc, principal, 0, radius, conductance)
        response = core.solve(disc, magnetic, core.mv.xy, conductivity)
        powers.append(response.heating.sites.sum().kernel)
    # Midpoint edge integration and the polygonal rim both approach the smooth-disc problem.
    assert np.all(abs(powers[2] - powers[1]) < 0.4 * abs(powers[1] - powers[0]))


def test_isotropic_material_recovers_the_cotangent_stiffness():
    radius, rings, sectors, conductance = 0.1, 6, 36, 2.5
    disc = placed(shapes.disk(radius, rings, sectors))
    face_ones = core.as_scalar(np.ones(disc.topology.n_elements[2])).field()
    gradient = (core.Vector ^ core.mv.t)(disc.reconstruction)(disc.topology.d[0] * core.Scalar)
    outflow = gradient.adjugate()(core.spdiag(face_ones * (core.Current & core.time)))
    material = disc.triangle_areas * conducting(conductance * equal_principal)
    balance = outflow(material(gradient))
    # Each edge's cotangent weight, half the cotangents of the angles facing it: at each corner, one edge's slice
    # read along the other, over the size of the plane they span.
    tails, heads = disc.topology.corners[2, 1]
    after, before = disc.triangle_edges[tails], disc.triangle_edges[heads]
    cotangents = -(after.dual_inverse() & before) / after.commutator(before).norm()
    weights = (~disc.topology.incidence[2, 1] * (cotangents / 2)).sum(axis=0)
    cotangent = (~disc.topology.d[0] * core.spdiag(conductance * weights) * disc.topology.d[0]) * core.Scalar
    potentials = core.as_scalar(np.random.default_rng(3).normal(size=disc.topology.n_vertices)).field()
    # The face gradient is constant, so integrating its material pairing at the centroid is exact.
    np.testing.assert_allclose(balance(potentials).kernel, cotangent(potentials).kernel, atol=1e-10)


def test_material_frame_heating_matches_a_solve_on_the_rotated_world_mesh():
    radius, rings, sectors = 0.1, 4, 24
    strength, width, offset, conductance = 0.2, 0.02, 0.055, 5.8e4
    inertia, speed, angle, step, steps = 0.005, 2.0, 0.4, 0.1, 1
    disc = placed(shapes.disk(radius, rings, sectors))
    tilt = (core.mv.xy * -0.2).exp()
    plane = tilt >> core.mv.xy
    centre = tilt >> (core.mv.x * offset).dual()
    orientation = (plane * (-angle / 2)).exp() * tilt
    spin = plane * speed
    conductivity = radial_conductivity(disc, conductance * np.array([1.9, 0.1]))
    heat = disc.triangle_areas * 0
    states = list(core.braking(disc, centre, plane, strength, width,
                               conductivity[None], inertia, orientation[None],
                               spin[None], heat[None], step, steps))

    # Solve independently in world coordinates, leaving the magnet fixed as the mesh turns.
    midpoint = (spin * (-step / 4)).exp() * orientation
    world = core.disc(disc.topology, midpoint >> disc.vertices)
    magnetic = (plane * strength * (-(world.edge_midpoints - centre).squared() / (2 * width ** 2)).exp())
    material = midpoint >> conductivity(midpoint << core.Field)
    response = core.solve(world, magnetic, plane, material)
    drag = -(plane & response.torque)
    midpoint_spin = spin / (1 + drag * step / (2 * inertia))
    gained_heat = step * midpoint_spin.scalar_norm_squared() * response.heating
    np.testing.assert_allclose((states[-1].heat[0] - gained_heat).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((states[-1].spin[0] - (2 * midpoint_spin - spin)).kernel, 0, atol=1e-12)


def test_inductive_braking_steps_and_draws():
    disc, conductivity = scenarios.materials(n_divisions=3)
    states = scenarios.inductive_materials(disc, conductivity, n_modes=2, n_harmonics=2, n_inductive_steps=1,
                                           n_substeps=1)
    fibres = (render.radial_fibres(scenarios.fibre_inner_radius, scenarios.fibre_outer_radius, 2, 4),)
    shown = np.ones((len(scenarios.radial_conductivity), 1), dtype=bool)
    next(render.animate_induction(disc.topology, disc.vertices, disc.triangle_areas, list(states)[1:], scenarios.centre,
                                  scenarios.field_width, scenarios.inductive_magnet_field, fibres, shown))
