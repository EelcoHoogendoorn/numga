"""The nematic's head–tail symmetry and energy derivative under bivector turns."""

import numpy as np

from numga.backend.jax import derivative
from examples.fluids.liquid_crystals import core, scenarios
from examples.fluids.liquid_crystals import render


def test_bivector_derivative_is_the_work_of_the_restoring_couple():
    topology, vertices = scenarios.film(7, 3.0)
    centres = core.mv.vector([[-0.8, 0.15], [0.9, -0.12]]).dual()
    order = core.defect_pair(vertices, centres, 0.4, core.mv.xy * 0.2)
    areas, gradients, cells = core.triangles(topology, vertices)
    stiffness = core.elasticity(topology, areas, gradients)
    molecular = core.molecular_field(stiffness, cells, 0.7, 2.0, order)
    slope = core.turn_slope(order, molecular)
    couple = core.work.solve(-slope)
    trial = core.mv.xy * 0.17

    def turned_energy(turns):
        rotor = (-turns / 2).exp()
        turned = rotor >> order(rotor << core.Point)
        return core.free_energy(stiffness, cells, 0.7, 2.0, turned)

    # A spatially varying turn exercises the full fibre derivative, not just rigid rotation of the film.
    steps = trial * (core.mv.x & vertices)
    exact = derivative(turned_energy)(steps * 0)(steps)
    predicted = (cells * slope(steps)).sites.sum()
    np.testing.assert_allclose(exact.kernel, predicted.kernel, atol=1e-8)
    np.testing.assert_allclose((couple & steps).kernel, -slope(steps).kernel, atol=1e-8)


def test_alignment_has_no_chosen_end_and_turns_with_its_director():
    director = (core.mv.xy * -0.31).exp() >> core.mv.x.dual()
    order = core.alignment(director, 0.8)
    np.testing.assert_allclose(order.kernel, core.alignment(-director, 0.8).kernel, atol=1e-8)
    np.testing.assert_allclose(order.trace().kernel, 0, atol=1e-8)
    np.testing.assert_allclose((2 * order(order).trace()).kernel, [0.64], atol=1e-8)


def test_passive_motion_descends_energy_and_holds_the_boundary():
    topology, vertices = scenarios.film(9, 3.0)
    centres = core.mv.vector([[-0.8, 0.15], [0.9, -0.12]]).dual()
    order = core.defect_pair(vertices, centres, 0.4, core.mv.xy * 0.2)[None].broadcast_to((2,))
    areas, gradients, cells = core.triangles(topology, vertices)
    stiffness = core.elasticity(topology, areas, gradients)
    molecular = core.molecular_field(stiffness, cells, 0.7, 2.0, order)
    change = core.rate(topology, areas, gradients, cells, stiffness, 0.7, 2.0, np.array([0.0, 0.8]), order)
    energy_rate = -(cells * molecular(change).batch().trace().field()).sites.sum()
    assert energy_rate[0] < 0
    boundary = topology.reading(topology.exterior[0], 0)
    np.testing.assert_allclose((boundary * change).kernel, 0, atol=1e-8)
    np.testing.assert_allclose(change.batch().trace().kernel, 0, atol=1e-8)


def test_the_pair_has_opposite_half_charges():
    topology, vertices = scenarios.film(25, 3.0)
    centres = core.mv.vector([[-0.8, 0.15], [0.9, -0.12]]).dual()
    order = core.defect_pair(vertices, centres, 0.4, core.mv.xy * 0.2)
    charges = core.windings(topology, order, core.mv.x.dual()).scalar_product(~core.mv.xy)
    nonzero = charges.to_array()[np.abs(charges.to_array()) > 0.1]
    np.testing.assert_allclose(np.sort(nonzero), [-0.5, 0.5], atol=1e-8)


def test_mirroring_the_film_mirrors_its_motion():
    topology, vertices = scenarios.film(7, 3.0)
    centres = core.mv.vector([[-0.8, 0.15], [0.9, -0.12]]).dual()
    order = core.defect_pair(vertices, centres, 0.4, core.mv.xy * 0.2)
    areas, gradients, cells = core.triangles(topology, vertices)
    stiffness = core.elasticity(topology, areas, gradients)
    change = core.rate(topology, areas, gradients, cells, stiffness, 0.7, 2.0, np.array(0.8), order)
    mirror = core.mv.x
    reflected = mirror >> order(mirror >> core.Point)
    mirror_areas, mirror_gradients, mirror_cells = core.triangles(topology, mirror >> vertices)
    mirror_stiffness = core.elasticity(topology, mirror_areas, mirror_gradients)
    mirror_change = core.rate(topology, mirror_areas, mirror_gradients, mirror_cells, mirror_stiffness,
                               0.7, 2.0, np.array(0.8), reflected)
    np.testing.assert_allclose(mirror_change.kernel, (mirror >> change(mirror >> core.Point)).kernel, atol=1e-8)


def test_rotational_mobility_dissipates_energy_at_fixed_order():
    topology, vertices = scenarios.film(9, 3.0)
    centres = core.mv.vector([[-0.8, 0.15], [0.9, -0.12]]).dual()
    order = core.defect_pair(vertices, centres, 0.4, core.mv.xy * 0.2)
    areas, gradients, cells = core.triangles(topology, vertices)
    stiffness = core.elasticity(topology, areas, gradients)
    molecular = core.molecular_field(stiffness, cells, 0.7, 2.0, order)
    slope = core.turn_slope(order, molecular)
    friction = -core.Turn.scalar_product(core.Turn)
    angular_velocity = friction.solve(-slope)
    changing = order.commutator(angular_velocity) - order(core.Point.commutator(angular_velocity))
    assert (cells * slope(angular_velocity)).sites.sum() < 0
    np.testing.assert_allclose(order(changing).batch().trace().kernel, 0, atol=1e-8)


def test_defect_readout_follows_a_zero_across_triangles():
    topology, vertices = scenarios.film(9, 2.0)
    centres = core.mv.vector([[0.03, 0.11], [0.23, 0.11], [0.43, 0.11], [0.63, 0.11]]).dual()
    first = core.alignment(core.mv.x.dual(), 1.0)
    turn = (core.mv.xy * (-np.pi / 8)).exp()
    second = turn >> first(turn << core.Point)
    for centre in centres:
        relative = vertices - centre
        order = first * (core.mv.x & relative) + second * (core.mv.y & relative)
        positions, charges = render.defect_positions(topology, vertices, order)
        assert len(positions) == 1
        np.testing.assert_allclose((positions[0] - centre).norm().to_array(), 0, atol=1e-12)
        np.testing.assert_allclose(charges.norm().to_array(), 0.5, atol=1e-12)
