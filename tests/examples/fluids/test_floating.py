"""Hydrostatic stability coverage tests: closed-form buoyancy, metacentric heights and wall-sided righting
arms of a rectangular box, the face-up stability band of a uniform cube, and one rendered frame of a coarse
released hull."""

import numpy as np

from examples.fluids.floating import core, render, scenarios

mv = core.mv
box_sizes = np.array([2.0, 1.0, 1.0])
box_draft = 0.37
gravity_height = 0.3


def test_box_buoyancy_metacentric_heights_and_righting_arm_match_closed_forms():
    topology, vertices = scenarios.box_surface(box_sizes)
    box = topology.incidence[2, 0] * vertices
    length, beam, depth = box_sizes
    surface = core.water - (box_draft - depth / 2) * mv.w
    push, growth = core.buoyancy(box, surface, 1.0)
    assert (((push & core.heave) - length * beam * box_draft).abs() < 1e-12).all()
    assert (((growth & core.heave) - length * beam).abs() < 1e-12).all()
    weight = core.unit * (length * beam * box_draft)
    centre = mv.point([0, 0, gravity_height - depth / 2])
    # Metacentric heights across beam and along length from waterplane second moment and vertical separation.
    across = box_draft / 2 + beam ** 2 / (12 * box_draft) - gravity_height
    along = box_draft / 2 + length ** 2 / (12 * box_draft) - gravity_height
    stiffness = core.tilt_stiffness(box, core.unit * (box_draft - depth / 2), weight, centre, 1.0)
    assert ((stiffness.eigvalsh() / weight - np.array([across, along])).abs() < 1e-12).all()
    # Wall-sided Scribanti formula before the deck edge dips: sin(heel) * (GM + BM / 2 * tan(heel)**2).
    heels = np.radians([10.0, 20.0, 30.0])
    turns = (core.heel_axis * (-heels / 2)).exp()
    heights = core.equilibrium_heights(box, turns, weight, core.unit * np.zeros(len(heels)), 1.0, scenarios.n_newton_steps)
    push, weighing = core.body_forces(box, turns, heights, weight, centre, 1.0)
    righting = -((push + weighing) & core.heel_axis) / weight
    expected = np.sin(heels) * (across + beam ** 2 / (12 * box_draft) / 2 * np.tan(heels) ** 2)
    assert ((righting - expected).abs() < 1e-12).all()


def test_cube_floats_face_up_outside_the_unstable_density_band():
    topology, vertices = scenarios.box_surface(np.ones(3))
    cube = topology.incidence[2, 0] * vertices
    bounds = np.array([(3 - np.sqrt(3)) / 6, (3 + np.sqrt(3)) / 6])
    test_densities = (bounds[:, None] + np.array([-1e-3, 1e-3])).reshape(-1)
    stiffness = core.tilt_stiffness(
        cube,
        core.unit * (test_densities - 0.5),
        core.unit * test_densities,
        core.origin,
        1.0,
    )
    upright = stiffness(core.heel_axis, core.heel_axis)
    assert np.array_equal(upright > 0, [True, False, False, True])


def test_coarse_hull_steps_and_draws_one_frame():
    topology, vertices, weight, centres = scenarios.ship_scene(2)
    hull = topology.incidence[2, 0] * vertices                          # [3] Point-[F]
    couples, _ = scenarios.righting_curves(hull, weight, centres)
    states = scenarios.release_states(hull, weight, centres[scenarios.released_loads])
    next(states)
    next(
        render.animate_release(
            topology,
            vertices,
            centres[scenarios.released_loads],
            scenarios.heels,
            couples[:, scenarios.released_loads],
            weight,
            [next(states)],
        )
    )
