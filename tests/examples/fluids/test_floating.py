"""Hydrostatic stability: a box's buoyancy, metacentric heights and righting arm agree with their closed
forms, a cube floats face up exactly outside the band of densities between (3 - sqrt(3)) / 6 and
(3 + sqrt(3)) / 6, and a coarse hull is released, stepped one frame and drawn."""

import numpy as np

from examples.fluids.floating import core, render, scenarios

mv = core.mv
# A box two by one by one, floating with its waterline on no edge, its centre of gravity this high over its
# bottom.
SIZES = np.array([2.0, 1.0, 1.0])
DRAFT = 0.37
GRAVITY_HEIGHT = 0.3


def test_box_buoyancy_metacentric_heights_and_righting_arm_are_the_closed_forms():
    topology, vertices = core.box(SIZES)
    box = topology.incidence[2, 0] * vertices
    length, beam, depth = SIZES
    surface = core.water - (DRAFT - depth / 2) * mv.w
    push, growth = core.buoyancy(box, surface, 1.0)
    assert abs((push & core.heave).to_array() - length * beam * DRAFT) < 1e-12
    assert abs((growth & core.heave).to_array() - length * beam) < 1e-12
    weight = mv.scalar([length * beam * DRAFT])
    centre = mv.point([0, 0, GRAVITY_HEIGHT - depth / 2])
    # The metacentric heights: the centre of buoyancy's height, plus the waterplane's second moment over the
    # displaced volume, less the centre of gravity's height.
    across = DRAFT / 2 + beam ** 2 / (12 * DRAFT) - GRAVITY_HEIGHT
    along = DRAFT / 2 + length ** 2 / (12 * DRAFT) - GRAVITY_HEIGHT
    stiffness = core.stiffness(box, mv.scalar([DRAFT - depth / 2]), weight, centre, 1.0)
    assert np.abs((stiffness.eigvalsh() / weight).to_array() - [across, along]).max() < 1e-12
    # Wall-sided, until the deck edge dips: the righting arm is sin(heel) * (GM + BM / 2 * tan(heel)**2).
    heels = np.radians([10.0, 20.0, 30.0])
    turns = (mv.yz * (-heels / 2)).exp()
    heights = core.afloat(box, turns, weight, mv.scalar(np.zeros((len(heels), 1))), 1.0, scenarios.N_NEWTON_STEPS)
    push, weighing = core.forces(box, turns, heights, weight, centre, 1.0)
    righting = (-((push + weighing) & core.heel_axis) / weight).to_array()
    expected = np.sin(heels) * (across + beam ** 2 / (12 * DRAFT) / 2 * np.tan(heels) ** 2)
    assert np.abs(righting - expected).max() < 1e-12


def test_cube_floats_face_up_outside_the_band_of_densities():
    topology, vertices = core.box(np.ones(3))
    cube = topology.incidence[2, 0] * vertices
    edges = np.array([(3 - np.sqrt(3)) / 6, (3 + np.sqrt(3)) / 6])
    densities = (edges[:, None] + np.array([-1e-3, 1e-3])).reshape(-1)       # [densities]
    stiffness = core.stiffness(cube, mv.scalar((densities - 0.5)[:, None]), mv.scalar(densities[:, None]),
                               mv.w.dual(), 1.0)
    upright = stiffness(mv.yz, mv.yz).to_array()
    assert np.array_equal(upright > 0, [True, False, False, True])


def test_a_coarse_hull_is_released_and_draws():
    topology, vertices, weight, centres = scenarios.ship(n_cells=2)
    hull = topology.incidence[2, 0] * vertices                               # [corners] Point-[triangles]
    couples, _ = scenarios.righting_curves(hull, weight, centres)
    frames = scenarios.released(hull, weight, centres[scenarios.RELEASED])
    next(frames)
    next(render.animate_release(topology, vertices, centres[scenarios.RELEASED], [next(frames)], scenarios.HEELS,
                                couples[:, scenarios.RELEASED], weight))
