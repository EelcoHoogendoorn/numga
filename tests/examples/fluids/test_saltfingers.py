"""Salt fingers: gravity's pull on a parcel's mass alone, and the fingering tank on the smallest mesh, stepped one frame
and drawn."""

import numpy as np

from examples.fluids.saltfingers import core, render, scenarios


def test_gravity_pulls_mass_alone():
    pull = -core.mv.ym                                                        # [] Drive+
    parts = (core.mass, core.volume, core.entropy, core.salt)
    readings = [(core.mv.y & pull.commutator(part)).to_array() for part in parts]
    np.testing.assert_allclose(np.ravel(readings), [-1.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_the_tank_steps_one_frame_and_draws():
    closed, opened, opening, vertices, temperature, salinity, frames = scenarios.fingers(n_along=3, n_across=3,
                                                                                         n_flat=1)
    next(render.animate(opened, vertices, opening, scenarios.period, [next(frames)]))
