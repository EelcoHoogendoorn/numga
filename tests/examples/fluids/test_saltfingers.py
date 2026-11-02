"""Salt fingers: gravity's pull on a parcel's mass alone, and fingers that keep their salt and trade their warmth across
the middle of the tank."""

import numpy as np

from examples.fluids.saltfingers import core, render, scenarios


def test_gravity_pulls_mass_alone():
    pull = -core.mv.ym                                                        # [] Drive+
    parts = (core.mass, core.volume, core.entropy, core.salt)
    readings = [(core.mv.y & pull.commutator(part)).to_array() for part in parts]
    np.testing.assert_allclose(np.ravel(readings), [-1.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_fingers_keep_their_salt_and_trade_their_warmth():
    closed, opened, opening, vertices, temperature, salinity, frames = scenarios.fingers()
    frames = list(frames)
    middle = np.isclose((core.mv.y & vertices).to_array(), 0.5)
    _, temperatures, salinities = ((opening * value).to_array()[middle] for value in frames[-1])
    assert np.ptp(salinities) > 0.9
    assert temperatures.std() < salinities.std() / 2
    render.draw_tank(opened, vertices, opening, scenarios.period, temperature, salinity)
    assert len(list(render.animate(opened, vertices, opening, scenarios.period, frames[-1:]))) == 1
