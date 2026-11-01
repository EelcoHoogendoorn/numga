"""Salt fingers: gravity's pull on a parcel's mass alone, the fingers' limit at the ratio of the diffusivities, and
fingers that keep their salt and trade their warmth across the middle of the tank."""

import numpy as np

from examples.fluids.saltfingers import core, render, scenarios


def test_gravity_pulls_mass_alone():
    pull = -core.mv.yt                                                        # [] Drive+
    parts = (core.time, core.volume, core.entropy, core.salt)
    readings = [(core.mv.y & pull.commutator(part)).to_array() for part in parts]
    np.testing.assert_allclose(np.ravel(readings), [-1.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_fingers_grow_below_the_ratio_of_diffusivities():
    rates = scenarios.growth().to_array()
    assert rates[0] > 0 > rates[-1]
    crossing = np.interp(0.0, -rates, scenarios.growth_ratios)
    np.testing.assert_allclose(crossing, 1 / scenarios.salt_diffusivity, rtol=0.01)


def test_fingers_keep_their_salt_and_trade_their_warmth():
    topology, vertices, temperature, salinity, frames = scenarios.fingers()
    frames = list(frames)
    _, places, _ = core.tank(scenarios.n_along, scenarios.n_across, scenarios.period)
    middle = np.isclose(places[:, 1], 0.5)
    _, temperatures, salinities = (value.to_array()[middle] for value in frames[-1])
    assert np.ptp(salinities) > 0.9
    assert temperatures.std() < salinities.std() / 2
    render.draw_tank(topology, vertices, scenarios.n_across, scenarios.period, temperature, salinity)
    assert next(render.animate(topology, vertices, scenarios.n_across, scenarios.period, frames[-1:])).shape[-1] == 3
    render.draw_growth(scenarios.growth_ratios, core.mv.scalar(np.linspace(1, -1, 7)[:, None]),
                       scenarios.salt_diffusivity)
