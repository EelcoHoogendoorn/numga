"""A cellular flame: burning moves energy from the fuel into the heat and makes none, so that while the flame spreads
the disk's energy stays put and its fuel turns into heat."""

from itertools import islice

import numpy as np

from examples.thermodynamics.flame import core, render, scenarios

# The frames the flame is followed for.
n_frames = 4


def test_burning_moves_energy_from_fuel_to_heat():
    np.testing.assert_allclose((core.energy & core.reaction).to_array(), 0.0, atol=1e-15)
    topology, vertices, frames = scenarios.spreading()
    _, _, cells = core.triangles(topology, vertices)
    frames = list(islice(frames, n_frames))
    energies = np.array([(cells * (core.energy & parcels)).sites.sum().to_array() for parcels, _ in frames])
    heats = np.array([(cells * (core.mv.h & parcels)).sites.sum().to_array() for parcels, _ in frames])
    np.testing.assert_allclose(energies, energies[0], rtol=1e-12)
    assert np.all(np.diff(heats) > 0)
    picture = next(render.animate(topology, vertices, scenarios.radius, core.energy, frames[-1:]))
    assert picture.shape == (render.n_pixels, 2 * render.n_pixels + render.n_gap, 3)
