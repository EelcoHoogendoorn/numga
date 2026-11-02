"""A rotating Bose-Einstein condensate: it settles into a lattice of vortices that all wind once the same way, as many
within the cloud's bulk as Feynman's rule asks."""

import numpy as np

from examples.quantum.condensate import core, render, scenarios


def test_rotating_condensate_settles_into_feynmans_lattice():
    topology, vertices, frames = scenarios.settled()
    frames = list(frames)
    settled = frames[-1]
    turns = core.windings(topology, settled).scalar_product(~core.mv.ab).to_array()
    whole = turns[np.abs(turns) > scenarios.vortex_share]
    np.testing.assert_allclose(np.abs(whole), 1.0, atol=1e-6)
    assert len(np.unique(np.sign(whole))) == 1
    counted = scenarios.counted(topology, vertices, settled)
    bulk = scenarios.counting_radii == 4.0
    np.testing.assert_allclose(counted[bulk], scenarios.rotation * scenarios.counting_radii[bulk] ** 2, rtol=0.2)
    assert next(render.animate(topology, vertices, [settled.scalar_norm_squared()])).shape[-1] == 3
    render.draw_count(scenarios.counting_radii, counted, scenarios.rotation * scenarios.counting_radii ** 2)
