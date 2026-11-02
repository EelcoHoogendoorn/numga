"""The Meissner effect: a rod cooled in a field and the field switched off, as a superconductor and as a perfect
conductor, and the field's decay into the cold superconductor over the London penetration depth; and a ring trapping
whole flux quanta."""

import numpy as np

from examples.electromagnetism.superconductor import core, render, scenarios


def test_superconductor_expels_and_perfect_conductor_freezes():
    topology, vertices, in_rod, frames = scenarios.cooled()
    frames = list(frames)
    _, places, _ = scenarios.rod()
    centre = np.hypot(places[:, 0], places[:, 1]) < 0.6
    held = frames[(scenarios.n_normal + scenarios.n_cooling + scenarios.n_held - 1) // scenarios.stride][0]
    # Cold in the field, the superconductor's core holds no field and the perfect conductor's the applied one.
    potentials = held.to_array().reshape(2, -1)
    assert np.abs(potentials[0][centre]).max() < 0.02
    np.testing.assert_allclose(potentials[1][centre], places[centre, 1], atol=1e-6)
    # After the field is gone, the perfect conductor still holds it.
    last = frames[-1][0].to_array().reshape(2, -1)
    assert np.abs(last[0][centre]).max() < 0.02
    assert np.abs(last[1][centre] - places[centre, 1]).max() < 0.02
    # The field decays into the superconductor over the penetration depth, within a tenth.
    depths, strengths = scenarios.penetrating(topology, vertices, held[0])
    shallow = (depths > 0.05) & (depths < 0.4)
    decay = -1 / np.polyfit(depths[shallow].to_array(), np.log(strengths[shallow].to_array()), 1)[0]
    np.testing.assert_allclose(decay, scenarios.penetration, rtol=0.1)
    render.draw_rod(topology, vertices, in_rod)
    assert next(render.animate(topology, vertices, in_rod, scenarios.penetration, frames[-1:])).shape[-1] == 3
    render.draw_penetration(depths, strengths, scenarios.penetration)


def test_ring_traps_whole_flux_quanta():
    topology, places, in_ring, hole_faces, in_hole, rim = scenarios.ring()
    vertices = core.mv(core.Plane, places).dual().field()
    areas, gradients, cells = core.triangles(topology, vertices)
    trapped = core.ring_fields(topology, areas, gradients, in_ring, hole_faces, in_hole, rim, scenarios.penetration,
                               scenarios.trapped_quanta, 0 * scenarios.trapped_quanta)
    np.testing.assert_allclose((trapped * cells).sites.sum().to_array(), scenarios.trapped_quanta, rtol=0.02)
    # Cooled in a growing field and let go, the ring keeps none, then one, then two quanta, never fewer than before.
    kept = scenarios.kept_quanta(topology, areas, gradients, cells, in_ring, hole_faces, in_hole, rim)
    assert set(np.unique(kept)) == {0.0, 1.0, 2.0}
    assert np.all(np.diff(kept) >= 0)
    render.draw_trapped(topology, vertices, in_hole, trapped)
    render.draw_staircase(scenarios.trapped_quanta, (trapped * cells).sites.sum())
