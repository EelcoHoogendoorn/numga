"""Discrete shells: the forces are the energy's downhill direction and balance as internal forces, and the
scenes draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.shells import core, render, scenarios


def test_forces_are_the_energy_gradient_and_balance():
    """On a crumpled patch with natural curvature, every force matches minus the energy's derivative by
    central differences, and the force lines add up to nothing."""
    rng = np.random.default_rng(1)
    flat = core.lattice(0.3, 0.7)
    flat = flat[np.linalg.norm(flat, axis=-1) <= 0.7]
    mesh, points = core.flat_mesh(flat, 0.3)
    tails, heads = (end * points for end in mesh.edge_ends)
    lengths = (tails & heads).norm() * 1.1
    origin = core.mv.w.dual()
    along, across = origin & core.mv.x.dual(), origin & core.mv.y.dual()
    curvature = 2 * ((along | core.Line) * (along | core.Line) - (across | core.Line) * (across | core.Line))
    sheet = core.sheet(mesh, points, lengths, curvature, core.mv.z, 0.1)
    bent = np.concatenate([flat, np.zeros((len(flat), 1))], axis=-1) + 0.05 * rng.normal(size=(len(flat), 3))
    pushes = core.forces(sheet, core.mv.point(bent).field()).batch()
    shift = 1e-6
    for vertex in (0, len(bent) // 2):
        for axis in range(3):
            up, down = bent.copy(), bent.copy()
            up[vertex, axis] += shift
            down[vertex, axis] -= shift
            slope = (core.energy(sheet, core.mv.point(up).field()) - core.energy(sheet, core.mv.point(down).field())).to_array() / (2 * shift)
            np.testing.assert_allclose(pushes[vertex].cast(core.ga.subspace("wzy wxz wyx")).kernel[axis], -slope, rtol=1e-6, atol=1e-9)
    total = (core.mv.point(bent).field() & pushes.field()).sites.sum()
    assert np.abs(total.kernel).max() < 1e-12


def test_the_scenes_draw():
    disc, disc_frames = scenarios.disc()
    strip, strip_frames = scenarios.strip()
    plt.close(render.draw_settled([(disc.mesh.faces, disc_frames[-1]), (strip.mesh.faces, strip_frames[-1])]))
    assert len(list(render.animate(strip.mesh.faces, strip_frames[-1:]))) == 1
