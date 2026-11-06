"""A gas of vortices whose mirror image moves as its mirror image, and the leapfrogging pairs stepped once, their frame
and their worldlines drawn."""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from examples.fluids.vortices import core, render, scenarios


def run(vortices: core.Vortices, dt: float, steps: int) -> core.Vortices:
    for _ in range(steps):
        vortices = core.step(vortices, dt)
    return vortices


def test_the_mirror_image_of_the_gas_moves_as_its_mirror_image():
    vortices = scenarios.gas().vortices
    mirror = (core.mv.x + 0.4 * core.mv.y).normalized()
    # A mirror turns every circulation the other way, along with every centre.
    mirrored = replace(vortices, centres=mirror >> vortices.centres, circulations=mirror >> vortices.circulations, copies=mirror >> vortices.copies)
    np.testing.assert_allclose(((mirror >> run(vortices, 0.01, 10).centres) - run(mirrored, 0.01, 10).centres).kernel, 0.0, atol=1e-12)


def test_the_pairs_step_once_and_draw():
    scene = scenarios.leapfrog()
    *_, stepped = scenarios.frames(scene.vortices, 2, 1, scene.dt)
    next(render.animate([stepped], scene.half_height, (8, 12), scene.vorticity_scale))
    plt.close(render.braids(scenarios.braids(n_ring_steps=1, n_leapfrog_steps=1), scenarios.BRAID_HEIGHT))
