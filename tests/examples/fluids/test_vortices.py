"""Mirror equivariance of the vortex gas, and single-step rendering of leapfrogging frames and worldlines."""

import matplotlib.pyplot as plt

from examples.fluids.vortices import core, render, scenarios


def test_mirror_image_of_gas_moves_as_mirror_image():
    scene = scenarios.gas()
    mirror = (core.mv.x + 0.4 * core.mv.y).normalized()
    # Reflecting across a line reverses every circulation bivector along with every vortex centre.
    stepped = core.step(scene.centres, scene.circulations, scene.core_radius, scene.copies, 0.01)
    mirrored_stepped = core.step(
        mirror >> scene.centres,
        mirror >> scene.circulations,
        scene.core_radius,
        mirror >> scene.copies,
        0.01,
    )
    assert (((mirror >> stepped) - mirrored_stepped).norm() < 1e-9).all()


def test_pairs_step_once_and_draw():
    scene = scenarios.leapfrog()
    *_, final_centres = core.trajectory(
        scene.centres, scene.circulations, scene.core_radius, scene.copies, 2, 1, scene.dt
    )
    next(
        render.animate(
            scene.circulations,
            scene.core_radius,
            scene.copies,
            scene.half_height,
            (8, 12),
            scene.vorticity_scale,
            [final_centres],
        )
    )
    plt.close(render.braids(scenarios.braid_worldlines(1, 1), scenarios.braid_height))
