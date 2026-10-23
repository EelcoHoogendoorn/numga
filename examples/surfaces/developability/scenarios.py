"""Scenes for developability: the Stanford bunny flowed towards developable patches, as it creases."""

from __future__ import annotations

from itertools import islice

import numpy as np

from examples import MODEL_DIR
from examples.surfaces.developability import core

mv = core.mv
# The Stanford bunny, from the shared models, its size, and the quarter turn that stands it up: its y axis,
# up in the file, turned onto z.
BUNNY = MODEL_DIR / "bunny.obj"
SIZE = 2.0
UPRIGHT = ((mv.y ^ mv.z) * (-np.pi / 4)).exp()
# The flow's step, as a share of what each vertex's faces can take, its number of steps, the steps between
# kept states, and the duration of a frame.
STEP = 0.3
N_STEPS = 600
STRIDE = 10
DURATION_MS = 80


# --- math -----------------------------------------------------------------------------
def flowed() -> tuple[core.Mesh, list[tuple[core.Vector, core.Scalar]]]:
    """The bunny with the states of its flow: the vertices and each vertex's smallest eigenvalue."""
    mesh = bunny()
    return mesh, list(islice(core.flow(mesh, STEP), 0, N_STEPS + 1, STRIDE))


# --- plumbing -------------------------------------------------------------------------
def bunny() -> core.Mesh:
    """The bunny, upright, centred in a box of the given size."""
    mesh = core.Mesh.read(BUNNY, SIZE)
    return mesh.copy(UPRIGHT >> mesh.vertices)


def main() -> None:
    from examples.util.animation import save_animation
    from examples.surfaces.developability import render

    mesh, states = flowed()
    save_animation(render.animate_flow(mesh, states), "developability_flow", DURATION_MS)

    # checks
    # Every kept state has less energy than the one before, and the flow takes away nearly all of it.
    energies = np.array([values.sum(axis=0).to_array() for _, values in states])
    assert np.all(np.diff(energies) < 0)
    assert energies[-1] < 0.1 * energies[0]


if __name__ == "__main__":
    main()
