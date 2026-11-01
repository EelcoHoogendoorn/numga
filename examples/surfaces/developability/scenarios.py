"""Scenes for developability: the Stanford bunny flowed towards developable patches, as it creases."""

from __future__ import annotations

from itertools import islice

import numpy as np

from examples import MODEL_DIR
from examples.util import shapes
from examples.surfaces.developability import core

mv = core.mv
# The Stanford bunny, from the shared models, its size, and the quarter turn that stands it up: its y axis,
# up in the file, turned onto z.
BUNNY = MODEL_DIR / "bunny.obj"
SIZE = 2.0
upright = (mv.yz * (-np.pi / 4)).exp()
# The flow's step, as a share of what each vertex's faces can take, its number of steps, the steps between
# kept states, and the duration of a frame.
STEP = 0.3
N_STEPS = 600
STRIDE = 5
DURATION_MS = 40


# --- math -----------------------------------------------------------------------------
def flowed() -> tuple[core.Topology, list[tuple[core.Point, core.Scalar]]]:
    """The bunny with the states of its flow: the vertices and each vertex's smallest eigenvalue."""
    topology, vertices = bunny()
    return topology, list(islice(core.flow(topology, vertices, STEP), 0, N_STEPS + 1, STRIDE))


# --- plumbing -------------------------------------------------------------------------
def bunny() -> tuple[core.Topology, core.Point]:
    """The bunny's mesh, and its vertices upright, centred in a box of the given size."""
    topology, coordinates = shapes.surface(BUNNY, SIZE)
    vertices = mv.vector(coordinates).dual().field()                            # Point-[vertices]
    return topology, upright >> vertices


def main() -> None:
    from examples.util.animation import save_animation
    from examples.surfaces.developability import render

    topology, states = flowed()
    save_animation(render.animate_flow(topology, states), "developability_flow", DURATION_MS)

    # checks
    # Every kept state has less energy than the one before, and the flow takes away nearly all of it.
    energies = np.array([values.sum(axis=0).to_array() for _, values in states])
    assert np.all(np.diff(energies) < 0)
    assert energies[-1] < 0.1 * energies[0]


if __name__ == "__main__":
    main()
