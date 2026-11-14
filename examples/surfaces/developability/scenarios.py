"""Scenes for developability flow crease the triangulated Stanford bunny into piecewise developable panels."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples import MODEL_DIR
from examples.surfaces.developability import core
from examples.util import shapes
from numga import stack
from numga.topology import Topology

mv = core.mv
# Stanford bunny asset path, bounding-box scale, and quarter turn bringing the file's y-up axis onto z.
bunny_path = MODEL_DIR / "bunny.obj"
box_size = 2.0
upright_turn = (mv.yz * (-np.pi / 4)).exp()                                     # [] Rotor+
# Stiffness-scaled step fraction, total gradient steps, steps per recorded frame, and GIF frame duration.
step_scale = 0.3
n_steps = 600
n_stride = 5
n_states = n_steps // n_stride + 1
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def bunny_flow(n_kept: int, n_step_stride: int) -> tuple[Topology, Iterator[tuple[core.Point, core.Scalar]]]:
    """Streams `n_kept` states of the upright Stanford bunny (`Point-[V]`, `Scalar+[V]`) every `n_step_stride` developability steps."""
    topology, vertices = bunny_surface()
    return topology, core.developability_flow(topology, vertices, step_scale, n_kept, n_step_stride)


# --- plumbing -------------------------------------------------------------------------
def bunny_surface() -> tuple[Topology, core.Point]:
    """Loads the triangulated Stanford bunny centred in a box of width `box_size` with its upright vertex field (`Point-[V]`)."""
    topology, coordinates = shapes.surface(bunny_path, box_size)
    raw_vertices = mv.vector(coordinates).dual().field()                        # Point-[V]
    return topology, upright_turn >> raw_vertices


def main() -> None:
    """Streams the bunny developability flow animation to disk and checks monotone energy decay."""
    from examples.surfaces.developability import render
    from examples.util.animation import save_animation

    topology, state_stream = bunny_flow(n_states, n_stride)
    states: list[tuple[core.Point, core.Scalar]] = []

    def recorded() -> Iterator[tuple[core.Point, core.Scalar]]:
        for state in state_stream:
            states.append(state)
            yield state

    save_animation(render.flow_animation(topology, recorded()), "developability_flow", frame_duration_ms)

    # checks
    energies = stack([eigenvalues.batch().sum(axis=0) for _, eigenvalues in states])  # [n_states] Scalar+
    assert (energies[1:] < energies[:-1]).all()
    assert energies[-1] < 0.1 * energies[0]


if __name__ == "__main__":
    main()
