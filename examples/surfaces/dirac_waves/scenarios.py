"""Scenes for spinor waves on the sphere: a pulse spreading from the north pole and gathering again at
the south pole, and the standing waves of the first few frequencies, each over its period."""

from __future__ import annotations

from itertools import islice

import numpy as np

from numga import stack
from examples.util import shapes
from examples.surfaces.dirac_waves import core

mv = core.mv
# The sphere's subdivisions, the time step, a frame each, and the duration of a frame.
SUBDIVISIONS = 3
INTERVAL = 0.04
DURATION_MS = 40
# The pulse: its width, and the steps it runs, a little past the time to reach the far pole.
WIDTH = 0.25
N_STEPS = 84
# The standing waves: the frequencies shown, each held eight times its own, after the four constant
# fields of frequency zero; and the frames over each period.
N_LEVELS = 5
N_CONSTANT = 4
N_PHASES = 25


# --- math -----------------------------------------------------------------------------
def pulse(subdivisions: int = SUBDIVISIONS, n_steps: int = N_STEPS) -> tuple[core.Topology, core.Point, list[tuple[core.Even, core.Even]]]:
    """A bump of the xy plane at the north pole spreads over the sphere: the sphere and its points, with the vertex
    and face fields at every step, `Even+[vertices]` and `Even-[faces]`."""
    topology, coordinates = shapes.sphere(subdivisions)
    points = mv.vector(coordinates).dual().field()                              # Point-[vertices]
    start = ((points - mv.z.dual()).norm_squared() * (-0.5 / WIDTH**2)).exp() * mv.xy   # Even+[vertices]
    return topology, points, list(islice(core.spread(topology, points, start, INTERVAL), n_steps))


def modes(subdivisions: int = SUBDIVISIONS, n_levels: int = N_LEVELS, n_phases: int = N_PHASES) -> tuple[core.Topology, core.Point, core.Scalar, core.Even, core.Even]:
    """The standing waves of the first few frequencies, one of each, swing over their periods: the sphere and its
    points, the frequencies squared of all the waves found, and the vertex and face fields
    `[frequencies, phases] Even+[vertices]` and `[frequencies, phases] Even-[faces]`."""
    topology, coordinates = shapes.sphere(subdivisions)
    points = mv.vector(coordinates).dual().field()                              # Point-[vertices]
    sizes = N_CONSTANT + 8 * np.cumsum(np.arange(n_levels + 1))               # [levels + 1]
    # Of each frequency's waves, the part of a bump on the equator they hold, in all three planes: every
    # frequency drawn the same way round.
    probe = ((points - mv.x.dual()).norm_squared() * (-0.5 / WIDTH**2)).exp() * (mv.yz + mv.zx + mv.xy)  # Even+[vertices]
    frequencies = np.arange(1, n_levels + 1)
    phases = np.linspace(0.0, 2 * np.pi, n_phases, endpoint=False)
    squared, vertices, faces = core.modes(topology, points, sizes, probe, frequencies, phases)
    return topology, points, squared, vertices, faces


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.surfaces.dirac_waves import render

    topology, points, states = pulse()
    vertices, faces = stack([state[0] for state in states]), stack([state[1] for state in states])   # [steps] Even+[vertices], [steps] Even-[faces]
    save_animation(render.animate_pulse(topology, points, vertices, faces), "dirac_waves_pulse", DURATION_MS)
    _, _, squared, mode_vertices, mode_faces = modes()
    save_animation(render.animate_modes(topology, points, mode_vertices, mode_faces), "dirac_waves_modes", DURATION_MS)

    # checks
    # The leapfrog keeps the energy: the vertex field's area-weighted norm, and the face field's paired
    # across each step.
    edges = topology.boundary[2, 1] * (topology.d[0] * points)
    triangle_areas = edges[1].commutator(edges[0]).norm() / 2
    corner_mean = topology.incidence[2, 0].mean(axis=0)
    face_areas, vertex_areas = core.spdiag(triangle_areas), core.spdiag(~corner_mean * triangle_areas)
    energies = stack([(vertex_areas * vertices.scalar_norm_squared()).sites.sum()
                      + (face_areas * before.reverse().scalar_product(after)).sites.sum()
                      for (vertices, after), (_, before) in zip(states[1:], states)]).to_array()
    np.testing.assert_allclose(energies, energies[0], rtol=1e-12)
    # Half a turn round the sphere on, the pulse has gathered at the south pole.
    density = (vertex_areas * states[int(round(np.pi / INTERVAL))][0].scalar_norm_squared()).to_array()
    assert density[(points & mv.z) < -0.9].sum() > 0.5 * density.sum()
    # The frequencies are the whole numbers, each held eight times its own, after the constant fields,
    # up to the mesh's resolution.
    measured = squared.to_array()
    np.testing.assert_allclose(measured[:N_CONSTANT], 0.0, atol=1e-9)
    expected = np.repeat(np.arange(1, N_LEVELS + 1), 8 * np.arange(1, N_LEVELS + 1))
    np.testing.assert_allclose(np.sqrt(measured[N_CONSTANT:]), expected, rtol=0.03)


if __name__ == "__main__":
    main()
