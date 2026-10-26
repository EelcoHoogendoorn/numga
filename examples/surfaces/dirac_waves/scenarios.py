"""Scenes for spinor waves on the sphere: a pulse spreading from the north pole and gathering again at
the south pole, and the standing waves of the first few frequencies, each over its period."""

from __future__ import annotations

from itertools import islice

import numpy as np

from numga import stack
from examples.surfaces.spin_transformations.core import icosphere
from examples.surfaces.dirac_waves import core

mv = core.mv
# The sphere's subdivisions, the time step, and the steps between frames.
SUBDIVISIONS = 3
INTERVAL = 0.04
STRIDE = 2
# The pulse: its width, and the steps it runs, a little past the time to reach the far pole.
WIDTH = 0.25
N_STEPS = 84
# The standing waves: the frequencies shown, each held eight times its own, after the four constant
# fields of frequency zero; and the frames over each period.
N_LEVELS = 5
N_CONSTANT = 4
N_PHASES = 20


# --- math -----------------------------------------------------------------------------
def pulse() -> tuple[core.Mesh, core.Point, list[tuple[core.Even, core.Even]]]:
    """A bump of the xy plane at the north pole, spreading: the sphere and its points, and the vertex and face
    fields at every step `Even[V]` and `Even[F]`."""
    sphere, points = icosphere(SUBDIVISIONS)
    start = ((points - mv.z.dual()).norm_squared() * (-0.5 / WIDTH**2)).exp() * mv.xy   # Even[V]
    return sphere, points, list(islice(core.spread(sphere, points, start, INTERVAL), N_STEPS))


def modes() -> tuple[core.Mesh, core.Point, core.Scalar, core.Even, core.Even]:
    """The standing waves of the first few frequencies, one of each, over its period: the sphere and its
    points, the frequencies squared of all the waves found, and the vertex and face fields
    `[frequencies, phases] Even[V]` and `[frequencies, phases] Even[F]`."""
    sphere, points = icosphere(SUBDIVISIONS)
    sizes = N_CONSTANT + 8 * np.cumsum(np.arange(N_LEVELS + 1))               # [levels + 1]
    # Of each frequency's waves, the part of a bump on the equator they hold, in all three planes: every
    # frequency drawn the same way round.
    probe = ((points - mv.x.dual()).norm_squared() * (-0.5 / WIDTH**2)).exp() * (mv.yz + mv.zx + mv.xy)   # Even[V]
    frequencies = np.arange(1, N_LEVELS + 1)
    phases = np.linspace(0.0, 2 * np.pi, N_PHASES, endpoint=False)
    squared, vertices, faces = core.modes(sphere, points, sizes, probe, frequencies, phases)
    return sphere, points, squared, vertices, faces


# --- plumbing -------------------------------------------------------------------------


def main() -> None:
    from examples.util.animation import save_animation
    from examples.surfaces.dirac_waves import render

    sphere, points, states = pulse()
    frames = states[::STRIDE]
    vertices, faces = stack([state[0] for state in frames]), stack([state[1] for state in frames])   # [frames] Even[V], [frames] Even[F]
    save_animation(render.animate_pulse(sphere, points, vertices, faces), "dirac_waves_pulse", 80)
    _, _, squared, mode_vertices, mode_faces = modes()
    save_animation(render.animate_modes(sphere, points, mode_vertices, mode_faces), "dirac_waves_modes", 50)

    # checks
    # The leapfrog keeps the energy: the vertex field's area-weighted norm, and the face field's paired
    # across each step.
    edges = ((sphere.d0 * points).batch()[sphere.face_edges.T] * sphere.face_edge_orientation.T).field()
    triangle_areas = edges[1].commutator(edges[0]).norm() / 2
    thirds = core.as_ga_sparse(sphere.faces, core.as_scalar(np.ones_like(sphere.faces) / 3))
    face_areas, vertex_areas = core.spdiag(triangle_areas), core.spdiag(~thirds * triangle_areas)
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
