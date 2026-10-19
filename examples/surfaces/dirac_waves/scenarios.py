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
def pulse() -> tuple[core.Mesh, list[tuple[core.Even, core.Even]]]:
    """A bump of the xy plane at the north pole, spreading: the sphere, and the vertex and face fields at
    every step `Even[V]` and `Even[F]`."""
    sphere = icosphere(SUBDIVISIONS)
    start = bump(sphere, mv.z.dual()) * mv.xy                                         # Even[V]
    return sphere, list(islice(core.spread(sphere, start, INTERVAL), N_STEPS))


def modes() -> tuple[core.Mesh, core.Scalar, core.Even, core.Even]:
    """The standing waves of the first few frequencies, one of each, over its period: the sphere, the
    frequencies squared of all the waves found, and the vertex and face fields
    `[frequencies, phases] Even[V]` and `[frequencies, phases] Even[F]`."""
    sphere = icosphere(SUBDIVISIONS)
    sizes = N_CONSTANT + 8 * np.cumsum(np.arange(N_LEVELS + 1))               # [levels + 1]
    squared, waves = core.standing(sphere, int(sizes[-1]))                   # [modes] Scalar, [modes] Even[V]
    # Of each frequency's waves, the part of a bump on the equator they hold, in all three planes: every
    # frequency drawn the same way round.
    _, _, vertex_areas, _ = core.dirac(sphere)
    probe = bump(sphere, mv.x.dual()) * (mv.yz + mv.zx + mv.xy)                      # Even[V]
    shares = [waves[low:high] for low, high in zip(sizes[:-1], sizes[1:])]
    chosen = stack([(share * (vertex_areas * share.reverse().scalar_product(probe)).sites.sum()).sum(axis=0)
                    for share in shares])                                     # [levels] Even[V]
    frequencies = np.arange(1, N_LEVELS + 1).astype(float)
    phases = np.linspace(0.0, 2 * np.pi, N_PHASES, endpoint=False)
    vertices, faces = core.oscillation(sphere, chosen, mv.scalar(frequencies[:, None]), phases)
    return sphere, squared, vertices, faces


# --- plumbing -------------------------------------------------------------------------
def bump(sphere, centre: core.Point) -> core.Scalar:
    """A smooth bump on the sphere about the given point, of width WIDTH."""
    return ((sphere.vertices - centre).norm_squared() * (-0.5 / WIDTH**2)).exp()


def main() -> None:
    from examples.animation import save_animation
    from examples.surfaces.dirac_waves import render

    sphere, states = pulse()
    frames = states[::STRIDE]
    vertices, faces = stack([state[0] for state in frames]), stack([state[1] for state in frames])   # [frames] Even[V], [frames] Even[F]
    save_animation(render.animate(sphere, vertices[None], faces[None], 1), "dirac_waves_pulse", 80)
    _, squared, mode_vertices, mode_faces = modes()
    save_animation(render.animate(sphere, mode_vertices, mode_faces, N_LEVELS), "dirac_waves_modes", 50)

    # checks
    # The leapfrog keeps the energy: the vertex field's area-weighted norm, and the face field's paired
    # across each step.
    _, face_areas, vertex_areas, _ = core.dirac(sphere)
    energies = stack([(vertex_areas * vertices.scalar_norm_squared()).sites.sum()
                      + (face_areas * before.reverse().scalar_product(after)).sites.sum()
                      for (vertices, after), (_, before) in zip(states[1:], states)]).to_array()
    np.testing.assert_allclose(energies, energies[0], rtol=1e-12)
    # Half a turn round the sphere on, the pulse has gathered at the south pole.
    density = (vertex_areas * states[int(round(np.pi / INTERVAL))][0].scalar_norm_squared()).to_array()
    assert density[(sphere.vertices & mv.z) < -0.9].sum() > 0.5 * density.sum()
    # The frequencies are the whole numbers, each held eight times its own, after the constant fields,
    # up to the mesh's resolution.
    measured = squared.to_array()
    np.testing.assert_allclose(measured[:N_CONSTANT], 0.0, atol=1e-9)
    expected = np.repeat(np.arange(1, N_LEVELS + 1), 8 * np.arange(1, N_LEVELS + 1))
    np.testing.assert_allclose(np.sqrt(measured[N_CONSTANT:]), expected, rtol=0.03)


if __name__ == "__main__":
    main()
