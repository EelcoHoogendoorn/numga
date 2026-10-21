"""As rigid as possible: a bar held at one end and bent a quarter turn up at the other, as
rigid as possible beside Laplacian editing."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.surfaces.arap import core

mv = core.mv
# The bar: its squares per side of the cube it is stretched from, and its length.
N_DIVISIONS = 10
LENGTH = 6.0
# How stiffly the handles are held, how far the far end turns, and the frames over the bend with the
# iterations at each, each frame starting from the last.
STIFFNESS = 1e3
BEND = np.pi / 2
N_FRAMES = 24
N_ITERATIONS = 6
# The checkerboard's cell, one square of the bar at rest, and the duration of a frame in milliseconds.
CELL = np.array([LENGTH, 2.0, 2.0]) / N_DIVISIONS
DURATION_MS = 80


# --- math -----------------------------------------------------------------------------
def bending():
    """The bar bent over the frames, as rigid as possible and by Laplacian editing: the vertices at each
    frame `[frames] Point[V]`, both ways, and the bar at rest; then which vertices are handles, where
    they are moved at the last frame, and the energy over the iterations of the last frame."""
    bar = core.bar(N_DIVISIONS, LENGTH)
    along = mv.x & bar.vertices                                                # Scalar[V]
    near, far = along < -LENGTH / 2 + 1e-6, along > LENGTH / 2 - 1e-6
    moved = core.as_scalar(far).field()                                        # Scalar[V]
    poses = [bar.vertices + moved * (bent(bar.vertices, angle) - bar.vertices)   # Point[V]
             for angle in np.linspace(0.0, BEND, N_FRAMES + 1)[1:]]
    frames = list(core.deform(bar, near | far, poses, STIFFNESS, N_ITERATIONS))
    rigid, laplacian = stack([shape for shape, _, _ in frames]), stack([edited for _, edited, _ in frames])
    return bar, rigid, laplacian, near | far, poses[-1], frames[-1][2]


# --- plumbing -------------------------------------------------------------------------
def bent(vertices: core.Point, angle: float) -> core.Point:
    """The far end of the bar where an arc of its length would carry it, turned by the angle about y:
    its centre on the arc from the near end, curving up."""
    radius = LENGTH / angle
    centre = (mv.x * (-LENGTH / 2 + radius * np.sin(angle)) + mv.z * (radius * (1 - np.cos(angle)))).dual()   # [] Point
    turn = ((mv.x ^ mv.z) * (-angle / 2)).exp()                                # [] Rotor
    return centre + (turn >> (vertices - mv.x.dual() * (LENGTH / 2)))


def main() -> None:
    from examples.animation import save_animation
    from examples.surfaces.arap import render

    bar, rigid, laplacian, handles, target, last_energies = bending()
    save_animation(render.animate(bar, rigid, CELL), "arap_bending", DURATION_MS)

    # checks
    # At the last frame the energy falls with every iteration, and the handles are where they were moved to.
    energies = last_energies.to_array()
    assert np.all(np.diff(energies) <= 1e-9 * energies[0])
    assert (rigid[-1] - target).norm().batch()[handles].to_array().max() < 1e-3
    # The bar keeps its angles: its corners' cosines change a fraction as much as by Laplacian editing.
    change = [np.abs((bar.copy(vertices=shape).corner_cosines() - bar.corner_cosines()).to_array()).mean()
              for shape in (rigid[-1], laplacian[-1])]
    assert change[0] < 0.5 * change[1]
    # Left at rest, the bar stays as it is.
    (still, _, _), = core.deform(bar, handles, [bar.vertices], STIFFNESS, 2)
    np.testing.assert_allclose((still - bar.vertices).kernel, 0.0, atol=1e-9)


if __name__ == "__main__":
    main()
