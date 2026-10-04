"""Scenes for the field over the whole algebra: a pulse spreading without a mass and with one, a
pulse along a light ray meeting a disk of mass, and the standing waves a disk without mass holds.

Lengths are in lattice spacings, times in spacings over the speed of light.
"""

from __future__ import annotations

from itertools import islice

import numpy as np

from numga import stack
from examples.relativity.kahler_dirac import core

mv = core.mv
# The lattice's side, the time step, under the plane's limit of 1 / sqrt(2), and the steps between frames.
SIDE = 128
INTERVAL = 0.5
EVERY = 4
# The spreading pulses: their width, the masses compared, and the steps they run.
WIDTH = 3.0
MASSES = np.array([0.0, 0.3])
SPREAD_STEPS = 112
# The pulse along a light ray: where it starts along x, its width along x and along y; the disk of mass
# it meets, with and without it, the disk's radius and edge; and the steps it runs.
START = -36.0
WIDTHS = np.array([3.0, 8.0])
BARRIERS = np.array([0.0, 1.0])
BARRIER_RADIUS = 14.0
EDGE = 1.5
BARRIER_STEPS = 100
# The standing waves: the lattice's side, the disk without mass and the mass around it, how many, how
# many share each frequency, one for each of the four space blades, and the frames over each period.
MODE_SIDE = 40
WELL_RADIUS = 12.0
WELL_MASS = 1.0
MODES = 48
SHARED = 4
PHASES = 20


# --- math -----------------------------------------------------------------------------
def spreading():
    """A pulse in the scalar part, spreading without a mass and with each mass: the space and time
    parts at each frame `[masses, frames, cells]`."""
    grid = core.Grid(SIDE)
    pulse = (grid.positions.squared() * (-0.5 / WIDTH**2)).exp()               # [cells] Scalar
    start = mv(core.Time, np.zeros((len(grid.here), 4)))                       # [cells] Time
    runs = [list(islice(run(grid, mv.scalar(np.full((len(grid.here), 1), mass)), pulse, start, SPREAD_STEPS), 0, None, EVERY))
            for mass in MASSES]
    space = stack([stack([space for space, _ in frames]) for frames in runs])  # [masses, frames, cells] Space
    time = stack([stack([time for _, time in frames]) for frames in runs])     # [masses, frames, cells] Time
    densities = core.density(space, time)                                      # [masses, frames, cells] Scalar

    # --- checks
    # The field's energy is kept exactly by the leapfrog: the space part's norm plus the time part's
    # paired across each step.
    mass = mv.scalar(np.full((len(grid.here), 1), MASSES[-1]))
    states = list(run(grid, mass, pulse, start, EVERY * 8))
    energies = [space.scalar_norm_squared().sum(axis=0) - before.reverse().scalar_product(after).sum(axis=0)
                for (space, after), (_, before) in zip(states[1:], states)]
    totals = stack(energies).to_array()
    np.testing.assert_allclose(totals, totals[0], rtol=1e-12)
    # Without a mass the front runs at the speed of light: by the last frame nothing is left beyond it.
    reach = SPREAD_STEPS * INTERVAL + 4 * WIDTH
    outside = (grid.positions.norm() - reach).to_array() > 0
    last = densities[0, -1].to_array()
    assert last[outside].sum() < 1e-6 * last.sum()
    return grid, space, time


def barrier():
    """A pulse travelling along x, along the light ray x + t, meeting a disk of mass, without it and
    with it: the space and time parts at each frame `[barriers, frames, cells]`."""
    grid = core.Grid(SIDE)
    offset = grid.positions - mv.x * START                                     # [cells] Vector
    envelope = ((offset | mv.x).squared() * (-0.5 / WIDTHS[0]**2) + (offset | mv.y).squared() * (-0.5 / WIDTHS[1]**2)).exp()   # [cells] Scalar
    # Along the null vector x + t, times y: the space part in the plane xy, the time part in yt.
    space, time = envelope * (mv.x * mv.y), envelope * (mv.t * mv.y)            # [cells] Space, [cells] Time
    wall = core.disk(grid, mv.x * 0.0, BARRIER_RADIUS, EDGE)                  # [cells] Scalar
    runs = [list(islice(core.leapfrog(core.step(grid, wall * height, core.Space), core.step(grid, wall * height, core.Time),
                                      space, time, INTERVAL, BARRIER_STEPS), 0, None, EVERY))
            for height in BARRIERS]
    space = stack([stack([space for space, _ in frames]) for frames in runs])  # [barriers, frames, cells] Space
    time = stack([stack([time for _, time in frames]) for frames in runs])     # [barriers, frames, cells] Time
    densities = core.density(space, time)                                      # [barriers, frames, cells] Scalar

    # --- checks
    # Without the disk most of the pulse has run on past the middle; with it, most of it is held back.
    beyond = (grid.positions | mv.x).to_array() > 0
    shares = [(density.to_array() * beyond).sum() / density.to_array().sum() for density in densities[:, -1]]
    assert shares[0] > 0.6 and shares[1] < 0.4
    return grid, space, time


def standing():
    """The standing waves of least frequency held by a disk without mass, in a lattice with mass
    around it, one of each frequency, over its period: the space and time parts `[waves, phases,
    cells]`."""
    grid = core.Grid(MODE_SIDE)
    mass = (1 - core.disk(grid, mv.x * 0.0, WELL_RADIUS, EDGE)) * WELL_MASS     # [cells] Scalar
    frequencies, waves = core.modes(grid, mass, MODES)                         # [modes] Scalar, [modes, cells] Space
    # Of each frequency's waves, the part of a scalar bump right of the centre they hold: every
    # frequency drawn the same way round.
    shared = waves.reshape(MODES // SHARED, SHARED, -1)                        # [frequencies, shared, cells] Space
    bump = ((grid.positions - mv.x * (WELL_RADIUS / 2)).squared() * (-0.5 / EDGE**2)).exp()   # [cells] Scalar
    chosen = (shared * shared.reverse().scalar_product(bump).sum(axis=-1)[..., None]).sum(axis=1)   # [frequencies, cells] Space
    phases = np.linspace(0.0, 2 * np.pi, PHASES, endpoint=False)
    space, time = core.oscillation(core.step(grid, mass, core.Space), chosen, frequencies[::SHARED].square_root(), phases)

    # --- checks
    # The waves come in fours, one frequency each.
    squared = frequencies.to_array().reshape(-1, SHARED)
    np.testing.assert_allclose(squared.max(axis=-1), squared.min(axis=-1), rtol=1e-8)
    # Every wave is held: its frequency is below the mass around the disk, and it lies in the disk.
    assert frequencies.to_array().max() < WELL_MASS**2
    density = waves.scalar_norm_squared().to_array()
    inside = (grid.positions.norm() - (WELL_RADIUS + 3 * EDGE)).to_array() < 0
    assert ((density * inside).sum(axis=-1) / density.sum(axis=-1)).min() > 0.95
    # With a mass everywhere the same, the step twice over is the Klein-Gordon equation: the waves of
    # least frequency are the mass squared, four times, then the mass squared and the lattice's
    # least Laplacian eigenvalue.
    even = core.modes(grid, mv.scalar(np.full((len(grid.here), 1), WELL_MASS)), 8)[0].to_array()
    np.testing.assert_allclose(even, WELL_MASS**2 + np.repeat([0.0, 4 * np.sin(np.pi / MODE_SIDE) ** 2], 4), rtol=1e-10)
    # Over a period the space part turns into the time part and back: the density holds.
    density = core.density(space, time).sum(axis=-1).to_array()                # [frequencies, phases]
    np.testing.assert_allclose(density, density[:, :1] * np.ones(PHASES), rtol=1e-12)
    return grid, space, time


# --- plumbing -------------------------------------------------------------------------
def run(grid: core.Grid, mass: core.Scalar, pulse: core.Scalar, start: core.Time, count: int):
    """A pulse in the scalar part, stepped by the leapfrog."""
    return core.leapfrog(core.step(grid, mass, core.Space), core.step(grid, mass, core.Time), pulse,
                         start, INTERVAL, count)


if __name__ == "__main__":
    from examples.animation import save_animation
    from examples.relativity.kahler_dirac import render

    save_animation(render.animate(*spreading(), len(MASSES), 2), "kahler_dirac_spreading", 120)
    save_animation(render.animate(*barrier(), len(BARRIERS), 2), "kahler_dirac_barrier", 120)
    save_animation(render.animate(*standing(), 6, 3), "kahler_dirac_modes", 50)
