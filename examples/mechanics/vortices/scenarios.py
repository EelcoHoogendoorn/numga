"""Vortices in three settings: two pairs leapfrogging, a shear layer rolling up, and a gas of vortices
of both senses; and the worldlines of a ring of equal vortices and of the leapfrogging pairs, traced
through time."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from numga import stack
from examples.mechanics.vortices import core

# Two pairs on one axis, each pair turning in opposite senses so that it travels along x; the inner
# pair is wider than 0.38 of the outer, so the two take turns passing, and keep doing so.
LEAPFROG_SPANS = np.array([1.0, -1.0, 0.55, -0.55])
LEAPFROG_CIRCULATIONS = np.array([1.0, -1.0, 1.0, -1.0])
LEAPFROG_CORE = 0.15

# A shear layer: a row of equal vortices, repeating along x, displaced by a wave of two rolls per
# period and a weaker wave of one, which seeds the rolls' pairing.
LAYER_PERIOD = 1.0
N_LAYER_VORTICES = 64
N_LAYER_COPIES = 4
LAYER_JUMP = 1.0
LAYER_CORE = 0.05
LAYER_AMPLITUDE = 0.01
LAYER_SEED = 0.5

# A gas of vortices scattered over a disc, most turning one way: the excess keeps the gas together,
# and the few turning the other way pair up with neighbours and shoot through it.
N_GAS_TURNING = 32
N_GAS_COUNTER = 8
GAS_RADIUS = 1.5
GAS_CORE = 0.08
GAS_SEED = 3

# A walkabout: the inner pair at 0.35 of the outer, where leapfrogging is unstable, and one inner
# vortex nudged along x to break the mirror symmetry. One lap, and back to leapfrogging, on a strip
# around the whole trip.
WALKABOUT_SPANS = np.array([1.0, -1.0, 0.35, -0.35])
WALKABOUT_NUDGES = np.array([0.0, 0.0, 0.015, 0.0])
WALKABOUT_DT = 0.2
N_WALKABOUT_FRAMES, N_WALKABOUT_SUBSTEPS = 135, 5
WALKABOUT_ALONG, WALKABOUT_ACROSS = 10.0, 1.4
WALKABOUT_HALF_WIDTH, WALKABOUT_HALF_HEIGHT = 10.6, 2.9
WALKABOUT_SAMPLES_PER_UNIT = 8
WALKABOUT_VORTICITY_LIMIT = 8.0

# Worldlines: five equal vortices evenly spaced on a ring around the origin, turning rigidly, and the
# leapfrogging pairs, traced for one and a half turns of the ring and three periods of the leapfrog;
# the step, the steps of each, and the height of each braid's drawn time against its width.
N_RING = 5
RING_RADIUS = 1.0
RING_CORE = 0.05
WORLDLINE_DT = 0.2
N_RING_STEPS, N_LEAPFROG_STEPS = 150, 75
BRAID_HEIGHT = 2.5

DURATION_MS = 60


@dataclass(frozen=True)
class Scene:
    name: str
    vortices: core.Vortices
    half_width: float
    half_height: float
    n_columns: int
    n_rows: int
    dt: float
    n_substeps: int
    n_frames: int
    vorticity_limit: float
    swirl_limit: float


# --- math -----------------------------------------------------------------------------
def evolve(scene: Scene) -> Iterator[tuple[core.Vortices, core.Vector, core.Even, core.Scalar]]:
    """The vortices and the flow around them, frame by frame, on a window that follows them."""
    offsets = core.grid(scene.half_width, scene.half_height, scene.n_columns, scene.n_rows)   # [rows, columns] Vector
    vortices = scene.vortices
    for _ in range(scene.n_frames):
        points = vortices.centres.mean(axis=-1) + offsets                   # [rows, columns] Vector
        gradients = core.gradient(vortices, points)                          # [rows, columns] Vector <- Vector
        yield vortices, points, core.derivative(gradients), core.swirl(gradients)
        for _ in range(scene.n_substeps):
            vortices = core.step(vortices, scene.dt)


def tracked(vortices: core.Vortices, points: core.Vector, n_frames: int, n_substeps: int, dt: float) -> Iterator[tuple[core.Vortices, core.Even]]:
    """The vortices and the derivative of their flow on fixed points, frame by frame."""
    for _ in range(n_frames):
        yield vortices, core.derivative(core.gradient(vortices, points))
        for _ in range(n_substeps):
            vortices = core.step(vortices, dt)


def worldlines(vortices: core.Vortices, dt: float, n_steps: int) -> core.Vector:
    """The centres of the vortices at every step: one curve through the plane and time per vortex."""
    def stepped() -> Iterator[core.Vector]:
        current = vortices
        yield current.centres
        for _ in range(n_steps):
            current = core.step(current, dt)
            yield current.centres

    return stack(list(stepped()))                                            # [steps + 1, vortices] Vector


def braids() -> list[core.Vector]:
    """The worldlines `[steps + 1, vortices] Vector` of a ring of equal vortices, turning rigidly about its
    centre, and of the leapfrogging pairs."""
    turns = (core.PLANE * (-np.pi * np.arange(N_RING) / N_RING)).exp()   # [vortices] Rotor
    ring = core.Vortices(turns >> (core.mv.x * RING_RADIUS), np.ones(N_RING), RING_CORE, core.mv.vector([[0.0, 0.0]]))
    pairs = leapfrog().vortices
    turning = worldlines(ring, WORLDLINE_DT, N_RING_STEPS)          # [steps + 1, vortices] Vector
    passing = worldlines(pairs, WORLDLINE_DT, N_LEAPFROG_STEPS)     # [steps + 1, vortices] Vector

    return [turning, passing]


def walkabout() -> tuple[core.Vortices, core.Vector]:
    """The nudged pairs, and the strip around their trip."""
    centres = core.mv.y * WALKABOUT_SPANS + core.mv.x * WALKABOUT_NUDGES      # [vortices] Vector
    vortices = core.Vortices(centres, LEAPFROG_CIRCULATIONS, LEAPFROG_CORE, core.mv.vector([[0.0, 0.0]]))
    n_columns = round(2 * WALKABOUT_HALF_WIDTH * WALKABOUT_SAMPLES_PER_UNIT)
    n_rows = round(2 * WALKABOUT_HALF_HEIGHT * WALKABOUT_SAMPLES_PER_UNIT)
    centre = core.mv.x * WALKABOUT_ALONG + core.mv.y * WALKABOUT_ACROSS     # [] Vector
    return vortices, centre + core.grid(WALKABOUT_HALF_WIDTH, WALKABOUT_HALF_HEIGHT, n_columns, n_rows)


def leapfrog() -> Scene:
    centres = (core.mv.y * LEAPFROG_SPANS).cast(core.Vector)                 # [vortices] Vector
    vortices = core.Vortices(centres, LEAPFROG_CIRCULATIONS, LEAPFROG_CORE, core.mv.vector([[0.0, 0.0]]))
    return Scene(name="vortices_leapfrog", vortices=vortices, half_width=2.4, half_height=1.6, n_columns=300, n_rows=200,
                 dt=0.1, n_substeps=2, n_frames=100, vorticity_limit=8.0, swirl_limit=6.0)


def shear_layer() -> Scene:
    along = (np.arange(N_LAYER_VORTICES) + 0.5) / N_LAYER_VORTICES * LAYER_PERIOD
    phase = 2 * np.pi * along / LAYER_PERIOD
    across = LAYER_AMPLITUDE * (np.sin(2 * phase) + LAYER_SEED * np.sin(phase))
    centres = core.mv.x * (along - LAYER_PERIOD / 2) + core.mv.y * across  # [vortices] Vector
    # The layer's circulation per period is the jump in velocity across it times the period.
    circulations = np.full(N_LAYER_VORTICES, -LAYER_JUMP * LAYER_PERIOD / N_LAYER_VORTICES)
    copies = core.mv.x * (np.arange(-N_LAYER_COPIES, N_LAYER_COPIES + 1) * LAYER_PERIOD)   # [copies] Vector
    vortices = core.Vortices(centres, circulations, LAYER_CORE, copies)
    return Scene(name="vortices_shear_layer", vortices=vortices, half_width=LAYER_PERIOD, half_height=0.4, n_columns=120,
                 n_rows=48, dt=0.025, n_substeps=2, n_frames=160, vorticity_limit=10.0, swirl_limit=1.5)


def gas() -> Scene:
    rng = np.random.default_rng(GAS_SEED)
    n_gas = N_GAS_TURNING + N_GAS_COUNTER
    # Uniform over the disc: a radius growing as the square root, turned by a rotor to a random angle.
    radii = GAS_RADIUS * np.sqrt(rng.uniform(size=n_gas))
    turns = (core.PLANE * (-np.pi * rng.uniform(size=n_gas))).exp()         # [vortices] Rotor
    centres = turns >> (core.mv.x * radii)                                   # [vortices] Vector
    circulations = rng.permutation(np.repeat([1.0, -1.0], [N_GAS_TURNING, N_GAS_COUNTER]))
    vortices = core.Vortices(centres, circulations, GAS_CORE, core.mv.vector([[0.0, 0.0]]))
    return Scene(name="vortices_gas", vortices=vortices, half_width=3.6, half_height=3.0, n_columns=240, n_rows=200,
                 dt=0.01, n_substeps=10, n_frames=150, vorticity_limit=15.0, swirl_limit=15.0)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.mechanics.vortices import render

    last_states = []
    for scene in (leapfrog(), shear_layer(), gas()):
        states = list(evolve(scene))
        save_animation(render.animate(states, scene.vorticity_limit, scene.swirl_limit), scene.name, DURATION_MS)
        last_states.append(states[-1])

    walker, strip = walkabout()
    frames = tracked(walker, strip, N_WALKABOUT_FRAMES, N_WALKABOUT_SUBSTEPS, WALKABOUT_DT)
    save_animation(render.window(strip, frames, WALKABOUT_VORTICITY_LIMIT), "vortices_walkabout", DURATION_MS)

    turning, passing = braids()
    save_figure(render.braids([turning, passing], BRAID_HEIGHT), "vortices_worldlines")

    # checks
    # The derivative of the flow has no divergence, and its bivector is the blobs' vorticity.
    for vortices, points, derivative, _ in last_states:
        separations = points[..., None, None] - (vortices.centres[..., None] + vortices.copies)
        blob = (separations | separations) + vortices.core_radius**2
        vorticity = (vortices.circulations[:, None] * vortices.core_radius**2 / np.pi / (blob * blob)).sum(axis=-1).sum(axis=-1)
        np.testing.assert_allclose((derivative - vorticity * core.PLANE).kernel, 0.0, atol=1e-9)
    ring_circulations = np.ones(N_RING)
    for circulations, centres in ((ring_circulations, turning), (LEAPFROG_CIRCULATIONS, passing)):
        # Runge–Kutta keeps the linear impulse, the circulation-weighted sum of the centres, to
        # round-off; the angular impulse to the method's accuracy.
        linear = (circulations * centres).sum(axis=-1)                       # [steps + 1] Vector
        angular = (circulations * (centres | centres)).sum(axis=-1)          # [steps + 1] Scalar
        np.testing.assert_allclose((linear - linear[0]).kernel, 0.0, atol=1e-12)
        np.testing.assert_allclose((angular - angular[0]).kernel, 0.0, atol=1e-5)
    # The ring turns rigidly: each vortex's place times its first place is the same for every vortex.
    turned = turning * turning[0]                                            # [steps + 1, vortices] Even
    np.testing.assert_allclose((turned - turned[:, :1]).kernel, 0.0, atol=1e-12)

if __name__ == "__main__":
    main()
