"""Vortices in three settings: two pairs leapfrogging, a shear layer rolling up, and a gas of vortices
of both senses; and the worldlines of a ring of equal vortices and of the leapfrogging pairs, traced
through time."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from numga import stack
from examples.fluids.vortices import core

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
WALKABOUT_DT = 1 / 6
N_WALKABOUT_FRAMES, N_WALKABOUT_SUBSTEPS = 203, 4
WALKABOUT_ALONG, WALKABOUT_ACROSS = 10.0, 1.4
WALKABOUT_HALF_WIDTH, WALKABOUT_HALF_HEIGHT = 10.6, 2.9
WALKABOUT_PIXELS_PER_UNIT = 20
WALKABOUT_VORTICITY_SCALE = 14.0

# Worldlines: five equal vortices evenly spaced on a ring around the origin, turning rigidly, and the
# leapfrogging pairs, traced for one and a half turns of the ring and three periods of the leapfrog;
# the step, the steps of each, and the height of each braid's drawn time against its width.
N_RING = 5
RING_CIRCULATIONS = np.ones(N_RING)
RING_RADIUS = 1.0
RING_CORE = 0.05
WORLDLINE_DT = 0.2
N_RING_STEPS, N_LEAPFROG_STEPS = 150, 75
BRAID_HEIGHT = 2.5

# The walkabout's frame duration, and the animations' palette, enough colours for the fields' smooth depths.
DURATION_MS = 40
N_COLOURS = 64


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
    duration_ms: int
    vorticity_scale: float


# --- math -----------------------------------------------------------------------------
def frames(vortices: core.Vortices, n_frames: int, n_substeps: int, dt: float) -> Iterator[core.Vortices]:
    """The vortices at every frame, a few steps apart."""
    for _ in range(n_frames):
        yield vortices
        for _ in range(n_substeps):
            vortices = core.step(vortices, dt)


def worldlines(vortices: core.Vortices, dt: float, n_steps: int) -> core.Point:
    """The centres of the vortices at every step: one curve through the plane and time per vortex."""
    def stepped() -> Iterator[core.Point]:
        current = vortices
        yield current.centres
        for _ in range(n_steps):
            current = core.step(current, dt)
            yield current.centres

    return stack(list(stepped()))                                            # [steps + 1, vortices] Point-


def braids() -> list[core.Point]:
    """The worldlines `[steps + 1, vortices] Point` of a ring of equal vortices, turning rigidly about its
    centre, and of the leapfrogging pairs."""
    turns = (core.mv.xy * (-np.pi * np.arange(N_RING) / N_RING)).exp()       # [vortices] Rotor+
    ring = core.Vortices(turns >> (core.mv.x.dual() * RING_RADIUS), core.mv.xy * RING_CIRCULATIONS, RING_CORE, core.mv.vector([[0.0, 0.0]]).dual())
    pairs = leapfrog().vortices
    turning = worldlines(ring, WORLDLINE_DT, N_RING_STEPS)                   # [steps + 1, vortices] Point-
    passing = worldlines(pairs, WORLDLINE_DT, N_LEAPFROG_STEPS)              # [steps + 1, vortices] Point-

    return [turning, passing]


def walkabout() -> core.Vortices:
    """The nudged pairs."""
    centres = (core.mv.y * WALKABOUT_SPANS + core.mv.x * WALKABOUT_NUDGES).dual()   # [vortices] Point-
    return core.Vortices(centres, core.mv.xy * LEAPFROG_CIRCULATIONS, LEAPFROG_CORE, core.mv.vector([[0.0, 0.0]]).dual())


def leapfrog() -> Scene:
    centres = (core.mv.y * LEAPFROG_SPANS).dual().cast(core.Point)           # [vortices] Point-
    vortices = core.Vortices(centres, core.mv.xy * LEAPFROG_CIRCULATIONS, LEAPFROG_CORE, core.mv.vector([[0.0, 0.0]]).dual())
    return Scene(name="vortices_leapfrog", vortices=vortices, half_width=2.4, half_height=1.6, n_columns=300, n_rows=200,
                 dt=1 / 15, n_substeps=2, n_frames=150, duration_ms=40, vorticity_scale=14.0)


def shear_layer() -> Scene:
    along = (np.arange(N_LAYER_VORTICES) + 0.5) / N_LAYER_VORTICES * LAYER_PERIOD
    phase = 2 * np.pi * along / LAYER_PERIOD
    across = LAYER_AMPLITUDE * (np.sin(2 * phase) + LAYER_SEED * np.sin(phase))
    centres = (core.mv.x * (along - LAYER_PERIOD / 2) + core.mv.y * across).dual()   # [vortices] Point-
    # The layer's circulation per period is the jump in velocity across it times the period.
    circulations = np.full(N_LAYER_VORTICES, -LAYER_JUMP * LAYER_PERIOD / N_LAYER_VORTICES)
    copies = core.mv.x.dual() * (np.arange(-N_LAYER_COPIES, N_LAYER_COPIES + 1) * LAYER_PERIOD)   # [copies] Point-
    vortices = core.Vortices(centres, core.mv.xy * circulations, LAYER_CORE, copies)
    return Scene(name="vortices_shear_layer", vortices=vortices, half_width=LAYER_PERIOD, half_height=0.4, n_columns=300,
                 n_rows=120, dt=1 / 60, n_substeps=2, n_frames=240, duration_ms=40, vorticity_scale=10.0)


def gas() -> Scene:
    rng = np.random.default_rng(GAS_SEED)
    n_gas = N_GAS_TURNING + N_GAS_COUNTER
    # Uniform over the disc: a radius growing as the square root, turned by a rotor to a random angle.
    radii = GAS_RADIUS * np.sqrt(rng.uniform(size=n_gas))
    turns = (core.mv.xy * (-np.pi * rng.uniform(size=n_gas))).exp()          # [vortices] Rotor+
    centres = turns >> (core.mv.x.dual() * radii)                            # [vortices] Point-
    circulations = rng.permutation(np.repeat([1.0, -1.0], [N_GAS_TURNING, N_GAS_COUNTER]))
    vortices = core.Vortices(centres, core.mv.xy * circulations, GAS_CORE, core.mv.vector([[0.0, 0.0]]).dual())
    return Scene(name="vortices_gas", vortices=vortices, half_width=3.6, half_height=3.0, n_columns=300, n_rows=250,
                 dt=1 / 120, n_substeps=8, n_frames=225, duration_ms=40, vorticity_scale=50.0)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.vortices import render

    scenes = (leapfrog(), shear_layer(), gas())
    last = []
    for scene in scenes:
        states = list(frames(scene.vortices, scene.n_frames, scene.n_substeps, scene.dt))
        save_animation(render.animate(states, scene.half_height, (scene.n_rows, scene.n_columns), scene.vorticity_scale),
                       scene.name, scene.duration_ms, N_COLOURS)
        last.append(states[-1])

    # The walkabout on one strip around its whole trip.
    walk_centre = (core.mv.x * WALKABOUT_ALONG + core.mv.y * WALKABOUT_ACROSS).dual()   # [] Point-
    walk_shape = (round(2 * WALKABOUT_HALF_HEIGHT * WALKABOUT_PIXELS_PER_UNIT), round(2 * WALKABOUT_HALF_WIDTH * WALKABOUT_PIXELS_PER_UNIT))
    walk = frames(walkabout(), N_WALKABOUT_FRAMES, N_WALKABOUT_SUBSTEPS, WALKABOUT_DT)
    save_animation(render.window(walk, walk_centre, WALKABOUT_HALF_HEIGHT, walk_shape, WALKABOUT_VORTICITY_SCALE),
                   "vortices_walkabout", DURATION_MS, N_COLOURS)

    turning, passing = braids()
    save_figure(render.braids([turning, passing], BRAID_HEIGHT), "vortices_worldlines")

    # checks
    # The derivative of the flow has no divergence, and its bivector is the blobs' vorticity, the closed form the
    # pictures draw.
    for scene, vortices in zip(scenes, last):
        points = vortices.centres.mean(axis=-1) + core.grid(scene.half_width, scene.half_height, 12, 8)
        derivative = core.derivative(core.gradient(vortices, points))
        separations = points[..., None, None] - (vortices.centres[..., None] + vortices.copies)
        blob = separations.dot(separations) + vortices.core_radius**2
        vorticity = (vortices.circulations[:, None] * vortices.core_radius**2 / np.pi / (blob * blob)).sum(axis=-1).sum(axis=-1)
        np.testing.assert_allclose((derivative - vorticity).kernel, 0.0, atol=1e-9)
    for circulations, centres in ((RING_CIRCULATIONS, turning), (LEAPFROG_CIRCULATIONS, passing)):
        # Runge–Kutta keeps the linear impulse, the circulation-weighted sum of the centres, to
        # round-off; the angular impulse to the method's accuracy.
        linear = (circulations * centres).sum(axis=-1)                       # [steps + 1] Point-
        angular = (circulations * (centres | centres)).sum(axis=-1)          # [steps + 1] Scalar+
        np.testing.assert_allclose((linear - linear[0]).kernel, 0.0, atol=1e-12)
        np.testing.assert_allclose((angular - angular[0]).kernel, 0.0, atol=1e-5)
    # The ring turns rigidly: each vortex's place times its first place is the same for every vortex.
    turned = turning * turning[0]                                            # [steps + 1, vortices] Even+
    np.testing.assert_allclose((turned - turned[:, :1]).kernel, 0.0, atol=1e-12)

if __name__ == "__main__":
    main()
