"""Vortices in three settings: two pairs leapfrogging, a shear layer rolling up, and a gas of vortices
of both senses."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from examples.mechanics.vortices import core

# Two pairs on one axis, each pair turning in opposite senses so that it travels along x; the inner
# pair is wider than 0.38 of the outer, so the two take turns passing, and keep doing so.
LEAPFROG_SPANS = np.array([1.0, -1.0, 0.55, -0.55])
LEAPFROG_CIRCULATIONS = np.array([1.0, -1.0, 1.0, -1.0])
LEAPFROG_CORE = 0.15

# A shear layer: a row of equal vortices, repeating along x, displaced by a wave of two rolls per
# period and a weaker wave of one, which seeds the rolls' pairing.
LAYER_PERIOD = 1.0
LAYER_VORTICES = 64
LAYER_COPIES = 4
LAYER_JUMP = 1.0
LAYER_CORE = 0.05
LAYER_AMPLITUDE = 0.01
LAYER_SEED = 0.5

# A gas of vortices scattered over a disc, most turning one way: the excess keeps the gas together,
# and the few turning the other way pair up with neighbours and shoot through it.
GAS_TURNING = 32
GAS_COUNTER = 8
GAS_RADIUS = 1.5
GAS_CORE = 0.08
GAS_SEED = 3

# A walkabout: the inner pair at 0.35 of the outer, where leapfrogging is unstable, and one inner
# vortex nudged along x to break the mirror symmetry. One lap, and back to leapfrogging, on a strip
# around the whole trip.
WALKABOUT_SPANS = np.array([1.0, -1.0, 0.35, -0.35])
WALKABOUT_NUDGES = np.array([0.0, 0.0, 0.015, 0.0])
WALKABOUT_DT = 0.2
WALKABOUT_FRAMES, WALKABOUT_SUBSTEPS = 130, 5
WALKABOUT_ALONG, WALKABOUT_ACROSS = 9.2, 1.4
WALKABOUT_HALF_WIDTH, WALKABOUT_HALF_HEIGHT = 9.8, 2.9
WALKABOUT_SAMPLES_PER_UNIT = 8
WALKABOUT_VORTICITY_LIMIT = 8.0

DURATION_MS = 60


@dataclass(frozen=True)
class Scene:
    name: str
    vortices: core.Vortices
    half_width: float
    half_height: float
    columns: int
    rows: int
    dt: float
    substeps: int
    frames: int
    vorticity_limit: float
    swirl_limit: float


# --- math -----------------------------------------------------------------------------
def evolve(scene: Scene) -> Iterator[tuple[core.Vortices, core.Vector, core.Even, core.Scalar]]:
    """The vortices and the flow around them, frame by frame, on a window that follows them."""
    offsets = core.grid(scene.half_width, scene.half_height, scene.columns, scene.rows)   # [rows, columns] Vector
    vortices = scene.vortices
    for _ in range(scene.frames):
        points = vortices.centres.mean(axis=-1) + offsets                   # [rows, columns] Vector
        gradients = core.gradient(vortices, points)                          # [rows, columns] Vector <- Vector
        yield vortices, points, core.derivative(gradients), core.swirl(gradients)
        for _ in range(scene.substeps):
            vortices = core.step(vortices, scene.dt)


def tracked(vortices: core.Vortices, points: core.Vector, frames: int, substeps: int, dt: float) -> Iterator[tuple[core.Vortices, core.Even]]:
    """The vortices and the derivative of their flow on fixed points, frame by frame."""
    for _ in range(frames):
        yield vortices, core.derivative(core.gradient(vortices, points))
        for _ in range(substeps):
            vortices = core.step(vortices, dt)


def walkabout() -> tuple[core.Vortices, core.Vector]:
    """The nudged pairs, and the strip around their trip."""
    centres = core.mv.y * WALKABOUT_SPANS + core.mv.x * WALKABOUT_NUDGES      # [vortices] Vector
    vortices = core.Vortices(centres, LEAPFROG_CIRCULATIONS, LEAPFROG_CORE, core.mv.vector([[0.0, 0.0]]))
    columns = round(2 * WALKABOUT_HALF_WIDTH * WALKABOUT_SAMPLES_PER_UNIT)
    rows = round(2 * WALKABOUT_HALF_HEIGHT * WALKABOUT_SAMPLES_PER_UNIT)
    centre = core.mv.x * WALKABOUT_ALONG + core.mv.y * WALKABOUT_ACROSS     # [] Vector
    return vortices, centre + core.grid(WALKABOUT_HALF_WIDTH, WALKABOUT_HALF_HEIGHT, columns, rows)


def leapfrog() -> Scene:
    centres = (core.mv.y * LEAPFROG_SPANS).cast(core.Vector)                 # [vortices] Vector
    vortices = core.Vortices(centres, LEAPFROG_CIRCULATIONS, LEAPFROG_CORE, core.mv.vector([[0.0, 0.0]]))
    return Scene("vortices_leapfrog", vortices, 2.4, 1.6, 300, 200, 0.1, 2, 100, 8.0, 6.0)


def shear_layer() -> Scene:
    along = (np.arange(LAYER_VORTICES) + 0.5) / LAYER_VORTICES * LAYER_PERIOD
    phase = 2 * np.pi * along / LAYER_PERIOD
    across = LAYER_AMPLITUDE * (np.sin(2 * phase) + LAYER_SEED * np.sin(phase))
    centres = core.mv.x * (along - LAYER_PERIOD / 2) + core.mv.y * across  # [vortices] Vector
    # The layer's circulation per period is the jump in velocity across it times the period.
    circulations = np.full(LAYER_VORTICES, -LAYER_JUMP * LAYER_PERIOD / LAYER_VORTICES)
    copies = core.mv.x * (np.arange(-LAYER_COPIES, LAYER_COPIES + 1) * LAYER_PERIOD)   # [copies] Vector
    vortices = core.Vortices(centres, circulations, LAYER_CORE, copies)
    return Scene("vortices_shear_layer", vortices, LAYER_PERIOD, 0.4, 120, 48, 0.025, 2, 160, 10.0, 1.5)


def gas() -> Scene:
    rng = np.random.default_rng(GAS_SEED)
    count = GAS_TURNING + GAS_COUNTER
    # Uniform over the disc: a radius growing as the square root, turned by a rotor to a random angle.
    radii = GAS_RADIUS * np.sqrt(rng.uniform(size=count))
    turns = (core.PLANE * (-np.pi * rng.uniform(size=count))).exp()         # [vortices] Rotor
    centres = turns >> (core.mv.x * radii)                                   # [vortices] Vector
    circulations = rng.permutation(np.repeat([1.0, -1.0], [GAS_TURNING, GAS_COUNTER]))
    vortices = core.Vortices(centres, circulations, GAS_CORE, core.mv.vector([[0.0, 0.0]]))
    return Scene("vortices_gas", vortices, 3.6, 3.0, 240, 200, 0.01, 10, 150, 15.0, 15.0)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation
    from examples.mechanics.vortices import render

    for scene in (leapfrog(), shear_layer(), gas()):
        states = list(evolve(scene))
        save_animation(render.animate(states, scene.vorticity_limit, scene.swirl_limit), scene.name, DURATION_MS)

        # --- checks
        # The derivative of the flow has no divergence, and its bivector is the blobs' vorticity.
        vortices, points, derivative, _ = states[-1]
        separations = points[..., None, None] - (vortices.centres[..., None] + vortices.copies)
        blob = (separations | separations) + vortices.core_radius**2
        vorticity = (vortices.circulations[:, None] * vortices.core_radius**2 / np.pi / (blob * blob)).sum(axis=-1).sum(axis=-1)
        np.testing.assert_allclose((derivative - vorticity * core.PLANE).kernel, 0.0, atol=1e-9)

    walker, strip = walkabout()
    frames = tracked(walker, strip, WALKABOUT_FRAMES, WALKABOUT_SUBSTEPS, WALKABOUT_DT)
    save_animation(render.window(strip, frames, WALKABOUT_VORTICITY_LIMIT), "vortices_walkabout", DURATION_MS)


if __name__ == "__main__":
    main()
