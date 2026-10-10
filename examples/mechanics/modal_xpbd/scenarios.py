"""Girders constrained at their corners: a spliced cantilever, a hinged chain under gravity, and a spliced
beam fixed at both ends and compressed until it buckles. Each scene is built in the core it is given, an
instance of `core.py` for a context."""

from dataclasses import replace
from collections.abc import Iterator
from types import ModuleType

import numpy as np

from numga.algebras import PGA2D
from examples import instantiate
from numga.sparse import SparseExtensor

# The chain's girder: its cells, length and height, its bars' stiffness, its density, and the
# vibration modes it keeps.
CELLS, LENGTH, HEIGHT = 4, 1.0, 0.2
STIFFNESS, DENSITY, MODES = 300.0, 1.0, 8
# The chain's moving girders, their modes' damping ratio, and the acceleration of gravity, downward.
LINKS = 4
DAMPING = np.array([0.02])
GRAVITY = 4.0
# A constraint's compliance, rigid to round-off; it keeps the two constraints of a rigid splice solvable.
SPLICE = 1e-14
# The swing's time step, its frames and the steps between them, and the frame duration.
INTERVAL, FRAMES, SUBSTEPS = 0.002, 200, 15
DURATION_MS = 30
# The beam: its moving girders, their cells of unit length, their height, the bars' stiffness, the
# modes' damping ratio, the final end displacement, its frames and the steps between them.
BEAM_GIRDERS, BEAM_CELLS, BEAM_HEIGHT = 8, 32, 1.0
BEAM_STIFFNESS, BEAM_DAMPING = 1e11, np.array([3.0])
END_DISPLACEMENT, BEAM_FRAMES, BEAM_SUBSTEPS, BEAM_INTERVAL = 0.15, 120, 20, 0.002


# --- plumbing -------------------------------------------------------------------------
def girders(core: ModuleType, shape, fixed: np.ndarray, damping: np.ndarray, flexibility: np.ndarray):
    """Girders end to end along x, some fixed."""
    mv = core.mv
    count, modes, cases = len(fixed), shape.modes.shape[0], len(damping)
    rest = shape.rest.batch()                                              # [vertices] Point
    length = (rest[-1] - rest[1]).dual() | mv.x                            # [] Scalar
    motor = (mv.xw * ((np.arange(count) - 0.5) * length) * 0.5).exp().cast(core.Motor).broadcast_to((cases, count)).field()  # [cases] Motor[bodies]
    amplitudes = mv.scalar(np.zeros((cases, modes, count, 1))).field()     # [cases, modes] Scalar[bodies]
    moving = ~fixed
    return core.Bodies(
        motor=motor,
        rate=mv.bivector(np.zeros((cases, count, 3))).field(),
        amplitudes=amplitudes,
        rates=amplitudes,
        compliance=(shape.compliance[:, None] * flexibility[:, None, None] * moving).field(),
        frequencies=shape.frequencies[:, None].broadcast_to((cases, modes, count)).field(),
        damping=mv.scalar(damping[:, None, None, None]).broadcast_to((cases, modes, count)).field(),
        masses=shape.masses.batch().sum(axis=-1).broadcast_to((cases, count)).field(),
        inertia=shape.inertia.broadcast_to((cases, count)).field(),
        inverse_inertia=(shape.inertia.inverse() * moving).broadcast_to((cases, count)).field(),
    )


def constrained(core: ModuleType, shape, bodies: int, body_idx: np.ndarray, corner_idx: np.ndarray):
    """Constraints between the given bodies, of so many, at the given vertices."""
    compliance = core.mv.scalar([SPLICE]).broadcast_to(body_idx.shape[-1:]).field()  # Scalar[constraints]
    ends = SparseExtensor.selection(core.ctx, body_idx, bodies)              # [sides] [constraints, bodies] Scalar
    corners = SparseExtensor.selection(core.ctx, corner_idx, len(shape.rest.batch()))   # [sides] [constraints, vertices] Scalar
    return core.Constraints(body_idx, ends, corners * shape.rest, corners * shape.modes[:, None], compliance)


def splices(core: ModuleType, shape, count: int):
    """Two constraints at each joint of a row of girders."""
    vertices = shape.rest.batch().shape[-1]
    joints = np.arange(count - 1)
    body_idx = np.repeat(np.stack([joints, joints + 1]), 2, axis=-1)       # [sides, constraints]
    corner_idx = np.tile([[vertices - 2, vertices - 1], [0, 1]], (1, count - 1))  # [sides, constraints]
    return constrained(core, shape, count, body_idx, corner_idx)


def hinges(core: ModuleType, shape, count: int):
    """One constraint at each joint of a row of girders."""
    vertices = shape.rest.batch().shape[-1]
    joints = np.arange(count - 1)
    return constrained(core, shape, count, np.stack([joints, joints + 1]), np.tile([[vertices - 1], [1]], (1, count - 1)))


def cantilever(core: ModuleType, shape, count: int, damping: np.ndarray, flexibility: np.ndarray) -> tuple:
    """A fixed girder followed by spliced moving girders."""
    fixed = np.arange(count + 1) == 0
    return girders(core, shape, fixed, damping, flexibility), splices(core, shape, count + 1)


def hinged_chain(core: ModuleType, shape, count: int, damping: np.ndarray) -> tuple:
    """A fixed girder followed by hinged moving girders."""
    fixed = np.arange(count + 1) == 0
    return girders(core, shape, fixed, damping, np.ones_like(damping)), hinges(core, shape, count + 1)


def clamped_beam(core: ModuleType, shape, count: int, damping: np.ndarray) -> tuple:
    """Spliced moving girders between two fixed ones."""
    fixed = (np.arange(count + 2) == 0) | (np.arange(count + 2) == count + 1)
    return girders(core, shape, fixed, damping, np.ones_like(damping)), splices(core, shape, count + 2)


# --- math -----------------------------------------------------------------------------
def swing(core: ModuleType, shape, bodies, constraints, gravity, dt: float, frames: int, substeps: int) -> Iterator:
    """The bodies' points at every frame, under gravity."""
    for _ in range(frames):
        yield core.points(bodies, shape)                                   # [cases, bodies] Point[vertices]
        for _ in range(substeps):
            bodies = core.step(bodies, constraints, dt, gravity)


def displaced(core: ModuleType, bodies, rest, displacement: float):
    """The bodies with the last one displaced inward along x."""
    driven = rest.batch()[..., -1] * (core.mv.xw * (-displacement / 2)).exp()   # [cases] Motor
    return replace(bodies, motor=bodies.motor.batch().at[..., -1].set(driven).field().normalized())


def compress(core: ModuleType, shape, bodies, constraints, displacements: np.ndarray, dt: float, substeps: int) -> Iterator[tuple]:
    """The beam's points and midspan point at every frame."""
    rest, middle = bodies.motor, bodies.motor.batch().shape[-1] // 2
    weightless = (core.mv.y * 0.0).dual()                                  # [] Direction
    for displacement in displacements:
        bodies = displaced(core, bodies, rest, displacement)
        for _ in range(substeps):
            bodies = core.step(bodies, constraints, dt, weightless)
        centres = bodies.motor.batch()[..., middle - 1:middle + 1] >> core.mv.w.dual()   # [cases, 2] Point
        yield core.points(bodies, shape), centres.sum(axis=-1) / 2          # [cases, bodies] Point[vertices], [cases] Point


def critical(height: float, span: float) -> float:
    """The critical end displacement of a beam of two chords, clamped at both ends."""
    return np.pi**2 * height**2 / span


def main():
    from examples.animation import save_animation
    from . import render

    core = instantiate("examples.mechanics.modal_xpbd.core", PGA2D)
    shape = core.girder(CELLS, LENGTH, HEIGHT, STIFFNESS, DENSITY, MODES)
    bodies, constraints = hinged_chain(core, shape, LINKS, DAMPING)
    gravity = (core.mv.y * -GRAVITY).dual()                                # [] Direction
    geometry = swing(core, shape, bodies, constraints, gravity, INTERVAL, FRAMES, SUBSTEPS)
    save_animation(render.swinging_chain(geometry, shape.edges, core.points(bodies, shape)), "modal_xpbd_swing", DURATION_MS)

    beam = core.girder(BEAM_CELLS, float(BEAM_CELLS), BEAM_HEIGHT, BEAM_STIFFNESS, DENSITY, MODES)
    bodies, constraints = clamped_beam(core, beam, BEAM_GIRDERS, BEAM_DAMPING)
    displacements = END_DISPLACEMENT * np.arange(BEAM_FRAMES) / BEAM_FRAMES
    frames = [shapes for shapes, _ in compress(core, beam, bodies, constraints, displacements, BEAM_INTERVAL, BEAM_SUBSTEPS)]
    save_animation(render.beam(frames, beam.edges), "modal_xpbd_buckle", DURATION_MS)


if __name__ == "__main__":
    main()
