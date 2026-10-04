"""Flexible girders pinned together at their corners: a cantilever of spliced girders, a chain of
girders hinged at one corner swinging under gravity, and a beam of spliced girders clamped at both
ends and crushed until it buckles."""

from dataclasses import replace
from collections.abc import Iterator

import numpy as np

from .core import Bodies, Direction, Motor, Pins, Point, Shape, girder, mv, points, step

# The chain's girder: its cells, length and height, its bars' stiffness, its density, and the
# vibration modes it keeps.
CELLS, LENGTH, HEIGHT = 4, 1.0, 0.2
STIFFNESS, DENSITY, MODES = 300.0, 1.0, 8
# The chain's moving girders, their modes' damping ratio, and gravity.
LINKS = 4
DAMPING = np.array([0.02])
GRAVITY = (mv.y * -4).dual()                                               # [] Direction
# A pin's compliance: a weld stiff to round-off, which keeps a rigid splice's two pins solvable.
SPLICE = 1e-14
# The swing's time step, its frames and the steps between them, and the frame duration.
INTERVAL, FRAMES, SUBSTEPS = 0.002, 200, 15
DURATION_MS = 30
# The beam: its moving girders, each of unit cells as long as it has cells, as stiff as steel wire
# against its weight, overdamped; the end displacement it is crushed by, and its frames.
BEAM_GIRDERS, BEAM_CELLS, BEAM_HEIGHT = 8, 32, 1.0
BEAM_STIFFNESS, BEAM_DAMPING = 1e11, np.array([3.0])
CRUSH, BEAM_FRAMES, BEAM_SUBSTEPS, BEAM_INTERVAL = 0.15, 120, 20, 0.002
WEIGHTLESS = (mv.y * 0).dual()                                             # [] Direction


# --- plumbing -------------------------------------------------------------------------
def girders(shape: Shape, fixed: np.ndarray, damping: np.ndarray, flexibility: np.ndarray) -> Bodies:
    """Girders end to end along x, the first centred half a girder left of the origin, those marked
    fixed held still and rigid; one case for each damping ratio and flexibility."""
    count, modes, cases = len(fixed), shape.modes.shape[0], len(damping)
    length = (shape.rest[-1] - shape.rest[1]).dual() | mv.x               # [] Scalar
    motor = (mv.xw * ((np.arange(count) - 0.5) * length) * 0.5).exp().cast(Motor).broadcast_to((cases, count))  # [cases, bodies] Motor
    amplitudes = mv.scalar(np.zeros((cases, count, modes, 1)))             # [cases, bodies, modes] Scalar
    moving = ~fixed
    return Bodies(
        motor=motor,
        rate=mv.xy * np.zeros((cases, count)),
        amplitudes=amplitudes,
        rates=amplitudes,
        compliance=shape.compliance * flexibility[:, None, None] * moving[None, :, None],
        frequencies=shape.frequencies.broadcast_to((cases, count, modes)),
        damping=np.broadcast_to(damping[:, None, None], (cases, count, modes)),
        masses=np.full((cases, count), shape.masses.sum()),
        inertia=shape.inertia.broadcast_to((cases, count)),
        inverse_inertia=(shape.inertia.inverse() * moving).broadcast_to((cases, count)),
    )


def pinned(shape: Shape, bodies: np.ndarray, corners: np.ndarray) -> Pins:
    """Pins between the given bodies `[pins, ends]` at the given vertices of each."""
    modes = shape.modes.shape[0]
    compliance = mv.scalar([SPLICE]).broadcast_to(bodies.shape[:1])       # [pins] Scalar
    return Pins(bodies, shape.rest[corners], shape.modes[np.arange(modes), corners[..., None]], compliance)


def splices(shape: Shape, count: int) -> Pins:
    """Two pins between each of count girders and the next, at the lower and upper corners where
    they meet."""
    vertices = shape.rest.shape[0]
    joints = np.arange(count - 1)
    bodies = np.repeat(np.stack([joints, joints + 1], axis=-1), 2, axis=0)  # [pins, ends]
    corners = np.tile([[vertices - 2, 0], [vertices - 1, 1]], (count - 1, 1))  # [pins, ends]
    return pinned(shape, bodies, corners)


def hinges(shape: Shape, count: int) -> Pins:
    """One pin between each of count girders and the next, at the upper corners where they meet."""
    vertices = shape.rest.shape[0]
    joints = np.arange(count - 1)
    return pinned(shape, np.stack([joints, joints + 1], axis=-1), np.tile([[vertices - 1, 1]], (count - 1, 1)))


def cantilever(shape: Shape, count: int, damping: np.ndarray, flexibility: np.ndarray) -> tuple[Bodies, Pins]:
    """A fixed girder followed by count moving girders, spliced end to end."""
    fixed = np.arange(count + 1) == 0
    return girders(shape, fixed, damping, flexibility), splices(shape, count + 1)


def hinged_chain(shape: Shape, count: int, damping: np.ndarray) -> tuple[Bodies, Pins]:
    """A fixed girder followed by count moving girders, each hinged to the last at an upper corner."""
    fixed = np.arange(count + 1) == 0
    return girders(shape, fixed, damping, np.ones_like(damping)), hinges(shape, count + 1)


def clamped_beam(shape: Shape, count: int, damping: np.ndarray) -> tuple[Bodies, Pins]:
    """Count moving girders spliced end to end between two fixed ones, the clamps."""
    fixed = (np.arange(count + 2) == 0) | (np.arange(count + 2) == count + 1)
    return girders(shape, fixed, damping, np.ones_like(damping)), splices(shape, count + 2)


# --- math -----------------------------------------------------------------------------
def swing(shape: Shape, bodies: Bodies, pins: Pins, gravity: Direction, dt: float, frames: int, substeps: int) -> Iterator[Point]:
    """The bodies' points at every frame, stepped under gravity."""
    for _ in range(frames):
        yield points(bodies, shape)                                        # [cases, bodies, vertices] Point
        for _ in range(substeps):
            bodies = step(bodies, pins, dt, gravity)


def crushed(bodies: Bodies, rest: Motor, displacement: float) -> Bodies:
    """The bodies with the last, the moving clamp, driven inward along x from where it rests."""
    driven = rest[..., -1] * (mv.xw * (-displacement / 2)).exp()          # [cases] Motor
    return replace(bodies, motor=bodies.motor.at[..., -1].set(driven).normalized())


def crush(shape: Shape, bodies: Bodies, pins: Pins, crushing: np.ndarray, dt: float, substeps: int) -> Iterator[tuple[Point, Point]]:
    """The beam's points and its midspan, the middle of its two middle girders' centres, at every
    frame, its moving clamp driven in by each displacement in turn."""
    rest, middle = bodies.motor, bodies.motor.shape[-1] // 2
    for displacement in crushing:
        bodies = crushed(bodies, rest, displacement)
        for _ in range(substeps):
            bodies = step(bodies, pins, dt, WEIGHTLESS)
        centres = bodies.motor[..., middle - 1:middle + 1] >> mv.w.dual()  # [cases, 2] Point
        yield points(bodies, shape), centres.sum(axis=-1) / 2               # [cases, bodies, vertices] Point, [cases] Point


def critical(height: float, span: float) -> float:
    """The end displacement at which a beam clamped at both ends buckles, for two chords a height
    apart: the Euler load `4 pi**2 EI / span**2`, with `EI = EA height**2 / 2`, over the axial
    stiffness `2 EA / span`."""
    return np.pi**2 * height**2 / span


def main():
    from examples.animation import save_animation
    from . import render

    shape = girder(CELLS, LENGTH, HEIGHT, STIFFNESS, DENSITY, MODES)
    bodies, pins = hinged_chain(shape, LINKS, DAMPING)
    geometry = swing(shape, bodies, pins, GRAVITY, INTERVAL, FRAMES, SUBSTEPS)
    save_animation(render.swinging_chain(geometry, shape.edges, points(bodies, shape)), "modal_xpbd_swing", DURATION_MS)

    beam = girder(BEAM_CELLS, float(BEAM_CELLS), BEAM_HEIGHT, BEAM_STIFFNESS, DENSITY, MODES)
    bodies, pins = clamped_beam(beam, BEAM_GIRDERS, BEAM_DAMPING)
    crushing = CRUSH * np.arange(BEAM_FRAMES) / BEAM_FRAMES
    frames = [shapes for shapes, _ in crush(beam, bodies, pins, crushing, BEAM_INTERVAL, BEAM_SUBSTEPS)]
    save_animation(render.beam(frames, beam.edges), "modal_xpbd_buckle", DURATION_MS)


if __name__ == "__main__":
    main()
