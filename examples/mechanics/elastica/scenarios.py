"""Scenes for the elastica: a strip curled by a couple at its end from straight to a double ring; Euler's
elastica, a rod pinned at both ends under end thrust, at end angles from a shallow arch to a tight loop;
Kirchhoff's helices, a wire coiled by a couple oblique to it; a wire under a fixed couple, its tilt swept
from square to the couple to along it, from a ring to a straight twist; and a coiled telephone cord, hanging
under its own weight, its handset pulled out along the coil and let go."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import concatenate, stack
from examples.mechanics.elastica import core

mv = core.mv
# The rod's length, the pieces it is cut into, and its compliances about and along the cross-section's axes,
# across, up and along: to turning, bending across and up and twisting, and none to shifting, a rod that only
# bends and twists.
LENGTH = 1.0
N_PIECES = 200
TURNING = np.array([1.0, 1.0, 1.4])
SHIFTING = np.zeros(3)
# The curled strip: its couple in whole turns of its end.
CURLS = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0]
# Euler's elastica: the angle at which the rod leaves its first pin, in degrees, for the plate and for the
# frames of the sweep, and the duration of a frame.
END_ANGLES = [20.0, 60.0, 90.0, 110.0, 130.7, 150.0, 165.0, 175.0]
SWEEP = np.linspace(2.0, 176.0, 60)
DURATION_MS = 60
# Kirchhoff's helices: the tilt of the rod's start from the couple, upward, in degrees, the whole turns the
# couple coils the rod through, and the width of the ribbon drawn along the rod.
TILTS = [80.0, 60.0, 40.0, 20.0]
N_TURNS = 3
RIBBON = 0.015
# The uncoiling wire: the whole turns its couple coils it through, its tilt from the couple swept from square to
# along and back over the frames, the width of its ribbon, and the duration of a frame.
UNCOIL_TURNS = 1.5
N_UNCOIL_FRAMES = 90
UNCOIL_RIBBON = 0.035
UNCOIL_MS = 60
# The telephone cord: its length in metres, its coils, the segments per coil, and the angle of its wire to the
# plane of a coil; its rigidities against turning about the cross-section's axes, bending across and up and
# twisting, in newton square metres, and against shifting along them, shearing across and stretching along, in
# newtons; its mass per metre in
# kilograms, the radius its segments' mass is spread over as a share of a segment's length, gravity in metres
# per second squared, and its damping per second.
CORD_LENGTH, N_CORD_COILS, N_PER_COIL, CORD_PITCH = 1.0, 5, 6, 10.0
CORD_TURNING = np.array([1e-3, 1e-3, 7e-4])
CORD_SHIFTING = np.array([20.0, 20.0, 20.0])
CORD_MASS, CORD_SPREAD, GRAVITY, CORD_DAMPING = 0.05, 2.0, 9.81, 1.0
# The pull: the time the cord hangs first, the time the handset takes out and the distance, the time it is
# held, and the time it takes back, in seconds and metres; the time step, the steps per frame, the frames, the
# drawn frames between neighbouring segments, the width of the drawn ribbon, and the duration of a frame.
HANG, OUT, REACH, HOLD, BACK = 0.3, 0.9, 0.6, 0.3, 0.4
TIME_STEP, N_STEPS_PER_FRAME, N_CORD_FRAMES = 1e-3, 40, 70
N_DRAWN_BETWEEN, CORD_RIBBON, CORD_MS = 4, 0.025, 40


# --- math -----------------------------------------------------------------------------
def turned(degrees: np.ndarray) -> core.Motor:
    """The turns by each angle about the axis across the rod's start."""
    return (mv.yz * (-np.radians(degrees) / 2)).exp()


def curling() -> core.Motor:
    """The strip's frames under a couple about the axis across it, by each number of turns of its end
    `[curls, pieces + 1]`: the couple whose rigidity over it is the strip's length over the turns."""
    couples = (mv.y ^ mv.z).dual() * (2 * np.pi / (TURNING[0] * LENGTH) * np.array(CURLS))   # [curls] Forque
    return core.rod(couples, core.compliance(TURNING, SHIFTING), mv.rotor(), LENGTH, N_PIECES)


def euler(degrees: np.ndarray) -> core.Motor:
    """The frames of rods leaving their first pin at each angle to the line of thrust through both pins
    `[angles, pieces + 1]`, under the thrust that brings them back to the line at their full length."""
    thrust = (core.centre & mv.z.dual()) * thrusts(np.radians(degrees))     # [angles] Forque
    return core.rod(thrust, core.compliance(TURNING, SHIFTING), turned(degrees), LENGTH, N_PIECES)


def kirchhoff() -> core.Motor:
    """The frames of rods started at each tilt from an upward couple `[tilts, pieces + 1]`, the couple coiling
    them through whole turns."""
    couple = (mv.x ^ mv.y).dual() * (2 * np.pi * N_TURNS / (TURNING[0] * LENGTH))           # [] Forque
    return core.rod(couple, core.compliance(TURNING, SHIFTING), turned(np.array(TILTS)), LENGTH, N_PIECES)


def uncoiling() -> core.Motor:
    """The frames of a wire under a fixed upward couple, its start tilted from the couple by an angle swept
    from square to along it and back `[frames, pieces + 1]`: the couple bending it into a ring, coiling it,
    and twisting it straight."""
    tilts = 45 * (1 + np.cos(np.linspace(0, 2 * np.pi, N_UNCOIL_FRAMES, endpoint=False)))   # [frames] degrees
    couple = (mv.x ^ mv.y).dual() * (2 * np.pi * UNCOIL_TURNS / (TURNING[0] * LENGTH))    # [] Forque
    return core.rod(couple, core.compliance(TURNING, SHIFTING), turned(tilts), LENGTH, N_PIECES)


def cord_tilt() -> core.Motor:
    """The turn by the cord's pitch from the rod's cross-section towards its length: it tilts the coil's
    turn, about the line along x, towards the line along z."""
    return ((mv.x ^ mv.z) * (-np.radians(CORD_PITCH) / 2)).exp()             # [] Motor


def coil(n_pieces: int) -> tuple[core.Twist, core.Motor]:
    """The cord at rest: its turn per unit length, a shift along it with a bend and a twist that coil it, and
    the rest frames of its segments with one held frame beyond each end `[pieces + 2]`."""
    turning = 2 * np.pi * N_CORD_COILS / CORD_LENGTH
    rest = core.shift_along_rod + (cord_tilt() >> mv.yz) * turning          # [] Twist
    step = CORD_LENGTH / n_pieces
    return rest, (rest * (np.arange(-1, n_pieces + 1) * step / 2)).exp()   # [pieces + 2] Motor


def pulled() -> Iterator[core.Motor]:
    """The cord's frames at every frame of the scene: hanging from its base under its weight, its handset
    pulled out along the coil's axis, held, and let back, the cord following by its dynamics. A frame's steps run
    under `jax.lax.scan`: each is many small operations on a short cord, which the compiled loop does at once."""
    import jax
    from numga.backend.jax import JaxContext

    n_pieces = N_CORD_COILS * N_PER_COIL
    step = CORD_LENGTH / n_pieces
    rest, frames = coil(n_pieces)
    stiff = core.stiffness(CORD_TURNING, CORD_SHIFTING)
    mass = CORD_MASS * step
    inertia = core.segment_inertia(mass, step, CORD_SPREAD * step)            # [] Forque <- Twist
    inverse = inertia.inverse()                                              # [] Twist <- Forque
    weight = mv.y.dual() * (-GRAVITY * mass)                                 # [] Point
    # The coil's axis is the direction of its turn; the handset moves along it, keeping its turn.
    axis = cord_tilt() >> mv.x                                               # [] Vector
    handset = frames[-1]
    joint = (rest * (step / 2)).exp()                                        # [] Motor

    def tick(state: tuple[core.Motor, core.Twist], out: float) -> tuple[tuple[core.Motor, core.Twist], None]:
        frames, rates = state
        # The handset placed out along the coil's axis, then the cord moved by its dynamics.
        moved = concatenate([frames[:-1], (((axis ^ mv.w) * (out / 2)).exp() * handset)[None]])
        return core.moving(moved, rates, joint, stiff, step, inertia, inverse, weight, CORD_DAMPING, TIME_STEP), None

    # A frame's steps compiled into one loop over how far the handset is out at each, carrying the cord's segments as
    # plain motors and their rates as twists.
    frame = jax.jit(lambda state, outs: jax.lax.scan(tick, state, outs)[0])
    outs = reach(np.arange(N_CORD_FRAMES * N_STEPS_PER_FRAME) * TIME_STEP).reshape(N_CORD_FRAMES, N_STEPS_PER_FRAME)
    state = (frames.retyped(core.Motor).materialize(JaxContext(core.ga)),
             mv.bivector(np.zeros((n_pieces, 6))).retyped(core.Twist).materialize(JaxContext(core.ga)))
    for frame_outs in outs:
        yield state[0].materialize(core.mv.context)
        state = frame(state, frame_outs)


# --- plumbing -------------------------------------------------------------------------
def reach(time: float) -> float:
    """How far the handset is out at a time: eased out after the hang, held, and eased back."""
    out = np.clip((time - HANG) / OUT, 0, 1)
    back = np.clip((time - HANG - OUT - HOLD) / BACK, 0, 1)
    return REACH * (3 * out ** 2 - 2 * out ** 3) * (1 - (3 * back ** 2 - 2 * back ** 3))


def ribbon(frames: core.Motor, width: float) -> core.Point:
    """The edges of a ribbon of the given width across each cross-section `[..., 2]`."""
    return frames[..., None] >> mv.point([[-width / 2, 0, 0], [width / 2, 0, 0]])


def thrusts(angles: np.ndarray) -> np.ndarray:
    """The end thrust under which a pinned rod leaving its pin at each angle reaches the line of thrust again
    at its full length: twice the complete elliptic integral of the first kind of the sine of half the angle,
    over the length, squared, over the compliance to bending."""
    modulus = np.sin(angles / 2)
    # The complete elliptic integral by the arithmetic-geometric mean: a quarter turn over the mean.
    mean, geometric = np.ones_like(modulus), np.sqrt(1 - modulus ** 2)
    for _ in range(8):
        mean, geometric = (mean + geometric) / 2, np.sqrt(mean * geometric)
    return (np.pi / mean / LENGTH) ** 2 / TURNING[0]


def chords(angles: np.ndarray) -> np.ndarray:
    """The distance between the pins of Euler's elastica at each angle: the length times twice the ratio of
    the complete elliptic integrals of the second and first kind, less one."""
    modulus = np.sin(angles / 2)
    # Both integrals by the arithmetic-geometric mean, the second from its sum of squared differences.
    mean, geometric, weight, deficit = np.ones_like(modulus), np.sqrt(1 - modulus ** 2), 0.5, modulus ** 2 / 2
    for _ in range(8):
        mean, geometric, difference = (mean + geometric) / 2, np.sqrt(mean * geometric), (mean - geometric) / 2
        weight, deficit = 2 * weight, deficit + 2 * weight * difference ** 2
    return LENGTH * (2 * (1 - deficit) - 1)


def positions(points: core.Point) -> np.ndarray:
    """x, y and z of unit points, on the last axis: their pairings with the coordinate planes."""
    return (stack([mv.x, mv.y, mv.z]) & points[..., None]).to_array()


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.mechanics.elastica import render

    curls = curling()                                                        # [curls, pieces + 1] Motor
    save_figure(render.draw_curls(curls >> core.centre), "elastica_curls")
    rods = euler(np.array(END_ANGLES))                                       # [angles, pieces + 1] Motor
    save_figure(render.draw_plate(rods >> core.centre), "elastica_euler")
    save_animation(render.animate_sweep(euler(SWEEP) >> core.centre), "elastica_sweep", DURATION_MS)
    coiled = kirchhoff()                                                     # [tilts, pieces + 1] Motor
    save_figure(render.draw_helices(coiled >> core.centre, ribbon(coiled, RIBBON)), "elastica_helices")
    uncoiled = uncoiling()                                                   # [frames, pieces + 1] Motor
    save_animation(render.animate_uncoiling(uncoiled >> core.centre, ribbon(uncoiled, UNCOIL_RIBBON)),
                   "elastica_uncoiling", UNCOIL_MS)
    moved = stack(list(pulled()))                                            # [frames, pieces + 2] Motor
    cord = core.between(moved, N_DRAWN_BETWEEN)                                # [frames, drawn] Motor
    save_animation(render.animate_cord(cord >> core.centre, ribbon(cord, CORD_RIBBON)), "elastica_cord", CORD_MS)

    # checks
    # Curled by whole turns, the strip's end comes back to its start.
    ends = positions(curls[[0, 4, 5], -1] >> core.centre)
    assert np.abs(ends[1:]).max() < 1e-11 and abs(ends[0, 2] - LENGTH) < 1e-11
    # Euler's elastica reach the line of thrust again at their far ends, a chord away from the first pin; the
    # tolerance tests the steps along the rod.
    far = positions(rods[:, -1] >> core.centre)                              # [angles, 3]
    assert np.abs(far[:, :2]).max() < 1e-3
    assert np.abs(far[:, 2] - chords(np.radians(END_ANGLES))).max() < 1e-3
    # Coiled through whole turns, the wire's end lies straight above its start, the length times the cosine
    # of the tilt higher; the tolerance tests the steps along the rod.
    tips = positions(coiled[:, -1] >> core.centre)                           # [tilts, 3]
    assert np.abs(tips - LENGTH * np.cos(np.radians(TILTS))[:, None] * [0, 0, 1]).max() < 3e-3
    # The telephone cord bends and twists but hardly stretches or shears: its joints' shifts stay within a few
    # hundredths of their rest, the most where the handset springs back.
    rest, _ = coil(N_CORD_COILS * N_PER_COIL)
    joint = (rest * (CORD_LENGTH / (N_CORD_COILS * N_PER_COIL) / 2)).exp()
    deviations = (joint.reverse() * moved[:, :-1].reverse() * moved[:, 1:]).log() * 2   # [frames, joints] Twist
    shifts = deviations.cast(core.ga.subspace("wx wy wz")).kernel / (CORD_LENGTH / (N_CORD_COILS * N_PER_COIL))
    assert np.abs(shifts).max() < 0.1


if __name__ == "__main__":
    main()
