"""Scenes for the rattleback: released tipped from rest it rocks into spin; spun the unfavourable way, it
rocks, stops and spins the other way.

The body is a wooden boat with an ellipsoidal underside, a flat deck and two cuts across opposite corners of
the top, and a steel ball pressed into each side, the second of every cut and ball half a turn about the
vertical from the first. Every part is an intersection of quadrics, and the same quadrics give its mass: a grid
of points, each weighted by the density of the part it lies inside, carries the centre of mass and the inertia
the dynamics use, and the drawing traces the same quadrics.
"""

from __future__ import annotations

from itertools import islice

import numpy as np

from numga import stack
from numga.algebras import PGA3D
from examples import instantiate

core = instantiate("examples.mechanics.rattleback.core", PGA3D)
mv = core.mv
mirrors = stack((mv.x, mv.y, mv.z))                                          # [3] Plane+
# The shell's semi-axes, the body's mass and gravity.
SEMI_AXES = np.array([5, 17 / 16, 1])
MASS = 4
GRAVITY = 9.8
# The deck's height above the shell's centre; the chamfers' tip about the long axis and along it, and a point
# they pass through.
DECK_HEIGHT = 0.5
CHAMFER_ROLL = np.deg2rad(35)
CHAMFER_PITCH = np.deg2rad(30)
CHAMFER_THROUGH = [3.2, 0, 0.2]
# The steel balls' radius, their place along the shell, and the share of their diameter pressed into the wood.
BALL_RADIUS = 0.4
BALL_ALONG = 3
BALL_SINK = 0.6
WOOD_DENSITY = 0.7
STEEL_DENSITY = 7.85
# The spacing of the grid the mass is read from.
SAMPLE_SPACING = 0.15
# The two launches: tipped from rest, and spinning the unfavourable way with a little pitching.
TILTS = np.deg2rad([15, 0])
SPINS = np.array([0, -1.5])
PITCH_RATES = np.array([0, 0.1])
# The time step and the steps, the steps between frames drawn, and the time the rocking animation runs.
TIME_STEP = 0.04
N_STEPS = 250
FRAME_STRIDE = 2
ROCKING_DURATION = 8
# Gallery picture: this makes plots/rattleback.gif for docs/extensors.md, tuned as a visual for the gallery. Don't change it; to experiment, copy it.
# The gallery's reversal: its pixels across and the steps between frames drawn.
GALLERY_PIXELS = 260
GALLERY_STRIDE = 3


# --- math -----------------------------------------------------------------------------
def distribution(wood: core.Quadric, steel: core.Quadric) -> tuple[core.Point, core.Point, core.Inertia]:
    """The grid points inside the body, its centre of mass, and its inertia about that: each point weighted by
    the density of the part it lies inside, inside a part where it pairs negatively with each of its quadrics."""
    low = -SEMI_AXES - [0, 2 * BALL_RADIUS, 0]
    high = np.array([SEMI_AXES[0], SEMI_AXES[1] + 2 * BALL_RADIUS, DECK_HEIGHT])
    axes = [np.arange(start, stop, SAMPLE_SPACING) + SAMPLE_SPACING / 2 for start, stop in zip(low, high)]
    grid = mv.point(np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3))   # [grid] Point-
    in_wood = ((wood[:, None](grid) & grid) <= 0).all(axis=0)                 # [grid]
    in_steel = ((steel[:, None](grid) & grid) <= 0).any(axis=0)               # [grid]
    densities = WOOD_DENSITY * in_wood + STEEL_DENSITY * in_steel            # [grid]
    inside = densities > 0
    samples, weights = grid[inside], densities[inside] / densities[inside].sum()   # [samples] Point-, [samples]
    centre = (samples * weights).sum(axis=0)                                 # [] Point-
    # Each sample joined with its own motion under a rate is its momentum; the inertia is their weighted sum.
    relative = core.origin + (samples - centre)                              # [samples] Point-
    inertia = (relative & relative.commutator(core.Rate) * weights).sum(axis=0) * MASS   # Forque- <- Rate+
    return samples, centre, inertia


def body() -> tuple[core.Shape, core.Direction, core.Inertia, core.Body, core.Point]:
    """The shell's support map, the shell's centre seen from the centre of mass, the inertia about the centre
    of mass, and the body as built and the points its mass is read from, in the frame of its centre of mass."""
    directions = mirrors.dual()                                              # [3] Direction-
    outline = (directions * (mirrors & core.Direction) * SEMI_AXES).sum(axis=0)       # Direction- <- Direction-
    shape = (directions * (core.Plane & directions) * SEMI_AXES**2).sum(axis=0)       # Direction- <- Plane+
    # The unit ball's planes across its axes carried onto the shell: each pairs with a stretched direction as
    # the ball's plane does with the direction, so the stretch's adjugate carries it back.
    across = outline.adjugate().solve(mirrors)                               # [3] Plane+
    underside = (across * (across & core.Point)).sum(axis=0) - mv.w * (mv.w & core.Point)   # Cut+ <- Point-
    deck = half_space(mv.z - mv.w * DECK_HEIGHT)                             # Cut+ <- Point-

    # Every cut and ball has its partner half a turn about the vertical.
    sides = (mv.xy * np.array([0, np.pi]) / 2).exp()                         # [sides] Motor+
    tipped = (mv.yz * (-CHAMFER_ROLL / 2)).exp() * (mv.xz * (CHAMFER_PITCH / 2)).exp() >> mv.z   # Plane+
    corner = core.origin + mv.direction(CHAMFER_THROUGH)                     # Point-
    chamfers = half_space(sides >> (tipped - mv.w * (tipped & corner)))      # [sides] Cut+ <- Point-
    # A ball on the shell's side, its diameter pressed in by the sink.
    side = SEMI_AXES[1] * np.sqrt(1 - (BALL_ALONG / SEMI_AXES[0])**2)
    seat = core.origin + mv.direction([BALL_ALONG, side + BALL_RADIUS * (1 - 2 * BALL_SINK), 0])   # Point-
    balls = ball(sides >> seat, BALL_RADIUS)                                 # [sides] Cut+ <- Point-

    built = core.Body(underside, deck, chamfers, balls)
    samples, centre, inertia = distribution(built.wood(), balls)
    # Everything moved so that the centre of mass is the origin.
    shift = ((core.origin - centre) * core.origin.inverse() / 2).exp()       # [] Motor+
    placed = core.Body(*(shift >> part(shift << core.Point) for part in built))
    return shape, core.origin - centre, inertia, placed, shift >> samples


def launches(dt: float, n_steps: int) -> tuple[core.Body, core.Point, core.Motor, core.Spin, core.Scalar]:
    """Both launches from their start over the steps `[steps + 1, cases]`: the body and the points its mass is
    read from, the placements, the body-frame spins and the energies."""
    shape, centre, inertia, built, samples = body()
    up = mv.z.dual()                                                         # [] Direction-
    gravity = -up * GRAVITY                                                  # [] Direction-
    spin = mv.xy * SPINS - mv.xz * PITCH_RATES                               # [cases] Spin+
    tilt = (-mv.xz * TILTS / 2).exp()                                        # [cases] Motor+
    # Each launch lifted by its centre of mass's height, so that the shell rests on the table: the shell's
    # centre's height, less how far that centre sits above the centre of mass.
    normal = (tilt << up).dual_inverse()                                     # [cases] Plane+
    height = (normal & shape(normal)).square_root() - (normal & centre)     # [cases] Scalar+
    motor = (up * core.origin.inverse() * height / 2).exp() * tilt           # [cases] Motor+

    def motion(motor: core.Motor, spin: core.Spin) -> tuple[core.Rate, core.Spin]:
        return core.dynamics(motor, spin, shape, centre, inertia, MASS, gravity, up)

    states = islice(core.evolve(motor, spin, dt, motion), n_steps + 1)
    placements, spins = (stack(list(history)) for history in zip(*states))   # [steps + 1, cases] each
    energies = core.energy(placements, spins, shape, centre, inertia, MASS, gravity, up)   # [steps + 1, cases] Scalar+
    return built, samples, placements, spins, energies


# --- plumbing -------------------------------------------------------------------------
def half_space(cut: core.Cut) -> core.Quadric:
    """The points on a plane's negative side, as the quadric pairing them with the plane and with the plane at
    infinity, symmetrized."""
    return (cut * (mv.w & core.Point) + mv.w * (cut & core.Point)) / 2


def ball(centre: core.Point, radius: float) -> core.Quadric:
    """Solid balls about points of unit weight: the planes across the axes through each centre, each one at a
    radius from it, less the weight squared."""
    across = mirrors - mv.w * (mirrors & centre[..., None])                  # [..., 3] Cut+
    return (across * (across & core.Point)).sum(axis=-1) / radius**2 - mv.w * (mv.w & core.Point)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.mechanics.rattleback import render

    duration_ms = round(TIME_STEP * FRAME_STRIDE * 1000)
    built, samples, placements, spins, energies = launches(TIME_STEP, N_STEPS)
    rocking_frames = slice(0, round(ROCKING_DURATION / TIME_STEP) + 1, FRAME_STRIDE)
    reversal_frames = slice(None, None, FRAME_STRIDE)
    save_figure(render.draw_setup(built, samples, placements[0, 1]), "rattleback_body")
    save_animation(render.frames(placements[rocking_frames, 0], built, samples), "rattleback_rocking", duration_ms)
    save_animation(render.frames(placements[reversal_frames, 1], built, samples), "rattleback_reversal", duration_ms)
    # Gallery picture: this makes plots/rattleback.gif for docs/extensors.md, tuned as a visual for the gallery. Don't change it; to experiment, copy it.
    save_animation(render.sized_frames(placements[::GALLERY_STRIDE, 1], built, samples, GALLERY_PIXELS), "rattleback",
                   round(TIME_STEP * GALLERY_STRIDE * 1000))

    # checks
    # The tipped body spins up from rest, the spinning one reverses, and the energy is kept.
    spun = (-(placements >> spins).scalar_product(mv.xy)).to_array()          # [steps + 1, cases]
    assert abs(spun[0, 0]) < 1e-11 and spun[-1, 0] > 1.4
    assert spun[0, 1] < -1.4 and spun[-1, 1] > 1.4
    np.testing.assert_allclose((energies / energies[0]).to_array(), 1, rtol=1e-2)


if __name__ == "__main__":
    main()
