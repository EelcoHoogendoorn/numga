"""The hollow top's Boolean quadrics, on the table, and its inclination history."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice, tee

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.mechanics.tippe_top import core
from examples.util import charts
from examples.util.wgsl import image_figure, solids

mv, origin = solids.mv, solids.origin
up = mv.z.dual()
shell_colour = np.array([0.78, 0.30, 0.16])
light_shell_colour = np.array([0.93, 0.62, 0.42])
stem_colour = np.array([0.84, 0.65, 0.32])
floor_colour = np.array([0.80, 0.79, 0.75])
background = np.array([0.96, 0.95, 0.93])
curve_colours = ("#d26038", "#297b97")
n_pixels = 300
half_extent = np.array([1.6, 1.6])
view_centre = origin + up * 1.1
floor = solids.half_space(mv.z)[None, None]
# The table's checkerboard, drawn about wherever a top has travelled, on whole squares, out past the far edge of the
# view: a board without end at a fixed count of squares. Each dark square is a line as wide as a square along its
# column's middle, within its row.
square = 1.0
n_squares = 25
square_steps = np.arange(n_squares) - n_squares // 2
dark_square_colour = np.array([0.62, 0.61, 0.57])
# The camera follows each top's mass centre across the table, averaged over the states either side of each frame,
# weighted down linearly: a few rocking periods, so that the camera glides with the drift and ignores the rocking.
n_smoothing = 16
smoothing = (n_smoothing + 1 - np.abs(np.arange(-n_smoothing, n_smoothing + 1))) / (n_smoothing + 1)**2
# The body's halves either side of a plane through its axis, and above and below its equator: chequered blocks of the
# shell, fixed to the body, so that its spin shows.
one_side, other_side = solids.half_space(mv.x), solids.half_space(-mv.x)     # [] Plane+ <- Point-
lower, upper = solids.half_space(mv.z), solids.half_space(-mv.z)            # [] Plane+ <- Point-


# --- plumbing -------------------------------------------------------------------------
def picture(body: core.Body, motor: solids.Motor, travel: solids.Point, elevation: float) -> np.ndarray:
    """The shell, carved by its cavity and chequered in blocks, joined to the stem and the ball at its end, on the
    table's checkerboard, the view moved along by the top's travel across the table so that it stays in view."""
    shell, cavity, stem, tip = (part.materialize(mv.context) for part in (body.shell, body.cavity, body.stem, body.tip))
    light = stack((stack((shell, -cavity, one_side, lower)), stack((shell, -cavity, other_side, upper))))   # [blocks, bounds] Plane+ <- Point-
    dark = stack((stack((shell, -cavity, one_side, upper)), stack((shell, -cavity, other_side, lower))))    # [blocks, bounds] Plane+ <- Point-
    stem_and_tip = stack((stem, tip))[:, None]                               # [solids, bounds] Plane+ <- Point-
    # The translation that brings the top back into view.
    following = (-travel * origin.inverse() / 2).exp()                       # [] Motor+
    placement = following * motor                                            # [] Motor+
    placed = [(placement >> solid(placement << core.Point), colour)
              for solid, colour in ((light, light_shell_colour), (dark, shell_colour), (stem_and_tip, stem_colour))]
    # The dark squares about the square the top is on, where column and row differ in parity, moved along with it.
    under = np.floor((stack((mv.x, mv.y)) & travel).to_array() / square)     # [2]
    columns, rows = np.meshgrid(under[0] + square_steps, under[1] + square_steps, indexing="ij")
    dark = (columns + rows) % 2 == 1                                          # [columns, rows]
    middles = following >> (mv.x - mv.w * ((columns[dark] + 0.5) * square))  # [squares] Plane+
    sides = following >> (mv.y - mv.w * ((rows[dark][:, None] + np.array([0, 1])) * square))   # [squares, 2] Plane+
    # Negative between a row's two sides: the pairings with both, multiplied.
    within_rows = (sides[:, 0] * (sides[:, 1] & core.Point) + sides[:, 1] * (sides[:, 0] & core.Point)) / 2
    view = solids.View(solids.aim(view_centre, elevation, -0.65), half_extent, n_pixels)
    board = solids.Ink(3, solids.half_space(middles), within_rows, dark_square_colour, square / view.pixel, 1.0)
    return solids.draw(placed + [(floor, floor_colour)], view, background, [board], [], [])


def draw_setup(body: core.Body) -> plt.Figure:
    """An elevated view into the cavity shows its orientation relative to the outer shell."""
    motor, _ = core.launch(body, np.array([0.]), 0.)
    on_table = motor[0].materialize(mv.context)                               # [] Motor+
    centre = (on_table >> body.mass.materialize(mv.context)).normalized()     # [] Point-
    return image_figure(picture(body, on_table, centre - origin - up * (mv.z & centre), 0.85))


def animate(body: core.Body, motors: Iterable[core.Motor], first_frame: int, n_frames: int) -> Iterator[np.ndarray]:
    """Opposite-spin launches, from a stream of their placements the frames from `first_frame` on, each followed
    horizontally at the same camera scale, smoothly: by its mass centre's travel averaged over the states about it."""
    separator = np.full((n_pixels, 6, 3), 255, dtype=np.uint8)
    mass = body.mass.materialize(mv.context)                                 # [] Point-
    shown = islice(motors, first_frame - n_smoothing, first_frame + n_frames + n_smoothing)
    # Sliding windows of the states either side of each frame.
    copies = tee((motor.materialize(mv.context) for motor in shown), len(smoothing))
    for window in zip(*(islice(copy, offset, None) for offset, copy in enumerate(copies))):
        on_table = stack(window)                                              # [states, spins] Motor+
        centres = (on_table >> mass).normalized()                             # [states, spins] Point-
        travel = ((centres - origin - up * (mv.z & centres)) * smoothing[:, None]).sum(axis=0)   # [spins] Point-
        motor = on_table[n_smoothing]                                         # [spins] Motor+
        yield np.concatenate((picture(body, motor[0], travel[0], 0.24), separator,
                              picture(body, motor[1], travel[1], 0.24)), axis=1)


@charts.style()
def history(times: np.ndarray, vertical: core.Scalar) -> plt.Figure:
    """The body axis's vertical component distinguishes upright spinning from inversion."""
    figure, ax = charts.subplots(450, 180)
    for values, colour in zip(vertical.to_array().T, curve_colours):
        ax.plot(times, values, color=colour)
    ax.axhline(0, color="#99958e", linewidth=charts.FRAME_WIDTH)
    ax.set(xlim=(times[0], times[-1]), ylim=(-1.08, 1.08),
           xlabel="time", ylabel="vertical axis", xticks=(), yticks=(-1, 0, 1))
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure
