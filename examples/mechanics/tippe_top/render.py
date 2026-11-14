"""Ray-traced views and stem inclination charts depict the hollow tippe top and its opposite-spin launches."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice, tee

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from numga.render import solids
from examples.mechanics.spinning_top import core as contact
from examples.util import charts
from examples.util.panels import image_figure

mv, origin = solids.mv, solids.origin
up = mv.z.dual()                                                             # [] Point-
shell_colour = np.array([0.78, 0.30, 0.16])
light_shell_colour = np.array([0.93, 0.62, 0.42])
stem_colour = np.array([0.84, 0.65, 0.32])
floor_colour = np.array([0.80, 0.79, 0.75])
background = np.array([0.96, 0.95, 0.93])
curve_colours = ("#d26038", "#297b97")
n_pixels = 300
half_extent = np.array([1.6, 1.6])
view_centre = origin + up * 1.1                                              # [] Point-
floor = solids.half_space(mv.z)[None, None]                                  # [1, 1] Plane+ <- Point-
# Fixed-count checkerboard window centred on whichever square the top currently occupies.
square_size = 1.0
n_squares = 25
square_steps = np.arange(n_squares) - n_squares // 2
dark_square_colour = np.array([0.62, 0.61, 0.57])
# Triangular temporal weights smooth out rapid nutation so the tracking camera glides with horizontal drift.
n_smoothing = 16
n_window = 2 * n_smoothing + 1
smoothing_weights = (n_smoothing + 1 - np.abs(np.arange(-n_smoothing, n_smoothing + 1))) / (n_smoothing + 1)**2
# Body-fixed half-spaces divide the shell into alternating octants so spinning stays visible.
front_half, back_half = solids.half_space(mv.x), solids.half_space(-mv.x)    # [] Plane+ <- Point-, Plane+ <- Point-
lower_half, upper_half = solids.half_space(mv.z), solids.half_space(-mv.z)   # [] Plane+ <- Point-, Plane+ <- Point-


def picture(placed_parts: solids.Quadric, motor: solids.Motor, travel: solids.Point, elevation: float) -> np.ndarray:
    """Renders the chequered hollow shell, stem, and tip (`[4] Plane+ <- Point-`) on a camera-tracked checkerboard (`[n_pixels, n_pixels, 3]`)."""
    shell, cavity, stem, tip = placed_parts
    light_blocks = stack((
        stack((shell, -cavity, front_half, lower_half)),
        stack((shell, -cavity, back_half, upper_half)),
    ))                                                                       # [2, 4] Plane+ <- Point-
    dark_blocks = stack((
        stack((shell, -cavity, front_half, upper_half)),
        stack((shell, -cavity, back_half, lower_half)),
    ))                                                                       # [2, 4] Plane+ <- Point-
    stem_and_tip = stack((stem, tip))[:, None]                               # [2, 1] Plane+ <- Point-
    camera_shift = (-travel * origin.inverse() / 2).exp()                    # [] Motor+
    placement = camera_shift * motor                                         # [] Motor+
    placed_solids = [
        (placement >> solid(placement << solids.Point), colour)
        for solid, colour in ((light_blocks, light_shell_colour), (dark_blocks, shell_colour), (stem_and_tip, stem_colour))
    ]
    # Dark squares surround the tile under the top wherever column and row indices have opposite parity.
    base_tile = np.floor((stack((mv.x, mv.y)) & travel).to_array() / square_size)  # [2]
    columns, rows = np.meshgrid(base_tile[0] + square_steps, base_tile[1] + square_steps, indexing="ij")
    odd_squares = (columns + rows) % 2 == 1                                  # [n_squares, n_squares]
    unshifted_middles = mv.x - mv.w * ((columns[odd_squares] + 0.5) * square_size)  # [...] Plane+
    column_middles = camera_shift >> unshifted_middles                       # [...] Plane+
    unshifted_sides = mv.y - mv.w * ((rows[odd_squares][:, None] + np.array([0, 1])) * square_size)  # [..., 2] Plane+
    row_sides = camera_shift >> unshifted_sides                              # [..., 2] Plane+
    # Multiplying the pairings with a row's two bounding planes turns negative strictly inside the row strip.
    row_slabs = (row_sides[:, 0] * (row_sides[:, 1] & solids.Point) + row_sides[:, 1] * (row_sides[:, 0] & solids.Point)) / 2
    view = solids.View(solids.aim(view_centre, elevation, -0.65), half_extent, n_pixels)
    board = solids.Ink(3, solids.half_space(column_middles), row_slabs, dark_square_colour, square_size / view.pixel, 1.0)
    return solids.draw(placed_solids + [(floor, floor_colour)], view, background, [board], [], [])


def setup_figure(parts: solids.Quadric, mass: solids.Point) -> plt.Figure:
    """Renders an elevated view into the carved cavity (`[4] Plane+ <- Point-`) resting upright on the table."""
    placed_parts = solids.context.include_from_algebra(parts)                # [4] Plane+ <- Point-
    contacts = stack((placed_parts[0], placed_parts[3]))                     # [2] Plane+ <- Point-
    height = (mv.z & contact.lowest_point(contacts, up)).min(axis=0)         # [] Scalar+
    table_motor = (-up * origin.inverse() * height / 2).exp()                # [] Motor+
    centre = (table_motor >> solids.context.include_from_algebra(mass)).normalized()  # [] Point-
    return image_figure(picture(placed_parts, table_motor, centre - origin - up * (mv.z & centre), 0.85))


def frames(parts: solids.Quadric, mass: solids.Point, first_frame: int, n_frames: int,
           states: Iterable[tuple[solids.Motor, solids.Extensor]]) -> Iterator[np.ndarray]:
    """Yields `n_frames` side-by-side RGB frames starting at `first_frame` while consuming `states` in a single pass."""
    separator = np.full((n_pixels, 6, 3), 255, dtype=np.uint8)
    placed_parts = solids.context.include_from_algebra(parts)                # [4] Plane+ <- Point-
    mass_point = solids.context.include_from_algebra(mass)                   # [] Point-
    state_iterator = iter(states)
    window_slice = islice(state_iterator, first_frame - n_smoothing, first_frame + n_frames + n_smoothing)
    window_motors = (solids.context.include_from_algebra(motor) for motor, _ in window_slice)
    copies = tee(window_motors, n_window)
    for window in zip(*(islice(copy, offset, None) for offset, copy in enumerate(copies))):
        window_placements = stack(window)                                    # [n_window, 2] Motor+
        centres = (window_placements >> mass_point).normalized()             # [n_window, 2] Point-
        travel = ((centres - origin - up * (mv.z & centres)) * smoothing_weights[:, None]).sum(axis=0)  # [2] Point-
        frame_motors = window_placements[n_smoothing]                        # [2] Motor+
        yield np.concatenate((
            picture(placed_parts, frame_motors[0], travel[0], 0.24),
            separator,
            picture(placed_parts, frame_motors[1], travel[1], 0.24),
        ), axis=1)
    for _ in state_iterator:
        pass


@charts.style()
def inclination_figure(times: np.ndarray, vertical: solids.Scalar) -> plt.Figure:
    """Plots the stem's vertical component (`[n_states, 2] Scalar+`) across `times` for both spin directions."""
    figure, ax = charts.subplots(450, 180)
    for values, colour in zip(vertical.to_array().T, curve_colours):
        ax.plot(times, values, color=colour)
    ax.axhline(0, color="#99958e", linewidth=charts.FRAME_WIDTH)
    ax.set(xlim=(times[0], times[-1]), ylim=(-1.08, 1.08),
           xlabel="time", ylabel="vertical axis", xticks=(), yticks=(-1, 0, 1))
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure
