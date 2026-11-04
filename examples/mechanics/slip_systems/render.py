"""The stereographic chart of pull directions, coloured by the slip system that yields first, and the paths a pull's
axis takes through the crystal."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

from numga import stack

from examples.util import charts
from examples.util.animation import capture
from examples.util.wgsl import pinned, room, simplices
from examples.util.wgsl.camera import Camera, panel
from examples.mechanics.slip_systems import core

SYSTEM_MAP = "tab20"
# The Schmid factor at which a region is drawn white, and the one at which it is drawn in full colour.
FAINT, FULL = 0.25, 0.5
PATH_COLOUR = "0.1"
NORMAL_COLOUR = "0.15"
EDGE_COLOUR = "0.3"
# The corners of the standard triangle and their names.
NAMES = ("[001]", "[011]", "[111]")
corners = core.mv.vector([[0, 0, 1], [0, 1, 1], [1, 1, 1]]).normalized().dual()   # [corners] Direction-
# The directions a pull's axis heads for: the leading slip direction, and the corner where two systems share.
goals = core.mv.vector([[1, 0, 1], [1, 1, 2]]).normalized().dual()              # [goals] Direction-
GOAL_NAMES = ("[101]", "[112]")
# How much darker than its systems' colours a path is drawn.
PATH_SHADE = 0.55
# How far each corner's name sits above its corner, in points: [001] below, the others above.
LIFTS = (-10, 8, 8)
EDGE_SAMPLES = 59
RIM_SAMPLES = 180
# The windows of the stereographic disk the whole disk and the close-up show, left to right and bottom to top alike.
DISK = (-1.02, 1.02)
WINDOW = (-0.04, 0.46)
# The animation's side in pixels.
PIXELS = 300
# The marks drawn as squares of four strokes, in pixels from their middles to their corners: the planes' normals,
# squares along the picture's rows and columns, and the directions the axes head for, squares turned an eighth of a
# turn, diamonds.
NORMAL_REACH = 0.75 * charts.MARKER_PX * np.sqrt(2)
GOAL_REACH = charts.MARKER_PX


# --- plumbing -------------------------------------------------------------------------
def scalars(values: core.Scalar) -> np.ndarray:
    """The values of a batch of scalars, for drawing and checks."""
    return values.to_array()


def stereographic(directions: core.Direction) -> core.Direction:
    """Directions as the stereographic disk of the upper half of the sphere draws them: a direction and its opposite
    are the same pull, so each is taken on its upper side; its bisector with [001], read across [001] at unit
    reading along it by `disk_camera`, is the point of the plane across [001] that `lifted` lifts to it."""
    unit = directions.normalized()                                            # [...] Direction-
    upper = unit * np.where((core.mv.z & unit) < 0, -1, 1)                    # [...] Direction-
    return core.mv.z.dual() + upper                                           # [...] Direction-


def on_disk(directions: core.Direction) -> np.ndarray:
    """Where directions lie on the stereographic disk `[..., 2]`, for the names placed at them: their bisectors with
    [001] read across [001] at unit reading along it."""
    bisector = stereographic(directions)                                      # [...] Direction-
    return ((stack((core.mv.x, core.mv.y)) & bisector[..., None]) / (core.mv.z & bisector)[..., None]).to_array()


def squares(directions: core.Direction, reach: float, turn: float, camera: Camera) -> core.Direction:
    """Squares about where directions lie on the stereographic disk, their corners in turn, `[marks, corners]`:
    `reach` pixels from each direction's bisector with [001] at unit reading along it, the first corner turned `turn`
    from x in the plane across [001]."""
    angles = turn + np.pi / 2 * np.arange(4)                                  # [corners]
    corners = (core.mv.xy * (-angles / 2)).exp() >> core.mv.x.dual()         # [corners] Direction-
    bisector = stereographic(directions)                                      # [marks] Direction-
    places = bisector / (core.mv.z & bisector)                                # [marks] Direction-
    return places[:, None] + corners * (reach * camera.pixel)


def disk_camera(ax: plt.Axes) -> Camera:
    """The camera of a panel of the stereographic disk: a direction read across and up by the planes x and y, over
    its reading along [001], a central projection from the origin onto the plane across [001]."""
    return panel(ax, core.mv.x, core.mv.y, core.mv.z)


def lifted(window: tuple[float, float], pixels: int) -> core.Direction:
    """Pull directions on the middles of the pixels of a square window of the stereographic disk, rows down from
    the top `[rows, columns]`: each point of the plane across [001] lifts to the direction that projects onto it,
    [001] reflected in the mirror through [001] plus the point."""
    edges = np.linspace(*window, pixels + 1)
    across = (edges[:-1] + edges[1:]) / 2
    offsets = core.mv.x * across + core.mv.y * across[::-1, None]             # [rows, columns] Vector+
    return (core.mv.z + offsets).normalized() >> core.mv.z.dual()            # [rows, columns] Direction-


def system_colours(count: int) -> np.ndarray:
    """The colour of each slip system, `[systems, 3]`."""
    return plt.get_cmap(SYSTEM_MAP)(np.arange(count) % 20)[:, :3]


def shades(factors: core.Scalar) -> np.ndarray:
    """A colour per pull direction: the hue of the system that yields first, lightened where its Schmid factor
    is small."""
    values = np.abs(scalars(factors)).reshape(factors.shape[:2] + (-1,))      # [rows, columns, systems]
    hue = system_colours(values.shape[-1])[values.argmax(axis=-1)]
    strength = np.clip((values.max(axis=-1) - FAINT) / (FULL - FAINT), 0, 1)[..., None]
    return 1 - strength * (1 - hue)


def picture(ax: plt.Axes, factors: Callable[[core.Direction], core.Scalar], window: tuple[float, float]):
    """The colours of the systems over a square window of the disk, one pull direction per pixel of the room the
    axes get, held pixel for pixel, white beyond the rim; the image."""
    directions = lifted(window, min(room(ax)))                               # [rows, columns] Direction-
    # The reflection lifts a pixel inside the rim to a direction of the lower half, the opposite of its pull, and a
    # pixel beyond the rim to one of the upper half: there the disk ends.
    beyond = ((core.mv.z & directions) > 0)[..., None]
    return pinned(ax, np.where(beyond, 1.0, shades(factors(directions))), (*window, *window))


def triangle(camera: Camera) -> np.ndarray:
    """The standard triangle, between [001], [011] and [111], its edges arcs of great circles."""
    shares = np.linspace(0, 1, EDGE_SAMPLES, endpoint=False)
    # Each edge from a corner towards the next, by shares of the turn between them.
    tails, heads = (simplices.Topology.loops([len(corners)]).incidence[1, 0] * corners.field()).batch()   # [edges] Direction-, Direction-
    turns = ((heads * tails.inverse()).log()[:, None] * (shares / 2)).exp()  # [edges, samples] Rotor+
    arcs = stereographic(turns >> tails[:, None]).reshape(-1)                 # [edges * samples] Direction-
    return simplices.strokes(arcs, simplices.Topology.loops([EDGE_SAMPLES * len(corners)]), to_rgb(EDGE_COLOUR),
                             charts.LINE_PX, camera)


def draw_disk(ax: plt.Axes, factors: Callable[[core.Direction], core.Scalar], planes: core.Plane,
              slips: core.Direction) -> None:
    """The whole disk: each region in the colour of its first system, white outside the rim, the slip planes'
    normals as squares and the slip directions as dots, and the standard triangle."""
    ax.set(xlim=DISK, ylim=DISK)
    picture(ax, factors, DISK)
    camera = disk_camera(ax)
    # The rim, where the disk meets the directions square to [001], turned about it.
    angles = np.linspace(0, 2 * np.pi, RIM_SAMPLES, endpoint=False)
    rim = stereographic((core.mv.xy * (-angles / 2)).exp() >> core.mv.x.dual())   # [samples] Direction-
    # The planes' normals a little larger than the directions, so that the two kinds of mark read apart.
    # A plane's normal, the direction square to it, is its dual.
    normals = squares(planes.dual(), NORMAL_REACH, np.pi / 4, camera)         # [planes, corners] Direction-
    simplices.overlay(ax, [
        simplices.strokes(rim, simplices.Topology.loops([RIM_SAMPLES]), to_rgb(EDGE_COLOUR), charts.LINE_PX, camera),
        simplices.strokes(normals.reshape(-1), simplices.Topology.loops([normals.shape[1]] * normals.shape[0]),
                          to_rgb(NORMAL_COLOUR), charts.LINE_PX, camera),
        simplices.markers(stereographic(slips.reshape(-1)), to_rgb(NORMAL_COLOUR), charts.MARKER_PX / 2, camera),
        triangle(camera)])


def draw_close(ax: plt.Axes, factors: Callable[[core.Direction], core.Scalar]) -> list[np.ndarray]:
    """The standard triangle close up, in the colours of its systems, with the names of its corners and of the
    directions the axes head for; the pictures of the triangle and of the directions the axes head for."""
    ax.set(xlim=WINDOW, ylim=WINDOW)
    picture(ax, factors, WINDOW)
    for corner, name, lift in zip(on_disk(corners), NAMES, LIFTS):
        ax.annotate(name, corner, xytext=(0, lift), textcoords="offset points", ha="center", va="center",
                    color=EDGE_COLOUR)
    for goal, name in zip(on_disk(goals), GOAL_NAMES):
        ax.annotate(name, goal, xytext=(0, 9), textcoords="offset points", ha="center", va="center",
                    color=EDGE_COLOUR)
    camera = disk_camera(ax)
    diamonds = squares(goals, GOAL_REACH, 0.0, camera)                        # [goals, corners] Direction-
    return [triangle(camera), simplices.strokes(diamonds.reshape(-1),
                                                simplices.Topology.loops([diamonds.shape[1]] * diamonds.shape[0]),
                                                to_rgb(EDGE_COLOUR), charts.LINE_PX, camera)]


def paths(camera: Camera, axes: core.Direction, amounts: core.Scalar, count: int) -> list[np.ndarray]:
    """Each pull's axis's path over its first `count` places on the disk, each step in the colours of the systems
    that slipped in it, mixed by how much, darkened; and its place there."""
    weights = np.abs(scalars(amounts[:count - 1])).reshape((count - 1, amounts.shape[1], -1))   # [steps, starts, systems]
    mixed = weights @ system_colours(weights.shape[-1]) / weights.sum(axis=-1, keepdims=True)   # [steps, starts, 3]
    n_starts = axes.shape[1]
    # Each start's places in turn, a chain of them.
    tracks = stereographic(axes[np.arange(count), np.arange(n_starts)[:, None]])   # [starts, steps + 1] Direction-
    return [simplices.strokes(tracks.reshape(-1), simplices.Topology.chains([count] * n_starts),
                              np.moveaxis(mixed, 1, 0).reshape(-1, 3) * PATH_SHADE, charts.LINE_PX, camera),
            simplices.markers(tracks[:, count - 1], to_rgb(PATH_COLOUR), charts.MARKER_PX / 2, camera)]


@charts.style()
def draw_chart(factors: Callable[[core.Direction], core.Scalar], planes: core.Plane, slips: core.Direction,
               axes: core.Direction, amounts: core.Scalar) -> plt.Figure:
    """The whole disk, and the standard triangle with the paths, coloured by each system's Schmid factors for a
    pull direction, `factors`: each pull's axis's path so far and its current place, each step in the colours of the
    systems that slipped in it, mixed by how much, and the directions the axes head for."""
    # Nearly the whole column: two square panels, each an image of fine regions, one with the names of its corners.
    figure, (disk, close) = charts.subplots(760, 380, ncols=2, layout="constrained")
    # Both panels without axes before either takes its room.
    disk.set_axis_off()
    close.set_axis_off()
    draw_disk(disk, factors, planes, slips)
    triangle_and_goals = draw_close(close, factors)
    simplices.overlay(close, triangle_and_goals[:1] + paths(disk_camera(close), axes, amounts, axes.shape[0])
                      + triangle_and_goals[1:])
    return figure


def animate_paths(factors: Callable[[core.Direction], core.Scalar], axes: core.Direction, amounts: core.Scalar,
                  frames: int) -> Iterator[np.ndarray]:
    """The paths through the standard triangle growing, over so many frames, the triangle drawn once."""
    return animate_paths_sized(factors, axes, amounts, frames, PIXELS)


def animate_paths_sized(factors: Callable[[core.Direction], core.Scalar], axes: core.Direction,
                        amounts: core.Scalar, frames: int, pixels: int) -> Iterator[np.ndarray]:
    """The paths through the standard triangle growing, over so many frames, the triangle drawn once, the figure
    so many pixels across."""
    reached = np.linspace(2, axes.shape[0], frames).round().astype(int)
    # Laid out once under the charts' style; only the paths and the places move between frames.
    with charts.style():
        figure, ax = charts.subplots(pixels, pixels, layout="constrained")
        ax.set_axis_off()
        triangle_and_goals = draw_close(ax, factors)
        figure.canvas.draw()
        figure.set_layout_engine("none")
    camera = disk_camera(ax)
    image = simplices.overlay(ax, triangle_and_goals)
    for count in reached:
        image.set_data(simplices.composited(triangle_and_goals[:1] + paths(camera, axes, amounts, count)
                                            + triangle_and_goals[1:]))
        yield capture(figure)
    plt.close(figure)
