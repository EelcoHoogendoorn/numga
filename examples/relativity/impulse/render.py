"""Spacetime diagrams of the impulse examples: drawing only.

Coordinates leave the algebra here, at the plotting boundary. Every diagram is drawn with
x horizontal and ct vertical, in units of the proper length.
"""

from __future__ import annotations

from functools import partial

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import numpy as np

from numga import Extensor, stack

from examples.util import charts
from examples.util.wgsl import contours, planar, simplices
from examples.util.wgsl.camera import Camera, flat_panel, panel, preimage, whole
from examples.relativity.impulse.core import Point, mv

REAR, FRONT, IMPULSE = "#2475ba", "#db7220", "#258e59"
DOOR, ELASTIC, CONTACT = "#59616c", "#a64479", "#ba4242"
BARS, SPAN = "#333333", "#888888"
# The material's worldlines bold, twice a curve's width; the bars measuring a length across a slice four curves wide
# and faint; the contact events' discs, larger than the events' dots, and the half sides of the door closures' and the
# ships' squares, outlined a curve's width. Widths and radii in pixels for the strokes and markers, in points for the
# legends' lines.
WORLDLINE_PX, BAR_PX = 2 * charts.LINE_PX, 4 * charts.LINE_PX
WORLDLINE = charts.points(WORLDLINE_PX)
CONTACT_RADIUS = charts.MARKER_PX
CLOSURE_HALF_SIDE, SHIP_HALF_SIDE = 0.7 * charts.MARKER_PX, 1.15 * charts.MARKER_PX
# Dashes and dots, a dash and a gap in pixels: for a curve, for a bold worldline, and dotted; the radius of the dots at
# the tips of a rope's span.
DASHED, BOLD_DASHED, DOTTED = (5.5, 2.5), (11.0, 5.0), (1.5, 2.5)
SPAN_TIP = 3.0
# The diagrams' strokes and dots over their grids and under their texts and stars.
DRAWING_ZORDER = 2
# A faint white box under a label that sits over the lines, so that it reads; the room a two-line note takes under
# a panel's title, in points.
LABEL_BOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5}
NOTE_PAD = charts.points(3.3 * charts.TEXT_PX)
# The direction along space, which pairs to one with its slice `mv.x`.
space = mv.x.dual()                                                          # [] Point-


def ct_x(events: Point) -> np.ndarray:
    """(..., 2) the ct and x coordinates of events, where texts and stars go."""
    return (stack((mv.t, mv.x)) & events[..., None]).to_array()


def lengths(cuts: Point) -> np.ndarray:
    """The lengths of equal-time cuts across a body `[..., end]`: the step from one end to the other, read by the
    slice x."""
    return (mv.x & (cuts[..., 1] - cuts[..., 0])).to_array()


def middles(cuts: Point) -> np.ndarray:
    """(..., 2) the ct and x coordinates of the events halfway along cuts `[..., end]`, where their labels go."""
    return ct_x((cuts[..., 0] + cuts[..., 1]) / 2)


def header(fig: plt.Figure, lines: list[tuple[str, str]], handles: list[Line2D]) -> None:
    """The figure's title lines, each in its colour, and the legend of `handles` in one row under them, from the top
    in pixels; the panels laid out in the room left below."""
    height = fig.bbox.height
    line_px = 1.4 * charts.TEXT_PX
    for row, (text, colour) in enumerate(lines):
        fig.text(0.5, 1 - (4 + row * line_px) / height, text, ha="center", va="top", color=colour)
    legend_top = 4 + len(lines) * line_px
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - legend_top / height), ncol=len(handles),
               frameon=False)
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - (legend_top + 2 * line_px) / height))


def ink(colour: str) -> np.ndarray:
    """A colour's red, green and blue."""
    return np.array(to_rgb(colour))


def lines_of_ends(lines: Point, colours: tuple[str, ...], width: float, **style) -> partial:
    """Each end's polyline through events `[vertex, end]`, in its colour: strokes for a diagram's camera."""
    n_vertices, n_ends = lines.shape
    # The events end by end, each end's in its order along its worldline.
    by_end = lines[np.arange(n_vertices)[None, :], np.arange(n_ends)[:, None]].reshape(-1)   # [end * vertex] Point-
    return partial(simplices.strokes, by_end, simplices.Topology.chains([n_vertices] * n_ends),
                   np.repeat([ink(colour) for colour in colours], n_vertices - 1, axis=0), width, **style)


def chain(events: Point, colour: str, width: float, **style) -> partial:
    """The polylines through events `[..., event]`, one along the last axis for each of the others: strokes for a
    diagram's camera."""
    n_events = events.shape[-1]
    return partial(simplices.strokes, events.reshape(-1),
                   simplices.Topology.chains([n_events] * (int(np.prod(events.shape)) // n_events)), ink(colour), width,
                   **style)


def dots(events: Point, colour: str) -> partial:
    """A chart's dots at events, in a colour: markers for a diagram's camera."""
    return partial(simplices.markers, events.reshape(-1), ink(colour), charts.MARKER_PX / 2)


def squares(events: Extensor, colours: np.ndarray, half_side: float, camera: Camera) -> np.ndarray:
    """Squares about points `[events]` of any algebra the camera reads, their sides `half_side` pixels from each, four
    strokes a curve's width each, in one colour `[3]` or one each `[events, 3]`: their picture."""
    n_squares, n_corners = events.shape[0], 4
    # The steps the camera reads as one across and one up, whatever the events' own layout, and each square's corners in
    # turn, closed back to the first.
    _, across, up, _ = preimage(camera, whole(events, events.algebra))
    around = across * np.array([-1, 1, 1, -1, -1]) + up * np.array([-1, -1, 1, 1, -1])   # [corners + 1] Point-
    corners = events[:, None] + around * (half_side * camera.pixel)           # [events, corners + 1] Point-
    return simplices.strokes(corners.reshape(-1), simplices.Topology.chains([n_corners + 1] * n_squares),
                             np.repeat(np.broadcast_to(colours, (n_squares, 3)), n_corners, axis=0), charts.LINE_PX,
                             camera)


def length_bar(ends: Point, colour: str, opacity: float) -> partial:
    """A bar across the equal-time events at both ends `[end]`, faint: a stroke for a diagram's camera."""
    return chain(ends, colour, BAR_PX, cap=simplices.Cap.BUTT, opacity=opacity)


def diagram(ax: plt.Axes, layers: list[partial]) -> None:
    """A spacetime diagram's layers, in their order, drawn by the camera of its panel, which reads events by the planes
    x across and t up, over the panel's grid and under its texts; the panel's frame, ticks and labels stay."""
    camera = panel(ax, mv.x, mv.t)
    simplices.overlay(ax, [layer(camera=camera) for layer in layers]).set_zorder(DRAWING_ZORDER)
    ax.set_axis_on()


# --- relativistic impulse -------------------------------------------------------------
@charts.style()
def draw_impulse(
    worldlines: Point, kinks: Point, slices: Point, train: Point, steps: Point, train_slice: Point,
) -> plt.Figure:
    """The symmetric reversal, the same drawing boosted, and a train of ten small impulses."""
    # The whole column: three spacetime diagrams under a title and a legend, over a caption of two long lines.
    fig, axes = charts.subplots(charts.DESIGN_WIDTH, 420, ncols=3, layout="constrained")
    layers = {ax: [] for ax in axes}
    for index, (ax, title) in enumerate(zip(axes[:2], (
        "Symmetric frame: −0.5c → +0.5c", "Boosted frame: 0 → 0.8c",
    ))):
        events = ct_x(kinks[index])
        cut_lengths, cut_middles = lengths(slices[index]), middles(slices[index])
        layers[ax] += [lines_of_ends(worldlines[index], (REAR, FRONT), WORLDLINE_PX),
                       chain(kinks[index], IMPULSE, WORLDLINE_PX, dashes=BOLD_DASHED, cap=simplices.Cap.BUTT),
                       length_bar(slices[index], BARS, 0.35), dots(kinks[index], IMPULSE)]
        for length, (t, middle), prefix in zip(cut_lengths, cut_middles, ("Before", "After")):
            ax.text(middle, t+(-0.14 if t<0 else 0.12),
                    f"{prefix}: {length:.3f} L₀", ha="center", bbox=LABEL_BOX)
        if index == 0:
            ax.text(events[1, 1]/2, 0.12, "Simultaneous reversal", ha="center", color=IMPULSE, bbox=LABEL_BOX)
        else:
            ax.text(0.34, 0.41, "Same line, boosted", color=IMPULSE, bbox=LABEL_BOX)
        # The titles of the row level with the last, which has a note under it.
        ax.set_title(title, loc="left", pad=NOTE_PAD)
        ax.set(xlabel="x / L₀", ylabel="ct / L₀", xlim=(-0.16, 1.93), ylim=(-0.70, 1.53))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 0.5, 1.0, 1.5])
        ax.set_yticks([-0.5, 0, 0.5, 1.0, 1.5])
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)

    # Ten small impulses from repeated composition of the same affine map.
    ax = axes[2]
    end_time = float((mv.t & train[-1, 0]).to_array())
    layers[ax] += [lines_of_ends(train, (REAR, FRONT), WORLDLINE_PX),
                   chain(steps, IMPULSE, charts.LINE_PX, dashes=DASHED, cap=simplices.Cap.BUTT, opacity=0.75),
                   length_bar(train_slice, BARS, 0.35), dots(steps, IMPULSE)]

    sample_time, middle = middles(train_slice)
    ax.text(middle, sample_time+0.12, f"After: {lengths(train_slice):.3f} L₀", ha="center", bbox=LABEL_BOX)
    # The note between the title and the panel, clear of the worldlines.
    ax.text(0, 1.02, "Repeat one fixed map\nEqual ticks on the left clock", transform=ax.transAxes, va="bottom")
    ax.set_title("10 impulses: 0 → 0.8c", loc="left", pad=NOTE_PAD)
    ax.set(xlabel="x / L₀", ylabel="ct / L₀", xlim=(-0.16, 2.7), ylim=(-0.38, end_time+0.30))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)

    legend = [Line2D([], [], color=REAR, lw=WORLDLINE, label="Left end"),
              Line2D([], [], color=FRONT, lw=WORLDLINE, label="Right end"),
              Line2D([], [], color=IMPULSE, ls="--", lw=WORLDLINE, label="Velocity-step surface")]
    header(fig, [("Strain-preserving impulses: start with a symmetric reversal", "black")], legend)
    # The caption under the panels, laid out with them as the figure's label below.
    fig.supxlabel("Velocity changes at kinks; length changes continuously along connected worldlines.\n"
                  "Different observers take different time slices of the same drawing and agree on the material strain.",
                  color="#59616c", linespacing=1.5)
    for ax, drawn in layers.items():
        diagram(ax, drawn)
    return fig


# --- ladder paradox -------------------------------------------------------------------
def style_axis(ax, title, xlim, ylim, primed):
    suffix = "′" if primed else ""
    ax.set(xlabel=f"x{suffix} / L₀", ylabel=f"ct{suffix} / L₀",
           xlim=xlim, ylim=ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, pad=13)


def barn(doors: Point) -> list[partial]:
    """The barn between its door worldlines `[vertex, door]`, faintly filled, and the worldlines dashed: layers for a
    diagram's camera."""
    n_vertices, n_doors = doors.shape
    return [partial(contours.picture, doors.reshape(-1), contours.Topology.grid(n_vertices, n_doors),
                    np.zeros(n_vertices * n_doors), ink(DOOR), opacity=0.07),
            lines_of_ends(doors, (DOOR, DOOR), charts.LINE_PX, dashes=DASHED, cap=simplices.Cap.BUTT, opacity=0.65)]


def contacts(events: Point) -> list[partial]:
    """The contact events `[events]` as discs ringed in white, larger than the events' dots: layers for a diagram's
    camera."""
    return [partial(simplices.markers, events, np.ones(3), CONTACT_RADIUS + charts.FRAME_PX),
            partial(simplices.markers, events, ink(CONTACT), CONTACT_RADIUS)]


def closures(closure: Point) -> list[partial]:
    """The door-closure events as white squares outlined in the doors' ink, over the rest: layers for a diagram's
    camera."""
    return [partial(simplices.markers, closure, np.ones(3), CLOSURE_HALF_SIDE),
            partial(squares, closure, ink(DOOR), CLOSURE_HALF_SIDE)]


def bar_label(ax, ends: Point, label, offset, color) -> None:
    """A length bar's label, above or below the middle of its ends `[end]`."""
    time_at, middle = middles(ends)
    ax.text(middle, time_at + offset, label, ha="center",
            color=color, va="bottom" if offset > 0 else "top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5})


@charts.style()
def draw_ladder(
    barn_frame: tuple[Point, ...], ladder_frame: tuple[Point, ...],
    strain_preserving: tuple[Point, ...], ringing: tuple[Point, ...],
) -> plt.Figure:
    """The ladder in the barn frame and its own frame, a strain-preserving stop, and a stop inside."""
    # The whole column: four annotated spacetime diagrams, two by two, under a title, a subtitle and a legend.
    # The lower diagrams' time runs longest: their row the taller.
    fig, axes = charts.subplots(charts.DESIGN_WIDTH, 900, nrows=2, ncols=2, height_ratios=[1, 1.8], layout="constrained")

    # 1. The ordinary paradox concerns simultaneous door closure, without a stop.
    doors, closure, ladder, at_closure = barn_frame
    ax = axes[0, 0]
    style_axis(ax, "1  ·  In the barn frame: it fits at closure",
               (-0.46, 1.32), (-0.6, 0.65), False)
    # The barn frame's slice through the closures, across the panel.
    slice_ends = stack((ax.get_xlim()[0] * space, ax.get_xlim()[1] * space))   # [2] Point-
    first = (barn(doors) + [lines_of_ends(ladder, (REAR, FRONT), WORLDLINE_PX),
                            chain(slice_ends, DOOR, charts.FRAME_PX, cap=simplices.Cap.BUTT, opacity=0.3),
                            length_bar(at_closure, BARS, 0.28)] + closures(closure))
    bar_label(ax, at_closure, "Moving ladder: 0.60 L₀", 0.10, "#333333")
    ax.annotate("Doors close together", xy=(0.8, 0.0), xytext=(0.43, -0.36),
                ha="center", color=DOOR,
                arrowprops={"arrowstyle": "->", "color": DOOR})
    ax.text(0.23, 0.51, "Barn: 0.80 L₀", ha="center", color=DOOR)

    # 2. Boost the SAME closure events and incoming end worldlines.
    doors, closure, ladder, ladder_cut, barn_cut = ladder_frame
    ax = axes[0, 1]
    second = (barn(doors) + [lines_of_ends(ladder, (REAR, FRONT), WORLDLINE_PX), length_bar(ladder_cut, BARS, 0.28),
                             length_bar(barn_cut, DOOR, 0.28)] + closures(closure))
    bar_label(ax, ladder_cut, "Ladder: 1.00 L₀", 0.08, "#333333")
    bar_label(ax, barn_cut, "Barn: 0.48 L₀", 0.08, DOOR)
    closing = ct_x(closure)
    ax.annotate("Exit closes first", xy=closing[1, ::-1],
                xytext=(1.40, -0.97), color=DOOR,
                arrowprops={"arrowstyle": "->", "color": DOOR})
    ax.annotate("Entrance closes later", xy=closing[0, ::-1],
                xytext=(0.49, 0.13), color=DOOR,
                arrowprops={"arrowstyle": "->", "color": DOOR})
    style_axis(ax, "2  ·  In the incoming ladder frame: different times",
               (-0.31, 2.32), (-1.62, 0.30), True)

    # 3. Prescribed staggered braking: no proper-length mismatch to relax.
    doors, closure, ladder, kinks, at_closure, stopped, contact = strain_preserving
    limits = (-0.48, 2.12)
    ax = axes[1, 0]
    third = (barn(doors) + [lines_of_ends(ladder, (REAR, FRONT), WORLDLINE_PX),
                            chain(kinks, IMPULSE, WORLDLINE_PX, dashes=BOLD_DASHED, cap=simplices.Cap.BUTT),
                            length_bar(at_closure, BARS, 0.28), length_bar(stopped, BARS, 0.28), dots(kinks, IMPULSE)]
             + closures(closure) + contacts(contact[None]))
    ax.text(0.30, 0.57, "Kinks joined by\ntheir bisector", color=IMPULSE, ha="center")
    bar_label(ax, at_closure, "0.60 L₀ at closure", -0.13, "#333333")
    bar_label(ax, stopped, "At rest: 1.00 L₀", 0.11, "#333333")
    contact_time, contact_x = ct_x(contact)
    ax.annotate("Front reaches exit", xy=(contact_x, contact_time),
                xytext=(1.28, 0.85), ha="right", color=CONTACT,
                arrowprops={"arrowstyle": "->", "color": CONTACT})
    ax.text(0.59, 1.76, "Relaxed at rest:\ntoo long for the barn", ha="center")
    style_axis(ax, "3  ·  Strain-preserving stop: no longer fits",
               (-0.44, 1.44), limits, False)

    # 4. Stopping inside leaves compression, whose relaxation drives ringing.
    doors, closure, incoming, ring, relaxed, stop, contacts_at, relaxed_cut = ringing
    ax = axes[1, 1]
    fourth = (barn(doors) + [lines_of_ends(incoming, (REAR, FRONT), WORLDLINE_PX),
                             lines_of_ends(ring, (REAR, FRONT), WORLDLINE_PX),
                             lines_of_ends(relaxed, (REAR, FRONT), charts.LINE_PX, dashes=DOTTED,
                                           cap=simplices.Cap.BUTT, opacity=0.5),
                             chain(stop, ELASTIC, WORLDLINE_PX, dashes=BOLD_DASHED, cap=simplices.Cap.BUTT),
                             length_bar(stop, ELASTIC, 0.28), length_bar(relaxed_cut, BARS, 0.28),
                             dots(stop, ELASTIC)] + closures(closure) + contacts(contacts_at))
    bar_label(ax, stop, "Stopped at 0.60 L₀: compressed", -0.14, ELASTIC)
    touches = ct_x(contacts_at)
    ax.annotate("Expansion reaches both doors", xy=touches[1, ::-1],
                xytext=(0.4, 0.50), ha="center", color=CONTACT,
                arrowprops={"arrowstyle": "->", "color": CONTACT})
    bar_label(ax, relaxed_cut, "Relaxed length: 1.00 L₀", 0.08, "#333333")
    ax.text(0.4, 1.04, "Elastic ringing\nQuarter-critical\ndamping",
            ha="center", color=ELASTIC)
    style_axis(ax, "4  ·  Stop inside: compression and ringing",
               (-0.44, 1.44), limits, False)

    legend = [Line2D([], [], color=REAR, lw=WORLDLINE, label="Rear end"),
              Line2D([], [], color=FRONT, lw=WORLDLINE, label="Front end"),
              Line2D([], [], color=DOOR, ls="--", label="Door position"),
              Line2D([], [], color=DOOR, marker="s", markerfacecolor="white", ls="none", label="Door closes")]
    header(fig, [("The ladder paradox: fitting in motion, stopping in a shorter barn", "black"),
                 ("Relaxed ladder L₀ = 1   ·   Barn = 0.8 L₀   ·   Incoming speed = 0.8c", DOOR)], legend)
    # The caption under the panels, laid out with them as the figure's label below.
    fig.supxlabel("The ladder can fit for an instant; it cannot remain inside at its relaxed rest length.\n"
                  "Door crossings assume yielding doors; ringing illustrates an elastic mode, not a collision calculation.",
                  color=DOOR, linespacing=1.6)
    for ax, drawn in zip(axes.ravel(), (first, second, third, fourth)):
        diagram(ax, drawn)
    return fig


# --- Bell's spaceships ----------------------------------------------------------------
def rope_comparison(ax, gap, y, color, title, label) -> list[partial]:
    """A spatial slice in the final rest frame, aligned on the rear attachment, at a height `y` of its panel: the rope,
    the span of its length and the ships at its ends, white squares outlined in their colours, as layers for the
    panel's camera, with the texts."""
    ax.text(0.0, y + 0.36, title, color=color, weight="medium")
    ax.text(gap / 2, y - 0.29, label, color=color, ha="center", va="top")
    rope = planar.mv.point([[0.0, y], [gap, y]])                               # [2] Point-
    ships = np.stack((ink(REAR), ink(FRONT)))                                   # [ships, 3]
    # The span from its middle out to either end, a dot at each.
    span = planar.mv.point([[gap / 2, y - 0.17], [0.0, y - 0.17], [gap / 2, y - 0.17], [gap, y - 0.17]])   # [4] Point-
    return [partial(simplices.strokes, rope, simplices.Topology.chains([2]), ink(color), WORLDLINE_PX),
            partial(simplices.strokes, span, simplices.Topology.chains([2, 2]), ink(color), charts.LINE_PX,
                    cap=simplices.Cap.BUTT),
            partial(simplices.markers, span[1::2], ink(color), SPAN_TIP),
            partial(simplices.markers, rope, np.ones(3), SHIP_HALF_SIDE),
            partial(squares, rope, ships, SHIP_HALF_SIDE)]


@charts.style()
def draw_spaceships(tracks: Point, schedules: Point, slices: Point, final_events: Point) -> plt.Figure:
    """Both impulse schedules in the lab, and the rope each demands in the final rest frame."""
    # The whole column: two spacetime diagrams and the ropes beside them, under a title, a subtitle and a legend, over
    # a caption.
    fig, axes = charts.subplots(charts.DESIGN_WIDTH, 480, ncols=3, width_ratios=[1.0, 1.0, 1.05], layout="constrained")
    titles = ("Strain-preserving impulse train", "Bell: identical clock programs")
    end_time = float((mv.t & tracks[0, -1, 0]).to_array())
    layers = []
    for index, (color, ax) in enumerate(zip((IMPULSE, ELASTIC), axes[:2])):
        cuts = slices[index]                                                     # [slice, end] Point-
        layers.append([lines_of_ends(tracks[index], (REAR, FRONT), WORLDLINE_PX),
                       chain(schedules[index], color, charts.LINE_PX, dashes=DASHED, cap=simplices.Cap.BUTT,
                             opacity=0.8),
                       length_bar(cuts, BARS, 0.27), dots(schedules[index], color)])
        for length, (time_at, middle), label in zip(lengths(cuts), middles(cuts),
                                                    ("Initially: {:.2f} L₀", "Final lab gap: {:.2f} L₀")):
            ax.text(middle, time_at + (0.09 if time_at > 0 else -0.09),
                    label.format(length), ha="center", va="bottom" if time_at > 0 else "top",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
        # Each note in two lines between its title and its panel.
        note = ("Front impulses follow\nthe bisectors"
                if index == 0 else "Matching clock readings;\nhorizontal timing lines")
        ax.text(0.5, 1.025, note, transform=ax.transAxes, ha="center", va="bottom", color=color)
        ax.set_title(titles[index], pad=NOTE_PAD)
        ax.set(xlabel="x / L₀", ylabel="ct / L₀", xlim=(-0.20, 3.05), ylim=(-0.40, end_time + 0.18))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([0, 1, 2])
        ax.grid(alpha=0.13)
        ax.spines[["top", "right"]].set_visible(False)

    # In the final rest frame both ships stand still, so the step along x from the rear's final kink to the front's is
    # the gap, whether or not the kinks are simultaneous there.
    gaps = lengths(final_events)
    ax = axes[2]
    ax.set_title("In the final shared rest frame", pad=NOTE_PAD)
    # The attachment at the relaxed length, dotted, under both ropes.
    relaxed = planar.mv.point([[1.0, 0.65], [1.0, 2.25]])                      # [2] Point-
    ropes = ([partial(simplices.strokes, relaxed, simplices.Topology.chains([2]), ink(SPAN), charts.LINE_PX,
                      dashes=DOTTED, cap=simplices.Cap.BUTT)]
             + rope_comparison(ax, gaps[0], 2.05, IMPULSE, "Strain-preserving timing", "1.00 L₀ · no extension")
             + rope_comparison(ax, gaps[1], 0.82, ELASTIC, "Synchronized-clock timing",
                               f"{gaps[1]:.2f} L₀ · {100 * (gaps[1] - 1):.1f}% extension"))
    ax.text(0.0, 2.78, "Same final speed: 0.8c\nDifferent required rope lengths", color="#59616c")
    ax.text(0.0, 0.04, "Rope tension grows;\nfailure depends on its strength\nand elastic response.", color=ELASTIC,
            va="top")
    # The slice reaching right as far as its notes do.
    ax.set(xlim=(-0.22, 3.2), ylim=(-0.75, 3.03))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    legend = [Line2D([], [], color=REAR, lw=WORLDLINE, label="Rear ship"),
              Line2D([], [], color=FRONT, lw=WORLDLINE, label="Front ship"),
              Line2D([], [], color="#666666", ls="--", label="Matching impulse events")]
    header(fig, [("Bell’s spaceships: the timing across the rope matters", "black"),
                 ("10 matching impulses   ·   0 → 0.8c   ·   Same rear-ship history in both cases", "#59616c")], legend)
    # The caption under the panels, a sentence a line, laid out with them as the figure's label below.
    fig.supxlabel("The bisector train preserves proper spacing.\n"
                  "Identical synchronized clock programs preserve the initial lab gap instead.\n"
                  "Dashed lines specify timing; they are not signals or sound waves.\n"
                  "Rope extension is kinematic; elastic dynamics are not simulated.",
                  color="#59616c", linespacing=1.6)
    for ax, drawn in zip(axes[:2], layers):
        diagram(ax, drawn)
    camera = flat_panel(axes[2], planar.ga)
    simplices.overlay(axes[2], [layer(camera=camera) for layer in ropes]).set_zorder(DRAWING_ZORDER)
    return fig
