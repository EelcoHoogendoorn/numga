"""Spacetime diagrams of the impulse examples: drawing only.

Coordinates leave the algebra here, at the plotting boundary. Every diagram is drawn with
x horizontal and ct vertical, in units of the proper length.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from numga import stack

from examples.util import charts
from examples.relativity.impulse.core import Point, mv

REAR, FRONT, IMPULSE = "#2475ba", "#db7220", "#258e59"
DOOR, ELASTIC, CONTACT = "#59616c", "#a64479", "#ba4242"
# The material's worldlines bold, twice a curve's width; the bars measuring a length across a slice four curves wide
# and faint; the contact events' stars, and the ships, larger than the events' dots.
WORLDLINE = charts.points(2 * charts.LINE_PX)
BAR = charts.points(4 * charts.LINE_PX)
STAR = charts.points(2 * charts.MARKER_PX) ** 2
SHIP = charts.points(3 * charts.MARKER_PX) ** 2
# A faint white box under a label that sits over the lines, so that it reads; the room a two-line note takes under
# a panel's title, in points.
LABEL_BOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5}
NOTE_PAD = charts.points(3.3 * charts.TEXT_PX)


def ct_x(events: Point) -> np.ndarray:
    """(..., 2) the ct and x coordinates of events."""
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


def plot_worldlines(ax, lines: np.ndarray, colors: tuple[str, str], **style) -> None:
    """Draw each end's polyline from (vertex, end, 2) ct, x coordinates."""
    for end, color in enumerate(colors):
        ax.plot(lines[:, end, 1], lines[:, end, 0], color=color, **style)


# --- relativistic impulse -------------------------------------------------------------
@charts.style()
def draw_impulse(
    worldlines: Point, kinks: Point, slices: Point, train: Point, steps: Point, train_slice: Point,
) -> plt.Figure:
    """The symmetric reversal, the same drawing boosted, and a train of ten small impulses."""
    # The whole column: three spacetime diagrams under a title and a legend, over a caption of two long lines.
    fig, axes = charts.subplots(charts.DESIGN_WIDTH, 420, ncols=3, layout="constrained")
    for index, (ax, title) in enumerate(zip(axes[:2], (
        "Symmetric frame: −0.5c → +0.5c", "Boosted frame: 0 → 0.8c",
    ))):
        lines, events, cuts = ct_x(worldlines[index]), ct_x(kinks[index]), ct_x(slices[index])
        cut_lengths, cut_middles = lengths(slices[index]), middles(slices[index])
        for end, label in enumerate(("Left end", "Right end")):
            ax.plot(lines[:, end, 1], lines[:, end, 0], color=(REAR, FRONT)[end], lw=WORLDLINE, label=label)
        ax.plot(events[:, 1], events[:, 0], "--", color=IMPULSE, lw=WORLDLINE, label="Velocity-step surface")
        ax.scatter(events[:, 1], events[:, 0], color=IMPULSE, zorder=4)

        for cut, length, (t, middle), prefix in zip(cuts, cut_lengths, cut_middles, ("Before", "After")):
            ax.plot(cut[:, 1], cut[:, 0], color="#333333", lw=BAR, alpha=0.35)
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
    lines, events, cut = ct_x(train), ct_x(steps), ct_x(train_slice)
    end_time = lines[-1, 0, 0]
    plot_worldlines(ax, lines, (REAR, FRONT), lw=WORLDLINE)
    for step in events:
        ax.plot(step[:, 1], step[:, 0], "--", color=IMPULSE, alpha=0.75)
        ax.scatter(step[:, 1], step[:, 0], color=IMPULSE, zorder=4)

    sample_time, middle = middles(train_slice)
    ax.plot(cut[:, 1], cut[:, 0], color="#333333", lw=BAR, alpha=0.35)
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

    header(fig, [("Strain-preserving impulses: start with a symmetric reversal", "black")],
           axes[0].get_legend_handles_labels()[0])
    # The caption under the panels, laid out with them as the figure's label below.
    fig.supxlabel("Velocity changes at kinks; length changes continuously along connected worldlines.\n"
                  "Different observers take different time slices of the same drawing and agree on the material strain.",
                  color="#59616c", linespacing=1.5)
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


def barn_background(ax, doors: Point, closure: Point) -> None:
    """The barn between its door worldlines (time, door), and the door-closure events."""
    lines, events = ct_x(doors), ct_x(closure)
    ax.fill_betweenx(lines[:, 0, 0], lines[:, 0, 1], lines[:, 1, 1], color=DOOR, alpha=0.07)
    ax.plot(lines[..., 1], lines[..., 0], color=DOOR, ls="--", alpha=0.65)
    ax.scatter(events[:, 1], events[:, 0], marker="s", color=DOOR, edgecolor="white", linewidth=charts.FRAME_WIDTH,
               zorder=6)


def length_bar(ax, ends: Point, label, offset, color):
    """A bar across the equal-time events at both ends, labelled above or below."""
    events = ct_x(ends)
    time, middle = middles(ends)
    ax.plot(events[:, 1], events[:, 0], lw=BAR, color=color, alpha=0.28,
            solid_capstyle="butt")
    ax.text(middle, time + offset, label, ha="center",
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
    barn_background(ax, doors, closure)
    plot_worldlines(ax, ct_x(ladder), (REAR, FRONT), lw=WORLDLINE)
    ax.axhline(0.0, color=DOOR, lw=charts.FRAME_WIDTH, alpha=0.3)
    length_bar(ax, at_closure, "Moving ladder: 0.60 L₀", 0.10, "#333333")
    ax.annotate("Doors close together", xy=(0.8, 0.0), xytext=(0.43, -0.36),
                ha="center", color=DOOR,
                arrowprops={"arrowstyle": "->", "color": DOOR})
    ax.text(0.23, 0.51, "Barn: 0.80 L₀", ha="center", color=DOOR)
    style_axis(ax, "1  ·  In the barn frame: it fits at closure",
               (-0.46, 1.32), (-0.6, 0.65), False)

    # 2. Boost the SAME closure events and incoming end worldlines.
    doors, closure, ladder, ladder_cut, barn_cut = ladder_frame
    ax = axes[0, 1]
    barn_background(ax, doors, closure)
    plot_worldlines(ax, ct_x(ladder), (REAR, FRONT), lw=WORLDLINE)
    length_bar(ax, ladder_cut, "Ladder: 1.00 L₀", 0.08, "#333333")
    length_bar(ax, barn_cut, "Barn: 0.48 L₀", 0.08, DOOR)
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
    barn_background(ax, doors, closure)
    plot_worldlines(ax, ct_x(ladder), (REAR, FRONT), lw=WORLDLINE)
    events = ct_x(kinks)
    ax.plot(events[:, 1], events[:, 0], "--", color=IMPULSE, lw=WORLDLINE)
    ax.scatter(events[:, 1], events[:, 0], color=IMPULSE, zorder=5)
    ax.text(0.30, 0.57, "Kinks joined by\ntheir bisector", color=IMPULSE, ha="center")
    length_bar(ax, at_closure, "0.60 L₀ at closure", -0.13, "#333333")
    length_bar(ax, stopped, "At rest: 1.00 L₀", 0.11, "#333333")
    contact_time, contact_x = ct_x(contact)
    ax.scatter([contact_x], [contact_time], marker="*", color=CONTACT, s=STAR, zorder=7)
    ax.annotate("Front reaches exit", xy=(contact_x, contact_time),
                xytext=(1.28, 0.85), ha="right", color=CONTACT,
                arrowprops={"arrowstyle": "->", "color": CONTACT})
    ax.text(0.59, 1.76, "Relaxed at rest:\ntoo long for the barn", ha="center")
    style_axis(ax, "3  ·  Strain-preserving stop: no longer fits",
               (-0.44, 1.44), limits, False)

    # 4. Stopping inside leaves compression, whose relaxation drives ringing.
    doors, closure, incoming, ring, relaxed, stop, contacts, relaxed_cut = ringing
    ax = axes[1, 1]
    barn_background(ax, doors, closure)
    plot_worldlines(ax, ct_x(incoming), (REAR, FRONT), lw=WORLDLINE)
    plot_worldlines(ax, ct_x(ring), (REAR, FRONT), lw=WORLDLINE)
    plot_worldlines(ax, ct_x(relaxed), (REAR, FRONT), ls=":", alpha=0.5)
    events = ct_x(stop)
    ax.plot(events[:, 1], events[:, 0], "--", color=ELASTIC, lw=WORLDLINE)
    ax.scatter(events[:, 1], events[:, 0], color=ELASTIC, zorder=5)
    length_bar(ax, stop, "Stopped at 0.60 L₀: compressed", -0.14, ELASTIC)
    touches = ct_x(contacts)
    ax.scatter(touches[:, 1], touches[:, 0], marker="*", color=CONTACT, s=STAR, zorder=7)
    ax.annotate("Expansion reaches both doors", xy=touches[1, ::-1],
                xytext=(0.4, 0.50), ha="center", color=CONTACT,
                arrowprops={"arrowstyle": "->", "color": CONTACT})
    length_bar(ax, relaxed_cut, "Relaxed length: 1.00 L₀", 0.08, "#333333")
    ax.text(0.4, 1.04, "Elastic ringing\nQuarter-critical\ndamping",
            ha="center", color=ELASTIC)
    style_axis(ax, "4  ·  Stop inside: compression and ringing",
               (-0.44, 1.44), limits, False)

    legend = [Line2D([], [], color=REAR, lw=WORLDLINE, label="Rear end"),
              Line2D([], [], color=FRONT, lw=WORLDLINE, label="Front end"),
              Line2D([], [], color=DOOR, ls="--", label="Door position"),
              Line2D([], [], color=DOOR, marker="s", ls="none", label="Door closes")]
    header(fig, [("The ladder paradox: fitting in motion, stopping in a shorter barn", "black"),
                 ("Relaxed ladder L₀ = 1   ·   Barn = 0.8 L₀   ·   Incoming speed = 0.8c", DOOR)], legend)
    # The caption under the panels, laid out with them as the figure's label below.
    fig.supxlabel("The ladder can fit for an instant; it cannot remain inside at its relaxed rest length.\n"
                  "Door crossings assume yielding doors; ringing illustrates an elastic mode, not a collision calculation.",
                  color=DOOR, linespacing=1.6)
    return fig


# --- Bell's spaceships ----------------------------------------------------------------
def rope_comparison(ax, gap, y, color, title, label):
    """A spatial slice in the final rest frame, aligned on the rear attachment."""
    ax.text(0.0, y + 0.36, title, color=color, weight="medium")
    ax.plot([0.0, gap], [y, y], color=color, lw=WORLDLINE)
    for x, ship_color in ((0.0, REAR), (gap, FRONT)):
        ax.scatter([x], [y], marker=">", s=SHIP, facecolor="white", edgecolor=ship_color, zorder=5)
    ax.annotate("", xy=(gap, y - 0.17), xytext=(0.0, y - 0.17),
                arrowprops={"arrowstyle": "<->", "color": color})
    ax.text(gap / 2, y - 0.29, label, color=color, ha="center", va="top")


@charts.style()
def draw_spaceships(tracks: Point, schedules: Point, slices: Point, final_events: Point) -> plt.Figure:
    """Both impulse schedules in the lab, and the rope each demands in the final rest frame."""
    # The whole column: two spacetime diagrams and the ropes beside them, under a title, a subtitle and a legend, over
    # a caption.
    fig, axes = charts.subplots(charts.DESIGN_WIDTH, 480, ncols=3, width_ratios=[1.0, 1.0, 1.05], layout="constrained")
    titles = ("Strain-preserving impulse train", "Bell: identical clock programs")
    end_time = ct_x(tracks)[0, -1, 0, 0]
    for index, (color, ax) in enumerate(zip((IMPULSE, ELASTIC), axes[:2])):
        plot_worldlines(ax, ct_x(tracks[index]), (REAR, FRONT), lw=WORLDLINE)
        for step in ct_x(schedules[index]):
            ax.plot(step[:, 1], step[:, 0], "--", color=color, alpha=0.8)
            ax.scatter(step[:, 1], step[:, 0], color=color, zorder=4)

        cuts = slices[index]                                                     # [slice, end] Point-
        for cut, length, (time, middle), label in zip(ct_x(cuts), lengths(cuts), middles(cuts),
                                                      ("Initially: {:.2f} L₀", "Final lab gap: {:.2f} L₀")):
            ax.plot(cut[:, 1], cut[:, 0], color="#333333", lw=BAR, alpha=0.27)
            ax.text(middle, time + (0.09 if time > 0 else -0.09),
                    label.format(length), ha="center", va="bottom" if time > 0 else "top",
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
    rope_comparison(ax, gaps[0], 2.05, IMPULSE, "Strain-preserving timing",
                    "1.00 L₀ · no extension")
    rope_comparison(ax, gaps[1], 0.82, ELASTIC, "Synchronized-clock timing",
                    f"{gaps[1]:.2f} L₀ · {100 * (gaps[1] - 1):.1f}% extension")
    ax.plot([1.0, 1.0], [0.65, 2.25], color="#888888", ls=":")
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
    return fig
