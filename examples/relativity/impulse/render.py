"""Spacetime diagrams render endpoint worldlines, impulse schedules, and equal-time length cuts."""

from __future__ import annotations

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from examples.relativity.impulse.core import Point, mv, space
from examples.util import charts
from examples.util.panels import offset_panel, overlay, panel
from numga import Extensor, stack
from numga.render import Camera, contours, planar, preimage, simplices, whole
from numga.topology import Topology

rear_colour, front_colour, impulse_colour = "#2475ba", "#db7220", "#258e59"
door_colour, elastic_colour, contact_colour = "#59616c", "#a64479", "#ba4242"
bar_colour, span_colour = "#333333", "#888888"
worldline_px, bar_px = 2 * charts.LINE_PX, 4 * charts.LINE_PX
worldline_pt = charts.points(worldline_px)
contact_radius = charts.MARKER_PX
closure_half_side, ship_half_side = 0.7 * charts.MARKER_PX, 1.15 * charts.MARKER_PX
dashed_pattern, bold_dashed_pattern, dotted_pattern = (5.5, 2.5), (11.0, 5.0), (1.5, 2.5)
span_tip_px = 3.0
drawing_zorder = 2
label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5}
note_pad = charts.points(3.3 * charts.TEXT_PX)


def ct_x_coordinates(events: Point) -> np.ndarray:
    """Stacked time and space planes read out `(ct, x)` coordinates (`[..., 2]`) for text placement."""
    return (stack((mv.t, mv.x)) & events[..., None]).to_array()


def cut_lengths(cuts: Point) -> np.ndarray:
    """Pairing equal-time endpoint separations with the spatial plane gives measured lengths."""
    return (mv.x & (cuts[..., 1] - cuts[..., 0])).to_array()


def cut_midpoints(cuts: Point) -> np.ndarray:
    """Midpoints of endpoint pairs (`[..., 2] Point-`) locate length-bar labels in `(ct, x)` coordinates."""
    return ct_x_coordinates((cuts[..., 0] + cuts[..., 1]) / 2)


def figure_header(fig: plt.Figure, lines: list[tuple[str, str]], handles: list[Line2D]) -> None:
    """Title lines and a single-row legend reserve top margin above the figure panels."""
    height = fig.bbox.height
    line_px = 1.4 * charts.TEXT_PX
    for row, (text, colour) in enumerate(lines):
        fig.text(0.5, 1 - (4 + row * line_px) / height, text, ha="center", va="top", color=colour)
    legend_top = 4 + len(lines) * line_px
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - legend_top / height),
        ncol=len(handles), frameon=False,
    )
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - (legend_top + 2 * line_px) / height))


def ink_rgb(colour: str) -> np.ndarray:
    """Hex colour strings convert to RGB float arrays."""
    return np.array(to_rgb(colour))


def endpoint_polylines(lines: Point, colours: tuple[str, ...], width: float, **style) -> partial:
    """Endpoint event arrays (`[n_vertices, n_ends] Point-`) assemble into coloured polyline strokes."""
    n_vertices, n_ends = lines.shape
    by_end = lines[np.arange(n_vertices)[None, :], np.arange(n_ends)[:, None]].reshape(-1)  # [n_ends * n_vertices] Point-
    return partial(
        simplices.strokes, by_end, Topology.chains([n_vertices] * n_ends),
        np.repeat([ink_rgb(colour) for colour in colours], n_vertices - 1, axis=0), width, **style,
    )


def event_chain(events: Point, colour: str, width: float, **style) -> partial:
    """Contiguous event sequences along the trailing axis (`[...] Point-`) form stroke chains."""
    n_events = events.shape[-1]
    return partial(
        simplices.strokes, events.reshape(-1),
        Topology.chains([n_events] * (int(np.prod(events.shape)) // n_events)), ink_rgb(colour), width, **style,
    )


def event_dots(events: Point, colour: str) -> partial:
    """Event locations (`[...] Point-`) render as circular markers."""
    return partial(simplices.markers, events.reshape(-1), ink_rgb(colour), charts.MARKER_PX / 2)


def square_markers(events: Extensor, colours: np.ndarray, half_side: float, camera: Camera) -> np.ndarray:
    """Pixel-aligned square outlines frame each event (`[n_squares] Point-`) in the camera image."""
    n_squares, n_corners = events.shape[0], 4
    _, step_x, step_y, _ = preimage(camera, whole(events, events.algebra))
    corner_offsets = step_x * np.array([-1, 1, 1, -1]) + step_y * np.array([-1, -1, 1, 1])  # [4] Point-
    corners = events[:, None] + corner_offsets * (half_side * camera.pixel)     # [n_squares, 4] Point-
    return simplices.strokes(
        corners.reshape(-1), Topology.loops([n_corners] * n_squares),
        np.repeat(np.broadcast_to(colours, (n_squares, 3)), n_corners, axis=0), charts.LINE_PX, camera,
    )


def length_bar(ends: Point, colour: str, opacity: float) -> partial:
    """Equal-time endpoint pairs (`[..., 2] Point-`) render as translucent horizontal length bars."""
    return event_chain(ends, colour, bar_px, cap=simplices.Cap.BUTT, opacity=opacity)


def spacetime_diagram(ax: plt.Axes, layers: list[partial]) -> None:
    """Composited layer callables render into `ax` through an `(x, ct)` spacetime camera."""
    camera = offset_panel(ax, mv.x, mv.t)
    overlay(ax, [layer(camera=camera) for layer in layers]).set_zorder(drawing_zorder)
    ax.set_axis_on()


@charts.style()
def impulse_figure(
    worldlines: Point, kinks: Point, slices: Point, train: Point, steps: Point, train_slice: Point,
) -> plt.Figure:
    """Three spacetime diagrams compare a symmetric reversal, its boosted view, and a ten-step impulse train."""
    fig, axes = charts.subplots(charts.DESIGN_WIDTH, 420, ncols=3, layout="constrained")
    layers = {ax: [] for ax in axes}
    for index, (ax, title) in enumerate(zip(axes[:2], (
        "Symmetric frame: −0.5c → +0.5c", "Boosted frame: 0 → 0.8c",
    ))):
        events = ct_x_coordinates(kinks[index])
        lengths_at_cuts, middles = cut_lengths(slices[index]), cut_midpoints(slices[index])
        layers[ax] += [
            endpoint_polylines(worldlines[index], (rear_colour, front_colour), worldline_px),
            event_chain(kinks[index], impulse_colour, worldline_px, dashes=bold_dashed_pattern, cap=simplices.Cap.BUTT),
            length_bar(slices[index], bar_colour, 0.35),
            event_dots(kinks[index], impulse_colour),
        ]
        for length, (time_at, middle), prefix in zip(lengths_at_cuts, middles, ("Before", "After")):
            ax.text(middle, time_at + (-0.14 if time_at < 0 else 0.12), f"{prefix}: {length:.3f} L₀", ha="center", bbox=label_box)
        if index == 0:
            ax.text(events[1, 1] / 2, 0.12, "Simultaneous reversal", ha="center", color=impulse_colour, bbox=label_box)
        else:
            ax.text(0.34, 0.41, "Same line, boosted", color=impulse_colour, bbox=label_box)
        ax.set_title(title, loc="left", pad=note_pad)
        ax.set(xlabel="x / L₀", ylabel="ct / L₀", xlim=(-0.16, 1.93), ylim=(-0.70, 1.53))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 0.5, 1.0, 1.5])
        ax.set_yticks([-0.5, 0, 0.5, 1.0, 1.5])
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    end_time = float((mv.t & train[-1, 0]).to_array())
    layers[ax] += [
        endpoint_polylines(train, (rear_colour, front_colour), worldline_px),
        event_chain(steps, impulse_colour, charts.LINE_PX, dashes=dashed_pattern, cap=simplices.Cap.BUTT, opacity=0.75),
        length_bar(train_slice, bar_colour, 0.35),
        event_dots(steps, impulse_colour),
    ]

    sample_time, middle = cut_midpoints(train_slice)
    ax.text(middle, sample_time + 0.12, f"After: {cut_lengths(train_slice):.3f} L₀", ha="center", bbox=label_box)
    ax.text(0, 1.02, "Repeat one fixed map\nEqual ticks on the left clock", transform=ax.transAxes, va="bottom")
    ax.set_title("10 impulses: 0 → 0.8c", loc="left", pad=note_pad)
    ax.set(xlabel="x / L₀", ylabel="ct / L₀", xlim=(-0.16, 2.7), ylim=(-0.38, end_time + 0.30))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)

    legend = [
        Line2D([], [], color=rear_colour, lw=worldline_pt, label="Left end"),
        Line2D([], [], color=front_colour, lw=worldline_pt, label="Right end"),
        Line2D([], [], color=impulse_colour, ls="--", lw=worldline_pt, label="Velocity-step surface"),
    ]
    figure_header(fig, [("Strain-preserving impulses: start with a symmetric reversal", "black")], legend)
    fig.supxlabel(
        "Velocity changes at kinks; length changes continuously along connected worldlines.\n"
        "Different observers take different time slices of the same drawing and agree on the material strain.",
        color="#59616c", linespacing=1.5,
    )
    for ax, panel_layers in layers.items():
        spacetime_diagram(ax, panel_layers)
    return fig


def panel_axes(ax: plt.Axes, title: str, xlim: tuple[float, float], ylim: tuple[float, float], primed: bool) -> None:
    """Axis labels, equal aspect, faint grid, and title configure one ladder-paradox panel."""
    suffix = "′" if primed else ""
    ax.set(xlabel=f"x{suffix} / L₀", ylabel=f"ct{suffix} / L₀", xlim=xlim, ylim=ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, pad=13)


def barn_layers(doors: Point) -> list[partial]:
    """Shaded interior strip and dashed door worldlines (`[n_vertices, n_doors] Point-`) mark the barn."""
    n_vertices, n_doors = doors.shape
    return [
        partial(
            contours.picture, doors.reshape(-1), Topology.grid(n_vertices, n_doors),
            np.zeros(n_vertices * n_doors), ink_rgb(door_colour), opacity=0.07,
        ),
        endpoint_polylines(doors, (door_colour, door_colour), charts.LINE_PX, dashes=dashed_pattern, cap=simplices.Cap.BUTT, opacity=0.65),
    ]


def contact_layers(events: Point) -> list[partial]:
    """White-ringed red discs mark door-contact events (`[...] Point-`)."""
    return [
        partial(simplices.markers, events, np.ones(3), contact_radius + charts.FRAME_PX),
        partial(simplices.markers, events, ink_rgb(contact_colour), contact_radius),
    ]


def closure_layers(closure: Point) -> list[partial]:
    """White-filled outlined squares mark door-closure events (`[2] Point-`)."""
    return [
        partial(simplices.markers, closure, np.ones(3), closure_half_side),
        partial(square_markers, closure, ink_rgb(door_colour), closure_half_side),
    ]


def bar_label(ax: plt.Axes, ends: Point, label: str, offset: float, color: str) -> None:
    """Centered text above or below a length bar (`[2] Point-`) annotates its measured span."""
    time_at, middle = cut_midpoints(ends)
    ax.text(
        middle, time_at + offset, label, ha="center", color=color,
        va="bottom" if offset > 0 else "top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )


@charts.style()
def ladder_figure(
    barn_frame: tuple[Point, ...], ladder_frame: tuple[Point, ...],
    strain_preserving: tuple[Point, ...], ringing: tuple[Point, ...],
) -> plt.Figure:
    """Four spacetime diagrams contrast simultaneous door closure with strain-preserving and compressive stops."""
    fig, axes = charts.subplots(charts.DESIGN_WIDTH, 900, nrows=2, ncols=2, height_ratios=[1, 1.8], layout="constrained")

    doors, closure, ladder, at_closure = barn_frame
    ax = axes[0, 0]
    panel_axes(ax, "1  ·  In the barn frame: it fits at closure", (-0.46, 1.32), (-0.6, 0.65), False)
    slice_ends = stack((ax.get_xlim()[0] * space, ax.get_xlim()[1] * space))    # [2] Point-
    first_layers = (
        barn_layers(doors)
        + [
            endpoint_polylines(ladder, (rear_colour, front_colour), worldline_px),
            event_chain(slice_ends, door_colour, charts.FRAME_PX, cap=simplices.Cap.BUTT, opacity=0.3),
            length_bar(at_closure, bar_colour, 0.28),
        ]
        + closure_layers(closure)
    )
    bar_label(ax, at_closure, "Moving ladder: 0.60 L₀", 0.10, "#333333")
    ax.annotate(
        "Doors close together", xy=(0.8, 0.0), xytext=(0.43, -0.36), ha="center",
        color=door_colour, arrowprops={"arrowstyle": "->", "color": door_colour},
    )
    ax.text(0.23, 0.51, "Barn: 0.80 L₀", ha="center", color=door_colour)

    doors, closure, ladder, ladder_cut, barn_cut = ladder_frame
    ax = axes[0, 1]
    second_layers = (
        barn_layers(doors)
        + [
            endpoint_polylines(ladder, (rear_colour, front_colour), worldline_px),
            length_bar(ladder_cut, bar_colour, 0.28),
            length_bar(barn_cut, door_colour, 0.28),
        ]
        + closure_layers(closure)
    )
    bar_label(ax, ladder_cut, "Ladder: 1.00 L₀", 0.08, "#333333")
    bar_label(ax, barn_cut, "Barn: 0.48 L₀", 0.08, door_colour)
    closing = ct_x_coordinates(closure)
    ax.annotate("Exit closes first", xy=closing[1, ::-1], xytext=(1.40, -0.97), color=door_colour, arrowprops={"arrowstyle": "->", "color": door_colour})
    ax.annotate("Entrance closes later", xy=closing[0, ::-1], xytext=(0.49, 0.13), color=door_colour, arrowprops={"arrowstyle": "->", "color": door_colour})
    panel_axes(ax, "2  ·  In the incoming ladder frame: different times", (-0.31, 2.32), (-1.62, 0.30), True)

    doors, closure, ladder, kinks, at_closure, stopped, contact = strain_preserving
    limits = (-0.48, 2.12)
    ax = axes[1, 0]
    third_layers = (
        barn_layers(doors)
        + [
            endpoint_polylines(ladder, (rear_colour, front_colour), worldline_px),
            event_chain(kinks, impulse_colour, worldline_px, dashes=bold_dashed_pattern, cap=simplices.Cap.BUTT),
            length_bar(at_closure, bar_colour, 0.28),
            length_bar(stopped, bar_colour, 0.28),
            event_dots(kinks, impulse_colour),
        ]
        + closure_layers(closure)
        + contact_layers(contact[None])
    )
    ax.text(0.30, 0.57, "Kinks joined by\ntheir bisector", color=impulse_colour, ha="center")
    bar_label(ax, at_closure, "0.60 L₀ at closure", -0.13, "#333333")
    bar_label(ax, stopped, "At rest: 1.00 L₀", 0.11, "#333333")
    contact_time, contact_x = ct_x_coordinates(contact)
    ax.annotate(
        "Front reaches exit", xy=(contact_x, contact_time), xytext=(1.28, 0.85), ha="right",
        color=contact_colour, arrowprops={"arrowstyle": "->", "color": contact_colour},
    )
    ax.text(0.59, 1.76, "Relaxed at rest:\ntoo long for the barn", ha="center")
    panel_axes(ax, "3  ·  Strain-preserving stop: no longer fits", (-0.44, 1.44), limits, False)

    doors, closure, incoming, ring, relaxed, stop, contacts_at, relaxed_cut = ringing
    ax = axes[1, 1]
    fourth_layers = (
        barn_layers(doors)
        + [
            endpoint_polylines(incoming, (rear_colour, front_colour), worldline_px),
            endpoint_polylines(ring, (rear_colour, front_colour), worldline_px),
            endpoint_polylines(relaxed, (rear_colour, front_colour), charts.LINE_PX, dashes=dotted_pattern, cap=simplices.Cap.BUTT, opacity=0.5),
            event_chain(stop, elastic_colour, worldline_px, dashes=bold_dashed_pattern, cap=simplices.Cap.BUTT),
            length_bar(stop, elastic_colour, 0.28),
            length_bar(relaxed_cut, bar_colour, 0.28),
            event_dots(stop, elastic_colour),
        ]
        + closure_layers(closure)
        + contact_layers(contacts_at)
    )
    bar_label(ax, stop, "Stopped at 0.60 L₀: compressed", -0.14, elastic_colour)
    touches = ct_x_coordinates(contacts_at)
    ax.annotate(
        "Expansion reaches both doors", xy=touches[1, ::-1], xytext=(0.4, 0.50), ha="center",
        color=contact_colour, arrowprops={"arrowstyle": "->", "color": contact_colour},
    )
    bar_label(ax, relaxed_cut, "Relaxed length: 1.00 L₀", 0.08, "#333333")
    ax.text(0.4, 1.04, "Elastic ringing\nQuarter-critical\ndamping", ha="center", color=elastic_colour)
    panel_axes(ax, "4  ·  Stop inside: compression and ringing", (-0.44, 1.44), limits, False)

    legend = [
        Line2D([], [], color=rear_colour, lw=worldline_pt, label="Rear end"),
        Line2D([], [], color=front_colour, lw=worldline_pt, label="Front end"),
        Line2D([], [], color=door_colour, ls="--", label="Door position"),
        Line2D([], [], color=door_colour, marker="s", markerfacecolor="white", ls="none", label="Door closes"),
    ]
    figure_header(
        fig,
        [
            ("The ladder paradox: fitting in motion, stopping in a shorter barn", "black"),
            ("Relaxed ladder L₀ = 1   ·   Barn = 0.8 L₀   ·   Incoming speed = 0.8c", door_colour),
        ],
        legend,
    )
    fig.supxlabel(
        "The ladder can fit for an instant; it cannot remain inside at its relaxed rest length.\n"
        "Door crossings assume yielding doors; ringing illustrates an elastic mode, not a collision calculation.",
        color=door_colour, linespacing=1.6,
    )
    for ax, panel_layers in zip(axes.ravel(), (first_layers, second_layers, third_layers, fourth_layers)):
        spacetime_diagram(ax, panel_layers)
    return fig


def rope_comparison(ax: plt.Axes, gap: float, y: float, color: str, title: str, label: str) -> list[partial]:
    """Planar stroke and square-marker layers depict one final-rest-frame ship separation `gap` at height `y`."""
    ax.text(0.0, y + 0.36, title, color=color, weight="medium")
    ax.text(gap / 2, y - 0.29, label, color=color, ha="center", va="top")
    rope = planar.mv.point([[0.0, y], [gap, y]])                                # [2] Point-
    ship_inks = np.stack((ink_rgb(rear_colour), ink_rgb(front_colour)))         # [2, 3]
    span = planar.mv.point([[gap / 2, y - 0.17], [0.0, y - 0.17], [gap / 2, y - 0.17], [gap, y - 0.17]])  # [4] Point-
    return [
        partial(simplices.strokes, rope, Topology.chains([2]), ink_rgb(color), worldline_px),
        partial(simplices.strokes, span, Topology.chains([2, 2]), ink_rgb(color), charts.LINE_PX, cap=simplices.Cap.BUTT),
        partial(simplices.markers, span[1::2], ink_rgb(color), span_tip_px),
        partial(simplices.markers, rope, np.ones(3), ship_half_side),
        partial(square_markers, rope, ship_inks, ship_half_side),
    ]


@charts.style()
def spaceships_figure(tracks: Point, schedules: Point, slices: Point, final_events: Point) -> plt.Figure:
    """Two lab spacetime diagrams and a shared final-rest-frame comparison contrast Bell's spaceships with strain-preserving timing."""
    fig, axes = charts.subplots(charts.DESIGN_WIDTH, 480, ncols=3, width_ratios=[1.0, 1.0, 1.05], layout="constrained")
    titles = ("Strain-preserving impulse train", "Bell: identical clock programs")
    end_time = float((mv.t & tracks[0, -1, 0]).to_array())
    layers = []
    for index, (color, ax) in enumerate(zip((impulse_colour, elastic_colour), axes[:2])):
        cuts = slices[index]                                                    # [2, 2] Point-
        layers.append([
            endpoint_polylines(tracks[index], (rear_colour, front_colour), worldline_px),
            event_chain(schedules[index], color, charts.LINE_PX, dashes=dashed_pattern, cap=simplices.Cap.BUTT, opacity=0.8),
            length_bar(cuts, bar_colour, 0.27),
            event_dots(schedules[index], color),
        ])
        for length, (time_at, middle), label in zip(cut_lengths(cuts), cut_midpoints(cuts), ("Initially: {:.2f} L₀", "Final lab gap: {:.2f} L₀")):
            ax.text(
                middle, time_at + (0.09 if time_at > 0 else -0.09), label.format(length),
                ha="center", va="bottom" if time_at > 0 else "top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
            )
        note = "Front impulses follow\nthe bisectors" if index == 0 else "Matching clock readings;\nhorizontal timing lines"
        ax.text(0.5, 1.025, note, transform=ax.transAxes, ha="center", va="bottom", color=color)
        ax.set_title(titles[index], pad=note_pad)
        ax.set(xlabel="x / L₀", ylabel="ct / L₀", xlim=(-0.20, 3.05), ylim=(-0.40, end_time + 0.18))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([0, 1, 2])
        ax.grid(alpha=0.13)
        ax.spines[["top", "right"]].set_visible(False)

    gaps = cut_lengths(final_events)
    ax = axes[2]
    ax.set_title("In the final shared rest frame", pad=note_pad)
    relaxed = planar.mv.point([[1.0, 0.65], [1.0, 2.25]])                       # [2] Point-
    ropes = (
        [partial(simplices.strokes, relaxed, Topology.chains([2]), ink_rgb(span_colour), charts.LINE_PX, dashes=dotted_pattern, cap=simplices.Cap.BUTT)]
        + rope_comparison(ax, gaps[0], 2.05, impulse_colour, "Strain-preserving timing", "1.00 L₀ · no extension")
        + rope_comparison(ax, gaps[1], 0.82, elastic_colour, "Synchronized-clock timing", f"{gaps[1]:.2f} L₀ · {100 * (gaps[1] - 1):.1f}% extension")
    )
    ax.text(0.0, 2.78, "Same final speed: 0.8c\nDifferent required rope lengths", color="#59616c")
    ax.text(0.0, 0.04, "Rope tension grows;\nfailure depends on its strength\nand elastic response.", color=elastic_colour, va="top")
    ax.set(xlim=(-0.22, 3.2), ylim=(-0.75, 3.03))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    legend = [
        Line2D([], [], color=rear_colour, lw=worldline_pt, label="Rear ship"),
        Line2D([], [], color=front_colour, lw=worldline_pt, label="Front ship"),
        Line2D([], [], color="#666666", ls="--", label="Matching impulse events"),
    ]
    figure_header(
        fig,
        [
            ("Bell’s spaceships: the timing across the rope matters", "black"),
            ("10 matching impulses   ·   0 → 0.8c   ·   Same rear-ship history in both cases", "#59616c"),
        ],
        legend,
    )
    fig.supxlabel(
        "The bisector train preserves proper spacing.\n"
        "Identical synchronized clock programs preserve the initial lab gap instead.\n"
        "Dashed lines specify timing; they are not signals or sound waves.\n"
        "Rope extension is kinematic; elastic dynamics are not simulated.",
        color="#59616c", linespacing=1.6,
    )
    for ax, panel_layers in zip(axes[:2], layers):
        spacetime_diagram(ax, panel_layers)
    camera = panel(axes[2], planar.mv.x, planar.mv.y, planar.mv.w)
    overlay(axes[2], [layer(camera=camera) for layer in ropes]).set_zorder(drawing_zorder)
    return fig
