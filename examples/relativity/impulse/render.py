"""Spacetime diagrams of the impulse examples: drawing only.

Coordinates leave the algebra here, at the plotting boundary. Every diagram is drawn with
x horizontal and ct vertical, in units of the proper length.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from examples.relativity.impulse.core import Spacetime, Vector

REAR, FRONT, IMPULSE = "#2475ba", "#db7220", "#258e59"
DOOR, ELASTIC, CONTACT = "#59616c", "#a64479", "#ba4242"


def ct_x(events: Vector) -> np.ndarray:
    """(..., 2) the ct and x coordinates of events."""
    return events.cast(Spacetime.subspace("t x")).kernel


def plot_worldlines(ax, lines: np.ndarray, colors: tuple[str, str], **style) -> None:
    """Draw each end's polyline from (vertex, end, 2) ct, x coordinates."""
    for end, color in enumerate(colors):
        ax.plot(lines[:, end, 1], lines[:, end, 0], color=color, **style)


# --- relativistic impulse -------------------------------------------------------------
def draw_impulse(
    worldlines: Vector, kinks: Vector, slices: Vector, train: Vector, steps: Vector, train_slice: Vector,
) -> plt.Figure:
    """The symmetric reversal, the same drawing boosted, and a train of ten small impulses."""
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 7.1), dpi=160)
    for index, (ax, title) in enumerate(zip(axes[:2], (
        "Symmetric frame: −0.5c → +0.5c", "Boosted frame: 0 → 0.8c",
    ))):
        lines, events, cuts = ct_x(worldlines[index]), ct_x(kinks[index]), ct_x(slices[index])
        for end, label in enumerate(("Left end", "Right end")):
            ax.plot(lines[:, end, 1], lines[:, end, 0], color=(REAR, FRONT)[end], lw=2.5, label=label)
        ax.plot(events[:, 1], events[:, 0], "--", color=IMPULSE, lw=2.5, label="Velocity-step surface")
        ax.scatter(events[:, 1], events[:, 0], color=IMPULSE, s=32, zorder=4)

        for cut, prefix in zip(cuts, ("Before", "After")):
            t, (a, b) = cut[0, 0], cut[:, 1]
            ax.plot([a, b], [t, t], color="#333333", lw=5, alpha=0.35)
            ax.text((a+b)/2, t+(-0.14 if t<0 else 0.12),
                    f"{prefix}: {b-a:.3f} L₀", ha="center", fontsize=10)
        if index == 0:
            ax.text(events[1, 1]/2, 0.12, "Simultaneous reversal", ha="center", color=IMPULSE, fontsize=10)
        else:
            ax.text(0.34, 0.41, "Same line, boosted", color=IMPULSE, fontsize=10)
        ax.set_title(title, fontsize=12, loc="left", pad=14)
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
    plot_worldlines(ax, lines, (REAR, FRONT), lw=2.5)
    for step in events:
        ax.plot(step[:, 1], step[:, 0], "--", color=IMPULSE, lw=1.4, alpha=0.75)
        ax.scatter(step[:, 1], step[:, 0], color=IMPULSE, s=12, zorder=4)

    sample_time, (a, b) = cut[0, 0], cut[:, 1]
    ax.plot([a, b], [sample_time, sample_time], color="#333333", lw=5, alpha=0.35)
    ax.text((a+b)/2, sample_time+0.12, f"After: {b-a:.3f} L₀", ha="center", fontsize=10)
    ax.text(0.04, 0.96, "Repeat one fixed map\nEqual ticks on the left clock",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.set_title("10 impulses: 0 → 0.8c", fontsize=12, loc="left", pad=14)
    ax.set(xlabel="x / L₀", ylabel="ct / L₀", xlim=(-0.16, 2.7), ylim=(-0.38, end_time+0.30))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Strain-preserving impulses: start with a symmetric reversal", y=0.99)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.945), ncol=3, frameon=False, fontsize=10)
    fig.text(0.5, 0.025,
             "Velocity changes at kinks; length changes continuously along connected worldlines.\n"
             "Different observers take different time slices of the same drawing and agree on the material strain.",
             ha="center", fontsize=9.5, color="#59616c", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.13, 1, 0.86))
    return fig


# --- ladder paradox -------------------------------------------------------------------
def style_axis(ax, title, xlim, ylim, primed):
    suffix = "′" if primed else ""
    ax.set(xlabel=f"x{suffix} / L₀", ylabel=f"ct{suffix} / L₀",
           xlim=xlim, ylim=ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=12, pad=13)


def barn_background(ax, doors: Vector, closure: Vector) -> None:
    """The barn between its door worldlines (time, door), and the door-closure events."""
    lines, events = ct_x(doors), ct_x(closure)
    ax.fill_betweenx(lines[:, 0, 0], lines[:, 0, 1], lines[:, 1, 1], color=DOOR, alpha=0.07)
    ax.plot(lines[..., 1], lines[..., 0], color=DOOR, ls="--", lw=1.3, alpha=0.65)
    ax.scatter(events[:, 1], events[:, 0], marker="s", s=48,
               color=DOOR, edgecolor="white", linewidth=0.7, zorder=6)


def length_bar(ax, ends: Vector, label, offset, color):
    """A bar across the equal-time events at both ends, labelled above or below."""
    events = ct_x(ends)
    positions, time = events[:, 1], events[0, 0]
    ax.plot(positions, [time, time], lw=5, color=color, alpha=0.28,
            solid_capstyle="butt")
    ax.text(np.mean(positions), time + offset, label, ha="center", fontsize=10,
            color=color, va="bottom" if offset > 0 else "top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5})


def draw_ladder(
    barn_frame: tuple[Vector, ...], ladder_frame: tuple[Vector, ...],
    strain_preserving: tuple[Vector, ...], ringing: tuple[Vector, ...],
) -> plt.Figure:
    """The ladder in the barn frame and its own frame, a strain-preserving stop, and a stop inside."""
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 12.0), dpi=170)

    # 1. The ordinary paradox concerns simultaneous door closure, without a stop.
    doors, closure, ladder, at_closure = barn_frame
    ax = axes[0, 0]
    barn_background(ax, doors, closure)
    plot_worldlines(ax, ct_x(ladder), (REAR, FRONT), lw=2.6)
    ax.axhline(0.0, color=DOOR, lw=1, alpha=0.3)
    length_bar(ax, at_closure, "Moving ladder: 0.60 L₀", 0.10, "#333333")
    ax.annotate("Doors close together", xy=(0.8, 0.0), xytext=(0.43, -0.36),
                ha="center", fontsize=10, color=DOOR,
                arrowprops={"arrowstyle": "->", "color": DOOR})
    ax.text(0.23, 0.51, "Barn: 0.80 L₀", ha="center", color=DOOR, fontsize=10)
    style_axis(ax, "1  ·  In the barn frame: it fits at closure",
               (-0.46, 1.32), (-0.6, 0.65), False)

    # 2. Boost the SAME closure events and incoming end worldlines.
    doors, closure, ladder, ladder_cut, barn_cut = ladder_frame
    ax = axes[0, 1]
    barn_background(ax, doors, closure)
    plot_worldlines(ax, ct_x(ladder), (REAR, FRONT), lw=2.6)
    length_bar(ax, ladder_cut, "Ladder: 1.00 L₀", 0.08, "#333333")
    length_bar(ax, barn_cut, "Barn: 0.48 L₀", 0.08, DOOR)
    closing = ct_x(closure)
    ax.annotate("Exit closes first", xy=closing[1, ::-1],
                xytext=(1.40, -0.97), fontsize=10, color=DOOR,
                arrowprops={"arrowstyle": "->", "color": DOOR})
    ax.annotate("Entrance closes later", xy=closing[0, ::-1],
                xytext=(0.49, 0.13), fontsize=10, color=DOOR,
                arrowprops={"arrowstyle": "->", "color": DOOR})
    style_axis(ax, "2  ·  In the incoming ladder frame: different times",
               (-0.31, 2.32), (-1.62, 0.30), True)

    # 3. Prescribed staggered braking: no proper-length mismatch to relax.
    doors, closure, ladder, kinks, at_closure, stopped, contact = strain_preserving
    limits = (-0.48, 2.12)
    ax = axes[1, 0]
    barn_background(ax, doors, closure)
    plot_worldlines(ax, ct_x(ladder), (REAR, FRONT), lw=2.6)
    events = ct_x(kinks)
    ax.plot(events[:, 1], events[:, 0], "--", color=IMPULSE, lw=2.5)
    ax.scatter(events[:, 1], events[:, 0], color=IMPULSE, s=30, zorder=5)
    ax.text(0.30, 0.57, "Kinks joined by\ntheir bisector",
            fontsize=10, color=IMPULSE, ha="center")
    length_bar(ax, at_closure, "0.60 L₀ at closure", -0.13, "#333333")
    length_bar(ax, stopped, "At rest: 1.00 L₀", 0.11, "#333333")
    contact_time, contact_x = ct_x(contact)
    ax.scatter([contact_x], [contact_time], marker="*", color=CONTACT, s=125, zorder=7)
    ax.annotate("Front reaches exit", xy=(contact_x, contact_time),
                xytext=(1.28, 0.85), ha="right", fontsize=10, color=CONTACT,
                arrowprops={"arrowstyle": "->", "color": CONTACT})
    ax.text(0.59, 1.76, "Relaxed at rest:\ntoo long for the barn", ha="center", fontsize=10)
    style_axis(ax, "3  ·  Strain-preserving stop: no longer fits",
               (-0.44, 1.44), limits, False)

    # 4. Stopping inside leaves compression, whose relaxation drives ringing.
    doors, closure, incoming, ring, relaxed, stop, contacts, relaxed_cut = ringing
    ax = axes[1, 1]
    barn_background(ax, doors, closure)
    plot_worldlines(ax, ct_x(incoming), (REAR, FRONT), lw=2.6)
    plot_worldlines(ax, ct_x(ring), (REAR, FRONT), lw=2.6)
    plot_worldlines(ax, ct_x(relaxed), (REAR, FRONT), ls=":", lw=1.3, alpha=0.5)
    events = ct_x(stop)
    ax.plot(events[:, 1], events[:, 0], "--", color=ELASTIC, lw=2.6)
    ax.scatter(events[:, 1], events[:, 0], color=ELASTIC, s=32, zorder=5)
    length_bar(ax, stop, "Stopped at 0.60 L₀: compressed", -0.14, ELASTIC)
    touches = ct_x(contacts)
    ax.scatter(touches[:, 1], touches[:, 0], marker="*", color=CONTACT, s=125, zorder=7)
    ax.annotate("Expansion reaches both doors", xy=touches[1, ::-1],
                xytext=(0.4, 0.50), ha="center", fontsize=10, color=CONTACT,
                arrowprops={"arrowstyle": "->", "color": CONTACT})
    length_bar(ax, relaxed_cut, "Relaxed length: 1.00 L₀", 0.08, "#333333")
    ax.text(0.4, 1.04, "Elastic ringing\nQuarter-critical\ndamping",
            ha="center", fontsize=10, color=ELASTIC)
    style_axis(ax, "4  ·  Stop inside: compression and ringing",
               (-0.44, 1.44), limits, False)

    fig.suptitle("The ladder paradox: fitting in motion, stopping in a shorter barn", fontsize=17, y=0.98)
    fig.text(0.5, 0.948, "Relaxed ladder L₀ = 1   ·   Barn = 0.8 L₀   ·   Incoming speed = 0.8c",
             ha="center", fontsize=11, color=DOOR)
    legend = [Line2D([], [], color=REAR, lw=2.5, label="Rear end"),
              Line2D([], [], color=FRONT, lw=2.5, label="Front end"),
              Line2D([], [], color=DOOR, ls="--", lw=1.3, label="Door position"),
              Line2D([], [], color=DOOR, marker="s", ls="none", label="Door closes")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.931),
               ncol=4, frameon=False, fontsize=10)
    fig.text(0.5, 0.035,
             "The ladder can fit for an instant; it cannot remain inside at its relaxed rest length.\n"
             "Door crossings assume yielding doors; ringing illustrates an elastic mode, not a collision calculation.",
             ha="center", fontsize=10, color=DOOR, linespacing=1.6)
    fig.tight_layout(rect=(0.015, 0.075, 0.99, 0.90), h_pad=3.2, w_pad=3.0)
    return fig


# --- Bell's spaceships ----------------------------------------------------------------
def rope_comparison(ax, gap, y, color, title, label):
    """A spatial slice in the final rest frame, aligned on the rear attachment."""
    ax.text(0.0, y + 0.36, title, color=color, fontsize=11, weight="medium")
    ax.plot([0.0, gap], [y, y], color=color, lw=3)
    for x, ship_color in ((0.0, REAR), (gap, FRONT)):
        ax.scatter([x], [y], marker=">", s=160, facecolor="white",
                   edgecolor=ship_color, linewidth=2.0, zorder=5)
    ax.annotate("", xy=(gap, y - 0.17), xytext=(0.0, y - 0.17),
                arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.2})
    ax.text(gap / 2, y - 0.29, label, color=color, ha="center", va="top", fontsize=10)


def draw_spaceships(tracks: Vector, schedules: Vector, slices: Vector, final_events: Vector) -> plt.Figure:
    """Both impulse schedules in the lab, and the rope each demands in the final rest frame."""
    fig, axes = plt.subplots(1, 3, figsize=(16.7, 7.5), dpi=170,
                             gridspec_kw={"width_ratios": [1.0, 1.0, 0.85]})
    titles = ("Strain-preserving impulse train", "Bell: identical clock programs")
    end_time = ct_x(tracks)[0, -1, 0, 0]
    for index, (color, ax) in enumerate(zip((IMPULSE, ELASTIC), axes[:2])):
        plot_worldlines(ax, ct_x(tracks[index]), (REAR, FRONT), lw=2.6)
        for step in ct_x(schedules[index]):
            ax.plot(step[:, 1], step[:, 0], "--", color=color, lw=1.35, alpha=0.8)
            ax.scatter(step[:, 1], step[:, 0], color=color, s=11, zorder=4)

        for cut, label in zip(ct_x(slices[index]), ("Initially: {:.2f} L₀", "Final lab gap: {:.2f} L₀")):
            time, endpoints = cut[0, 0], cut[:, 1]
            ax.plot(endpoints, [time, time], color="#333333", lw=5, alpha=0.27)
            ax.text(np.mean(endpoints), time + (0.09 if time > 0 else -0.09),
                    label.format(endpoints[1] - endpoints[0]), ha="center", va="bottom" if time > 0 else "top",
                    fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
        note = ("Front impulses follow the bisectors"
                if index == 0 else "Matching clock readings; horizontal timing lines")
        ax.text(0.5, 1.025, note, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color=color)
        ax.set_title(titles[index], fontsize=12, pad=38)
        ax.set(xlabel="x / L₀", ylabel="ct / L₀", xlim=(-0.20, 3.05), ylim=(-0.40, end_time + 0.18))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([0, 1, 2])
        ax.grid(alpha=0.13)
        ax.spines[["top", "right"]].set_visible(False)

    # The rear's final kink sits at the origin of the final frame, so the front's position in that
    # frame is the gap.
    gaps = ct_x(final_events)[:, 1, 1]
    ax = axes[2]
    ax.set_title("In the final shared rest frame", fontsize=12, pad=38)
    rope_comparison(ax, gaps[0], 2.05, IMPULSE, "Strain-preserving timing",
                    "1.00 L₀ · no extension")
    rope_comparison(ax, gaps[1], 0.82, ELASTIC, "Synchronized-clock timing",
                    f"{gaps[1]:.2f} L₀ · {100 * (gaps[1] - 1):.1f}% extension")
    ax.plot([1.0, 1.0], [0.65, 2.25], color="#888888", ls=":", lw=1.2)
    ax.text(0.0, 2.78, "Same final speed: 0.8c\nDifferent required rope lengths", fontsize=10, color="#59616c")
    ax.text(0.0, 0.04, "Rope tension grows; failure depends\non its strength and elastic response.",
            fontsize=10, color=ELASTIC)
    ax.set(xlim=(-0.22, 2.05), ylim=(-0.35, 3.03))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    fig.suptitle("Bell’s spaceships: the timing across the rope matters", fontsize=17, y=0.99)
    fig.text(0.5, 0.937, "10 matching impulses   ·   0 → 0.8c   ·   Same rear-ship history in both cases",
             ha="center", fontsize=11, color="#59616c")
    legend = [Line2D([], [], color=REAR, lw=2.6, label="Rear ship"),
              Line2D([], [], color=FRONT, lw=2.6, label="Front ship"),
              Line2D([], [], color="#666666", ls="--", lw=1.4, label="Matching impulse events")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.903),
               ncol=3, frameon=False, fontsize=10)
    fig.text(0.5, 0.045,
             "The bisector train preserves proper spacing. Identical synchronized clock programs preserve the initial lab gap instead.\n"
             "Dashed lines specify timing; they are not signals or sound waves. Rope extension is kinematic; elastic dynamics are not simulated.",
             ha="center", fontsize=10, color="#59616c", linespacing=1.6)
    fig.subplots_adjust(left=0.055, right=0.98, bottom=0.19, top=0.75, wspace=0.28)
    return fig
