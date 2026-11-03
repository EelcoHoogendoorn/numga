"""Fluxons in a planar Josephson junction: a fluxon and an antifluxon passing through each other along a narrow
junction, drawn as a row of pendula; boosted fluxon lines, whose fields contract as moving rods do; a ring of flux
collapsing under its own tension; and an oblique fluxon line reflected from the junction's edge as an antifluxon.

The sine-Gordon equation of a long junction, its kink, the fluxon's Lorentz contraction and a fluxon reflected from an
open edge as an antifluxon follow the Stockholm lecture notes on the Josephson effect and on fluxons in Josephson
junctions.
"""

from __future__ import annotations

import numpy as np

from numga.sparse import spdiag

from examples.electromagnetism.josephson import core

mv = core.mv

# The meeting: the fluxon's and the antifluxon's starting distance from the middle and their rapidity, the steps, the
# steps between frames, every how many places a pendulum is drawn, and how far from the middle they are drawn.
meeting_start, meeting_rapidity = 8.0, 0.6
n_meeting, meeting_stride, pendulum_every, pendulum_reach = 1500, 15, 6, 12.0
# The boosted fluxons: a strip, in Josephson penetration depths, its length, its height and its places along and
# across; their rapidities, where they start, the time step in inverse plasma frequencies, and the steps they run.
strip_length, strip_height, n_along, n_strip_rows = 40.0, 0.1, 801, 3
rapidities = np.array([0.0, 0.4, 0.8, 1.2, 1.6])
start, strip_dt, n_running = -12.0, 0.02, 800
# The ring: a square junction's side and its places to a side, the ring's radius, the time step, the steps and the
# steps between frames.
square_side, n_square, ring_radius = 24.0, 97, 8.0
ring_dt, n_ring_steps, ring_stride = 0.1, 240, 2
# The reflection: a junction's width and height and its places across and up, the line's tilt from the y axis in
# radians, its starting distance from the middle and rapidity, the time step, the steps and the steps between frames.
wide, high, n_wide, n_high = 32.0, 16.0, 129, 65
tilt, line_start, line_rapidity = 0.35, 4.0, 0.9
line_dt, n_line_steps, line_stride = 0.1, 360, 2
frame_duration_ms = 40


def operators(topology: core.Topology, places: np.ndarray):
    """A junction's places of unit weight `Point-[places]`, its triangles' hat gradients and the field's tension,
    `Scalar+[places <- places]`."""
    vertices = (core.axis * places[:, 0] + core.across * places[:, 1]).field()   # Point-[places]
    areas, gradients, cells = core.triangles(topology, vertices)
    # Each hat's gradient read against the supercurrent another's drives, over the triangle.
    local = (gradients[..., :, None] & core.supercurrent(gradients[..., None, :])) * areas[..., None, None]
    stiffness = core.assembled(topology, local)                                # Scalar+[places <- places]
    return vertices, gradients, spdiag(1 / cells) * stiffness


def started(turns_now: core.Turn, turns_next: core.Turn, dt: float) -> tuple[core.Phase, core.Rate]:
    """The phase directions `[...] Phase+[places]` of these turns now, and their rates `[...] Rate+[places]` from these
    and the turns a step later: the phase's turn over the step, the wedge of the two directions, over the step."""
    now, later = turns_now >> mv.a, turns_next >> mv.a
    return now, (now ^ later) / dt


def everywhere(places: np.ndarray) -> core.Scalar:
    """Tunnelling of unit strength at every place `Scalar+[places]`."""
    return mv.scalar(np.ones((len(places), 1))).field()


def meeting():
    """A fluxon and an antifluxon running into each other along the strip, their turns composed so that their phases
    add: the pendula's pivots, every so many places along the strip's middle `[pendula] Point-`, and their phase
    directions at each frame `[frames] Phase+[pendula]`."""
    topology, places = core.junction(n_along, n_strip_rows, strip_length, strip_height)
    vertices, gradients, tension = operators(topology, places)
    events = [(core.axis * places[:, 0] + core.across * places[:, 1] + core.time * moment).field()
              for moment in (0.0, strip_dt)]
    pairs = [core.fluxon(moment_events + core.axis * meeting_start, meeting_rapidity, 1.0) *
             core.fluxon(moment_events - core.axis * meeting_start, -meeting_rapidity, -1.0) for moment_events in events]
    phases, rates = started(*pairs, strip_dt)
    middle_row = np.isclose(places[:, 1], 0.0, atol=1e-9) & (np.abs(places[:, 0]) <= pendulum_reach)
    shown = np.flatnonzero(middle_row)[::pendulum_every]                     # [pendula]
    frames = core.evolving(tension, everywhere(places), phases, rates, strip_dt, n_meeting, meeting_stride)
    return vertices.batch()[shown], (phases.batch()[shown] for phases in frames)


def contracting() -> tuple[np.ndarray, core.Scalar, core.Scalar]:
    """The boosted fluxons' speeds `[rapidities]`, and the squared strength of their field in each triangle
    `[rapidities] Scalar+[faces]` at the start and after their run."""
    topology, places = core.junction(n_along, n_strip_rows, strip_length, strip_height)
    vertices, gradients, tension = operators(topology, places)
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- places]
    phases, rates = started(*(core.fluxon((core.axis * (places[:, 0] - start) + core.across * places[:, 1] +
                                           core.time * moment).field(), rapidities, 1.0)
                              for moment in (0.0, strip_dt)), strip_dt)
    runs = list(core.evolving(tension, everywhere(places), phases, rates, strip_dt, n_running, n_running))
    # The field's strength: the norm of the phase's gradient, in the mixed planes of place and phase.
    strengths = [(gradients ^ (corners * run[..., None])).sum(axis=-1).scalar_norm_squared() for run in runs]
    return np.tanh(rapidities), strengths[0], strengths[-1]


def collapsing():
    """The ring of flux at rest, collapsing: the square junction's topology, its places `Point-[places]`, and the
    tunnelling current at each frame, the bias direction wedged with the phase, `[frames] Rate+[places]`."""
    topology, places = core.junction(n_square, n_square, square_side, square_side)
    vertices, gradients, tension = operators(topology, places)
    # The resting kink across the ring, by each place's distance from the middle less the ring's radius: four times the
    # arctangent of its exponential.
    distances = vertices.squared().abs().square_root()                        # Scalar+[places]
    angles = 4 * (distances - ring_radius).exp().arctan()                     # Scalar+[places]
    phases = (mv.ab * (-angles / 2)).exp() >> mv.a                            # Phase+[places]
    frames = core.evolving(tension, everywhere(places), phases, 0 * (mv.a ^ phases), ring_dt, n_ring_steps, ring_stride)
    return topology, vertices, (mv.a ^ phases for phases in frames)


def reflecting():
    """The oblique fluxon line running into the junction's edge: the junction's topology, its places `Point-[places]`,
    and the tunnelling current at each frame `[frames] Rate+[places]`."""
    topology, places = core.junction(n_wide, n_high, wide, high)
    vertices, gradients, tension = operators(topology, places)
    # The line's events: the junction's events turned by the tilt about the middle, so that a line along y lies along
    # the tilted line.
    turn = (mv.xy * (tilt / 2)).exp()                                         # [] Rotor+
    turns = [core.fluxon(turn >> (core.axis * places[:, 0] + core.across * places[:, 1] +
                                  core.axis * (-line_start) + core.time * moment).field(), line_rapidity, 1.0)
             for moment in (0.0, line_dt)]
    phases, rates = started(*turns, line_dt)
    frames = core.evolving(tension, everywhere(places), phases, rates, line_dt, n_line_steps, line_stride)
    return topology, vertices, (mv.a ^ phases for phases in frames)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.electromagnetism.josephson import render

    save_animation(render.animate_pendula(*meeting()), "josephson_pendula", frame_duration_ms)
    speeds, before, after = contracting()
    save_figure(render.draw_contraction(speeds, before, after), "josephson_contraction")
    save_animation(render.animate(*collapsing()), "josephson_ring", frame_duration_ms)
    save_animation(render.animate(*reflecting()), "josephson_reflection", frame_duration_ms)


if __name__ == "__main__":
    main()
