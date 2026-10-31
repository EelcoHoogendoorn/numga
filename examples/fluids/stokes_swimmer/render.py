"""Drawing for low-Reynolds-number swimming in PGA2D: the Purcell swimmer and its surrounding Stokes
flow field beside the reciprocal flapper and their paths in hinge-angle space."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack
from examples.util import charts
from examples.util.wgsl import flow as ink, simplices
from examples.util.wgsl.camera import Camera, flat_panel
from examples.util.animation import capture
from examples.fluids.stokes_swimmer import core

purcell_colour = "#1d3557"
scallop_colour = "#c84b31"
trail_colour = "#e63946"
flow_colour = "#94a3b8"
force_colour = "#2a9d8f"
cross_colour, label_colour = "#cbd5e1", "0.4"
# The margin around the strokes in hinge-angle space, as a share of their extent, across and up: up, the room for the
# name running down the line.
hinge_margin, hinge_room = 0.12, 0.5
# Half the height of the water each animated panel shows, and how far it reaches past the swimmer's track and the
# origin either way.
water_half_height = 1.55
water_margin = 2.3
# The swimmer's sizes in points, drawn at each figure's pixels a point: its chain's width, its beads' radius and their
# white rims, its hinges' radius and their coloured rims; the trail's width.
chain_width, bead_radius, bead_rim, hinge_radius, hinge_rim = 2.6, 3.25, 0.6, 2.55, 1.4
summary_trail_width, animation_trail_width = charts.points(charts.LINE_PX), 1.6
# The bead forces' arrows: their width and the radius of the dots at their tips in pixels, and the force per width of
# the flow's panel, the arrows' scale.
force_width, force_tip, force_scale = 1.75, 2.6, 1.4


def rgb(colour: str) -> np.ndarray:
    """A named colour's red, green and blue."""
    return np.array(to_rgb(colour))


def _cross(ax, across: str, up: str) -> None:
    """No scales: a faint cross through the origin, its lines named at their ends, each along itself,
    the line across named just past its end."""
    ax.axhline(0.0, color=cross_colour, linewidth=charts.FRAME_WIDTH, zorder=0)
    ax.axvline(0.0, color=cross_colour, linewidth=charts.FRAME_WIDTH, zorder=0)
    ax.text(1, 0, f" {across}", transform=ax.get_yaxis_transform(), ha="left", va="center", color=label_colour)
    ax.text(0, 1, f"{up} ", transform=ax.get_xaxis_transform(), ha="right", va="bottom", color=label_colour,
            rotation=90, rotation_mode="anchor")
    ax.set_axis_off()


def _zero_axes(ax, across: str, up: str) -> None:
    """Axes of a quantity through its zero and the parameter's, named at their ends, each along itself, without tick
    marks or numbers, the axis across named just past its end."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_position(("data", 0))
    ax.set(xticks=[], yticks=[])
    ax.text(1, 0, f" {across}", transform=ax.get_yaxis_transform(), ha="left", va="center", color=label_colour)
    ax.text(0, 1, f"{up} ", transform=ax.get_xaxis_transform(), ha="right", va="bottom", color=label_colour,
            rotation=90, rotation_mode="anchor")


def _links(beads: core.Point) -> tuple[core.Point, core.Point]:
    """The three-link body through its nine beads `[9]`: the chain through the beads and the hinges `[11] Point-`,
    and the two hinges `[2] Point-`."""
    # Left arm (beads 0, 1, 2), centre bar (beads 3, 4, 5), right arm (beads 6, 7, 8); each hinge half a bead's
    # spacing beyond the end of the centre bar.
    left_hinge = 1.5 * beads[3:4] - 0.5 * beads[4:5]                             # [1] Point-
    right_hinge = 1.5 * beads[5:6] - 0.5 * beads[4:5]                            # [1] Point-
    chain = concatenate([beads[0:3], left_hinge, beads[3:6], right_hinge, beads[6:9]])
    return chain, concatenate([left_hinge, right_hinge])


def _swimmer(beads: core.Point, colour: str, camera: Camera, pixels: float) -> list[np.ndarray]:
    """The three-link body and its nine hydrodynamic beads `[9] Point-`: the chain, the beads in white rims and the
    hinges, white in coloured rims, at `pixels` a point."""
    chain, hinges = _links(beads)
    return [simplices.strokes(chain, simplices.Mesh.chains([len(chain)]), rgb(colour), chain_width * pixels, camera),
            simplices.markers(beads, np.ones(3), (bead_radius + bead_rim / 2) * pixels, camera),
            simplices.markers(beads, rgb(colour), (bead_radius - bead_rim / 2) * pixels, camera),
            simplices.markers(hinges, rgb(colour), (hinge_radius + hinge_rim / 2) * pixels, camera),
            simplices.markers(hinges, np.ones(3), (hinge_radius - hinge_rim / 2) * pixels, camera)]


def _trail(track: core.Point, width: float, camera: Camera) -> np.ndarray:
    """A swimmer's centre's trail through its places `[steps] Point-`, `width` pixels wide."""
    return simplices.strokes(track, simplices.Mesh.chains([len(track)]), rgb(trail_colour), width, camera)


@charts.style()
def draw_summary(
    angles: np.ndarray,
    positions: core.Point,
    world_sites: core.Point,
    world_forces: core.Force,
    grid: core.Point,
    flow: core.Direction,
) -> plt.Figure:
    """Left: instantaneous Stokes flow around Purcell's swimmer with bead forces and centre trail;
    middle: centre displacement along x over the three cycles for both strokes; right: the two strokes
    in hinge-angle space."""
    # The whole column: the surrounding flow and two trajectory plots.
    figure, (ax_flow, ax_path, ax_hinge) = charts.subplots(800, 260, ncols=3)

    mv = core.mv
    beads = world_sites.batch()                                                 # [steps, swimmers, beads] Point-
    along = (mv.x & positions).to_array()                                       # [steps + 1, swimmers]

    # The flow's panel spans the grid, from its first sample to its last.
    limits = (stack((mv.x, mv.y))[:, None] & stack((grid[0, 0], grid[-1, -1]))).to_array()   # [2, 2]
    ax_flow.set(xlim=limits[0], ylim=limits[1])
    ax_flow.set_aspect("equal")
    ax_flow.set_axis_off()

    t = np.linspace(0.0, 1.0, len(along))
    ax_path.plot(t, along[:, 0], color=purcell_colour)
    ax_path.plot(t, along[:, 1], color=scallop_colour, linestyle="--")
    _zero_axes(ax_path, "time", "$x$")

    ax_hinge.plot(angles[:, 0, 0], angles[:, 0, 1], color=purcell_colour)
    ax_hinge.plot(angles[:, 1, 0], angles[:, 1, 1], color=scallop_colour, linestyle="--")
    ax_hinge.set_aspect("equal")
    # Room above the loop for the name of the line up.
    ax_hinge.margins(x=hinge_margin, y=hinge_room)
    _cross(ax_hinge, "left hinge", "right hinge")

    figure.tight_layout()
    mesh = ink.Mesh.grid(*grid.shape)
    points, directions = grid.reshape(-1), flow[0].reshape(-1)
    reference_speed = float(directions.dual().norm().max().to_array())
    camera = flat_panel(ax_flow, core.ga)
    # Each bead's force drawn along the direction it pushes, one panel's width for `force_scale` of force, a stroke
    # with a dot at its tip.
    sites = beads[0, 0]                                                         # [beads] Point-
    panel_width = mv.x & (grid[0, -1] - grid[0, 0])                             # [] Scalar+
    pushes = world_forces.batch()[0, 0].dual() * (panel_width / force_scale)   # [beads] Direction-
    n_beads = len(sites)
    simplices.overlay(ax_flow, [
        ink.layer(points, mesh, directions, camera, reference_speed, rgb(flow_colour)),
        _trail(positions[:, 0], summary_trail_width * charts.DPI / 72, camera),
        *_swimmer(sites, purcell_colour, camera, charts.DPI / 72),
        simplices.strokes(stack((sites, sites + pushes), axis=-1).reshape(-1), simplices.Mesh.chains([2] * n_beads),
                          rgb(force_colour), force_width, camera),
        simplices.markers(sites + pushes, rgb(force_colour), force_tip, camera)])
    return figure


def animate_swimmers(
    positions: core.Point,
    world_sites: core.Point,
    grid: core.Point,
    flows: Iterable[core.Direction],
    every: int,
) -> Iterator[np.ndarray]:
    """Both swimmers' instantaneous velocities share fixed seeds and a fixed magnitude scale."""
    beads = world_sites.batch()[::every]                                        # [frames, swimmers, beads] Point-
    # Each swimmer's centre's places, swimmer by swimmer.
    paths = stack([positions[:, swimmer] for swimmer in range(positions.shape[1])])   # [swimmers, steps + 1] Point-
    mesh = ink.Mesh.grid(*grid.shape)
    points = grid.reshape(-1)                                                  # [samples] Point-
    flows = iter(flows)
    first = next(flows)
    directions = first.reshape(first.shape[0], -1)
    reference_speed = float(np.percentile(directions.dual().norm().to_array(), 95))
    # The stretch of water the Purcell swimmer's track and the origin span along x, the margin either way.
    along = core.mv.x & positions[:, 0]                                        # [steps + 1] Scalar+
    left = float((along.min() - water_margin).clip(-np.inf, -water_margin).to_array())
    right = float((along.max() + water_margin).clip(water_margin, np.inf).to_array())
    # As wide as the stretch of water each panel shows, at the height the two panels share.
    figure_height = 4.6
    figure_width = figure_height / 2 * (right - left) / (2 * water_half_height)
    dpi = 55
    figure, axes = plt.subplots(2, 1, figsize=(figure_width, figure_height), dpi=dpi)
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, hspace=0.06)
    for ax in axes:
        ax.set(xlim=(left, right), ylim=(-water_half_height, water_half_height), aspect="equal")
        ax.set_axis_off()
    views = [flat_panel(ax, core.ga) for ax in axes]
    textures = [simplices.overlay(ax, [np.zeros(view.shape + (4,))]) for ax, view in zip(axes, views)]
    for frame, fields in enumerate(chain((first,), flows)):
        directions = fields.reshape(fields.shape[0], -1)
        step_end = frame * every + 1
        for swimmer, (texture, view, field, colour) in enumerate(
                zip(textures, views, directions, (purcell_colour, scallop_colour))):
            texture.set_data(simplices.composited([
                ink.layer(points, mesh, field, view, reference_speed, rgb(flow_colour)),
                _trail(paths[swimmer, :step_end], animation_trail_width * dpi / 72, view),
                *_swimmer(beads[frame, swimmer], colour, view, dpi / 72)]))
        yield capture(figure)
    plt.close(figure)
