"""GPU ray casting and a companion bar chart display the deforming cube, its area-normal arrows, its face areas, and its volume."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.container import BarContainer

from numga import stack
from examples.util import charts
from examples.util.animation import capture
from examples.mechanics.area_transport import core
from numga.topology import Topology
from numga.render import simplices, solids
from examples.util.panels import fitted, show

face_quads = np.array([(4, 5, 7, 6), (2, 3, 7, 6), (1, 3, 7, 5),
                       (0, 2, 3, 1), (0, 1, 5, 4), (0, 4, 6, 2)])              # [6, 4]
# Number the eight corners by their bits along x, y, and z and connect pairs differing in one bit.
edge_pairs = np.array([(corner, corner | bit) for bit in (4, 2, 1) for corner in range(8) if not corner & bit])  # [12, 2]
axis_colours = ("#2563eb", "#0d9488", "#f97316")
edge_colour = "#334155"
arrow_scale = 0.36
background = np.ones(3)
# Set elevation and azimuth in radians, with azimuth zero looking along +y, and pad the framed cube.
elevation, azimuth = np.radians([23.0, -57.0 + 90.0])
margin = 0.16
# Expand the first frame's camera bounds across and up to leave room for the sideways shear and arrows.
view_room = np.array([1.35, 1.04])
face_opacity = 0.1
edge_radius = 0.87
arrow_width = 2.84
chevron_length = 11.4
# Multiplying an intrinsic patch by the unit volume and taking the dual yields its outward area-normal displacement.
unit_volume = core.mv.xyz.dual()                                                # [] Scalar-


# --- plumbing -------------------------------------------------------------------------
def arrow_offsets(patches: core.Patch) -> core.Point:
    """Multiplying each oriented face patch (`[...] Patch-`) by the unit volume and taking the dual gives its outward area-normal displacement (`[...] Point-`)."""
    return (patches * unit_volume).dual() * arrow_scale


def view(vertices: core.Point, centres: core.Point, patches: core.Patch) -> solids.View:
    """One shared camera view frames both the cube corners (`[8] Point-`) and the patch-arrow tips (`[6] Point-`)."""
    corners = solids.origin + solids.context.include_from_algebra(vertices)     # [8] Point-
    tips = solids.origin + solids.context.include_from_algebra(centres + arrow_offsets(patches))  # [6] Point-
    return solids.view([corners, tips], 0.0, elevation, azimuth, margin, 1)


def picture(vertices: core.Point, centres: core.Point, patches: core.Patch, camera_view: solids.View) -> np.ndarray:
    """Translucent face plates, opaque edge rods, and glossy chevron arrows depict one deformed cube state."""
    places = solids.origin + solids.context.include_from_algebra(vertices)      # [8] Point-
    offsets = arrow_offsets(patches)                                            # [6] Point-
    # Omit faces and arrows whose crushed area falls below one pixel.
    lengths = offsets.norm()                                                    # [6] Scalar+
    visible = lengths > camera_view.pixel
    n_visible = int(visible.sum())
    colours = np.array([to_rgb(colour) for colour in axis_colours * 2])         # [6, 3]
    starts = solids.origin + solids.context.include_from_algebra(centres[visible])  # [n_visible] Point-
    ends = starts + solids.context.include_from_algebra(offsets[visible])       # [n_visible] Point-
    arrows = simplices.tubes(stack((starts, ends), axis=-1).reshape(-1), Topology.chains([2] * n_visible),
                             colours[visible], arrow_width, camera_view, gloss=solids.gloss,
                             chevrons=np.ones(n_visible), chevron=chevron_length)
    rods, joints = solids.edges(places[edge_pairs[:, 0]], places[edge_pairs[:, 1]], edge_radius * camera_view.pixel)
    rod_colour = np.array(to_rgb(edge_colour))
    opaque = [(rods, rod_colour), (joints, rod_colour)]
    plates = solids.plates(places[face_quads[visible]], camera_view.pixel)      # [n_visible, 6] Plane+
    return solids.draw(opaque, camera_view, background, inks=[],
                       translucent=[(plates, colours[visible], face_opacity)], meshes=[(arrows, 1.0)])


def heights(patches: core.Patch, volume: core.Scalar) -> np.ndarray:
    """The norms of the first three face patches and the scalar volume form the four bar heights (`[4]`)."""
    return np.concatenate([patches[:3].norm().to_array(), volume.to_array()[None]])


def measures(ax: Axes, patches: core.Patch, volume: core.Scalar) -> BarContainer:
    """Four bars compare the distinct face areas and the volume against the dotted unit-cube level."""
    values = heights(patches, volume)
    bars = ax.bar(np.arange(len(values)), values, color=(*axis_colours, edge_colour), width=0.58)
    ax.axhline(1, color="#94a3b8", linestyle=":")
    ax.set(xticks=np.arange(len(values)), xticklabels=["face $x$", "face $y$", "face $z$", "volume"], yticks=[1],
           ylim=(0, 1.35))
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    return bars


def animate(frames: Iterable[tuple[core.Point, core.Point, core.Patch, core.Scalar]]) -> Iterator[np.ndarray]:
    """Yields side-by-side RGB frames of the deforming cube and the face-area and volume bars."""
    states = iter(frames)
    first = next(states)
    initial_view = view(*first[:3])
    with charts.style():
        figure, (cube_ax, bar_ax) = charts.subplots(600, 288, ncols=2, layout="constrained")
        bars = measures(bar_ax, *first[2:])
    cube_ax.set_axis_off()
    camera_view = fitted(initial_view._replace(half=initial_view.half * view_room), cube_ax)
    image = show(cube_ax, np.ones((*camera_view.shape, 3)), camera_view)
    for vertices, centres, patches, volume in chain([first], states):
        image.set_data(picture(vertices, centres, patches, camera_view))
        for bar, height in zip(bars, heights(patches, volume)):
            bar.set_height(height)
        yield capture(figure)
    plt.close(figure)
