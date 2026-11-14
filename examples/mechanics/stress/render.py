"""GPU cube views and Mohr circle diagrams draw the deformed cube, face tractions, and principal
axes across reference frames.

The deformed cube renders as a translucent blue solid with dark edge rods inside a pale wireframe of
the undeformed cube; blue arrows show normal tractions, red arrows show shear tractions, and green
rods mark the principal directions through the cube centre."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.util import charts
from examples.util.animation import capture
from examples.mechanics.stress import core
from numga.topology import Topology
from numga.render import Camera, drawing, Scene, simplices, solids
from examples.util.panels import fitted, offset_panel, overlay, show
from numga.render.drawing import Draw

# Cube edge pairs and outward quad faces indexed into `core.corners`, where corner `4*i + 2*j + k` sits at `(-0.5, 0.5)` along x, y, z.
edge_pairs = np.array([(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)])
face_quads = np.array([(4, 5, 7, 6), (0, 2, 3, 1), (2, 6, 7, 3), (0, 1, 5, 4), (1, 3, 7, 5), (0, 4, 6, 2)])
# Arrow lengths per unit traction and frame diameter colours on Mohr's diagram.
normal_scale, shear_scale = 0.4, 0.5
frame_colours = ("#0284c7", "#10b981", "#e11d48")
# Mohr circle inks, sample count around each circle, and inner-circle dot-gap dash pattern in pixels.
outer_colour, inner_colour = np.array(to_rgb("#1e293b")), np.array(to_rgb("#94a3b8"))
n_samples = 180
circle_dots = (1.5, 2.0)
# Half-length of principal-axis rods past the cube, and extra framing room for the rotating animation.
principal_reach = 0.75
turning_room = 0.05
# Colours of the deformed cube, its edges, the undeformed wireframe, the tractions, and the principal rods.
cube_colour, edge_colour, ghost_colour = (np.array(to_rgb(colour)) for colour in ("#38bdf8", "#0f172a", "#94a3b8"))
normal_colour, shear_colour, principal_colour = (np.array(to_rgb(colour)) for colour in ("#0284c7", "#e11d48", "#10b981"))
background = np.ones(3)
# Camera elevation and azimuth in radians, plus relative margin around the framed geometry.
elevation, azimuth = np.radians([22.0, 25.0 + 90.0])
margin = 0.14
# Opacities, white-blended tints, rod radii, and arrow stroke widths and chevron lengths in pixels.
face_opacity, ghost_opacity, principal_opacity = 0.14, 0.7, 0.8
ghost_tint, principal_tint = (opacity * colour + (1 - opacity) for opacity, colour in
                              ((ghost_opacity, ghost_colour), (principal_opacity, principal_colour)))
edge_radius, ghost_radius, principal_radius = 0.9, 0.6, 0.6
arrow_width, chevron_length = 2.2, 8.8
# Stroke-width scale for the narrower panel of the rotating-frame animation.
turning_thickness = 0.62


def cube_view(corners: core.Point, centres: core.Point, normal: core.Point, shear: core.Point,
              principal: core.Point, room: float) -> solids.View:
    """Camera view framing the undeformed and deformed cubes, scaled traction arrows, and principal
    rods with `room` to spare."""
    points = [core.corners, corners, centres + normal * normal_scale, centres + shear * shear_scale,
              principal_reach * principal]
    # Embed Euclidean offsets from the cube centre as projective points around `solids.origin`.
    framed = [solids.origin + solids.context.include_from_algebra(batch) for batch in points]
    return solids.view(framed, room, elevation, azimuth, margin, 1)


def traction_arrows(starts: core.Point, offsets: core.Point, colour: np.ndarray, view: solids.View,
                    thickness: float) -> Draw:
    """Chevron-tipped tube arrows from `starts` (`[6] Point-`) along `offsets` (`[6] Point-`),
    omitting arrows shorter than one pixel."""
    lengths = offsets.norm()                                                   # [6] Scalar+
    visible = lengths > view.pixel
    tails = solids.origin + solids.context.include_from_algebra(starts[visible])   # [...] Point-
    tips = tails + solids.context.include_from_algebra(offsets[visible])       # [...] Point-
    n_visible = tails.shape[0]
    return simplices.tubes(stack((tails, tips), axis=-1).reshape(-1), Topology.chains([2] * n_visible), colour,
                           thickness * arrow_width, view, gloss=solids.gloss, chevrons=np.ones(n_visible),
                           chevron=thickness * chevron_length)


def ghost_cube(view: solids.View, thickness: float) -> list[tuple[solids.Quadric, np.ndarray]]:
    """Pale wireframe rods and corner joints of the undeformed unit cube."""
    corners = solids.origin + solids.context.include_from_algebra(core.corners)   # [8] Point-
    edges, joints = solids.edges(corners[edge_pairs[:, 0]], corners[edge_pairs[:, 1]],
                                 thickness * ghost_radius * view.pixel)
    return [(edges, ghost_tint), (joints, ghost_tint)]


def deformed_cube(corners: core.Point, centres: core.Point, normal: core.Point, shear: core.Point,
                  principal: core.Point, view: solids.View,
                  thickness: float) -> tuple[list[tuple[solids.Quadric, np.ndarray]],
                                             list[tuple[solids.Quadric, np.ndarray, float]],
                                             list[tuple[Draw, float]]]:
    """Opaque edge and principal rods, translucent polyhedron faces, and normal/shear traction
    arrows for one cube frame."""
    places = solids.origin + solids.context.include_from_algebra(corners)      # [8] Point-
    directions = solids.context.include_from_algebra(principal_reach * principal)   # [3] Point-
    tractions = [(traction_arrows(centres, normal * normal_scale, normal_colour, view, thickness), 1.0),
                 (traction_arrows(centres, shear * shear_scale, shear_colour, view, thickness), 1.0)]
    edges, joints = solids.edges(places[edge_pairs[:, 0]], places[edge_pairs[:, 1]],
                                 thickness * edge_radius * view.pixel)
    # Draw each principal axis as a rod passing both ways through the cube centre.
    principal_rods = solids.rods(solids.origin - directions, solids.origin + directions,
                                 thickness * principal_radius * view.pixel)
    parts = [(edges, edge_colour), (joints, edge_colour), (principal_rods, principal_tint)]
    body = solids.polyhedra(places, face_quads)[None]                          # [1, 6] Plane+
    return parts, [(body, cube_colour, face_opacity)], tractions


def cube_image(corners: core.Point, centres: core.Point, normal: core.Point, shear: core.Point,
               principal: core.Point, view: solids.View) -> np.ndarray:
    """Rendered RGB image of one frame's deformed cube, ghost wireframe, tractions, and principal
    rods."""
    parts, translucent, tractions = deformed_cube(corners, centres, normal, shear, principal, view, 1.0)
    return solids.draw(ghost_cube(view, 1.0) + parts, view, background, inks=[], translucent=translucent,
                       meshes=tractions)


def mohr_place(normal: core.Scalar, shear: core.Scalar) -> core.Point:
    """Point on Mohr's diagram with normal traction along the first axis and shear traction along
    the second."""
    x, y, _ = core.axes
    return x.dual() * normal + y.dual() * shear


def mohr_axes(ax, stresses: core.Scalar) -> None:
    """Axis limits and principal-stress labels on `ax` framing the outer Mohr circle of `stresses`
    (`[3] Scalar+`)."""
    ordered = np.sort(stresses.to_array())
    # Place the outer two labels just outside the outer circle and the middle label below its marker.
    for stress, label, shift, align in zip(ordered, ["\u03c3\u2081", "\u03c3\u2082", "\u03c3\u2083"], (-5, 0, 5),
                                           ("right", "center", "left")):
        ax.annotate(label, (stress, 0), xytext=(shift / 2, -2), textcoords="offset points", ha=align, va="top",
                    color="#1e293b")
    centre, radius = (ordered[0] + ordered[2]) / 2, (ordered[2] - ordered[0]) / 2
    span = 1.3 * radius
    ax.set_xlim(centre - span, centre + span)
    ax.set_ylim(-span, span)
    ax.text(centre + 0.95 * span, 0.04, "\u03c3", ha="right", va="bottom", color="#64748b")
    ax.text(0.04, 0.95 * span, "\u03c4", ha="left", va="top", color="#64748b")
    ax.set_aspect("equal")
    ax.set_axis_off()


def mohr_backdrop(ax, stresses: core.Scalar, camera: Camera) -> list[np.ndarray]:
    """Rendered RGBA layers for the three Mohr circles of `stresses` (`[3] Scalar+`), the origin
    crosshairs, and the principal-stress markers."""
    x, _, _ = core.axes
    ordered = stresses[stresses.argsort()]                                     # [3] Scalar+
    # Sample the outer circle and the two inner circles by turning radii around their midpoints on the normal-stress axis.
    lows, highs = ordered[[0, 0, 1]], ordered[[2, 1, 2]]                       # [3] Scalar+, [3] Scalar+
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    turns = (core.mv.xy * (-angles / 2)).exp()                                 # [n_samples] Rotor+
    radii = x.dual() * ((highs - lows) / 2)[:, None]                           # [3, 1] Point-
    rims = x.dual() * ((lows + highs) / 2)[:, None] + (turns >> radii)         # [3, n_samples] Point-
    (left, right), (bottom, top) = ax.get_xlim(), ax.get_ylim()
    cross = mohr_place(np.array([left, right, 0, 0]), np.array([0, 0, bottom, top]))   # [4] Point-
    return [simplices.strokes(rims[0], Topology.loops([n_samples]), outer_colour, charts.LINE_PX, camera),
            simplices.strokes(rims[1:].reshape(-1), Topology.loops([n_samples] * 2), inner_colour,
                              charts.FRAME_PX, camera, dashes=circle_dots, cap=simplices.Cap.BUTT),
            simplices.strokes(cross, Topology.chains([2, 2]), inner_colour, charts.FRAME_PX, camera,
                              cap=simplices.Cap.BUTT),
            simplices.markers(x.dual() * ordered, outer_colour, charts.MARKER_PX / 2, camera)]


def frame_diameter(normal: core.Point, shear: core.Point) -> core.Point:
    """Endpoints (`[2] Point-`) of one frame's Mohr diameter read from the `+x` and `+y` faces of
    `normal` and `shear` (`[6] Point-`)."""
    x, y, _ = core.axes
    sliding = y & shear[0]                                                     # [] Scalar+
    return stack((mohr_place(x & normal[0], sliding), mohr_place(y & normal[1], -sliding)))


def diameter_layers(ends: core.Point, colour: str, camera: Camera) -> list[np.ndarray]:
    """Rendered RGBA layers for a diameter segment between `ends` (`[2] Point-`), with a filled dot
    for the `+x` face and an open ring for the `+y` face."""
    ink = np.array(to_rgb(colour))
    rim = charts.LINE_PX / 2
    return [simplices.strokes(ends, Topology.chains([2]), ink, charts.LINE_PX, camera),
            simplices.markers(ends[:1], ink, charts.MARKER_PX / 2, camera),
            simplices.markers(ends[1:], ink, charts.MARKER_PX / 2 + rim, camera),
            simplices.markers(ends[1:], np.ones(3), charts.MARKER_PX / 2 - rim, camera)]


@charts.style()
def frames_figure(views: tuple[core.Point, core.Point, core.Point, core.Point, core.Point],
                  stresses: core.Scalar) -> plt.Figure:
    """Side-by-side panels of the deformed cube in each frame of `views` followed by Mohr's circles
    with each frame's diameter."""
    corners, centres, normal, shear, principal = views
    n_views = corners.shape[0]
    figure, panels = charts.subplots(190 * (n_views + 1), 190, ncols=n_views + 1)
    cube_panels = panels[:n_views]
    circles = panels[n_views]
    mohr_axes(circles, stresses)
    for ax in cube_panels:
        ax.set_axis_off()
    figure.tight_layout()
    shared_view = fitted(cube_view(*views, 0.0), cube_panels[0])
    for idx, ax in enumerate(cube_panels):
        show(ax, cube_image(corners[idx], centres[idx], normal[idx], shear[idx], principal[idx], shared_view),
             shared_view)
    x, y, _ = core.axes
    camera = offset_panel(circles, x, y)
    overlay(circles, mohr_backdrop(circles, stresses, camera) + [
        layer for idx, colour in zip(range(n_views), frame_colours * n_views)
        for layer in diameter_layers(frame_diameter(normal[idx], shear[idx]), colour, camera)])
    return figure


def turning_frames(stresses: core.Scalar,
                   states: Iterable[tuple[core.Point, core.Point, core.Point, core.Point, core.Point]]
                   ) -> Iterator[np.ndarray]:
    """Yields RGB frames of the deformed cube and its rotating diameter on Mohr's circles across
    `states`, reusing a single GPU scene for the ghost cube and static circle backdrop."""
    iterator = iter(states)
    first = next(iterator)
    x, y, _ = core.axes
    with charts.style():
        figure, (cube_ax, circles) = charts.subplots(480, 230, ncols=2)
        shared_view = fitted(cube_view(*first, turning_room), cube_ax)
        scene = Scene(shared_view, background, ghost_cube(shared_view, turning_thickness), [], inks=[], translucent=[])
        picture = show(cube_ax, np.zeros((*shared_view.shape, 3), dtype=np.uint8), shared_view)
        mohr_axes(circles, stresses)
        camera = offset_panel(circles, x, y)
        backdrop = mohr_backdrop(circles, stresses, camera)
        diagram = overlay(circles, backdrop)
    for corners, centres, normal, shear, principal in chain([first], iterator):
        parts, translucent, tractions = deformed_cube(corners, centres, normal, shear, principal, shared_view,
                                                      turning_thickness)
        picture.set_data(scene.draw(parts, inks=[], translucent=translucent, meshes=tractions))
        diagram.set_data(drawing.composited(backdrop + diameter_layers(frame_diameter(normal, shear), frame_colours[0],
                                                                       camera)))
        yield capture(figure)
    plt.close(figure)
