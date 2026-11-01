"""Drawing for the stressed cube: the cube seen from each frame with its face tractions, on the GPU, and Mohr's
circles.

The deformed cube is a translucent solid, its edges dark rods, and the undeformed cube a ghost of thin pale rods; the normal traction on each face is a blue arrow from the face's centre, the shear a red one, and the principal
directions are thin green rods through the centre."""

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
from examples.util.wgsl import fitted, show, simplices, solids
from examples.util.wgsl.camera import Camera, flat_panel
from examples.util.wgsl.meshes import Samples
from examples.util.wgsl.scene import Scene

# The cube's edges and faces, by the corners' indices: corner 4 i + 2 j + k sits at the i-th, j-th and
# k-th of -0.5 and 0.5 along x, y and z.
EDGES = np.array([(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)])
FACES = np.array([(4, 5, 7, 6), (0, 2, 3, 1), (2, 6, 7, 3), (0, 1, 5, 4), (1, 3, 7, 5), (0, 4, 6, 2)])
# The arrows' lengths per unit traction, and the colours of the frames' diameters on Mohr's circle.
NORMAL_SCALE, SHEAR_SCALE = 0.4, 0.5
COLOURS = ("#0284c7", "#10b981", "#e11d48")
# Mohr's circles: the outer one's colour and the inner ones' and the cross's, the samples round each circle, and the
# inner circles' dots, a dot and a gap, in pixels.
outer_colour, inner_colour = np.array(to_rgb("#1e293b")), np.array(to_rgb("#94a3b8"))
n_circle = 181
circle_dots = (1.5, 2.0)
# The principal directions are drawn out to this distance from the centre, just past the cube; the room the view of
# a turning frame leaves around the first frame's cube and arrows, for them to move in.
REACH = 0.75
TURNING_ROOM = 0.05
# The colours of the cube, its edges, the undeformed cube, the tractions and the principal directions.
cube_colour, edge_colour, ghost_colour = (np.array(to_rgb(colour)) for colour in ("#38bdf8", "#0f172a", "#94a3b8"))
normal_colour, shear_colour, principal_colour = (np.array(to_rgb(colour)) for colour in ("#0284c7", "#e11d48", "#10b981"))
background = np.ones(3)
# The view, elevation and azimuth in radians, azimuth zero looking along +y, and the margin around the cubes as a
# share of the larger side.
elevation, azimuth = np.radians([22.0, 25.0 + 90.0])
margin = 0.14
# How much of what lies behind them each of the cube's faces hides, and how strongly the undeformed cube and the
# principal directions show over the white page; the radii of the
# edges, the undeformed cube's edges and the principal directions, and the arrows' width with the length of the strokes
# of the chevrons at their tips, in pixels.
face_opacity, ghost_opacity, principal_opacity = 0.14, 0.7, 0.8
ghost_tint, principal_tint = (opacity * colour + (1 - opacity) for opacity, colour in
                              ((ghost_opacity, ghost_colour), (principal_opacity, principal_colour)))
edge_radius, ghost_radius, principal_radius = 0.9, 0.6, 0.6
arrow_width, chevron_length = 2.2, 8.8
# The turning animation's radii and widths as a share of those above, its picture that much narrower than a frame's.
turning_thickness = 0.62


def view(corners: core.Point, centres: core.Point, normal: core.Point, shear: core.Point,
         principal: core.Point, room: float) -> solids.View:
    """A view framing every cube, arrow and principal direction given, with `room` to spare around each."""
    points = [core.corners, corners, centres + normal * NORMAL_SCALE, centres + shear * SHEAR_SCALE,
              REACH * principal]
    # The projective embedding: places of space, offsets from the cube's centre, as the renderer's points about its origin.
    framed = [solids.origin + solids.context.include_from_algebra(batch) for batch in points]
    return solids.view(framed, room, elevation, azimuth, margin, 1)


def arrows(starts: core.Point, offsets: core.Point, colour: np.ndarray, view: solids.View,
           thickness: float) -> Samples:
    """Arrows from points along offsets `[arrows]`, tubes ending in chevrons for `solids.draw` to compose among the
    solids, their sizes a share `thickness` of the ones set above; those shorter than a pixel, a traction that
    vanishes, left out."""
    lengths = offsets.norm()                                                   # [arrows] Scalar+
    shown = lengths > view.pixel
    tails = solids.origin + solids.context.include_from_algebra(starts[shown])         # [shown] Point-
    tips = tails + solids.context.include_from_algebra(offsets[shown])                 # [shown] Point-
    n_shown = tails.shape[0]
    return simplices.tubes(stack((tails, tips), axis=-1).reshape(-1), simplices.Topology.chains([2] * n_shown), colour,
                           thickness * arrow_width, view, gloss=solids.gloss, chevrons=np.ones(n_shown),
                           chevron=thickness * chevron_length)


def ghost(view: solids.View, thickness: float) -> list[tuple[solids.Quadric, np.ndarray]]:
    """The undeformed cube, a ghost of thin pale rods, their radius a share `thickness` of the one set above."""
    corners = solids.origin + solids.context.include_from_algebra(core.corners)          # [corners] Point-
    edges, joints = solids.edges(corners[EDGES[:, 0]], corners[EDGES[:, 1]], thickness * ghost_radius * view.pixel)
    return [(edges, ghost_tint), (joints, ghost_tint)]


def cube(corners: core.Point, centres: core.Point, normal: core.Point, shear: core.Point, principal: core.Point,
         view: solids.View, thickness: float) -> tuple[list[tuple[solids.Quadric, np.ndarray]],
                                                       list[tuple[solids.Quadric, np.ndarray, float]],
                                                       list[tuple[Samples, float]]]:
    """The deformed cube seen from one frame, its edges solid and its faces translucent, the normal traction on each
    face in blue and the shear in red, and the principal directions in green; the radii and widths a share
    `thickness` of those set above."""
    # The projective embedding: places of space, offsets from the cube's centre, as the renderer's points about its origin.
    places = solids.origin + solids.context.include_from_algebra(corners)                # [corners] Point-
    directions = solids.context.include_from_algebra(REACH * principal)                  # [3] Point-
    tractions = [(arrows(centres, normal * NORMAL_SCALE, normal_colour, view, thickness), 1.0),
                 (arrows(centres, shear * SHEAR_SCALE, shear_colour, view, thickness), 1.0)]
    edges, joints = solids.edges(places[EDGES[:, 0]], places[EDGES[:, 1]], thickness * edge_radius * view.pixel)
    # Each principal direction a rod through the centre, both ways.
    principal_rods = solids.rods(solids.origin - directions, solids.origin + directions,
                                 thickness * principal_radius * view.pixel)
    parts = [(edges, edge_colour), (joints, edge_colour), (principal_rods, principal_tint)]
    body = solids.polyhedra(places, FACES)[None]                              # [1, 6] Plane+
    return parts, [(body, cube_colour, face_opacity)], tractions


def draw_cube(corners: core.Point, centres: core.Point, normal: core.Point, shear: core.Point, principal: core.Point,
              view: solids.View) -> np.ndarray:
    """The cube seen from one frame: the undeformed cube a ghost, the deformed one translucent, the normal
    traction on each face in blue and the shear in red, and the principal directions in green."""
    parts, translucent, tractions = cube(corners, centres, normal, shear, principal, view, 1.0)
    return solids.draw(ghost(view, 1.0) + parts, view, background, inks=[], translucent=translucent, meshes=tractions)


def mohr_place(normal: core.Scalar, shear: core.Scalar) -> core.Point:
    """A place on Mohr's diagram, its normal traction across, along x, and its shear up, along y."""
    x, y, _ = core.axes
    return x.dual() * normal + y.dual() * shear


def draw_circles(ax, values: core.Scalar) -> None:
    """The frame of Mohr's circles of the principal stresses, on axes framing the largest: its limits and names."""
    stresses = np.sort(values.to_array())
    # The outer two outside the circle that passes through them, the middle one below its dot.
    for stress, name, shift, align in zip(stresses, ["$\\sigma_1$", "$\\sigma_2$", "$\\sigma_3$"], (-5, 0, 5),
                                          ("right", "center", "left")):
        ax.annotate(name, (stress, 0), xytext=(shift / 2, -2), textcoords="offset points", ha=align, va="top",
                    color="#1e293b")
    centre, radius = (stresses[0] + stresses[2]) / 2, (stresses[2] - stresses[0]) / 2
    margin = 1.3 * radius
    ax.set_xlim(centre - margin, centre + margin)
    ax.set_ylim(-margin, margin)
    ax.text(centre + 0.95 * margin, 0.04, "$\\sigma$", ha="right", va="bottom", color="#64748b")
    ax.text(0.04, 0.95 * margin, "$\\tau$", ha="left", va="top", color="#64748b")
    ax.set_aspect("equal")
    ax.set_axis_off()


def circles_drawn(ax, values: core.Scalar, camera: Camera) -> list[np.ndarray]:
    """Mohr's circles of the principal stresses, the outer one at a chart's line width and the inner two dotted as
    thin as its frame, under the cross through the origin and the principal stresses' dots: their pictures."""
    x, _, _ = core.axes
    stresses = values[values.argsort()]                                       # [3] Scalar+
    # Each circle between two principal stresses, the outer one first: about their mean on the axis of normal
    # traction, out to half their difference, turned all the way round in the plane of the diagram.
    lows, highs = stresses[[0, 0, 1]], stresses[[2, 1, 2]]                    # [circles] Scalar+, [circles] Scalar+
    turns = (core.mv.xy * (-np.linspace(0, 2 * np.pi, n_circle) / 2)).exp()  # [samples] Rotor+
    rims = x.dual() * ((lows + highs) / 2)[:, None] + (turns >> x.dual() * ((highs - lows) / 2)[:, None])   # [circles, samples] Point-
    (left, right), (bottom, top) = ax.get_xlim(), ax.get_ylim()
    cross = mohr_place(np.array([left, right, 0, 0]), np.array([0, 0, bottom, top]))   # [4] Point-
    return [simplices.strokes(rims[0], simplices.Topology.chains([n_circle]), outer_colour, charts.LINE_PX, camera),
            simplices.strokes(rims[1:].reshape(-1), simplices.Topology.chains([n_circle] * 2), inner_colour,
                              charts.FRAME_PX, camera, dashes=circle_dots, cap=simplices.Cap.BUTT),
            simplices.strokes(cross, simplices.Topology.chains([2, 2]), inner_colour, charts.FRAME_PX, camera,
                              cap=simplices.Cap.BUTT),
            simplices.markers(x.dual() * stresses, outer_colour, charts.MARKER_PX / 2, camera)]


def diameter(normal: core.Point, shear: core.Point) -> core.Point:
    """The ends of a frame's diameter on Mohr's diagram, from the normal and shear traction on its faces `[6]`, read by
    pairing: the face facing x, its normal traction along x and its shear along y, and the face facing y, with the
    opposite shear, `[2] Point-`."""
    x, y, _ = core.axes
    sliding = y & shear[0]                                                    # [] Scalar+
    return stack((mohr_place(x & normal[0], sliding), mohr_place(y & normal[1], -sliding)))


def diameter_drawn(ends: core.Point, colour: str, camera: Camera) -> list[np.ndarray]:
    """A frame's diameter between its ends `[2]`, the face facing x a filled dot and the face facing y a ring: their
    pictures."""
    ink = np.array(to_rgb(colour))
    rim = charts.LINE_PX / 2
    # The ring a white dot over a larger one in the diameter's colour.
    return [simplices.strokes(ends, simplices.Topology.chains([2]), ink, charts.LINE_PX, camera),
            simplices.markers(ends[:1], ink, charts.MARKER_PX / 2, camera),
            simplices.markers(ends[1:], ink, charts.MARKER_PX / 2 + rim, camera),
            simplices.markers(ends[1:], np.ones(3), charts.MARKER_PX / 2 - rim, camera)]


@charts.style()
def draw_frames(views: tuple[core.Point, core.Point, core.Point, core.Point, core.Point],
                values: core.Scalar) -> plt.Figure:
    """The cube seen from each frame `[frames]`, all from one camera, and Mohr's circles with each frame's
    diameter."""
    corners, centres, normal, shear, principal = views
    count = corners.shape[0]
    # Nearly the whole column: a square panel a frame's cube, and Mohr's circles beside them.
    figure, panels = charts.subplots(190 * (count + 1), 190, ncols=count + 1)
    cubes = panels[:count]
    circles = panels[count]
    draw_circles(circles, values)
    # The figure laid out with the cubes' axes bare, as their pictures show them; their rooms alike, one view fitted
    # to the first holds them all.
    for ax in cubes:
        ax.set_axis_off()
    figure.tight_layout()
    seen = fitted(view(*views, 0.0), cubes[0])
    for frame, ax in enumerate(cubes):
        show(ax, draw_cube(corners[frame], centres[frame], normal[frame], shear[frame], principal[frame], seen), seen)
    # Mohr's circles, and each frame's diameter in its colour.
    camera = flat_panel(circles, core.ga)
    simplices.overlay(circles, circles_drawn(circles, values, camera) + [
        picture for frame, colour in zip(range(count), COLOURS * count)
        for picture in diameter_drawn(diameter(normal[frame], shear[frame]), colour, camera)])
    return figure


def animate_turning(values: core.Scalar, frames: Iterable[tuple[core.Point, core.Point, core.Point, core.Point,
                                                                   core.Point]]) -> Iterator[np.ndarray]:
    """Each frame of the cube seen from a turning frame, all from one camera framing the first with room around it,
    and its diameter on Mohr's circles; the undeformed cube and the circles set up once."""
    frames = iter(frames)
    first = next(frames)
    # Three fifths of the column: the cube beside Mohr's circles, which are drawn once under the charts' style; only
    # the cube's picture and the diameter change between frames.
    with charts.style():
        figure, (cube_ax, circles) = charts.subplots(480, 230, ncols=2)
        seen = fitted(view(*first, TURNING_ROOM), cube_ax)
        scene = Scene(seen, background, ghost(seen, turning_thickness), [], inks=[], translucent=[])
        picture = show(cube_ax, np.zeros((*seen.shape, 3), dtype=np.uint8), seen)
        draw_circles(circles, values)
        camera = flat_panel(circles, core.ga)
        backdrop = circles_drawn(circles, values, camera)
        diagram = simplices.overlay(circles, backdrop)
    for corners, centres, normal, shear, principal in chain([first], frames):
        parts, translucent, tractions = cube(corners, centres, normal, shear, principal, seen, turning_thickness)
        picture.set_data(scene.draw(parts, inks=[], translucent=translucent, meshes=tractions))
        diagram.set_data(simplices.composited(backdrop + diameter_drawn(diameter(normal, shear), COLOURS[0], camera)))
        yield capture(figure)
    plt.close(figure)
