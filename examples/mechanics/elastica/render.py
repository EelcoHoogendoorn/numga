"""Drawing for the elastica: the curled strip from straight to a double ring, Euler's elastica between their
pins on the line of thrust, the sweep through them, and Kirchhoff's helices with a ribbon showing the twist of
the cross-sections."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.util.animation import capture
from examples.mechanics.elastica import core
from examples.util.wgsl import image_figure, meshes, simplices, solids
from examples.util.wgsl.camera import Camera, panel

mv = core.mv
# The colours of the rods by case, of the line of thrust and the pins, and of the ribbon; the margin around
# the drawn rods, the spacing of the helices as a share of the widest, and their view as (elevation, azimuth), in
# radians, azimuth zero looking along +y.
ROD_MAP = "viridis"
ROD, THRUST, RIBBON = np.array(to_rgb("#1f6f99")), np.array(to_rgb("0.55")), np.array(to_rgb("#7cc4e4"))
PIN = np.array(to_rgb("0.2"))
MARGIN = 0.08
SPACING = 1.4
VIEW = np.radians([12.0, 10.0])
# The uncoiling wire's view: its elevation, and the azimuths the camera turns through over the frames.
UNCOIL_ELEVATION = np.radians(16.0)
UNCOIL_AZIMUTHS = np.radians([-30.0, 150.0])
# The telephone cord's view as (elevation, azimuth), and the colour of its clamps.
CORD_VIEW = np.radians([20.0, 26.0])
CLAMP = np.full(3, 0.2)
# The pixels across the helices, the uncoiling wire and the cord; the width of a rod in pixels, the radius of a
# clamp, and the background.
N_HELIX_PIXELS, N_UNCOIL_PIXELS, N_CORD_PIXELS = 330, 160, 675
ROD_WIDTH, CLAMP_RADIUS = 3.0, 5.0
BACKGROUND = np.ones(3)
# In the plane of bending, in pixels: the widths of the rods, of the clamp and of the line of thrust, and the pins'
# radius.
PLANE_ROD_WIDTH, PLANE_CLAMP_WIDTH, THRUST_WIDTH, PIN_RADIUS = 1.7, 3.0, 1.0, 3.0
# The points around each rim of the cylinder that frames the uncoiling wire: a multiple of four, so that the rims'
# points across and along every view lie among them.
N_FRAMING_RIM = 4


# --- plumbing -------------------------------------------------------------------------
def extent(points: core.Point) -> tuple[np.ndarray, np.ndarray]:
    """The lowest and the highest readings of points in the plane of bending, along the unbent rod and up `[2]` each:
    their pairings with the planes z and y over their weight."""
    points = points.reshape(-1)                                              # [points] Point-
    readings = (stack((mv.z, mv.y))[:, None] & points) / (mv.w & points)      # [2, points] Scalar+
    return readings.min(axis=-1).to_array(), readings.max(axis=-1).to_array()


def frame_plane(ax: plt.Axes, low: np.ndarray, high: np.ndarray) -> Camera:
    """The panel of the plane of bending, framed about the extent from `low` to `high`, and its camera, reading points
    along the unbent rod and up by the planes z and y."""
    ax.set(xlim=(low[0] - MARGIN, high[0] + MARGIN), ylim=(low[1] - MARGIN, high[1] + MARGIN), aspect="equal")
    ax.set_axis_off()
    return panel(ax, mv.z, mv.y)


def draw_curls(centres: core.Point) -> plt.Figure:
    """The strip curled by each couple `[curls, pieces + 1]`, from straight, darkest, to a double ring, with
    its clamp."""
    n_curls, n_points = centres.shape
    colours = plt.get_cmap(ROD_MAP)(np.linspace(0.05, 0.9, n_curls))[:, :3]
    figure, ax = plt.subplots(figsize=(7.5, 4.0), dpi=55)
    ax.set_axis_off()
    figure.tight_layout()
    camera = frame_plane(ax, *extent(centres))
    # The clamp across the whole panel, along the plane z = 0 where the strip is held.
    clamp = mv.point([[0, height, 0] for height in ax.get_ylim()])           # [ends] Point-
    simplices.overlay(ax, [
        simplices.strokes(centres.reshape(-1), simplices.Mesh.chains([n_points] * n_curls),
                          np.repeat(colours, n_points - 1, axis=0), PLANE_ROD_WIDTH, camera),
        simplices.strokes(clamp, simplices.Mesh.chains([2]), PIN, PLANE_CLAMP_WIDTH, camera, cap=simplices.Cap.BUTT)])
    return figure


def rod_pictures(curve: core.Point, ax: plt.Axes, camera: Camera) -> list[np.ndarray]:
    """One of Euler's elastica `[pieces + 1]`: the line of thrust across the panel, the rod, and its pins at its two
    ends."""
    thrust = mv.point([[0, 0, along] for along in ax.get_xlim()])            # [ends] Point-
    return [simplices.strokes(thrust, simplices.Mesh.chains([2]), THRUST, THRUST_WIDTH, camera),
            simplices.strokes(curve, simplices.Mesh.chains([len(curve)]), ROD, PLANE_ROD_WIDTH, camera),
            simplices.markers(curve[[0, -1]], PIN, PIN_RADIUS, camera)]


def animate_sweep(centres: core.Point) -> Iterator[np.ndarray]:
    """Euler's elastica as the end angle grows `[frames, pieces + 1]`, all to one scale, in one figure whose rod and
    pins move."""
    low, high = extent(centres)
    figure, ax = plt.subplots(figsize=(5.0, 5.0 * (high[1] - low[1] + 2 * MARGIN) / (high[0] - low[0] + 2 * MARGIN)),
                              dpi=60)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    camera = frame_plane(ax, low, high)
    rod = simplices.overlay(ax, rod_pictures(centres[0], ax, camera))
    for curve in centres:
        rod.set_data(simplices.composited(rod_pictures(curve, ax, camera)))
        yield capture(figure)
    plt.close(figure)


def draw_helices(centres: core.Point, edges: core.Point) -> plt.Figure:
    """The wires coiled by their couples `[tilts, pieces + 1]`, side by side in one view of space, each with a
    ribbon across its cross-sections `[tilts, pieces + 1, 2]` showing how they twist."""
    # The wires set side by side across, each a little more than the widest wire apart.
    readings = stack((mv.x, mv.y))[None, :, None] & edges.reshape(centres.shape[0], -1)[:, None]
    spans = readings.max(axis=-1) - readings.min(axis=-1)
    spacing = SPACING * spans.max()
    across = mv.x.dual() * (spacing * np.arange(centres.shape[0]))           # [tilts] Point-
    curves, ribbons = centres + across[:, None], edges + across[:, None, None]
    view = solids.view([ribbons], 0.0, *VIEW, MARGIN, N_HELIX_PIXELS)
    return image_figure(scene(curves, ribbons, view))


def animate_uncoiling(centres: core.Point, edges: core.Point) -> Iterator[np.ndarray]:
    """The wire through the sweep of its tilt `[frames, pieces + 1]`, with its ribbon `[frames, pieces + 1, 2]`,
    the camera turning slowly around it, all to one scale."""
    places = edges.reshape(-1)                                               # [points] Point-
    frame = stack((mv.x, mv.y, mv.z))
    readings = frame[:, None] & places
    low = frame & places[readings.argmin(axis=-1)]
    high = frame & places[readings.argmax(axis=-1)]
    # The upright cylinder about the coil's axis holding every frame, seen alike from every azimuth: its axis the
    # vertical line through the middle of every frame's extent, its radius the farthest any point lies from that
    # line, the size of the plane through the two, and its rims at the bottom and the top of the extent.
    middle = mv.w.dual() + (frame.dual() * ((low + high) / 2)).sum()           # [] Point-
    axis = (middle & mv.z.dual()).normalized()                               # [] Line-
    radii = (axis & edges).norm().reshape(-1)
    radius = radii.max()
    half_height = (high - low)[2] / 2
    rims = middle + mv.z.dual() * (half_height * np.array([-1.0, 1.0]))      # [2] Point-
    # Each rim the directions across, turned a full turn in the level plane, as far from its middle as the radius.
    turns = (mv.xy * (-np.linspace(0, 2 * np.pi, N_FRAMING_RIM, endpoint=False) / 2)).exp()   # [rim] Rotor+
    circles = rims[:, None] + (turns >> mv.x.dual() * radius)                # [2, rim] Point-
    framing = solids.view([circles], 0.0, UNCOIL_ELEVATION, 0.0, MARGIN, N_UNCOIL_PIXELS)
    azimuths = np.linspace(*UNCOIL_AZIMUTHS, centres.shape[0], endpoint=False)
    for curve, ribbon, azimuth in zip(centres, edges, azimuths):
        yield scene(curve[None], ribbon[None], framing._replace(camera=solids.aim(middle, UNCOIL_ELEVATION, azimuth)))


def animate_cord(centres: core.Point, edges: core.Point) -> Iterator[np.ndarray]:
    """The telephone cord at each frame, as `sized_cord` draws it `N_CORD_PIXELS` across."""
    return sized_cord(centres, edges, N_CORD_PIXELS)


def sized_cord(centres: core.Point, edges: core.Point, pixels: int) -> Iterator[np.ndarray]:
    """The telephone cord at each frame `[frames, drawn]`, as a ribbon `[frames, drawn, 2]` with its base and its
    handset, `pixels` across, all from one fixed view and to one scale."""
    view = solids.view([edges], 0.0, *CORD_VIEW, MARGIN, pixels)
    # The clamps at the base and the handset: small balls at the cord's two ends.
    clamps = solids.ball(centres[:, [0, -1]], CLAMP_RADIUS * view.pixel)[..., None]   # [frames, 2, 1] Plane+ <- Point-
    for curve, ribbon, ends in zip(centres, edges, clamps):
        yield solids.draw([(ends, CLAMP)], view, BACKGROUND, inks=[], translucent=[],
                          meshes=wires(curve[None], ribbon[None], view))


def wires(curves: core.Point, ribbons: core.Point, view: solids.View) -> list[tuple[meshes.Samples, float]]:
    """Rods along curves `[rods, n]` as strokes, through their ribbons `[rods, n, 2]`: strips between the edges, two
    triangles to each piece of rod; each opaque, sample by sample."""
    index = np.arange(np.prod(ribbons.shape)).reshape(ribbons.shape)         # [rods, n, 2]
    corners = np.stack([index[:, :-1, 0], index[:, 1:, 0], index[:, 1:, 1], index[:, :-1, 1]], axis=-1)
    faces = corners[..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)
    edges = ribbons.reshape(-1)                                              # [vertices] Point-
    strips = meshes.samples(edges, faces, RIBBON, meshes.smooth(edges, meshes.Mesh(len(edges), faces)), view, face_values=np.zeros(4),
                            colour=meshes.plain, gloss=solids.gloss)
    rods = simplices.tubes(curves.reshape(-1), simplices.Mesh.chains([curves.shape[1]] * curves.shape[0]), ROD,
                           ROD_WIDTH, view, gloss=solids.gloss)
    return [(strips, 1.0), (rods, 1.0)]


def scene(curves: core.Point, ribbons: core.Point, view: solids.View) -> np.ndarray:
    """Rods and their ribbons, as `wires`, on the background, `[rows, columns, 3]` bytes."""
    return solids.draw([], view, BACKGROUND, inks=[], translucent=[], meshes=wires(curves, ribbons, view))
