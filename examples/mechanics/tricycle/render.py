"""Drawing for the tricycle: the plane from above and space in perspective, with the tricycle's frame and
wheels, each axle drawn on to the turning centre, and the trails of the middle of the rear axle and of the
front wheel.

The geometry of the drawing is written once for both algebras, on the points of whichever the drive is in; the
plane is read out for matplotlib by pairing, and space, in the GPU library's own algebra, is passed as it is."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import Extensor, concatenate, stack
from examples.util.animation import capture
from examples.util.wgsl import meshes, solids, strokes

FRAME = "#2c3e50"
WHEEL = "#222222"
AXLE = "#c0392b"
CENTRE = "#c0392b"
TRAIL = "#7f8c8d"
FRONT_TRAIL = "#2980b9"
WHEEL_RADIUS = 0.25
# Half the width of the view, which follows the middle of the rear axle.
VIEW = 3.0
BACKGROUND = np.ones(3)
# The plane view: the width in points of the axles, drawn dashed, and the lengths of their dashes and gaps in stroke
# widths.
AXLE_WIDTH, AXLE_DASHES = 0.9, (4.0, 3.0)
# Space: the view's elevation and azimuth in radians, azimuth zero looking along +y, the margin around the square
# of ground it follows as a share of the larger side, and the pixels across it; the length of a dash along an axle
# and of the gap after it, and the points around a rim; the widths in pixels of the axles' dashes, the frame, the
# post, the rims and the trail.
ELEVATION, AZIMUTH = np.radians(35.0), np.radians(25.0)
MARGIN, N_PIXELS = 0.04, 300
DASH = 0.12
N_RIM = 48
DASH_WIDTH, FRAME_WIDTH, POST_WIDTH, RIM_WIDTH, TRAIL_WIDTH = 2.0, 3.0, 2.0, 2.0, 2.0
# The corners of the square of ground the view of space follows, across x and y, as shares of its half width.
CORNERS = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])


# --- plumbing -------------------------------------------------------------------------
def ground(points: Extensor) -> np.ndarray:
    """The coordinates along x and y of points on the ground: their pairings with the planes x = 0 and
    y = 0 over their weights."""
    mv = points.context.multivector
    return ((stack((mv.x, mv.y)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def axles(contacts: Extensor, centre: Extensor) -> Extensor:
    """Where the axles are drawn from, `[axles]`: the rear wheels share one axle, drawn once from the rear wheel nearer
    the turning centre, a point of unit weight, and the front wheel has its own; every axle runs on to the centre."""
    # How far each rear wheel lies from the centre: the size of the line through them.
    nearer = (contacts[:2] & centre).norm().argmin()
    return contacts[[nearer, 2]]


def frame(contacts: Extensor) -> tuple[Extensor, Extensor]:
    """The tricycle's frame, from where to where `[bars]` each: across between the rear wheels, and from the middle of
    the rear axle to the front wheel."""
    middle = (contacts[0] + contacts[1]) / 2
    return stack([contacts[0], middle]), stack([contacts[1], contacts[2]])


def wheel_steps(contacts: Extensor, forward: Extensor) -> Extensor:
    """Each wheel's step along the way it rolls, as long as a wheel's radius: towards the point ahead of it, over the
    distance to that point, the size of the line through the two."""
    return (forward - contacts) * (WHEEL_RADIUS / (contacts & forward).norm())


def plane(contacts: Extensor, forward: Extensor, centre: Extensor, middle: Extensor) -> np.ndarray:
    """The plane from above, around the tricycle."""
    mv = centre.context.multivector
    # The turning centre, an extrinsic meet, as the intrinsic point it names, at unit weight.
    centre = (centre * mv.antipseudoscalar([1.0])).normalized()
    steps = wheel_steps(contacts, forward)
    extent = following(ground(middle))
    figure, ax = plt.subplots(figsize=(3, 3))
    ax.plot(*ground(middle).T, color=TRAIL, linewidth=0.6)
    for start in ground(axles(contacts, centre)):
        ax.plot(*np.stack([start, ground(centre)]).T, color=AXLE, linewidth=AXLE_WIDTH, linestyle=(0, AXLE_DASHES))
    ax.scatter(*ground(centre), color=CENTRE, s=11, zorder=4)
    for start, end in zip(*(ground(ends) for ends in frame(contacts))):
        ax.plot(*np.stack([start, end]).T, color=FRAME, linewidth=1.5)
    for start, end in zip(ground(contacts - steps), ground(contacts + steps)):
        ax.plot(*np.stack([start, end]).T, color=WHEEL, linewidth=3, solid_capstyle="round")
    ax.set(xlim=extent[:, 0], ylim=extent[:, 1], aspect="equal")
    ax.axis("off")
    figure.tight_layout()
    pixels = capture(figure)
    plt.close(figure)
    return pixels


def space(contacts: solids.Point, forward: solids.Point, centre: solids.Line, middle: solids.Point) -> np.ndarray:
    """Space around the tricycle, drawn on the GPU: the frame and the axles' dashes rods, the wheels' rims and the
    trail strokes, and the turning centre a vertical post, met with the ground where it is drawn. The view follows
    the middle of the rear axle."""
    mv = centre.context.multivector
    # Up from the ground, and where the turning centre meets the ground, as an intrinsic point of unit weight.
    up = mv.z.dual()                                                                     # [] Point-
    foot = ((centre ^ mv.z) * mv.antipseudoscalar([1.0])).normalized()                    # [] Point-
    view = space_view(middle[-1])
    # Each axle's dashes, at the height of the hubs: stretches of one length along it from its wheel, as many as
    # fit before the turning centre.
    froms = axles(contacts, foot)                                                         # [axles] Point-
    lengths = (froms & foot).norm()                                                       # [axles] Scalar+
    directions = (foot - froms) / lengths                                                 # [axles] Point-
    shares = [np.arange(0, length - DASH, 2 * DASH) for length in lengths.to_array()]     # [axles] of [dashes]
    starts = concatenate([start + direction * share for start, direction, share in zip(froms, directions, shares)])
    dash_ends = concatenate([start + direction * (share + DASH)
                             for start, direction, share in zip(froms, directions, shares)])
    dashes = solids.rods(starts + up * WHEEL_RADIUS, dash_ends + up * WHEEL_RADIUS, DASH_WIDTH / 2 * view.pixel)
    hubs = contacts + up * WHEEL_RADIUS                                                   # [wheels] Point-
    bars = solids.rods(*(ends + up * WHEEL_RADIUS for ends in frame(contacts)), FRAME_WIDTH / 2 * view.pixel)
    post = solids.rods(foot, foot + up * (4 * WHEEL_RADIUS), POST_WIDTH / 2 * view.pixel)
    parts = [(dashes, to_rgb(AXLE)), (bars, to_rgb(FRAME)), (post[None], to_rgb(CENTRE))]
    # Each rim: a circle about its axle, the line through its hub across the way the wheel rolls, the step turned a
    # quarter turn in the ground, from the point of the rim ahead of the hub.
    steps = wheel_steps(contacts, forward)                                                # [wheels] Point-
    sideways = steps.commutator(mv.xy)                                                    # [wheels] Point-
    rims = meshes.circles(hubs + steps, (hubs & sideways) * solids.flip, N_RIM)          # [wheels, rim + 1] Point-
    tyres = strokes.polylines([(rim, to_rgb(WHEEL)) for rim in rims], RIM_WIDTH, view)
    trail = strokes.polylines([(middle, to_rgb(TRAIL))], TRAIL_WIDTH, view)
    return solids.draw(parts, view, BACKGROUND, inks=[], translucent=[], meshes=[(tyres, 1.0), (trail, 1.0)])


def space_view(place: solids.Point) -> solids.View:
    """A view of the square of ground following a place on it, up to the top of the turning centre's post."""
    mv = place.context.multivector
    corners = place + (mv.x.dual() * CORNERS[:, 0] + mv.y.dual() * CORNERS[:, 1]) * VIEW  # [corners] Point-
    return solids.view([corners, corners + mv.z.dual() * (4 * WHEEL_RADIUS)], 0.0, ELEVATION, AZIMUTH, MARGIN,
                       N_PIXELS)


def following(trail: np.ndarray) -> np.ndarray:
    """The square view centred on the last point of a trail: [corners, 2]."""
    return trail[-1] + np.array([[-VIEW, -VIEW], [VIEW, VIEW]])


def animate(draw, scenes) -> Iterator[np.ndarray]:
    """One frame per scene, drawn by the given view."""
    for scene in scenes:
        yield draw(*scene)
