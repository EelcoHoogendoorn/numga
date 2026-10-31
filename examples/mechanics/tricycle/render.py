"""Drawing for the tricycle: the plane from above and space in perspective, with the tricycle's frame and
wheels, each axle drawn on to the turning centre, and the trails of the middle of the rear axle and of the
front wheel.

The geometry of the drawing is written once for both algebras, on the points of whichever the drive is in; the
plane is drawn in its own algebra, read by a camera of its planes x and y, and space, in the GPU library's own
algebra, is passed as it is."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import Extensor, concatenate, stack
from examples.util.animation import capture
from examples.util.wgsl import meshes, simplices, solids
from examples.util.wgsl.camera import panel

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
# The plane view, in pixels: the widths of the trail, the axles, drawn dashed, with the lengths of their dashes and
# gaps, the frame and the wheels; the turning centre's radius.
PLANE_TRAIL_WIDTH, AXLE_WIDTH, AXLE_DASHES, PLANE_FRAME_WIDTH, WHEEL_WIDTH = 1.0, 1.5, (6.0, 4.5), 2.0, 4.0
CENTRE_RADIUS = 2.3
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


def pairs(n_pairs: int) -> simplices.Mesh:
    """An edge from each of the first `n_pairs` points to the one `n_pairs` after it."""
    return simplices.Mesh(2 * n_pairs, np.stack([np.arange(n_pairs), n_pairs + np.arange(n_pairs)], axis=-1))


def plane(contacts: Extensor, forward: Extensor, centre: Extensor, middle: Extensor) -> np.ndarray:
    """The plane from above, around the tricycle, drawn on the GPU in the drive's own algebra."""
    mv = centre.context.multivector
    # The turning centre, an extrinsic meet, as the intrinsic point it names, at unit weight.
    centre = (centre * mv.antipseudoscalar([1.0])).normalized()
    steps = wheel_steps(contacts, forward)
    across, up = following(middle[-1])
    figure, ax = plt.subplots(figsize=(3, 3))
    ax.set(xlim=across, ylim=up, aspect="equal")
    ax.axis("off")
    figure.tight_layout()
    camera = panel(ax, mv.x, mv.y)
    # Each axle on to the turning centre, the last of the points.
    froms = axles(contacts, centre)                                          # [axles] Point-
    spokes = simplices.Mesh(len(froms) + 1, np.stack([np.arange(len(froms)), np.full(len(froms), len(froms))], -1))
    starts, ends = frame(contacts)                                           # [bars] Point-, Point-
    simplices.overlay(ax, [
        simplices.strokes(middle, simplices.Mesh.chains([len(middle)]), to_rgb(TRAIL), PLANE_TRAIL_WIDTH, camera),
        simplices.strokes(concatenate([froms, centre[None]]), spokes, to_rgb(AXLE), AXLE_WIDTH, camera,
                          dashes=AXLE_DASHES, cap=simplices.Cap.BUTT),
        simplices.markers(centre[None], to_rgb(CENTRE), CENTRE_RADIUS, camera),
        simplices.strokes(concatenate([starts, ends]), pairs(len(starts)), to_rgb(FRAME), PLANE_FRAME_WIDTH, camera),
        simplices.strokes(concatenate([contacts - steps, contacts + steps]), pairs(len(contacts)), to_rgb(WHEEL),
                          WHEEL_WIDTH, camera)])
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
    tyres = simplices.tubes(rims.reshape(-1), simplices.Mesh.chains([rims.shape[1]] * rims.shape[0]), to_rgb(WHEEL),
                            RIM_WIDTH, view, gloss=solids.gloss)
    trail = simplices.tubes(middle, simplices.Mesh.chains([len(middle)]), to_rgb(TRAIL), TRAIL_WIDTH, view,
                            gloss=solids.gloss)
    return solids.draw(parts, view, BACKGROUND, inks=[], translucent=[], meshes=[(tyres, 1.0), (trail, 1.0)])


def space_view(place: solids.Point) -> solids.View:
    """A view of the square of ground following a place on it, up to the top of the turning centre's post."""
    mv = place.context.multivector
    corners = place + (mv.x.dual() * CORNERS[:, 0] + mv.y.dual() * CORNERS[:, 1]) * VIEW  # [corners] Point-
    return solids.view([corners, corners + mv.z.dual() * (4 * WHEEL_RADIUS)], 0.0, ELEVATION, AZIMUTH, MARGIN,
                       N_PIXELS)


def following(place: Extensor) -> np.ndarray:
    """The square view centred on a place on the ground, its limits along x and along y `[2, 2]`: the place's
    pairings with the planes x = 0 and y = 0 over its weight, the view's half width either side."""
    mv = place.context.multivector
    middle = (stack((mv.x, mv.y)) & place) / (mv.w & place)                  # [2] Scalar+
    return stack([middle - VIEW, middle + VIEW], axis=-1).to_array()


def animate(draw, scenes) -> Iterator[np.ndarray]:
    """One frame per scene, drawn by the given view."""
    for scene in scenes:
        yield draw(*scene)
