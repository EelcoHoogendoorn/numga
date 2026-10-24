"""Three views of the spherical conic: its foci, its tangent great circles, its cone and polhodes, drawn on the GPU.

The sphere is traced: clear in the first view, where a mesh on the sampled grid colours it by the polarity's form, a
faint ball in the second, and in the third a pale sheet. The curves on it are inked where a quadric cuts it: a great
circle where a plane through the centre does, the oval where the cone of the form does, and the polhodes where the
form's level quadrics do; the geodesics are tubes. The cone itself is traced, clear, its sheet painted on it as far
as it was sampled. Points are balls and poles arrows.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_rgb
import numpy as np

from numga import stack
from examples.quadrics.spherical_quadrics.core import Plane, Point, Polarity, Scalar, ga, mv, points
from examples.util.wgsl import meshes, show, solids, strokes

sm = solids.mv
OVAL, FOCUS, SAMPLE = np.array(to_rgb("gold")), np.array(to_rgb("red")), np.ones(3)
GEODESICS = np.array([to_rgb("lime"), to_rgb("cyan")])
SPHERE, CONE, DUAL = np.array(to_rgb("slategray")), np.array(to_rgb("khaki")), np.array(to_rgb("magenta"))
POLHODE_SHEET = np.full(3, 0.93)
BELOW, ABOVE = np.array(to_rgb("salmon")), np.array(to_rgb("deepskyblue"))
background = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin around the scene as a share
# of the larger side, and the pixels across each panel.
ELEVATION, AZIMUTH = np.radians(30.0), np.radians(30.0)
MARGIN, N_PIXELS = 0.25, 600
# Opacities: the sphere coloured by the form, each surface of the faint sphere and its great circles, the antipodal
# loop, the tangent circles, each surface of the pale sheet of the polhodes, and each of the cone.
FORM_OPACITY, BALL_OPACITY, MERIDIAN_OPACITY, ANTIPODE_OPACITY = 0.7, 0.05, 0.35, 0.7
CIRCLE_OPACITY, SHEET_OPACITY, CONE_OPACITY = 0.6, 0.55, 0.23
# Widths in pixels: the oval, the antipodal loop, the geodesics, the tangent circles, the sphere's great circles, the
# poles' shafts and heads, and the polhodes' ink; the samples along a dash of the antipodal loop.
OVAL_WIDTH, ANTIPODE_WIDTH, GEODESIC_WIDTH, CIRCLE_WIDTH, MERIDIAN_WIDTH = 5.0, 3.5, 3.5, 3.0, 3.0
POLE_WIDTH, HEAD_WIDTH, POLHODE_WIDTH = 3.0, 10.0, 3.0
N_DASH = 8
# Radii in pixels: the foci, the sample point and the dots of the dual conic; every how many samples a dot.
FOCUS_RADIUS, SAMPLE_RADIUS, DOT_RADIUS, DOT_EVERY = 7.0, 8.0, 2.5, 3
# The poles' length, and their heads' share of it.
POLE_LENGTH, HEAD_SHARE = 0.35, 0.3
# The polhodes: the levels drawn, as fractions of the form's least and greatest values on the sphere.
FRACTIONS = np.linspace(0.0, 0.7, 4)[1:]


# --- plumbing -------------------------------------------------------------------------
def xyz(points: Point) -> np.ndarray:
    """Coordinates (..., 3) of points: their pairings with the planes x, y and z."""
    return (stack((mv.x, mv.y, mv.z)) & points[..., None]).to_array()


def normals(planes: Plane) -> np.ndarray:
    """Normals (..., 3) of planes on the basis planes x, y and z."""
    return planes.cast(ga.subspace("x y z")).kernel


def grid_faces(n_rows: int, n_columns: int) -> np.ndarray:
    """The triangles `[faces, 3]` of a grid of rows and columns of vertices, two to each quad."""
    index = np.arange(n_rows * n_columns).reshape(n_rows, n_columns)
    quads = np.stack([index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]], axis=-1).reshape(-1, 4)
    return quads[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)


def sphere_mesh(sphere: np.ndarray) -> tuple[solids.Point, np.ndarray, dict]:
    """The unit sphere's grid `[rows, columns, 3]` as a mesh: its points, its triangles, and at each corner the
    sphere's own normal, the plane through the centre square to the radius there."""
    places = sm.point(sphere.reshape(-1, 3))
    faces = grid_faces(*sphere.shape[:2])
    radial = -(places - solids.origin).dual()                                 # [vertices] Plane
    return places, faces, dict(a=radial[faces[:, 0]], b=radial[faces[:, 1]], c=radial[faces[:, 2]])


def dashes(curve: np.ndarray) -> np.ndarray:
    """Every other run of `N_DASH` samples of a curve `[n, 3]`, each run one sample longer to meet the next gap,
    `[dashes, N_DASH + 1, 3]`."""
    starts = np.arange(0, len(curve) - N_DASH, 2 * N_DASH)
    return curve[starts[:, None] + np.arange(N_DASH + 1)]


def form(conic: Polarity) -> solids.Quadric:
    """The polarity's form, `P & conic(P)`, as a quadric of the traced scene: the cone through the centre where it
    vanishes, and on the unit sphere the form itself."""
    # The form's matrix: each basis point's polar plane, read on the basis planes.
    matrix = normals(conic(points(np.eye(3))))                                  # [3, 3]
    return (solids.axes[:, None] * (solids.axes & solids.Point)[None, :] * matrix).sum(axis=(-1, -2))


def through_centre(directions: solids.Point) -> solids.Plane:
    """The planes through the centre square to directions."""
    return -directions.dual()


def chord_balls(starts: solids.Point, ends: solids.Point) -> solids.Quadric:
    """The balls on the chords from starts to ends of unit weight: of a circle through both ends, each holds the
    shorter arc between them and no more of it, and of a short run of a smooth curve, the run."""
    radii = (ends - starts).dual().norm().to_array() / 2                      # [...]
    return solids.placed(solids.ellipsoid(np.repeat(radii[..., None], 3, axis=-1)),
                         solids.translator((starts + ends) / 2 - solids.origin))


def draw_spherical_conic(
    curve: Point, foci: Point, arcs: Point, sample: Point, tangents: Plane,
    touching: Plane, circles: Point, surface: Point, grid: Point, potential: Scalar, conic: Polarity,
) -> plt.Figure:
    """The oval with its foci and the geodesics from them to one point, over the polarity's form; the oval
    as the envelope of its tangent great circles, with their poles; and the cone through it, with the
    level sets of the form, the polhodes."""
    pk, fk, ak, sk = xyz(curve), xyz(foci), xyz(arcs), xyz(sample)
    sphere, cone = xyz(grid), xyz(surface)
    level = potential.to_array()
    ck = xyz(circles)
    poles = ck[:, 0] + POLE_LENGTH * normals(touching)
    view = solids.view([sm.point(sphere), sm.point(cone), sm.point(poles)], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    places, faces, radial = sphere_mesh(sphere)
    unit_ball = solids.ball(solids.origin, 1.0)[None, None]
    # The form's cone cuts the sphere in the oval and in its antipodal loop; the oval is its part on the side of the
    # centre the oval lies on.
    cone_form = form(conic)                                                    # [] Quadric
    nappe = solids.half_space(through_centre(sm.direction(-pk.mean(axis=0))))  # [] Quadric

    def oval(part: int) -> solids.Ink:
        """The oval inked on a part's surfaces."""
        return solids.Ink(part, cone_form[None], nappe[None], OVAL, OVAL_WIDTH, 1.0)

    # Primal spherical oval, its antipodal loop, the foci and geodesics to one point, over the form on the sphere.
    shades = cm.coolwarm((level - level.min()) / (level.max() - level.min()))[..., :3].reshape(-1, 3)
    shown_form = meshes.samples(places, faces, shades[faces], radial, view, gloss=0.1)
    marks = [(solids.ball(sm.point(fk), FOCUS_RADIUS * view.pixel)[:, None], FOCUS),
             (solids.ball(sm.point(sk), SAMPLE_RADIUS * view.pixel)[None, None], SAMPLE)]
    geodesics = strokes.polylines([(sm.point(arc), colour) for arc, colour in zip(ak, GEODESICS)], GEODESIC_WIDTH, view)
    # The sphere itself, clear, is the first part after the opaque ones and carries the oval and its antipodal loop,
    # dashed: each dash the loop within the ball on its chord.
    runs = sm.point(dashes(-pk))                                               # [dashes, N_DASH + 1] Point
    antipode = solids.Ink(len(marks), cone_form[None].broadcast_to(runs.shape[:1]), chord_balls(runs[:, 0], runs[:, -1]),
                          OVAL, ANTIPODE_WIDTH, ANTIPODE_OPACITY)
    primal = solids.draw(marks, view, background, [antipode, oval(len(marks))], translucent=[(unit_ball, SPHERE, 0.0)],
                         meshes=[(shown_form, FORM_OPACITY), (geodesics, 1.0)])

    # Plane-based dual view: the oval as the envelope of its tangent great circles, each with its pole, and the dual
    # conic the poles trace, dotted.
    colours = cm.plasma(np.linspace(0.1, 0.9, len(ck)))[:, :3]
    shafts, heads = solids.arrows(sm.point(ck[:, 0]), sm.point(poles), POLE_WIDTH / 2 * view.pixel,
                                  HEAD_SHARE * POLE_LENGTH, HEAD_WIDTH / 2 * view.pixel)
    dual_conic = solids.ball(sm.point(normals(tangents)[::DOT_EVERY]), DOT_RADIUS * view.pixel)[:, None]
    pointers = [(shafts, colours), (heads, colours), (dual_conic, DUAL)]
    # The faint sphere is the first part after the opaque ones: its great circles square to x, y and z, and the
    # tangent great circles, inked where their planes cut it.
    meridians = solids.Ink(len(pointers), solids.half_space(solids.axes), solids.everywhere[None].broadcast_to((3,)),
                           SPHERE, MERIDIAN_WIDTH, MERIDIAN_OPACITY)
    tangent_planes = through_centre(sm.direction(normals(touching)))           # [touching] Plane
    tangent_circles = solids.Ink(len(pointers), solids.half_space(tangent_planes),
                                 solids.everywhere[None].broadcast_to(tangent_planes.shape), colours, CIRCLE_WIDTH,
                                 CIRCLE_OPACITY)
    envelope = solids.draw(pointers, view, background, [meridians, tangent_circles, oval(len(pointers))],
                           translucent=[(unit_ball, SPHERE, BALL_OPACITY)])

    # The quadratic cone through the oval, and the polhodes: level sets of P & C(P), below zero and above, where the
    # form less each level cuts the pale sheet. The cone is clear, its sheet painted on it as far from the centre as
    # it was sampled, by a fill through a plane across the cone, which cuts it nowhere but the apex.
    levels = np.concatenate([level.min() * FRACTIONS[::-1], level.max() * FRACTIONS])
    inks = np.repeat(np.stack([BELOW, ABOVE]), len(FRACTIONS), axis=0)
    polhodes = solids.Ink(0, cone_form + solids.everywhere * levels, solids.everywhere[None].broadcast_to(levels.shape),
                          inks, POLHODE_WIDTH, 1.0)
    reach = np.linalg.norm(cone, axis=-1).max()
    sheet = solids.Ink(1, solids.half_space(through_centre(sm.direction(pk.mean(axis=0))))[None],
                       solids.ball(solids.origin, reach)[None], CONE, np.inf, CONE_OPACITY)
    solid_cone = stack([cone_form, nappe])[None]                               # [1, 2] Quadric
    polhode_view = solids.draw([], view, background, [polhodes, oval(0), sheet],
                               translucent=[(unit_ball, POLHODE_SHEET, SHEET_OPACITY), (solid_cone, CONE, 0.0)])

    fig = plt.figure(figsize=(3 * N_PIXELS / 100, view.shape[0] / 100), dpi=100)
    for index, image in enumerate((primal, envelope, polhode_view)):
        show(fig.add_axes((index / 3, 0.0, 1 / 3, 1.0)), image, view)
    return fig
