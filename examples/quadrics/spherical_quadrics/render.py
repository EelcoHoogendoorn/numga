"""Three views of the spherical conic: its foci, its tangent great circles, its cone and polhodes, drawn on the GPU.

The sphere is a mesh on the sampled grid: coloured by the polarity's form in the first view, a faint ball in the
second, and in the third a pale sheet on which the polhodes, the level sets of the form, are inked by its colour
function from the form at each corner, how fast it changes across a pixel deciding how wide the ink lies. Curves are
tubes, points balls and poles arrows.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_rgb
import numpy as np

from numga import stack
from examples.quadrics.spherical_quadrics.core import Plane, Point, Scalar, ga, mv
from examples.wgsl import compose, meshes, show, solids, vec3

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
# Opacities: the sphere coloured by the form, the faint sphere and its great circles, the antipodal loop, the
# tangent circles, the pale sheet of the polhodes, and the cone.
FORM_OPACITY, BALL_OPACITY, MERIDIAN_OPACITY, ANTIPODE_OPACITY = 0.7, 0.1, 0.35, 0.7
CIRCLE_OPACITY, SHEET_OPACITY, CONE_OPACITY = 0.6, 0.7, 0.4
# Widths in pixels: the oval, the antipodal loop, the geodesics, the tangent circles, the sphere's great circles, the
# poles' shafts and heads, and the polhodes' ink; the samples along a dash of the antipodal loop.
OVAL_WIDTH, ANTIPODE_WIDTH, GEODESIC_WIDTH, CIRCLE_WIDTH, MERIDIAN_WIDTH = 5.0, 3.5, 3.5, 3.0, 3.0
POLE_WIDTH, HEAD_WIDTH, POLHODE_WIDTH = 3.0, 10.0, 3.0
N_DASH = 8
# Radii in pixels: the foci, the sample point and the dots of the dual conic; every how many samples a dot.
FOCUS_RADIUS, SAMPLE_RADIUS, DOT_RADIUS, DOT_EVERY = 7.0, 8.0, 2.5, 3
# The poles' length, and their heads' share of it; the samples around the faint sphere's great circles.
POLE_LENGTH, HEAD_SHARE, N_MERIDIAN = 0.35, 0.3, 120
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


def polhode_colour(levels: np.ndarray, inks: np.ndarray) -> str:
    """WGSL colouring the sphere pale, inked where the form, interpolated across each face from its corners, crosses
    one of the levels: by how many pixels away the level lies, the form's change across a pixel the measure."""
    lines = "\n".join(f"    shade = mix(shade, {vec3(ink)}, clamp({POLHODE_WIDTH / 2 + 0.5} - abs(values.x - {level:.6f}) / "
                      f"step, 0.0, 1.0));" for level, ink in zip(levels, inks))
    return f"""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec3<f32> {{
    let step = oversampling * fwidth(values.x);
    var shade = {vec3(POLHODE_SHEET)};
{lines}
    return shade;
}}
"""


def dashes(curve: np.ndarray) -> np.ndarray:
    """Every other run of `N_DASH` samples of a curve `[n, 3]`, each run one sample longer to meet the next gap,
    `[dashes, N_DASH + 1, 3]`."""
    starts = np.arange(0, len(curve) - N_DASH, 2 * N_DASH)
    return curve[starts[:, None] + np.arange(N_DASH + 1)]


def draw_spherical_conic(
    curve: Point, foci: Point, arcs: Point, sample: Point, tangents: Plane,
    touching: Plane, circles: Point, surface: Point, grid: Point, potential: Scalar,
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
    oval = (*meshes.tubes([(pk, OVAL)], OVAL_WIDTH, view), 1.0)

    # Primal spherical oval, its antipodal loop, the foci and geodesics to one point, over the form on the sphere.
    shades = cm.coolwarm((level - level.min()) / (level.max() - level.min()))[..., :3].reshape(-1, 3)
    form = meshes.layer(places, faces, shades[faces], radial, view, gloss=0.1)
    antipode = meshes.tubes([(dash, OVAL) for dash in dashes(-pk)], ANTIPODE_WIDTH, view)
    geodesics = meshes.tubes(list(zip(ak, GEODESICS)), GEODESIC_WIDTH, view)
    marks = solids.layer([(solids.ball(sm.point(fk), FOCUS_RADIUS * view.pixel)[:, None], FOCUS),
                          (solids.ball(sm.point(sk), SAMPLE_RADIUS * view.pixel)[None, None], SAMPLE)], view)
    primal = compose([(*form, FORM_OPACITY), oval, (*antipode, ANTIPODE_OPACITY), (*geodesics, 1.0), (*marks, 1.0)],
                     background)

    # Plane-based dual view: the oval as the envelope of its tangent great circles, each with its pole, and the dual
    # conic the poles trace, dotted.
    ball = (*solids.layer([(solids.ball(solids.origin, 1.0)[None, None], SPHERE)], view), BALL_OPACITY)
    meridians = meshes.tubes([(circle, SPHERE) for circle in meshes.circles(np.zeros((3, 3)), np.ones(3), np.eye(3),
                                                                            N_MERIDIAN)], MERIDIAN_WIDTH, view)
    colours = cm.plasma(np.linspace(0.1, 0.9, len(ck)))[:, :3]
    tangent_circles = meshes.tubes(list(zip(ck, colours)), CIRCLE_WIDTH, view)
    shafts, heads = solids.arrows(sm.point(ck[:, 0]), sm.point(poles), POLE_WIDTH / 2 * view.pixel,
                                  HEAD_SHARE * POLE_LENGTH, HEAD_WIDTH / 2 * view.pixel)
    dual_conic = solids.ball(sm.point(normals(tangents)[::DOT_EVERY]), DOT_RADIUS * view.pixel)[:, None]
    pointers = solids.layer([(shafts, colours), (heads, colours), (dual_conic, DUAL)], view)
    envelope = compose([ball, (*meridians, MERIDIAN_OPACITY), oval, (*tangent_circles, CIRCLE_OPACITY),
                        (*pointers, 1.0)], background)

    # The quadratic cone through the oval, and the polhodes: level sets of P & C(P), below zero and above.
    levels = np.concatenate([level.min() * FRACTIONS[::-1], level.max() * FRACTIONS])
    inks = np.repeat(np.stack([BELOW, ABOVE]), len(FRACTIONS), axis=0)
    sheet = meshes.layer(places, faces, level.reshape(-1)[faces][..., None], radial, view,
                         colour=polhode_colour(levels, inks), gloss=0.1)
    cone_places = sm.point(cone.reshape(-1, 3))
    cone_faces = grid_faces(*cone.shape[:2])
    cone_layer = meshes.layer(cone_places, cone_faces, CONE, meshes.smooth(cone_places, cone_faces), view)
    polhodes = compose([(*sheet, SHEET_OPACITY), (*cone_layer, CONE_OPACITY), oval], background)

    fig = plt.figure(figsize=(3 * N_PIXELS / 100, view.shape[0] / 100), dpi=100)
    for index, image in enumerate((primal, envelope, polhodes)):
        show(fig.add_axes((index / 3, 0.0, 1 / 3, 1.0)), image, view)
    return fig
