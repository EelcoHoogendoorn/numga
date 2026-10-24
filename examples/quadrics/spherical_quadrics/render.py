"""Three views of the spherical conic on the unit sphere: its foci, its tangent great circles, its cone and polhodes,
drawn on the GPU by evaluating the conic at the points of the sphere each pixel shows.

A view looks at the sphere along parallel rays, and the ray through a place on the screen crosses it at a point of its
far side and one of its near side. The shader evaluates the polarity's form `point & conic(point)` at both: its sign
colours the sphere, and where it vanishes on the oval's side of the centre the oval is inked, the form over how fast it
changes across the view being the distance across the view from the curve. Great circles are inked where their planes
pair to nothing with the point, the polhodes where the form equals each level, and the cone is where the form vanishes
along the ray itself. The far side shows through the near side.
"""

from __future__ import annotations

from functools import cache

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_rgb
import numpy as np

from numga import concatenate, stack
from examples.quadrics.spherical_quadrics.core import Plane, Point, Polarity, Scalar, ga, mv
from examples.util.wgsl import Shader, image_figure, supersampled

# The view's elevation and azimuth in radians, azimuth zero looking along +y; half the image's height in the sphere's
# radii, and the pixels across and up each view.
ELEVATION, AZIMUTH = np.radians(30.0), np.radians(30.0)
HALF_HEIGHT, N_PIXELS = 1.45, 300
PIXEL = 2 * HALF_HEIGHT / N_PIXELS
# The turn from the view's own frame, x right, y up and z toward the viewer, to the sphere's.
camera = (mv.xy * (-AZIMUTH / 2)).exp() * (mv.yz * (-(np.pi / 2 - ELEVATION) / 2)).exp()     # [] Rotor
# The points of the sphere right and up across the view and toward the viewer; the planes through the centre whose
# pairings with a point are how far right and up it lies, and how far toward the viewer; and the lamp, up and to the
# right of the viewer and in front, a plane through the centre: its pairing with a point is how squarely the point
# faces it.
right, up, toward = camera >> stack((mv.x, mv.y, mv.z)).dual()               # [3] Point
across_axes, depth_axis = camera >> stack((mv.x, mv.y)), camera >> mv.z      # [2] Plane, [] Plane
lamp = (camera >> (mv.x * 0.35 + mv.y * 0.5 + mv.z * 0.8)).normalized()     # [] Plane
# The background; the form's colours below zero, at zero and above, and the polhodes' below and above; the oval, the
# foci, the point the geodesics run to and the geodesics; the faint sphere and its great circles, the dual conic, the
# pale sheet under the polhodes, the cone, and the rims of the marks.
BACKGROUND = np.ones(3)
BELOW, NEUTRAL, ABOVE = (np.array(cm.coolwarm(share)[:3]) for share in (0.0, 0.5, 1.0))
POLHODE_INKS = np.array([to_rgb("salmon"), to_rgb("deepskyblue")])
OVAL, FOCUS, SAMPLE = np.array(to_rgb("gold")), np.array(to_rgb("red")), np.ones(3)
GEODESICS = np.array([to_rgb("lime"), to_rgb("cyan")])
SPHERE, DUAL, SHEET = np.array(to_rgb("slategray")), np.array(to_rgb("magenta")), np.full(3, 0.93)
CONE, RIM = np.array(to_rgb("khaki")), np.full(3, 0.2)
# Opacities: the form on the near side of the sphere, the antipodal loop, each side of the faint sphere, its great
# circles and the tangent great circles, each side of the pale sheet, and each crossing of the cone.
FORM_OPACITY, ANTIPODE_OPACITY, BALL_OPACITY, MERIDIAN_OPACITY, CIRCLE_OPACITY = 0.7, 0.7, 0.05, 0.35, 0.6
SHEET_OPACITY, CONE_OPACITY = 0.55, 0.23
# The lighting's floor, the dashes of the antipodal loop around the oval's centre, the cone's reach from the centre,
# the tangent great circles drawn, and every how many tangents a dot of the dual conic.
AMBIENT, N_DASHES, REACH, N_TOUCHING, DOT_EVERY = 0.7, 12, 1.25, 16, 3
# The arrows toward the poles: their length along the sphere, and their heads' share of it.
POLE_LENGTH, HEAD_SHARE = 0.35, 0.3
# The polhodes: the levels drawn, as fractions of the form's least and greatest values on the sphere.
FRACTIONS = np.linspace(0.0, 0.7, 4)[1:]
# Widths and radii in pixels: the oval, the antipodal loop, the geodesics, the great circles, the polhodes, the
# arrows' shafts and heads; the foci, the sample point, their rims, and the dots of the dual conic.
OVAL_WIDTH, ANTIPODE_WIDTH, GEODESIC_WIDTH, CIRCLE_WIDTH, POLHODE_WIDTH = 2.5, 1.75, 1.75, 1.5, 1.5
SHAFT_WIDTH, HEAD_WIDTH = 1.5, 5.0
FOCUS_RADIUS, SAMPLE_RADIUS, RIM_WIDTH, DOT_RADIUS = 3.5, 4.0, 0.75, 1.5


common = """
alias Sample = vec3<f32>;

// The polarity, and the oval's centre, which tells the oval from its antipodal twin.
var<uniform> conic: Polarity;
var<uniform> centre: Point;

// A point of the sphere at a place across the view and a depth toward the viewer, and how it moves as the place moves
// right and up across the view, keeping to the sphere.
struct Seen {
    point: Point,
    rightward: Point,
    upward: Point,
}

fn on_sphere(across: vec2<f32>, depth: f32) -> Seen {
    return Seen(right * across.x + up * across.y + toward * depth, right - toward * (across.x / depth),
                up - toward * (across.y / depth));
}

// The share of a pixel inside an edge, the pixel's centre `distance` inside it, in the sphere's radii.
fn cover(distance: f32) -> f32 {
    return clamp(0.5 + distance / pixel, 0.0, 1.0);
}

// The share of a pixel a line `width` across the view covers about where a value vanishes, from the value at a point
// and its change as the point steps right and up across the view: the value over that change is how far across the
// view the point lies from the line.
fn ink(value: f32, change: vec2<f32>, width: f32) -> f32 {
    return cover(width / 2 - abs(value) / length(change));
}

// The oval's ink at a seen point: where the form `point & conic(point)` vanishes, its gradient twice the polar, on
// the oval's side of the centre.
fn oval(seen: Seen) -> f32 {
    let polar = conic(seen.point);
    let change = 2 * vec2<f32>(polar & seen.rightward, polar & seen.upward);
    return select(0.0, ink(seen.point & polar, change, oval_width), -(seen.point | centre) > 0);
}

// The share of a pixel a disc about a place covers at a point of the sphere, round on the screen: by their offset
// across the view, where both lie on the same side of the plane of the screen through the centre.
fn disc(place: Point, point: Point, radius: f32) -> f32 {
    let near = (depth_axis & place) * (depth_axis & point) > 0;
    return select(0.0, cover(radius - length(across_axes & (place - point))), near);
}

// How brightly the lamp lights a point of the sphere: how squarely the point faces it, over an ambient floor.
fn lit(point: Point) -> f32 {
    return ambient + (1 - ambient) * max(lamp & point, 0.0);
}

// The colour at a place on the screen: within the sphere's outline, its far side seen through its near side.
fn sphere(across: vec2<f32>) -> vec3<f32> {
    let depth = sqrt(1 - dot(across, across));
    return select(background, side(side(background, on_sphere(across, -depth)), on_sphere(across, depth)), depth > 0);
}
"""

primal_source = """
// The form's largest size on the sphere, which its colours span.
var<uniform> extent: f32;

// A disc about a point of the sphere, in its colour and of its radius.
struct Mark {
    place: Point,
    colour: vec3<f32>,
    radius: f32,
}

// A geodesic from one point to another, in its colour.
struct Arc {
    start: Point,
    end: Point,
    colour: vec3<f32>,
}

var<storage, read> marks: array<Mark>;
var<storage, read> arcs: array<Arc>;

fn side(beneath: vec3<f32>, seen: Seen) -> vec3<f32> {
    let point = seen.point;
    let polar = conic(point);
    let form = point & polar;
    let change = 2 * vec2<f32>(polar & seen.rightward, polar & seen.upward);
    // The form's sign colours the sphere's near side, from the neutral colour at zero toward `below` or `above`.
    let shade = mix(neutral, select(below, above, form > 0), abs(form) / extent) * lit(point);
    var colour = mix(beneath, shade, select(0.0, form_opacity, (depth_axis & point) > 0));
    // A geodesic lies on the great circle through its ends, their join; a point between the ends joins each of them
    // in the sense the ends join each other.
    for (var index = 0u; index < arrayLength(&arcs); index++) {
        let arc = arcs[index];
        let circle = arc.start & arc.end;
        let between = ((arc.start & point) | circle) > 0 && ((point & arc.end) | circle) > 0;
        let line = ink(circle & point, vec2<f32>(circle & seen.rightward, circle & seen.upward), geodesic_width);
        colour = mix(colour, arc.colour, select(0.0, line, between));
    }
    // Where the form vanishes: the oval on the centre's side, and across it the antipodal loop in dashes, by the
    // point's turn about the centre from the great circle through the centre and the viewer.
    let spoke = centre & point;
    let reference = centre & toward;
    let dash = fract(atan2((spoke ^ reference) | centre, spoke | reference) * dashes_per_turn) < 0.5;
    let twin = select(0.0, antipode_opacity * ink(form, change, antipode_width), dash);
    colour = mix(colour, oval_colour, select(twin, ink(form, change, oval_width), -(point | centre) > 0));
    // The marks, each over a darker rim.
    for (var index = 0u; index < arrayLength(&marks); index++) {
        let mark = marks[index];
        colour = mix(colour, rim_colour, disc(mark.place, point, mark.radius));
        colour = mix(colour, mark.colour, disc(mark.place, point, mark.radius - rim_width));
    }
    return colour;
}

fn sample(place: vec2<f32>) -> Sample {
    return sphere(screen(place) * half_height);
}
"""

envelope_source = """
// A great circle by the plane through the centre it lies on, in its colour and opacity.
struct Circle {
    plane: Plane,
    colour: vec3<f32>,
    opacity: f32,
}

// An arrow painted on the sphere from a point of the oval toward the pole of its tangent great circle, a quarter turn
// away, in its colour.
struct Arrow {
    start: Point,
    pole: Point,
    colour: vec3<f32>,
}

var<storage, read> circles: array<Circle>;
var<storage, read> arrows: array<Arrow>;
// The poles of tangent great circles, dots along the dual conic.
var<storage, read> poles: array<Point>;

// The share of a pixel an arrow covers at a point of the sphere: the point's turn from the start toward the pole,
// read by its pairings with the two, and its distance off the great circle through them, their join; a shaft, then a
// head narrowing to the tip.
fn arrow_share(arrow: Arrow, point: Point) -> f32 {
    let along = atan2(-(point | arrow.pole), -(point | arrow.start));
    let off = abs((arrow.start & arrow.pole) & point);
    let head = (arrow_length - along) / head_length * head_width / 2;
    let half_width = select(shaft_width / 2, head, along > arrow_length - head_length);
    return select(0.0, cover(half_width - off), along > 0 && along < arrow_length);
}

fn side(beneath: vec3<f32>, seen: Seen) -> vec3<f32> {
    let point = seen.point;
    var colour = mix(beneath, sphere_colour, ball_opacity);
    // Each great circle where its plane pairs to nothing with the point; then the oval, the arrows and the dots.
    for (var index = 0u; index < arrayLength(&circles); index++) {
        let circle = circles[index];
        let change = vec2<f32>(circle.plane & seen.rightward, circle.plane & seen.upward);
        colour = mix(colour, circle.colour, circle.opacity * ink(circle.plane & point, change, circle_width));
    }
    colour = mix(colour, oval_colour, oval(seen));
    for (var index = 0u; index < arrayLength(&arrows); index++) {
        colour = mix(colour, arrows[index].colour, arrow_share(arrows[index], point));
    }
    for (var index = 0u; index < arrayLength(&poles); index++) {
        colour = mix(colour, dual_colour, disc(poles[index], point, dot_radius));
    }
    return colour;
}

fn sample(place: vec2<f32>) -> Sample {
    return sphere(screen(place) * half_height);
}
"""

polhode_source = """
// A polhode: where the form equals its level, in its colour.
struct Level {
    value: f32,
    colour: vec3<f32>,
}

var<storage, read> levels: array<Level>;

fn side(beneath: vec3<f32>, seen: Seen) -> vec3<f32> {
    let point = seen.point;
    let polar = conic(point);
    let change = 2 * vec2<f32>(polar & seen.rightward, polar & seen.upward);
    var colour = mix(beneath, sheet_colour * lit(point), sheet_opacity);
    for (var index = 0u; index < arrayLength(&levels); index++) {
        let level = levels[index];
        colour = mix(colour, level.colour, ink((point & polar) - level.value, change, polhode_width));
    }
    return mix(colour, oval_colour, oval(seen));
}

// The cone's sheet where the ray through a place on the screen crosses it between two depths toward the viewer, on
// the oval's side of the centre and within `reach` of it. Along the ray, from its point on the plane of the screen
// through the centre, the form is a quadratic in the depth, `a * depth**2 + 2 * b * depth + c`, whose roots are the
// crossings.
fn cone(beneath: vec3<f32>, across: vec2<f32>, low: f32, high: f32) -> vec3<f32> {
    let start = right * across.x + up * across.y;
    let polar = conic(toward);
    let a = toward & polar;
    let b = start & polar;
    let c = start & conic(start);
    let root = sqrt(b * b - a * c);
    var colour = beneath;
    for (var which = -1.0; which <= 1.0; which += 2.0) {
        let depth = (which * root - b) / a;
        let point = start + toward * depth;
        let crossed = b * b > a * c && depth > low && depth <= high && -(point | centre) > 0 && point.norm() < reach;
        colour = mix(colour, cone_colour, select(0.0, cone_opacity, crossed));
    }
    return colour;
}

// From the far end of the ray to the near: the cone behind the sphere, the sphere's far side, the cone within it, the
// sphere's near side, and the cone in front of it.
fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place) * half_height;
    let depth = sqrt(max(1 - dot(across, across), 0.0));
    var colour = cone(background, across, -reach, -depth);
    if (depth > 0) {
        colour = side(colour, on_sphere(across, -depth));
        colour = cone(colour, across, -depth, depth);
        colour = side(colour, on_sphere(across, depth));
    }
    return cone(colour, across, depth, reach);
}
"""


# --- plumbing -------------------------------------------------------------------------
constants = dict(
    right=right, up=up, toward=toward, across_axes=across_axes, depth_axis=depth_axis, lamp=lamp,
    half_height=HALF_HEIGHT, pixel=PIXEL, ambient=AMBIENT, background=BACKGROUND, below=BELOW, neutral=NEUTRAL,
    above=ABOVE, oval_colour=OVAL, rim_colour=RIM, sphere_colour=SPHERE, dual_colour=DUAL, sheet_colour=SHEET, cone_colour=CONE, form_opacity=FORM_OPACITY,
    antipode_opacity=ANTIPODE_OPACITY, ball_opacity=BALL_OPACITY, sheet_opacity=SHEET_OPACITY,
    cone_opacity=CONE_OPACITY, dashes_per_turn=N_DASHES / (2 * np.pi), reach=REACH,
    arrow_length=POLE_LENGTH, head_length=HEAD_SHARE * POLE_LENGTH, oval_width=OVAL_WIDTH * PIXEL,
    antipode_width=ANTIPODE_WIDTH * PIXEL, geodesic_width=GEODESIC_WIDTH * PIXEL, circle_width=CIRCLE_WIDTH * PIXEL,
    polhode_width=POLHODE_WIDTH * PIXEL, shaft_width=SHAFT_WIDTH * PIXEL, head_width=HEAD_WIDTH * PIXEL,
    rim_width=RIM_WIDTH * PIXEL, dot_radius=DOT_RADIUS * PIXEL,
)


@cache
def shader(source: str) -> Shader:
    """The shader of one view, its source after what the views share."""
    return Shader(supersampled() + common + source, ga, dict(Point=Point, Plane=Plane, Polarity=Polarity), constants)


def picture(source: str, **values) -> np.ndarray:
    """One view, `[rows, columns, 3]` bytes."""
    image = shader(source)((N_PIXELS, N_PIXELS), **values)["image"]
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)


def draw_spherical_conic(curve: Point, foci: Point, sample: Point, tangents: Plane, potential: Scalar,
                         conic: Polarity) -> plt.Figure:
    """The oval with its foci and the geodesics from them to one point, over the polarity's form; the oval as the
    envelope of its tangent great circles, with arrows toward their poles and the dual conic the poles trace, dotted;
    and the cone through it, with the level sets of the form, the polhodes."""
    # The oval's centre, the unit point its points sum toward; the form's least and greatest values on the sphere.
    shared = dict(conic=conic, centre=curve.sum(axis=0).normalized())
    least, greatest = potential.to_array().min(), potential.to_array().max()
    marks = dict(place=concatenate([foci, sample[None]]), colour=np.stack([FOCUS, FOCUS, SAMPLE]),
                 radius=np.array([FOCUS_RADIUS, FOCUS_RADIUS, SAMPLE_RADIUS]) * PIXEL)
    arcs = dict(start=foci, end=sample[None].broadcast_to(foci.shape), colour=GEODESICS)
    primal = picture(primal_source, extent=max(-least, greatest), marks=marks, arcs=arcs, **shared)

    # The great circles of the planes x, y and z, and some tangent great circles, each with an arrow from its point
    # of the oval toward its pole.
    touching = np.linspace(0, curve.shape[0] - 1, N_TOUCHING, dtype=int)
    colours = cm.plasma(np.linspace(0.1, 0.9, N_TOUCHING))[:, :3]
    circles = dict(plane=concatenate([stack((mv.x, mv.y, mv.z)).cast(Plane), tangents[touching]]),
                   colour=np.concatenate([np.broadcast_to(SPHERE, (3, 3)), colours]),
                   opacity=np.repeat([MERIDIAN_OPACITY, CIRCLE_OPACITY], [3, N_TOUCHING]))
    arrows = dict(start=curve[touching], pole=tangents[touching].dual(), colour=colours)
    envelope = picture(envelope_source, circles=circles, arrows=arrows, poles=tangents[::DOT_EVERY].dual(), **shared)

    # The polhodes, below zero and above.
    levels = dict(value=np.concatenate([least * FRACTIONS[::-1], greatest * FRACTIONS]),
                  colour=np.repeat(POLHODE_INKS, len(FRACTIONS), axis=0))
    polhodes = picture(polhode_source, levels=levels, **shared)
    return image_figure(np.concatenate([primal, envelope, polhodes], axis=1))
