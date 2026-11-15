"""Three GPU views of a spherical conic showing its foci, its tangent great circles, its cone, and its polhodes.

An orthographic view looks at the sphere along parallel rays, and the ray through each pixel crosses the far and near
sides of the sphere. The shader evaluates the polarity's quadratic form `conic(point) & point` at both crossings: its
sign colours the sphere, and where it vanishes on the oval's side of the centre the oval is inked by dividing the form
by its screen-space gradient magnitude. Great circles are inked where their planes pair to zero with the point, the
polhodes where the form matches each level, and the quadratic cone where the form vanishes along the ray itself.
"""

from __future__ import annotations

from functools import cache

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_rgb
import numpy as np

from numga import concatenate, stack
from examples.quadrics.spherical_quadrics.core import Plane, Point, Polarity, ga, mv, sphere_grid
from numga.wgsl import Shader, supersampled
from examples.util.panels import image_figure

# Sphere sampling counts for the shading range of the polarity's quadratic form.
n_grid_longitudes, n_grid_latitudes = 119, 60

# Camera elevation and azimuth in radians, half-height in sphere radii, and image resolution in pixels.
elevation, azimuth = np.radians(30.0), np.radians(30.0)
half_height, n_pixels = 1.45, 300
pixel = 2 * half_height / n_pixels
# The turn from the screen frame (x right, y up, z toward the viewer) into the sphere's frame.
camera = (mv.xy * (-azimuth / 2)).exp() * (mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()  # [] Rotor+
# Screen-frame poles on the sphere, screen-frame coordinate planes, and the lamp plane.
right, up, toward = camera >> stack((mv.x, mv.y, mv.z)).dual()                 # [] Point-, [] Point-, [] Point-
across_axes, depth_axis = camera >> stack((mv.x, mv.y)), camera >> mv.z        # [2] Plane+, [] Plane+
lamp_plane = camera >> (mv.x * 0.35 + mv.y * 0.5 + mv.z * 0.8)                 # [] Plane+
lamp = lamp_plane.normalized()                                                 # [] Plane+
# A screen-plane turn of 15 degrees lays the two strokes of an arrow chevron to either side of its shaft.
chevron_bivector = camera >> mv.xy                                             # [] Bivector+
chevron_turn = (chevron_bivector * (-np.pi / 12)).exp()                        # [] Rotor+

background_colour = np.ones(3)
below_colour, neutral_colour, above_colour = (np.array(cm.coolwarm(share)[:3]) for share in (0.0, 0.5, 1.0))
polhode_inks = np.array([to_rgb("salmon"), to_rgb("deepskyblue")])
oval_colour, focus_colour, sample_colour = np.array(to_rgb("gold")), np.array(to_rgb("red")), np.ones(3)
geodesic_colours = np.array([to_rgb("lime"), to_rgb("cyan")])
sphere_colour, dual_colour, sheet_colour = np.array(to_rgb("slategray")), np.array(to_rgb("magenta")), np.full(3, 0.93)
cone_colour, rim_colour = np.array(to_rgb("khaki")), np.full(3, 0.2)

form_opacity, antipode_opacity, ball_opacity, meridian_opacity, circle_opacity = 0.7, 0.7, 0.05, 0.35, 0.6
sheet_opacity, cone_opacity = 0.55, 0.23
ambient_floor, n_dashes, cone_reach, n_touching, dot_stride = 0.7, 12, 1.25, 16, 3
pole_arrow_length = 0.35
level_fractions = np.linspace(0.0, 0.7, 4)[1:]

oval_width, antipode_width, geodesic_width, circle_width, polhode_width = 2.5, 1.75, 1.75, 1.5, 1.5
shaft_width, chevron_length = 1.5, 6.0
focus_radius, sample_radius, rim_width, dot_radius = 3.5, 4.0, 0.75, 1.5


common = """
alias Sample = vec3<f32>;

// The polarity and the oval's centre, which distinguishes the oval from its antipodal twin.
var<uniform> conic: Polarity;
var<uniform> centre: Point;

// A sphere point at a screen position and depth, together with its screen-space tangent derivatives.
struct Seen {
    point: Point,
    rightward: Point,
    upward: Point,
}

fn on_sphere(across: vec2<f32>, depth: f32) -> Seen {
    return Seen((directions * vec3<f32>(across, depth)).sum(), right - toward * (across.x / depth),
                up - toward * (across.y / depth));
}

fn cover(distance: f32) -> f32 {
    return clamp(0.5 + distance / pixel, 0.0, 1.0);
}

fn ink(value: f32, change: f32, width: f32) -> f32 {
    return cover(width / 2 - abs(value) / change);
}

fn oval(seen: Seen) -> f32 {
    let polar = conic(seen.point);
    let change = 2 * (right.dual_inverse() * (polar & seen.rightward) + up.dual_inverse() * (polar & seen.upward));
    return select(0.0, ink(polar & seen.point, change.norm(), oval_width), (seen.point.reverse() | centre) > 0);
}

fn disc(place: Point, point: Point, radius: f32) -> f32 {
    let near = (depth_axis & place) * (depth_axis & point) > 0;
    return select(0.0, cover(radius - screen_plane(place - point).norm()), near);
}

fn lit(point: Point) -> f32 {
    return ambient + (1 - ambient) * max(lamp & point, 0.0);
}

fn sphere(across: vec2<f32>) -> vec3<f32> {
    let foot = (across_axes.dual() * across).sum();
    let depth = sqrt(1 - foot.scalar_norm_squared());
    return select(background, side(side(background, on_sphere(across, -depth)), on_sphere(across, depth)), depth > 0);
}
"""

primal_source = """
var<uniform> extent: f32;

struct Mark {
    place: Point,
    colour: vec3<f32>,
    radius: f32,
}

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
    let form = polar & point;
    let change = 2 * (right.dual_inverse() * (polar & seen.rightward) + up.dual_inverse() * (polar & seen.upward));
    let shade = mix(neutral, select(below, above, form > 0), abs(form) / extent) * lit(point);
    var colour = mix(beneath, shade, select(0.0, form_opacity, (depth_axis & point) > 0));
    for (var index = 0u; index < arrayLength(&arcs); index++) {
        let arc = arcs[index];
        let circle = arc.start & arc.end;
        let between = ((arc.start & point) | circle) > 0 && ((point & arc.end) | circle) > 0;
        let line = ink((circle & point).norm(), (right.dual_inverse() * (circle & seen.rightward) + up.dual_inverse() * (circle & seen.upward)).norm(), geodesic_width);
        colour = mix(colour, arc.colour, select(0.0, line, between));
    }
    let spoke = centre & point;
    let reference = centre & toward;
    let dash = fract(atan2((spoke ^ reference) | centre, spoke | reference) * dashes_per_turn) < 0.5;
    let twin = select(0.0, antipode_opacity * ink(form, change.norm(), antipode_width), dash);
    colour = mix(colour, oval_colour, select(twin, ink(form, change.norm(), oval_width), (point.reverse() | centre) > 0));
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
struct Circle {
    plane: Plane,
    colour: vec3<f32>,
    opacity: f32,
}

struct Arrow {
    start: Point,
    pole: Point,
    tip: Point,
    colour: vec3<f32>,
}

var<storage, read> circles: array<Circle>;
var<storage, read> arrows: array<Arrow>;
var<storage, read> poles: array<Point>;

fn barb(place: Point, direction: Point, length: f32, point: Point) -> f32 {
    let near = (depth_axis & place) * (depth_axis & point) > 0;
    let offset = screen_plane(point - place);
    let nearest = direction * clamp(offset.reverse() | direction, 0.0, length);
    return select(0.0, cover(shaft_width / 2 - (offset - nearest).norm()), near);
}

fn arrow_share(arrow: Arrow, point: Point) -> f32 {
    let sweep_arc = atan2(point.reverse() | arrow.pole, point.reverse() | arrow.start);
    let off = ((arrow.start & arrow.pole) & point).norm();
    let stroke = select(0.0, cover(shaft_width / 2 - off), sweep_arc > 0 && sweep_arc < arrow_length);
    let back = -screen_plane(arrow.pole - arrow.tip * (arrow.tip.reverse() | arrow.pole)).normalized();
    let length = min(chevron_length, screen_plane(arrow.tip - arrow.start).norm());
    let barbs = max(barb(arrow.tip, chevron_turn >> back, length, point), barb(arrow.tip, chevron_turn << back, length, point));
    return max(stroke, barbs);
}

fn side(beneath: vec3<f32>, seen: Seen) -> vec3<f32> {
    let point = seen.point;
    var colour = mix(beneath, sphere_colour, ball_opacity);
    for (var index = 0u; index < arrayLength(&circles); index++) {
        let circle = circles[index];
        let change = right.dual_inverse() * (circle.plane & seen.rightward) + up.dual_inverse() * (circle.plane & seen.upward);
        colour = mix(colour, circle.colour, circle.opacity * ink(circle.plane & point, change.norm(), circle_width));
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
struct Level {
    value: f32,
    colour: vec3<f32>,
}

var<storage, read> levels: array<Level>;

fn side(beneath: vec3<f32>, seen: Seen) -> vec3<f32> {
    let point = seen.point;
    let polar = conic(point);
    let change = 2 * (right.dual_inverse() * (polar & seen.rightward) + up.dual_inverse() * (polar & seen.upward));
    var colour = mix(beneath, sheet_colour * lit(point), sheet_opacity);
    for (var index = 0u; index < arrayLength(&levels); index++) {
        let level = levels[index];
        colour = mix(colour, level.colour, ink((polar & point) - level.value, change.norm(), polhode_width));
    }
    return mix(colour, oval_colour, oval(seen));
}

fn cone(beneath: vec3<f32>, across: vec2<f32>, low: f32, high: f32) -> vec3<f32> {
    let start = right * across.x + up * across.y;
    let polar = conic(toward);
    let quadratic_a = polar & toward;
    let quadratic_b = polar & start;
    let quadratic_c = conic(start) & start;
    let root = sqrt(quadratic_b * quadratic_b - quadratic_a * quadratic_c);
    var colour = beneath;
    for (var which = -1.0; which <= 1.0; which += 2.0) {
        let depth = (which * root - quadratic_b) / quadratic_a;
        let point = start + toward * depth;
        let crossed = quadratic_b * quadratic_b > quadratic_a * quadratic_c && depth > low && depth <= high && (point.reverse() | centre) > 0 && point.norm() < reach;
        colour = mix(colour, cone_colour, select(0.0, cone_opacity, crossed));
    }
    return colour;
}

fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place) * half_height;
    let foot = (across_axes.dual() * across).sum();
    let depth = sqrt(max(1 - foot.scalar_norm_squared(), 0.0));
    // Mesa's software Vulkan driver crashes on a runtime-length loop inside a branch, so sphere sides use `select`.
    let inside = depth > 0;
    var colour = cone(background, across, -reach, -depth);
    colour = select(colour, side(colour, on_sphere(across, -depth)), inside);
    colour = select(colour, cone(colour, across, -depth, depth), inside);
    colour = select(colour, side(colour, on_sphere(across, depth)), inside);
    return cone(colour, across, depth, reach);
}
"""


# --- plumbing -------------------------------------------------------------------------
constants = dict(
    directions=stack((right, up, toward)),
    screen_plane=Point - toward * (depth_axis & Point),
    right=right,
    up=up,
    toward=toward,
    across_axes=across_axes,
    depth_axis=depth_axis,
    lamp=lamp,
    half_height=half_height,
    pixel=pixel,
    ambient=ambient_floor,
    background=background_colour,
    below=below_colour,
    neutral=neutral_colour,
    above=above_colour,
    oval_colour=oval_colour,
    rim_colour=rim_colour,
    sphere_colour=sphere_colour,
    dual_colour=dual_colour,
    sheet_colour=sheet_colour,
    cone_colour=cone_colour,
    form_opacity=form_opacity,
    antipode_opacity=antipode_opacity,
    ball_opacity=ball_opacity,
    sheet_opacity=sheet_opacity,
    cone_opacity=cone_opacity,
    dashes_per_turn=n_dashes / (2 * np.pi),
    reach=cone_reach,
    arrow_length=pole_arrow_length,
    oval_width=oval_width * pixel,
    antipode_width=antipode_width * pixel,
    geodesic_width=geodesic_width * pixel,
    circle_width=circle_width * pixel,
    polhode_width=polhode_width * pixel,
    shaft_width=shaft_width * pixel,
    chevron_length=chevron_length * pixel,
    chevron_turn=chevron_turn,
    rim_width=rim_width * pixel,
    dot_radius=dot_radius * pixel,
)


@cache
def view_shader(source: str) -> Shader:
    """Compiles the supersampled WGSL shader for one spherical-conic panel."""
    return Shader(supersampled() + common + source, ga, dict(Point=Point, Plane=Plane, Polarity=Polarity), constants)


def panel_image(source: str, **values) -> np.ndarray:
    """Renders one `[n_pixels, n_pixels, 3]` RGB byte image on the GPU."""
    image = view_shader(source)((n_pixels, n_pixels), **values)["image"]
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)


def conic_figure(oval_points: Point, foci: Point, sample_point: Point, tangents: Plane, conic: Polarity) -> plt.Figure:
    """Renders the three-panel figure of the spherical oval, its tangent envelope and dual conic, and its cone with polhodes."""
    shared = dict(conic=conic, centre=oval_points.sum(axis=0).normalized())
    grid = sphere_grid(n_grid_longitudes, n_grid_latitudes)                     # [n_grid_latitudes, n_grid_longitudes] Point-
    form_values = conic(grid) & grid                                            # [n_grid_latitudes, n_grid_longitudes] Scalar+
    least, greatest = float(form_values.min().to_array()), float(form_values.max().to_array())
    marks = dict(
        place=concatenate([foci, sample_point[None]]),
        colour=np.stack([focus_colour, focus_colour, sample_colour]),
        radius=np.array([focus_radius, focus_radius, sample_radius]) * pixel,
    )
    arcs = dict(start=foci, end=sample_point[None].broadcast_to(foci.shape), colour=geodesic_colours)
    primal = panel_image(primal_source, extent=max(-least, greatest), marks=marks, arcs=arcs, **shared)

    touching_indices = np.linspace(0, oval_points.shape[0] - 1, n_touching, dtype=int)
    colours = cm.plasma(np.linspace(0.1, 0.9, n_touching))[:, :3]
    circles = dict(
        plane=concatenate([stack((mv.x, mv.y, mv.z)).cast(Plane), tangents[touching_indices]]),
        colour=np.concatenate([np.broadcast_to(sphere_colour, (3, 3)), colours]),
        opacity=np.repeat([meridian_opacity, circle_opacity], [3, n_touching]),
    )
    # Each arrow turns its starting point on the oval partway toward the tangent great circle's pole a quarter turn away.
    starts, poles = oval_points[touching_indices], tangents[touching_indices].dual()  # [n_touching] Point-, [n_touching] Point-
    quarter_turns = (1 + poles * starts.inverse()).normalized()                 # [n_touching] Rotor+
    arrow_turns = (quarter_turns.log() * (pole_arrow_length / (np.pi / 2))).exp()  # [n_touching] Rotor+
    tips = arrow_turns >> starts                                                # [n_touching] Point-
    arrows = dict(start=starts, pole=poles, tip=tips, colour=colours)
    envelope = panel_image(envelope_source, circles=circles, arrows=arrows, poles=tangents[::dot_stride].dual(), **shared)

    levels = dict(
        value=np.concatenate([least * level_fractions[::-1], greatest * level_fractions]),
        colour=np.repeat(polhode_inks, len(level_fractions), axis=0),
    )
    polhodes = panel_image(polhode_source, levels=levels, **shared)
    return image_figure(np.concatenate([primal, envelope, polhodes], axis=1))
