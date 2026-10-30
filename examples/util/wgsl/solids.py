"""Solids bounded by quadrics in PGA3D, ray traced on the GPU: balls, ellipsoids, rods and boxes, each lit by a
lamp, on a plain background, seen along parallel rays.

A solid is the intersection of a few quadrics, each a map from points to planes, `Plane <- Point`, whose
pairing with a point, `quadric(point) & point`, is negative inside and zero on its surface, where the plane
it gives is the tangent plane. A ball is one quadric, a rod a cylinder about its line between two half-spaces,
a box six half-spaces. Every ray tests every quadric of every solid: the surface it sees first is a root of
one solid's quadric that lies inside all its others.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from numga import NumpyContext, concatenate, stack, where
from numga.algebras import PGA3D as ga
from examples.util.wgsl import Shader, supersampled
from examples.util.wgsl.library import samples_per_side
from examples.util.wgsl.runner import Resident

if TYPE_CHECKING:
    from examples.util.wgsl.meshes import Samples

context = NumpyContext(ga)
mv = context.multivector
Point = ga.gatype.antivector()
Scalar = ga.gatype.scalar()
Plane = ga.gatype.vector()
Motor = ga.gatype.rotor()
Line = ga.gatype.bivector()
Quadric = ga.gatype((Plane, Point))                                          # Plane <- Point
DualQuadric = ga.gatype((Point, Plane))                                      # Point <- Plane
# The planes of the frame, and of every camera's in its own: a point's pairings with them are how far it lies across
# to the right, up, and toward the viewer, the camera looking along its own -z; a place so far along each is the
# origin moved by their duals, `origin + (axes.dual() * place).sum()`.
axes = stack((mv.x, mv.y, mv.z))                                             # [3] Plane
origin = mv.w.dual()                                                         # [] Point
# The intrinsic scalar one: the join of points, intrinsic, times it is the plane or line through them, extrinsic.
flip = mv.antipseudoscalar([1.0])                                            # [] Antipseudoscalar-
# The direction square to each plane through the origin: a form on directions read through it is measured as the
# scene's lengths are.
normal_direction = Plane.derive.nondegenerate.dual()                         # Point <- Plane
# The light that falls everywhere, and the glint's share and sharpness.
ambient, gloss, shininess = 0.22, 0.45, 32.0
# How much brighter the light that falls everywhere is on faces turned up than on faces turned sideways, and dimmer
# on faces turned down: a sky above and a ground below, so the form reads away from the lamp too.
fill_tilt = 0.5
# The share of the light that comes from the viewer, so that every face the camera sees keeps its colour, by how
# squarely it is turned to the viewer; the lamp gives the rest.
headlight = 0.25


def lamp_shading() -> str:
    """WGSL lighting a surface by the lamp, for the solids, meshes and strokes alike: the lamp's uniforms, as
    `lighting` gives them, and from a tangent plane and the direction of the view, `facing`, the plane at unit weight
    turned to the viewer, `brightness`, how brightly the lamp, the sky and the viewer light it, `glint`, and `lit`, a
    colour lit with its glint."""
    return """
// The lamp's direction, the light that falls everywhere, how much more of it falls on faces turned up, the share of
// the light that comes from the viewer, and the glint's share and sharpness.
var<uniform> lamp: Point;
var<uniform> ambient: f32;
var<uniform> fill_tilt: f32;
var<uniform> headlight: f32;
var<uniform> gloss: f32;
var<uniform> shininess: f32;

// A tangent plane at unit weight, facing a viewer looking along `direction`: a surface is lit on the side the camera
// sees.
fn facing(tangent: Plane, direction: Point) -> Plane {
    return (tangent * select(1.0, -1.0, (tangent & direction) > 0)).normalized();
}

// How brightly a surface is lit, its normal a plane at unit weight facing the viewer: by the lamp, by the sky above
// and the ground below, by the normal's pairing with the upward direction, and by the viewer.
fn brightness(normal: Plane, direction: Point) -> f32 {
    let light = max(normal & lamp, 0.0);
    let fill = ambient * (1 + fill_tilt * (normal & mv.z.dual())) + headlight * -(normal & direction);
    return fill + (1 - ambient - headlight) * light;
}

// The glint where the view ray, reflected in the tangent plane, meets the lamp, on a surface the lamp lights:
// `-d.dual()` is the plane square to d.
fn glint(normal: Plane, direction: Point) -> f32 {
    return pow(max(-((normal >> direction).dual() & lamp), 0.0), shininess) * select(0.0, 1.0, (normal & lamp) > 0);
}

// A colour on a surface with this tangent plane, lit, with its glint.
fn lit(colour: vec3<f32>, tangent: Plane, direction: Point) -> vec3<f32> {
    let normal = facing(tangent, direction);
    return colour * brightness(normal, direction) + gloss * glint(normal, direction);
}
"""


def quadric_crossings() -> str:
    """WGSL for where a ray crosses a quadric's surface, for the solids and strokes alike: `along_ray`, the quadric's
    pairing with the ray's points as a quadratic in the distance along it, `crossings`, its roots, and `entry`, the
    root where the ray enters the solid."""
    return """
const far = 1e30;

// A quadric along a ray, from `start` along `direction`: its pairing with the ray's point, a quadratic in the distance
// t along it, a t^2 + 2 b t + c.
struct Quadratic {
    a: f32,
    b: f32,
    c: f32,
}

fn along_ray(quadric: Quadric, start: Point, direction: Point) -> Quadratic {
    return Quadratic(quadric(direction) & direction, ((quadric(direction) & start) + (quadric(start) & direction)) / 2,
                     quadric(start) & start);
}

// Where a ray crosses a quadric's surface, the roots of the quadratic along it, where it has them: both without
// cancellation; a half-space, with no square term, has only the second, the first at `far`.
fn crossings(along: Quadratic) -> vec2<f32> {
    let discriminant = along.b * along.b - along.a * along.c;
    let q = -(along.b + select(-1.0, 1.0, along.b >= 0) * sqrt(discriminant));
    return vec2<f32>(select(q / along.a, far, along.a == 0), along.c / q);
}

// Where a ray enters a solid quadric: the crossing where the pairing falls through zero, its slope there, 2 (a t + b),
// twice the negative square root of the discriminant; the first root where b is positive, the second where it is
// negative, and `far` where the ray misses the surface or only leaves a half-space.
fn entry(along: Quadratic) -> f32 {
    let roots = crossings(along);
    return select(far, select(roots.y, roots.x, along.b >= 0), along.b * along.b - along.a * along.c >= 0);
}
"""


def quadric_regions() -> str:
    """WGSL intersecting a ray with an intersection of quadrics: its first surface between two distances,
    and the test that a point lies inside all the other quadrics. Scene traversal and surface ordering stay with
    the caller; both use the same crossings and containment test."""
    return quadric_crossings() + """
var<storage, read> quadrics: array<Quadric>;

struct SurfaceHit {
    distance: f32,
    quadric: u32,
    tangent: Plane,
}

fn inside_quadrics(first: u32, end: u32, own: u32, point: Point, tolerance: f32) -> bool {
    for (var index = first; index < end; index++) {
        if (index != own && (quadrics[index](point) & point) > tolerance) {
            return false;
        }
    }
    return true;
}

fn region_hit(first: u32, end: u32, start: Point, direction: Point, nearest: f32, farthest: f32,
              tolerance: f32) -> SurfaceHit {
    var hit = SurfaceHit(farthest, 0u, mv.z);
    for (var own = first; own < end; own++) {
        let quadric = quadrics[own];
        let along = along_ray(quadric, start, direction);
        if (along.b * along.b - along.a * along.c < 0) {
            continue;
        }
        let roots = crossings(along);
        for (var which = 0u; which < 2u; which++) {
            let distance = roots[which];
            let point = start + direction * distance;
            if (distance > nearest && distance < hit.distance && inside_quadrics(first, end, own, point, tolerance)) {
                hit = SurfaceHit(distance, own, quadric(point));
            }
        }
    }
    return hit;
}
"""


def quadric_tracing() -> str:
    """WGSL tracing rays through solids bounded by quadrics, as `scene` and `tiles` lay them out: `first_hit`, the
    first point a ray meets on a solid's surface after the last surface seen, by `crossings`, where it crosses each
    quadric, and `inside_others`."""
    return quadric_regions() + """
// A solid: the intersection of its quadrics, `quadrics[first]` up to `quadrics[end]`, and whether they are all
// half-spaces, bounded by planes alone.
struct Solid {
    first: u32,
    end: u32,
    flat: u32,
}

var<storage, read> solids: array<Solid>;
// The solids each tile of the picture, `tile` pixels a side, may see: the solids `tile_solids[tile_starts[tile]]` up to
// `tile_solids[tile_starts[tile + 1]]`, in their order; the tiles across the picture.
var<storage, read> tile_starts: array<u32>;
var<storage, read> tile_solids: array<u32>;
var<uniform> tiles_across: u32;
const tile = 16.0;
// A pixel's size in the scene.
var<uniform> pixel: f32;

// How far a point may lie outside a quadric and still count as inside, for points on two surfaces.
const tolerance = 1e-5;
// Surfaces closer along a ray than this share of a pixel lie at one depth, and are taken in the order of their solids,
// then their meshes: every surface is seen once, coincident ones included.
const tie = 0.01;

// The surface a ray meets: how far along it, its tangent plane there, and its solid.
struct Hit {
    distance: f32,
    tangent: Plane,
    solid: u32,
}

// Whether a surface, by its distance along a ray and its place in the order of surfaces, comes after another.
fn after(distance: f32, order: u32, last_distance: f32, last_order: u32) -> bool {
    let close = tie * pixel;
    return distance > last_distance + close || (abs(distance - last_distance) <= close && order > last_order);
}

// Whether a surface comes before another, among those after the last seen.
fn before(distance: f32, order: u32, other_distance: f32, other_order: u32) -> bool {
    let close = tie * pixel;
    return distance < other_distance - close || (abs(distance - other_distance) <= close && order < other_order);
}

// The nearer of a hit and a solid bounded by planes alone, after the last surface seen: along the ray each plane's
// pairing is linear, 2 b t + c, and the ray is inside where every one is negative, from the last plane it enters to the
// first it leaves; a cost in the number of planes, not in its square.
fn planes_hit(solid: Solid, index: u32, start: Point, direction: Point, last_distance: f32, last_order: u32,
              nearest: Hit) -> Hit {
    var hit = nearest;
    var entry = -far;
    var exit = far;
    var entering = solid.end;
    var leaving = solid.end;
    for (var own = solid.first; own < solid.end; own++) {
        let along = along_ray(quadrics[own], start, direction);
        // Along a plane, the ray is inside it everywhere or nowhere.
        if (along.b == 0) {
            if (along.c > 0) {
                return hit;
            }
            continue;
        }
        let crossing = -along.c / (2 * along.b);
        if (along.b < 0 && crossing > entry) {
            entry = crossing;
            entering = own;
        }
        if (along.b > 0 && crossing < exit) {
            exit = crossing;
            leaving = own;
        }
    }
    if (entry >= exit) {
        return hit;
    }
    let distances = vec2<f32>(entry, exit);
    let planes = vec2<u32>(entering, leaving);
    for (var which = 0u; which < 2u; which++) {
        let distance = distances[which];
        if (planes[which] < solid.end && after(distance, index, last_distance, last_order)
            && before(distance, index, hit.distance, hit.solid)) {
            hit = Hit(distance, quadrics[planes[which]](start + direction * distance), index);
        }
    }
    return hit;
}

// The first point a ray meets on a solid's surface after the last surface seen, by distance and then by the solids'
// order, among the solids its tile of the picture may see: on one of its quadrics, and inside all the others.
fn first_hit(start: Point, direction: Point, last_distance: f32, last_order: u32, seen: u32) -> Hit {
    var hit = Hit(far, mv.z, 0u);
    for (var listed = tile_starts[seen]; listed < tile_starts[seen + 1]; listed++) {
        let index = tile_solids[listed];
        let solid = solids[index];
        if (solid.flat == 1u) {
            hit = planes_hit(solid, index, start, direction, last_distance, last_order, hit);
            continue;
        }
        for (var own = solid.first; own < solid.end; own++) {
            let quadric = quadrics[own];
            let along = along_ray(quadric, start, direction);
            if (along.b * along.b - along.a * along.c < 0) {
                // Outside this quadric all along the ray, the ray never enters the solid: its other quadrics need no
                // test. Inside it all along, this quadric has no surface on the ray.
                if (along.a > 0) {
                    break;
                }
                continue;
            }
            let roots = crossings(along);
            for (var which = 0u; which < 2u; which++) {
                let distance = roots[which];
                let point = start + direction * distance;
                if (after(distance, index, last_distance, last_order) && before(distance, index, hit.distance, hit.solid)
                    && inside_quadrics(solid.first, solid.end, own, point, tolerance)) {
                    // The quadric's plane at a point of its surface is the tangent plane there.
                    hit = Hit(distance, quadric(point), index);
                }
            }
        }
    }
    return hit;
}
"""


source = """
// Each solid's colour, and how much of what lies behind it it hides.
struct Look {
    colour: vec3<f32>,
    opacity: f32,
}

var<storage, read> looks: array<Look>;
// Lines inked on the surfaces of the solids `first` up to `end`, where the ink's quadric pairs to zero with the point,
// `width` pixels across, hiding `opacity` of what lies behind them.
struct Ink {
    first: u32,
    end: u32,
    colour: vec3<f32>,
    width: f32,
    opacity: f32,
}

var<storage, read> inks: array<Ink>;
var<storage, read> ink_quadrics: array<Quadric>;
// Where each ink shows: where its region's quadric pairs negatively with the point.
var<storage, read> ink_regions: array<Quadric>;
// Meshes drawn at every sample of every pixel, mesh after mesh, each sample one cell of a grid `samples_per_side` to a
// pixel's side, rows first, each row `mesh_row` cells long: its colour, premultiplied by how much of what lies behind
// it it hides, a byte a channel packed in a word, red first, and its depth along the view, both zero where the mesh
// shows nothing.
var<storage, read> mesh_colours: array<u32>;
var<storage, read> mesh_depths: array<f32>;
var<uniform> mesh_row: u32;
// How much of what each mesh covers it hides, a share of its colour's own.
var<storage, read> mesh_opacities: array<f32>;
var<uniform> n_meshes: u32;
// A camera looking along its own -z, its rays parallel, starting `back` behind the middle of the view; the
// image half as high as this. Its frame, `axes`, is given from Python.
var<uniform> camera: Motor;
var<uniform> back: f32;
var<uniform> half_height: f32;
// What shows through where the surfaces let light pass: black, nothing, for a layer to compose with others.
var<uniform> background: vec3<f32>;

// A sample's colour and its depth along the view, in the first channel, each with its coverage.
struct Sample {
    colour: vec4<f32>,
    depth: vec4<f32>,
}

const origin = mv.w.dual();
// The most surfaces a ray sees through, nearest first, and how little light may pass for it to stop looking.
const n_layers = 8u;
const opaque = 0.004;

// A surface's colour and opacity, and whether an ink covers it there.
struct Painted {
    colour: vec4<f32>,
    inked: bool,
}

// The share of an ink's own colour it keeps whatever the light, so that a line on a surface turned from the lamp
// still reads in its colour, while the rest takes the surface's shading.
const ink_own = 0.5;

// A hit's colour and opacity with the lines inked on its solid, and whether one covers it: the distance along the
// surface to an ink's zero set is its pairing with the point over the part of its gradient along the surface, the
// quadric's plane there less its part along the normal.
fn inked(hit: Hit, point: Point, normal: Plane) -> Painted {
    let look = looks[hit.solid];
    var colour = vec4<f32>(look.colour, look.opacity);
    var painted = false;
    for (var index = 0u; index < arrayLength(&inks); index++) {
        let ink = inks[index];
        // The cheap tests first: the ink's solids, then its region.
        if (hit.solid < ink.first || hit.solid >= ink.end || (ink_regions[index](point) & point) > tolerance) {
            continue;
        }
        let gradient = ink_quadrics[index](point) * 2;
        let along = gradient - normal * (gradient | normal);
        let distance = abs(ink_quadrics[index](point) & point) / along.norm();
        let covers = distance < ink.width / 2 * pixel;
        colour = select(colour, vec4<f32>(ink.colour, ink.opacity), covers);
        painted = painted || covers;
    }
    return Painted(colour, painted);
}

// A hit's colour, lit by the lamp, the sky and the viewer, and its opacity; an ink keeps a share of its own colour.
fn shade(hit: Hit, start: Point, direction: Point) -> vec4<f32> {
    let normal = facing(hit.tangent, direction);
    let surface = inked(hit, start + direction * hit.distance, normal);
    let own = select(0.0, ink_own, surface.inked);
    let shaded = mix(surface.colour.rgb * brightness(normal, direction), surface.colour.rgb, own);
    return vec4<f32>(shaded + gloss * glint(normal, direction), surface.colour.a);
}

// A mesh's surface at a sample: its colour, premultiplied by how much it hides, and its distance along the ray.
struct Seen {
    colour: vec4<f32>,
    distance: f32,
    order: u32,
}

// The nearest surface of the meshes at a sample's cell after the last surface seen, at `far` where there is none;
// the meshes come after the solids in the order of surfaces.
fn mesh_after(cell: u32, last_distance: f32, last_order: u32) -> Seen {
    let n_cells = mesh_row * pixels.y * samples_per_side;
    var nearest = Seen(vec4<f32>(0.0), far, 0u);
    for (var mesh = 0u; mesh < n_meshes; mesh++) {
        let colour = unpack4x8unorm(mesh_colours[mesh * n_cells + cell]) * mesh_opacities[mesh];
        let along = mesh_depths[mesh * n_cells + cell] + back;
        let order = arrayLength(&solids) + mesh;
        if (colour.a > 0 && after(along, order, last_distance, last_order) && before(along, order, nearest.distance, nearest.order)) {
            nearest = Seen(colour, along, order);
        }
    }
    return nearest;
}

// The surfaces along a ray, solids' and meshes' alike, nearest first, each hiding its share of what lies behind it;
// the depth is the nearest one's, weighted as the colour is by how much the surfaces hide.
fn sample(place: vec2<f32>) -> Sample {
    // The ray's start in the camera's frame: across and up the image, and `back` toward the viewer.
    let across = screen(place) * half_height;
    let start = camera >> (origin + (axes.dual() * vec3<f32>(across, back)).sum());
    let direction = camera >> -mv.z.dual();
    // The sample's cell in the meshes' grid, and its tile of the picture.
    let grid = vec2<u32>(floor(place * f32(samples_per_side)));
    let cell = grid.y * mesh_row + grid.x;
    let corner = vec2<u32>(floor(place / tile));
    let seen = corner.y * tiles_across + corner.x;
    // Nothing seen yet: every surface ahead of the camera comes after.
    var last_distance = 0.0;
    var last_order = 0xffffffffu;
    var hit = first_hit(start, direction, last_distance, last_order, seen);
    var nearest = far;
    var colour = vec3<f32>(0.0);
    var passing = 1.0;
    for (var layer = 0u; layer < n_layers && passing > opaque; layer++) {
        let mesh = mesh_after(cell, last_distance, last_order);
        if (min(mesh.distance, hit.distance) >= far) {
            break;
        }
        // Each surface seen once, the next one after it in distance and then in order.
        if (before(mesh.distance, mesh.order, hit.distance, hit.solid)) {
            colour += passing * mesh.colour.rgb;
            passing *= 1 - mesh.colour.a;
            nearest = min(nearest, mesh.distance);
            last_distance = mesh.distance;
            last_order = mesh.order;
        } else {
            let surface = shade(hit, start, direction);
            colour += passing * surface.a * surface.rgb;
            passing *= 1 - surface.a;
            nearest = min(nearest, hit.distance);
            last_distance = hit.distance;
            last_order = hit.solid;
            hit = first_hit(start, direction, last_distance, last_order, seen);
        }
    }
    if (nearest >= far) {
        return Sample(vec4<f32>(background, 0.0), vec4<f32>(0.0));
    }
    return Sample(vec4<f32>(colour + passing * background, 1 - passing),
                  vec4<f32>((nearest - back) * (1 - passing), 0.0, 0.0, 1.0));
}
"""


# --- shapes ---------------------------------------------------------------------------
def translator(offset: Point) -> Motor:
    """The motor carrying every point along a direction, a point at infinity."""
    return ((mv.w ^ offset.dual()) / 2).exp()


def placed(quadric: Quadric, motor: Motor) -> Quadric:
    """A quadric moved by a motor: each point moved back, measured, and its plane moved forward."""
    return motor >> quadric(motor << Point)


def ellipsoid(semi_axes: np.ndarray) -> Quadric:
    """Solid ellipsoids about the origin, their semi-axes along x, y and z on the last axis."""
    return (axes * (axes & Point) / semi_axes**2).sum(axis=-1) - mv.w * (mv.w & Point)


def solid_of_dual_quadric(shape: DualQuadric) -> Quadric:
    """The solids about the origin whose dual quadrics are `shape`, each plane through the origin to the direction of
    the point its tangent plane touches, as `context.include_from_algebra` carries the dual quadrics of the algebra of space:
    with the origin's own dyad taken away, the dual quadric of a closed surface, inverted into the quadric that pairs
    points to at most zero inside."""
    return (shape - origin * (Plane & origin)).inverse()


def ball(centres: Point, radius: Scalar) -> Quadric:
    """Solid balls about points, of any weight."""
    round_form = (axes * (axes & Point)).sum(axis=-1) / radius**2 - mv.w * (mv.w & Point)
    return placed(round_form, translator(centres / (mv.w & centres) - origin))


# All of space: the quadric pairing every point of unit weight to minus one.
everywhere = -mv.w * (mv.w & Point)


def half_space(plane: Plane) -> Quadric:
    """The points on a plane's negative side, as the quadric that pairs them with the plane and with the
    plane at infinity, symmetrized."""
    return (plane * (mv.w & Point) + mv.w * (plane & Point)) / 2


def aligned(starts: Point, ends: Point) -> tuple[Plane, np.ndarray, Motor]:
    """For segments between points of unit weight: the unit plane through the origin square to each, how z lies
    along it, `1` or `-1`, and the turn carrying z onto that, whichever way along the segment lies nearer z, so the
    turn never reverses z."""
    across = (-(ends - starts).dual()).normalized()                           # [...] Plane
    sense = np.where((across & mv.z.dual()) >= 0, 1.0, -1.0)
    return across, sense, (1 + across * sense * mv.z).normalized()


def rods(starts: Point, ends: Point, radius: float) -> Quadric:
    """Solid rods of one radius between points, `[..., 3]` quadrics each: a cylinder about the line through the
    ends, and the half-spaces past each end, square to the line."""
    starts, ends = starts / (mv.w & starts), ends / (mv.w & ends)
    across, _, turn = aligned(starts, ends)
    # The cylinder about z, turned onto the rod and moved to its start.
    cylinder = (mv.x * (mv.x & Point) + mv.y * (mv.y & Point)) / radius**2 - mv.w * (mv.w & Point)
    tube = placed(cylinder, translator(starts - origin) * turn)              # [...] Quadric
    # Past each end: the plane square to the rod through the end, facing out.
    beyond_end = half_space(across - mv.w * (across & ends))
    before_start = half_space(-across + mv.w * (across & starts))
    return stack([tube, beyond_end, before_start], axis=-1)


def arrows(starts: Point, ends: Point, radius: float, head_length: float, head_radius: float) -> tuple[Quadric, Quadric]:
    """Arrows between points: their shafts, rods `[..., 3]` up to where the heads begin, and their heads, cones
    `[..., 3]` from there to the ends, each the cone and the half-spaces past its base and its tip."""
    starts, ends = starts / (mv.w & starts), ends / (mv.w & ends)
    across, sense, turn = aligned(starts, ends)
    # A head's length back from each end, along the arrow: the dual of the plane square to it.
    bases = ends - across.dual() * head_length                                # [...] Point
    # The cone about z with its base at the origin and its tip a head's length along the arrow: the distance from
    # the axis against the radius, less what remains of the head's length.
    remaining = mv.w - mv.z * (sense / head_length)                           # [...] Plane
    cone = (mv.x * (mv.x & Point) + mv.y * (mv.y & Point)) / head_radius**2 - remaining * (remaining & Point)
    head = stack([cone, half_space(-mv.z * sense), half_space(mv.z * sense - mv.w * head_length)], axis=-1)
    return rods(starts, bases, radius), placed(head, (translator(bases - origin) * turn)[..., None])


def frame(length: float, radius: float, head_length: float, head_radius: float) -> tuple[Quadric, Quadric]:
    """The coordinate frame: arrows along x, y and z from the shared origin, a length long; their shafts and heads
    `[3, 3]` each."""
    return arrows(origin, origin + axes.dual() * length, radius, head_length, head_radius)


def boxes(motors: Motor, half_extents: np.ndarray) -> Plane:
    """Solid boxes placed by motors, their half extents along their own x, y and z on the last axis, bounded by planes
    alone, `[..., 6]` each: the planes of the faces, the box on their negative sides."""
    reach = mv.w * half_extents                                               # [..., 3] Plane
    faces = concatenate([axes - reach, -axes - reach], axis=-1)               # [..., 6] Plane
    return motors[..., None] >> faces


def outward(planes: Plane, middles: Point) -> Plane:
    """Planes turned so that points of unit weight lie on their negative sides."""
    side = planes & middles
    return planes * (np.asarray(side < 0, dtype=int) - np.asarray(side > 0, dtype=int))


def plates(corners: Point, thickness: float) -> Plane:
    """Flat convex polygons of a thickness, their corners `[..., corners]` in order around each, bounded by planes
    alone, `[..., corners + 2]` each: the two half the thickness either side of the polygon's plane, and one along each
    edge, facing away from the polygon's middle."""
    corners = corners / (mv.w & corners)
    middle = corners.mean(axis=-1)                                            # [...] Point
    # The polygon's plane, the join of its first three corners, at unit weight, through its middle.
    joined = ((corners[..., 0] & corners[..., 1] & corners[..., 2]) * flip).normalized()   # [...] Plane
    face = joined - mv.w * (joined & middle)
    # Along each edge, the plane through the edge and the direction square to the polygon's plane.
    square = (face - mv.w * (face & origin)).dual()                           # [...] Point
    following = concatenate([corners[..., 1:], corners[..., :1]], axis=-1)    # [..., corners] Point
    sides = outward((corners & following & square[..., None]) * flip, middle[..., None])   # [..., corners] Plane
    return concatenate([stack([face, -face], axis=-1) - mv.w * (thickness / 2), sides], axis=-1)


def polyhedra(corners: Point, faces: np.ndarray) -> Plane:
    """Convex polyhedra, their corners `[..., corners]` and three or more corners of each face `[faces, corners]`,
    bounded by planes alone, `[..., faces]` each: the planes of the faces, the joins of their first three corners,
    facing away from the middle."""
    corners = corners / (mv.w & corners)
    joined = (corners[..., faces[:, 0]] & corners[..., faces[:, 1]] & corners[..., faces[:, 2]]) * flip   # [..., faces] Plane
    return outward(joined, corners.mean(axis=-1)[..., None])


def edges(starts: Point, ends: Point, radius: float) -> tuple[Quadric, Quadric]:
    """Edges between points: rods `[..., 3]` and balls of the same radius at both ends `[ends, 1]`, so that edges
    meeting at a point join round."""
    return rods(starts, ends, radius), ball(stack([starts, ends], axis=-1).reshape(-1), radius)[:, None]


# --- plumbing -------------------------------------------------------------------------
# The side of a tile of the picture, in pixels, as the shader's `tile`.
TILE = 16


@cache
def shader() -> Shader:
    return Shader(supersampled() + quadric_tracing() + lamp_shading() + source, ga,
                  dict(Point=Point, Plane=Plane, Motor=Motor, Quadric=Quadric), dict(axes=axes))


class Ink(NamedTuple):
    """Lines inked on the surfaces of a part's solids, where each quadric `[lines]` pairs to zero with the point,
    within where each region `[lines]` pairs negatively with it (`everywhere` for no bound): their colours,
    `[lines, 3]` or one for all, their width in pixels, `np.inf` to fill the region, and how much of what lies
    behind them they hide. The parts count the opaque ones first, then the translucent."""
    part: int
    quadrics: Quadric
    regions: Quadric
    colours: np.ndarray
    width: float
    opacity: float


def spheres(solids: Quadric) -> tuple[Point, Scalar]:
    """A ball holding each of a batch of solids `[solids, quadrics]`: its centre `[solids]` and radius `[solids]`,
    infinite where no quadric bounds it. How a quadric grows along the directions, its form on them, `quadric(direction)
    & direction`, says what it bounds: one growing along all three, an ellipsoid, is held by a ball about its centre,
    the pole of the plane at infinity; one growing along two, a cylinder or a cone, by its axis between where the
    solid's quadrics that the axis runs straight through cut it; any other leaves the solid unbounded."""
    # The form on directions: its rates of growth, slowest first, and the directions it grows along at those rates.
    rates, normals = (solids(normal_direction) & normal_direction).eigh()   # [solids, quadrics, 3] Scalar, Plane
    directions = normals.dual()                                               # [solids, quadrics, 3] Point
    zero, one, infinity = mv.scalar([0]), mv.scalar([1]), mv.scalar([np.inf])
    # A rate is a growth where it exceeds round-off of the quadric's size, the largest gain of its map.
    size = solids.svdvals()[..., 0]                                          # [solids, quadrics] Scalar+
    growing = rates > 1e-9 * size[..., None]
    positive_rates = where(growing, rates, infinity)
    slowest = positive_rates.min(axis=-1)
    count = growing.sum(axis=-1)
    # The centre: the origin moved along each direction the quadric grows along, to where its pairing with the point
    # stops changing along it; the pole of the plane at infinity, `quadric.solve(mv.w)`, of an ellipsoid.
    slopes = ((solids(origin)[..., None] & directions) + (solids[..., None](directions) & origin)) / 2
    steps = where(growing, -slopes / where(growing, rates, one), zero)       # [solids, quadrics, 3] Scalar+
    centres = origin + (directions * steps).sum(axis=-1)                      # [solids, quadrics] Point

    def reach(points: Point) -> Scalar:
        """How far a quadric's solid reaches from points of unit weight across its slowest growth: its pairing
        with each, against that rate."""
        return (-(solids(points) & points).clip(-np.inf, 0) / slowest).square_root()

    # A cylinder's or a cone's axis, the direction it grows slowest along, through the centre, and where the other
    # quadrics of its solid that the axis runs straight through, linearly along it, b t + c, cut it.
    axis = directions[..., 0]                                                 # [solids, quadrics] Point
    others = solids[..., None, :]                                             # [solids, 1, quadrics] Quadric
    along = (others(centres[..., None]) & axis[..., None]) + (others(axis[..., None]) & centres[..., None])
    offset = others(centres[..., None]) & centres[..., None]                  # [solids, quadrics, quadrics] Scalar+
    straight = (others(axis[..., None]) & axis[..., None]).abs() <= 1e-9 * size[:, None, :]
    cuts = straight & ((along > 0) | (along < 0)) & ~np.eye(solids.shape[-1], dtype=bool)
    crossing = -offset / where(cuts, along, one)
    upper_cuts = where(cuts & (along > 0), crossing, infinity)
    lower_cuts = where(cuts & (along < 0), crossing, -infinity)
    upper = upper_cuts.min(axis=-1)
    lower = lower_cuts.max(axis=-1)
    cylinder = (count == 2) & upper.isfinite() & lower.isfinite()
    upper, lower = where(cylinder, upper, zero), where(cylinder, lower, zero)
    # Held across the widest of its two ends, from the middle between them.
    lower_reach, upper_reach = reach(centres + axis * lower), reach(centres + axis * upper)
    widest = where(lower_reach > upper_reach, lower_reach, upper_reach)
    capped_radius = (((upper - lower) / 2).squared() + widest.squared()).square_root()
    radius = where(count == 3, reach(centres), where(cylinder, capped_radius, infinity))
    centres = centres + axis * ((upper + lower) / 2)
    # The smallest ball any of its quadrics gives.
    chosen = radius.argmin(axis=-1)                                          # [solids]
    picked = np.arange(len(chosen))
    return centres[picked, chosen], radius[picked, chosen]


def tiles(centres: Point, radii: Scalar, view: View) -> dict:
    """The solids each tile of the picture may see, from the balls holding them, their centres and radii `[solids]`:
    each tile's range in a list of solids, the list, and the tiles across; a solid without a ball in every tile."""
    rows, columns = view.shape
    across, down = -(-columns // TILE), -(-rows // TILE)
    seen = view.across(centres)                                                # [solids, 2]
    reach = (radii / view.pixel + 1).to_array()                                # [solids] pixels
    column = (seen[:, 0] + view.half[0]) / view.pixel
    row = (view.half[1] - seen[:, 1]) / view.pixel
    bounded = np.isfinite(reach)
    first = np.where(bounded, np.floor((np.stack([column, row]) - reach) / TILE), 0).clip(0, [[across], [down]])
    last = np.where(bounded, np.floor((np.stack([column, row]) + reach) / TILE), [[across - 1], [down - 1]]).clip(-1, [[across - 1], [down - 1]])
    widths, heights = np.maximum(last - first + 1, 0).astype(int)              # [solids] each
    sizes = widths * heights
    solid = np.repeat(np.arange(len(radii)), sizes)
    within = np.arange(sizes.sum()) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    tile = ((first[1][solid] + within // np.maximum(widths[solid], 1)) * across + first[0][solid]
            + within % np.maximum(widths[solid], 1)).astype(int)
    order = np.lexsort((solid, tile))
    starts = np.concatenate([[0], np.cumsum(np.bincount(tile, minlength=across * down))])
    # A list never empty, for the shader's array.
    return dict(tile_starts=starts.astype(np.uint32), tile_solids=np.append(solid[order], 0).astype(np.uint32),
                tiles_across=across)


def scene(parts: list[tuple[Quadric, np.ndarray]], translucent: list[tuple[Quadric, np.ndarray, float]],
          inks: list[Ink]) -> tuple[Quadric, dict, dict, Quadric, Quadric, dict, tuple[Point, Scalar]]:
    """Every solid's quadrics, solid after solid, each solid's range of them and whether they are all half-spaces, and
    its colour and opacity, from parts: each a batch of solids, `[solids, quadrics]` Quadric, all of one kind, or
    `[solids, planes]` Plane for solids bounded by planes alone, on their negative sides, their colours, `[solids, 3]`
    or one for all, and for the translucent parts their opacity; and every ink's quadric, with its part's range of
    solids, colour and width."""
    parts = [(solids, colour, 1.0) for solids, colour in parts] + list(translucent)
    quadrics, counts, colours, opacities, flats, centres, radii = [], [], [], [], [], [], []
    for solids, colour, opacity in parts:
        # Planes, values rather than maps, bound their solids as half-spaces, solids no ball holds.
        flat = solids.arity == 0
        if flat:
            centre = origin[None].broadcast_to(solids.shape[:1])
            radius = mv.scalar(np.full((solids.shape[0], 1), np.inf))
            solids = half_space(solids)
        else:
            centre, radius = spheres(solids)
        quadrics.append(solids.reshape(-1))
        centres.append(centre)
        radii.append(radius)
        counts.append(np.full(solids.shape[0], solids.shape[1]))
        colours.append(np.broadcast_to(colour, (solids.shape[0], 3)))
        opacities.append(np.full(solids.shape[0], opacity))
        flats.append(np.full(solids.shape[0], flat))
    # A solid of no quadrics, which no ray meets, and a quadric no solid holds, so that the arrays are never empty.
    quadrics.append(everywhere[None])
    counts.append(np.zeros(1, dtype=int))
    colours.append(np.zeros((1, 3)))
    opacities.append(np.zeros(1))
    flats.append(np.zeros(1, dtype=bool))
    # It lies in no tile: a ball of no size, out of every picture.
    centres.append(origin[None])
    radii.append(mv.scalar([[-np.inf]]))
    ends = np.cumsum(np.concatenate(counts))
    # The solids each part spans, part after part.
    part_ends = np.cumsum([solids.shape[0] for solids, _, _ in parts])
    part_starts = part_ends - [solids.shape[0] for solids, _, _ in parts]
    # An ink over no solids stands for none, so that the inks' arrays are never empty.
    inks = list(inks) + [Ink(0, everywhere[None], everywhere[None], np.zeros(3), 0.0, 0.0)]
    ink_quadrics = concatenate([ink.quadrics.reshape(-1) for ink in inks])
    ink_regions = concatenate([ink.regions.reshape(-1) for ink in inks])
    sizes = [int(np.prod(ink.quadrics.shape)) for ink in inks]
    marked = dict(first=np.repeat([part_starts[ink.part] for ink in inks[:-1]] + [0], sizes),
                  end=np.repeat([part_ends[ink.part] for ink in inks[:-1]] + [0], sizes),
                  colour=np.concatenate([np.broadcast_to(ink.colours, (size, 3)) for ink, size in zip(inks, sizes)]),
                  width=np.repeat([ink.width for ink in inks], sizes), opacity=np.repeat([ink.opacity for ink in inks], sizes))
    return (concatenate(quadrics),
            dict(first=ends - np.concatenate(counts), end=ends, flat=np.concatenate(flats).astype(np.uint32)),
            dict(colour=np.concatenate(colours), opacity=np.concatenate(opacities)),
            ink_quadrics, ink_regions, marked, (concatenate(centres), concatenate(radii)))


class View(NamedTuple):
    """What a picture is seen from: a camera looking along its own -z with parallel rays, half the image's width and
    height in the scene's units, and the pixels across it."""
    camera: Motor
    half: np.ndarray
    pixels: int

    @property
    def pixel(self) -> float:
        """A pixel's size in the scene."""
        return 2 * self.half[0] / self.pixels

    @property
    def shape(self) -> tuple[int, int]:
        """The image's rows and columns."""
        return round(self.pixels * self.half[1] / self.half[0]), self.pixels

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """The image's left, right, bottom and top about its middle, for showing it in matplotlib axes beside places
        from `across`."""
        return -self.half[0], self.half[0], -self.half[1], self.half[1]

    def squared(self) -> View:
        """The view widened to a square about its middle."""
        return self._replace(half=np.full(2, self.half.max()))

    def across(self, points: Point) -> np.ndarray:
        """Points, of any weight, across and up the image from its middle `[..., 2]`: read in the camera's own frame
        by its planes x and y."""
        seen = self.camera << points
        return ((axes[:2] & seen[..., None]) / (mv.w & seen)[..., None]).to_array()


def view(points: list[Point], reach: float, elevation: float, azimuth: float, margin: float, pixels: int) -> View:
    """A view framing batches of points, of any weight, and `reach` around each, from an elevation and azimuth, in
    radians, z up and azimuth zero looking along +y, with a margin as a share of the larger side, `pixels` across:
    framed by the support planes square to the image's right and up, the planes that touch the points on either
    side."""
    turn = (mv.xy * (-azimuth / 2)).exp() * (mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()     # [] Motor
    # The planes through the origin square to the image's right and up, and the one across the view.
    screen = turn >> axes                                                     # [3] Plane
    seen = concatenate([(batch / (mv.w & batch)).reshape(-1) for batch in points])   # [points] Point
    # The outermost points across and up the image, where each support plane touches them, on either side.
    heights = screen[:2, None] & seen                                         # [2, points] Scalar
    low, high = heights.min(axis=-1), heights.max(axis=-1)                    # [2] Scalar+, Scalar+
    # The planes halfway between the support planes, met with the plane across the view through the origin.
    middles = screen[:2] - mv.w * (low + high) / 2                             # [2] Plane
    centre = (middles[0] ^ middles[1] ^ screen[2]) * flip                    # [] Point
    half = (high - low) / 2 + reach
    half = half + margin * half.max()
    return View(aim(centre, elevation, azimuth), half.to_array(), pixels)


def aim(centre: Point, elevation: float, azimuth: float) -> Motor:
    """A camera looking at a point, of any weight, from an elevation and azimuth, in radians, z up and azimuth zero
    looking along +y."""
    turn = (mv.xy * (-azimuth / 2)).exp() * (mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()     # [] Motor
    return translator(centre / (mv.w & centre) - origin) * turn


def lamp(camera: Motor) -> Point:
    """The lamp's direction for a camera: up and to the right of the viewer, and a little behind, so that strokes
    running up the image catch it as well as the faces turned up."""
    toward = camera >> mv.direction([0.35, 0.5, 0.8])
    return toward / toward.dual().norm()


def lighting(camera: Motor, glint: float) -> dict:
    """The uniforms of `lamp_shading` for a camera, with a glint of `glint`."""
    return dict(lamp=lamp(camera), ambient=ambient, fill_tilt=fill_tilt, headlight=headlight, gloss=glint,
                shininess=shininess)


def layer(parts: list[tuple[Quadric, np.ndarray]], view: View, inks: list[Ink],
          translucent: list[tuple[Quadric, np.ndarray, float]],
          meshes: list[tuple[Samples, float]]) -> tuple[np.ndarray, np.ndarray]:
    """The solids as the view sees them, with the lines inked on them, the translucent solids over and among them,
    each with its opacity, and the meshes among them, each its samples, as `meshes.samples` gives them, and its
    opacity, sample by sample: their colour, `[rows, columns, 4]`, and their depth along the view, `[rows, columns]`,
    each weighted by coverage. The meshes are drawn, traced among the solids and averaged over each pixel in one pass
    of the GPU's work, and only the picture's pixels come back."""
    return traced(parts, view, inks, translucent, meshes, np.zeros(3))


def traced(parts: list[tuple[Quadric, np.ndarray]], view: View, inks: list[Ink],
           translucent: list[tuple[Quadric, np.ndarray, float]], meshes: list[tuple[Samples, float]],
           background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The picture `layer` makes, over a background, which shows through each sample as much as its surfaces let
    light pass: over black, the layer itself."""
    tracer = shader()
    encoder = tracer.device.create_command_encoder()
    # The meshes' samples in the buffers the tracer keeps for any one picture.
    buffers = mesh_buffers(view, len(meshes), lambda name, size, usage: tracer.buffer_kept(("meshes", name), size, usage))
    sampled(encoder, [mesh for mesh, _ in meshes], view, 0, buffers)
    return trace(tracer, encoder, view, scene(parts, translucent, inks), buffers,
                 [opacity for _, opacity in meshes], background)


def trace(tracer: Shader, encoder: Any, view: View, solids: tuple, buffers: MeshBuffers,
          opacities: list[float], background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The solids, as `scene` lays them out, traced among the meshes whose samples the buffers hold, each with its
    opacity, over the background, on the encoder: the picture's colour and depth."""
    quadrics, solids, looks, ink_quadrics, ink_regions, marked, holding = solids
    # A mesh that shows nothing stands for none, so that the opacities are never empty.
    outputs = tracer.encode(encoder, view.shape, (), dict(
        quadrics=quadrics, solids=solids, looks=looks, inks=marked, ink_quadrics=ink_quadrics, ink_regions=ink_regions,
        mesh_colours=Resident(buffers.colours, buffers.size), mesh_depths=Resident(buffers.depths, buffers.size),
        mesh_row=buffers.row_bytes // 4, mesh_opacities=np.array(opacities or [0.0], dtype=np.float32),
        n_meshes=len(opacities), **tiles(*holding, view), pixel=view.pixel, camera=view.camera,
        back=4 * view.half.max(), half_height=view.half[1], background=background, **lighting(view.camera, gloss)))
    image = tracer.read(encoder, outputs, view.shape)["image"]
    return image["colour"], image["depth"][..., 0]


class MeshBuffers(NamedTuple):
    """The GPU buffers meshes' samples are copied into, slot after slot: their colours and their depths, the bytes
    they hold, and the bytes a row of samples takes, padded as a copy pads it."""
    colours: Any
    depths: Any
    size: int
    row_bytes: int


def mesh_buffers(view: View, n_slots: int, made: Any) -> MeshBuffers:
    """Buffers for the samples of `n_slots` meshes over the view's pixels, each from `made(name, size, usage)`."""
    import wgpu

    rows, columns = view.shape
    # A copy's rows start at multiples of 256 bytes; a sample's colour and its depth are four bytes each.
    row_bytes = -(-samples_per_side() * columns * 4 // 256) * 256
    size = max(n_slots * samples_per_side() * rows * row_bytes, 16)
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    return MeshBuffers(made("colours", size, usage), made("depths", size, usage), size, row_bytes)


def sampled(encoder: Any, drawings: list[Samples], view: View, first: int, buffers: MeshBuffers) -> None:
    """The meshes drawn at every sample of the view's pixels, recorded on an encoder, and copied there, on the GPU,
    mesh after mesh, into the buffers' slots from `first` on, the slots before it left as they are."""
    rows, columns = view.shape
    shape = (samples_per_side() * rows, samples_per_side() * columns)
    for index, drawing in enumerate(drawings, start=first):
        # A mesh with nothing to draw, such as a trail not yet begun, shows nothing: its slot cleared.
        if drawing.n_vertices == 0:
            for buffer in (buffers.colours, buffers.depths):
                encoder.clear_buffer(buffer, index * shape[0] * buffers.row_bytes, shape[0] * buffers.row_bytes)
            continue
        # Each mesh's values in buffers of its own: all are written before the encoder runs.
        targets = drawing.shader.encode_draw(encoder, shape, drawing.n_vertices, ("mesh", index), drawing.values)
        for target, buffer in zip(targets, (buffers.colours, buffers.depths)):
            encoder.copy_texture_to_buffer(
                {"texture": target},
                {"buffer": buffer, "offset": index * shape[0] * buffers.row_bytes, "bytes_per_row": buffers.row_bytes,
                 "rows_per_image": shape[0]},
                (shape[1], shape[0], 1))


def draw(parts: list[tuple[Quadric, np.ndarray]], view: View, background: np.ndarray, inks: list[Ink],
         translucent: list[tuple[Quadric, np.ndarray, float]], meshes: list[tuple[Samples, float]]) -> np.ndarray:
    """The solids as the view sees them on the background, with the lines inked on them, the translucent solids over
    and among them, and the meshes among them, as `layer`, `[rows, columns, 3]`: the background blended in on the GPU,
    sample by sample."""
    colour, _ = traced(parts, view, inks, translucent, meshes, np.asarray(background, dtype=float))
    return np.round(np.clip(colour[..., :3], 0, 1) * 255).astype(np.uint8)
