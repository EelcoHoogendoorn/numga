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
from PIL import Image

from numga import NumpyContext, concatenate, stack
from numga.algebras import PGA3D as ga
from examples.wgsl import Shader, supersampled
from examples.wgsl.library import drawn, samples_per_side
from examples.wgsl.runner import Resident

if TYPE_CHECKING:
    from examples.wgsl.meshes import Samples

mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Motor = ga.gatype.rotor()
Quadric = ga.gatype((Plane, Point))                                          # Plane <- Point
axes = stack((mv.x, mv.y, mv.z))                                             # [3] Plane
origin = mv.w.dual()                                                         # [] Point
# The light that falls everywhere, the glint's share and sharpness, and the lamp: from up and to the right
# of a viewer looking along -y.
ambient, gloss, shininess = 0.22, 0.45, 32.0
# How much brighter the light that falls everywhere is on faces turned up than on faces turned sideways, and dimmer
# on faces turned down: a sky above and a ground below, so the form reads away from the lamp too.
fill_tilt = 0.5
# The share of the light that comes from the viewer, so that every face the camera sees keeps its colour, by how
# squarely it is turned to the viewer; the lamp gives the rest.
headlight = 0.25

source = supersampled() + """
// A solid: the intersection of its quadrics, `quadrics[first]` up to `quadrics[end]`, its colour, how much of what
// lies behind it it hides, and whether its quadrics are all half-spaces, bounded by planes alone.
struct Solid {
    first: u32,
    end: u32,
    colour: vec3<f32>,
    opacity: f32,
    flat: u32,
}

var<storage, read> solids: array<Solid>;
// The solids each tile of the picture, `tile` pixels a side, may see: the solids `tile_solids[tile_starts[tile]]` up to
// `tile_solids[tile_starts[tile + 1]]`, in their order; the tiles across the picture.
var<storage, read> tile_starts: array<u32>;
var<storage, read> tile_solids: array<u32>;
var<uniform> tiles_across: u32;
const tile = 16.0;
var<storage, read> quadrics: array<Quadric>;
// Lines inked on the surfaces of the solids `first` up to `end`, where the ink's quadric pairs to zero with the point,
// `width` pixels across, each pixel `pixel` across in the scene, hiding `opacity` of what lies behind them.
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
var<uniform> pixel: f32;
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
// image half as high as this.
var<uniform> camera: Motor;
var<uniform> back: f32;
var<uniform> half_height: f32;
var<uniform> lamp: Point;
var<uniform> ambient: f32;
var<uniform> fill_tilt: f32;
var<uniform> headlight: f32;
var<uniform> gloss: f32;
var<uniform> shininess: f32;
// What shows through where the surfaces let light pass: black, nothing, for a layer to compose with others.
var<uniform> background: vec3<f32>;

// A sample's colour and its depth along the view, in the first channel, each with its coverage.
struct Sample {
    colour: vec4<f32>,
    depth: vec4<f32>,
}

const origin = mv.w.dual();
const far = 1e30;
// How far a point may lie outside a quadric and still count as inside, for points on two surfaces.
const tolerance = 1e-5;
// The most surfaces a ray sees through, nearest first, and how little light may pass for it to stop looking.
const n_layers = 8u;
const opaque = 0.004;
// Surfaces closer along a ray than this share of a pixel lie at one depth, and are taken in the order of their solids,
// then their meshes: every surface is seen once, coincident ones included.
const tie = 0.01;

struct Hit {
    distance: f32,
    tangent: Plane,
    colour: vec3<f32>,
    opacity: f32,
    solid: u32,
}

fn inside_others(solid: Solid, own: u32, point: Point) -> bool {
    for (var index = solid.first; index < solid.end; index++) {
        if (index != own && (quadrics[index](point) & point) > tolerance) {
            return false;
        }
    }
    return true;
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
        let quadric = quadrics[own];
        let b = ((quadric(direction) & start) + (quadric(start) & direction)) / 2;
        let c = quadric(start) & start;
        // Along a plane, the ray is inside it everywhere or nowhere.
        if (b == 0) {
            if (c > 0) {
                return hit;
            }
            continue;
        }
        let crossing = -c / (2 * b);
        if (b < 0 && crossing > entry) {
            entry = crossing;
            entering = own;
        }
        if (b > 0 && crossing < exit) {
            exit = crossing;
            leaving = own;
        }
    }
    if (entry >= exit) {
        return hit;
    }
    let crossings = vec2<f32>(entry, exit);
    let planes = vec2<u32>(entering, leaving);
    for (var which = 0u; which < 2u; which++) {
        let distance = crossings[which];
        if (planes[which] < solid.end && after(distance, index, last_distance, last_order)
            && before(distance, index, hit.distance, hit.solid)) {
            hit = Hit(distance, quadrics[planes[which]](start + direction * distance), solid.colour, solid.opacity, index);
        }
    }
    return hit;
}

// The first point a ray meets on a solid's surface after the last surface seen, by distance and then by the solids'
// order, among the solids its tile of the picture may see: on one of its quadrics, and inside all the others.
fn first_hit(start: Point, direction: Point, last_distance: f32, last_order: u32, seen: u32) -> Hit {
    var hit = Hit(far, mv.z, vec3<f32>(0.0), 0.0, 0u);
    for (var listed = tile_starts[seen]; listed < tile_starts[seen + 1]; listed++) {
        let index = tile_solids[listed];
        let solid = solids[index];
        if (solid.flat == 1u) {
            hit = planes_hit(solid, index, start, direction, last_distance, last_order, hit);
            continue;
        }
        for (var own = solid.first; own < solid.end; own++) {
            let quadric = quadrics[own];
            // Along the ray the quadric is a quadratic in the distance, a t^2 + 2 b t + c.
            let a = quadric(direction) & direction;
            let b = ((quadric(direction) & start) + (quadric(start) & direction)) / 2;
            let c = quadric(start) & start;
            let discriminant = b * b - a * c;
            if (discriminant < 0) {
                // Outside this quadric all along the ray, the ray never enters the solid: its other quadrics need no
                // test. Inside it all along, this quadric has no surface on the ray.
                if (a > 0) {
                    break;
                }
                continue;
            }
            // Both roots without cancellation; a half-space, with no square term, has only the second.
            let q = -(b + select(-1.0, 1.0, b >= 0) * sqrt(discriminant));
            let roots = vec2<f32>(select(q / a, far, a == 0), c / q);
            for (var which = 0u; which < 2u; which++) {
                let distance = roots[which];
                let point = start + direction * distance;
                if (after(distance, index, last_distance, last_order) && before(distance, index, hit.distance, hit.solid)
                    && inside_others(solid, own, point)) {
                    // The quadric's plane at a point of its surface is the tangent plane there.
                    hit = Hit(distance, quadric(point), solid.colour, solid.opacity, index);
                }
            }
        }
    }
    return hit;
}

// A hit's colour and opacity with the lines inked on its solid, and whether one covers it: the distance along the
// surface to an ink's zero set is its pairing with the point over the part of its gradient along the surface, the
// quadric's plane there less its part along the normal.
// A surface's colour and opacity, and whether an ink covers it there.
struct Painted {
    colour: vec4<f32>,
    inked: bool,
}

// The share of an ink's own colour it keeps whatever the light, so that a line on a surface turned from the lamp
// still reads in its colour, while the rest takes the surface's shading.
const ink_own = 0.5;

fn inked(hit: Hit, point: Point, normal: Plane) -> Painted {
    var colour = vec4<f32>(hit.colour, hit.opacity);
    var painted = false;
    for (var index = 0u; index < arrayLength(&inks); index++) {
        let ink = inks[index];
        // The cheap tests first: the ink's solids, then its region.
        if (hit.solid < ink.first || hit.solid >= ink.end || (ink_regions[index](point) & point) > tolerance) {
            continue;
        }
        let gradient = ink_quadrics[index](point) * 2;
        let along = gradient - normal * (gradient | normal);
        let distance = abs(ink_quadrics[index](point) & point) / (along | along).square_root();
        let covers = distance < ink.width / 2 * pixel;
        colour = select(colour, vec4<f32>(ink.colour, ink.opacity), covers);
        painted = painted || covers;
    }
    return Painted(colour, painted);
}

// A hit's colour, lit by the lamp, the sky and the viewer, and its opacity.
fn shade(hit: Hit, start: Point, direction: Point) -> vec4<f32> {
    // Facing the viewer: a tangent plane is lit on the side the ray comes from.
    let tangent = hit.tangent * select(1.0, -1.0, (hit.tangent & direction) > 0);
    let normal = tangent / (tangent | tangent).square_root();
    let light = max(normal & lamp, 0.0);
    // The view ray reflected in the tangent plane, against the lamp: `-d.dual()` is the plane square to d.
    let glint = pow(max(-((normal >> direction).dual() & lamp), 0.0), shininess) * select(0.0, 1.0, light > 0);
    let fill = ambient * (1 + fill_tilt * (normal | mv.z)) + headlight * -(normal & direction);
    let surface = inked(hit, start + direction * hit.distance, normal);
    let lit = surface.colour.rgb * (fill + (1 - ambient - headlight) * light);
    let own = select(0.0, ink_own, surface.inked);
    return vec4<f32>(mix(lit, surface.colour.rgb, own) + gloss * glint, surface.colour.a);
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
    let across = screen(place) * half_height;
    let start = camera >> (origin + mv.z.dual() * back + mv.x.dual() * across.x + mv.y.dual() * across.y);
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


def quadric(matrix: np.ndarray, centre: Point) -> Quadric:
    """The solids `(x - centre) @ matrix @ (x - centre) <= 1` of symmetric matrices `[..., 3, 3]`, about points."""
    at_origin = (axes[:, None] * (axes & Point)[None, :] * matrix).sum(axis=(-1, -2)) - mv.w * (mv.w & Point)
    return placed(at_origin, translator(centre - origin))


def ball(centres: Point, radius: float) -> Quadric:
    """Solid balls about points, of any weight."""
    return placed(ellipsoid(np.full(3, radius)), translator(centres / (mv.w & centres) - origin))


def plane(normals: np.ndarray, offsets: np.ndarray) -> Plane:
    """The planes `normal @ x == offset`, normals `[..., 3]` and offsets `[...]`, positive on the side the normal
    points to."""
    return (axes * normals).sum(axis=-1) - mv.w * offsets


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
    across = -(ends - starts).dual()                                          # [...] Plane
    across = across / (across | across).square_root()
    sense = np.where((across | mv.z).to_array() >= 0, 1.0, -1.0)
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
    return arrows(mv.point(np.zeros((3, 3))), mv.point(length * np.eye(3)), radius, head_length, head_radius)


def boxes(motors: Motor, half_extents: np.ndarray) -> Quadric:
    """Solid boxes placed by motors, their half extents along their own x, y and z on the last axis, `[..., 6]`
    quadrics each: the half-spaces inside each face."""
    reach = mv.w * half_extents                                               # [..., 3] Plane
    faces = concatenate([axes - reach, -axes - reach], axis=-1)               # [..., 6] Plane
    return placed(half_space(faces), motors[..., None])


def plates(corners: np.ndarray, thickness: float) -> Quadric:
    """Flat convex polygons of a thickness, their corners `[..., corners, 3]` in order around each, `[..., corners + 2]`
    quadrics each: the half-spaces within half the thickness either side of the polygon's plane, and one along each
    edge, facing away from the polygon's middle."""
    normal = np.cross(corners[..., 1, :] - corners[..., 0, :], corners[..., 2, :] - corners[..., 0, :])
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)                  # [..., 3]
    middle = corners.mean(axis=-2)                                            # [..., 3]
    outwards = np.cross(np.roll(corners, -1, axis=-2) - corners, normal[..., None, :])   # [..., corners, 3]
    outwards *= np.sign(np.sum(outwards * (corners - middle[..., None, :]), axis=-1, keepdims=True))
    across = np.stack([normal, -normal], axis=-2)                              # [..., 2, 3]
    normals = np.concatenate([across, outwards], axis=-2)
    offsets = np.concatenate([np.sum(across * middle[..., None, :], axis=-1) + thickness / 2,
                              np.sum(outwards * corners, axis=-1)], axis=-1)
    return half_space(plane(normals, offsets))


def polyhedra(corners: np.ndarray, faces: np.ndarray) -> Quadric:
    """Convex polyhedra, their corners `[..., corners, 3]` and three or more corners of each face `[faces, corners]`,
    `[..., faces]` quadrics each: the half-spaces inside the planes of the faces, facing away from the middle."""
    a, b, c = (corners[..., faces[:, index], :] for index in range(3))       # [..., faces, 3] each
    normals = np.cross(b - a, c - a)
    middle = corners.mean(axis=-2)                                            # [..., 3]
    normals *= np.sign(np.sum(normals * (a - middle[..., None, :]), axis=-1, keepdims=True))
    return half_space(plane(normals, np.sum(normals * a, axis=-1)))


def edges(starts: Point, ends: Point, radius: float) -> tuple[Quadric, Quadric]:
    """Edges between points: rods `[..., 3]` and balls of the same radius at both ends `[ends, 1]`, so that edges
    meeting at a point join round."""
    return rods(starts, ends, radius), ball(stack([starts, ends], axis=-1).reshape(-1), radius)[:, None]


# --- plumbing -------------------------------------------------------------------------
# The side of a tile of the picture, in pixels, as the shader's `tile`.
TILE = 16


@cache
def shader() -> Shader:
    return Shader(source, ga, dict(Point=Point, Plane=Plane, Motor=Motor, Quadric=Quadric))


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


def spheres(solids: Quadric) -> tuple[np.ndarray, np.ndarray]:
    """A ball holding each of a batch of solids `[solids, quadrics]`: its centre `[solids, 3]` and radius `[solids]`,
    infinite where no quadric bounds it. A quadric is a form x S x + 2 l x + c in a point's place x: one whose S is
    positive definite, an ellipsoid, is held by a ball about its centre; one positive but for its axis, a cylinder,
    is held between the half-spaces of the solid that cross its axis; any other leaves the solid unbounded."""
    basis = concatenate([mv.direction(np.eye(3)), origin[None]])                # [4] Point
    gram = (solids[..., None, None](basis[:, None]) & basis[None, :]).to_array()   # [solids, quadrics, 4, 4]
    gram = (gram + np.swapaxes(gram, -1, -2)) / 2
    square, linear, constant = gram[..., :3, :3], gram[..., :3, 3], gram[..., 3, 3]
    values, vectors = np.linalg.eigh(square)                                   # [solids, quadrics, 3], [..., 3, 3]
    scale = np.abs(gram).reshape(gram.shape[:-2] + (16,)).max(axis=-1, initial=0.0)[..., None]
    positive = values > 1e-9 * scale                                           # [solids, quadrics, 3]
    flat = np.all(np.abs(values) <= 1e-9 * scale, axis=-1)                     # [solids, quadrics]
    # The centre and the reach of the form's level set along its positive directions.
    inverse = np.einsum("...ij,...j,...kj->...ik", vectors, np.where(positive, 1 / np.where(positive, values, 1), 0), vectors)
    centre = -np.einsum("...ij,...j->...i", inverse, linear)                   # [solids, quadrics, 3]
    level = np.maximum(np.einsum("...i,...i->...", linear, -centre) - constant, 0)   # [solids, quadrics]
    across = np.sqrt(level / np.where(positive, values, np.inf).min(axis=-1))   # [solids, quadrics]
    count = positive.sum(axis=-1)
    radius = np.where(count == 3, across, np.inf)
    # A cylinder's axis, the direction its form does not grow along, cut by the solid's half-spaces 2 l x + c <= 0.
    axis = vectors[..., :, 0]                                                  # [solids, quadrics, 3]
    rate = np.einsum("...ki,...ji->...jk", linear, axis)                       # [solids, cylinders, half-spaces]
    offset = np.einsum("...ki,...ji->...jk", linear, centre)
    reach = -(2 * offset + constant[..., None, :]) / np.where(rate == 0, 1, 2 * rate)
    cuts = flat[..., None, :] & (rate != 0)
    upper = np.where(cuts & (rate > 0), reach, np.inf).min(axis=-1)           # [solids, quadrics]
    lower = np.where(cuts & (rate < 0), reach, -np.inf).max(axis=-1)
    cylinder = (count == 2) & np.isfinite(upper) & np.isfinite(lower)
    upper, lower = np.where(cylinder, upper, 0), np.where(cylinder, lower, 0)
    middle = centre + axis * ((upper + lower) / 2)[..., None]
    radius = np.where(cylinder, np.hypot((upper - lower) / 2, across), radius)
    centre = np.where((count == 3)[..., None], centre, middle)
    # The smallest ball any of its quadrics gives.
    chosen = np.argmin(radius, axis=-1)[..., None]                              # [solids, 1]
    return (np.take_along_axis(centre, chosen[..., None], axis=-2)[..., 0, :],
            np.take_along_axis(radius, chosen, axis=-1)[..., 0])


def tiles(centres: np.ndarray, radii: np.ndarray, view: View) -> dict:
    """The solids each tile of the picture may see, from the balls holding them `[solids, 3]`, `[solids]`: each tile's
    range in a list of solids, the list, and the tiles across; a solid without a ball in every tile."""
    rows, columns = view.shape
    across, down = -(-columns // TILE), -(-rows // TILE)
    seen = view.across(mv.point(centres))                                      # [solids, 2]
    reach = radii / view.pixel + 1                                             # [solids] pixels
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


def bounded_by_planes(solids: Quadric) -> np.ndarray:
    """Which of a batch of solids `[solids, quadrics]` are bounded by planes alone: every quadric pairing every
    direction with itself to nothing, its square part gone, against the size of its coefficients `[solids]`."""
    # Six directions settle a symmetric form on three dimensions.
    directions = mv.direction(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]))   # [6] Point
    squares = np.abs((solids[..., None](directions) & directions).to_array())  # [solids, quadrics, 6]
    coefficients = np.abs(np.asarray(solids.kernel))
    sizes = coefficients.reshape(solids.shape + (int(np.prod(coefficients.shape[len(solids.shape):])),)).max(axis=-1,
                                                                                                        initial=0.0)   # [solids, quadrics]
    return np.all(squares <= 1e-9 * sizes[..., None], axis=(-1, -2))


def scene(parts: list[tuple[Quadric, np.ndarray]], translucent: list[tuple[Quadric, np.ndarray, float]],
          inks: list[Ink]) -> tuple[Quadric, dict, Quadric, Quadric, dict, tuple[np.ndarray, np.ndarray]]:
    """Every solid's quadrics, solid after solid, and each solid's range of them, colour and opacity, from parts: each
    a batch of solids, `[solids, quadrics]` Quadric, all of one kind, their colours, `[solids, 3]` or one for all,
    and for the translucent parts their opacity; and every ink's quadric, with its part's range of solids, colour and
    width."""
    parts = [(solids, colour, 1.0) for solids, colour in parts] + list(translucent)
    quadrics, counts, colours, opacities, flats, centres, radii = [], [], [], [], [], [], []
    for solids, colour, opacity in parts:
        quadrics.append(solids.reshape(-1))
        centre, radius = spheres(solids)
        centres.append(centre)
        radii.append(radius)
        counts.append(np.full(solids.shape[0], solids.shape[1]))
        colours.append(np.broadcast_to(colour, (solids.shape[0], 3)))
        opacities.append(np.full(solids.shape[0], opacity))
        flats.append(bounded_by_planes(solids))
    # A solid of no quadrics, which no ray meets, and a quadric no solid holds, so that the arrays are never empty.
    quadrics.append(everywhere[None])
    counts.append(np.zeros(1, dtype=int))
    colours.append(np.zeros((1, 3)))
    opacities.append(np.zeros(1))
    flats.append(np.zeros(1, dtype=bool))
    # It lies in no tile: a ball of no size, out of every picture.
    centres.append(np.zeros((1, 3)))
    radii.append(np.full(1, -np.inf))
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
    return (concatenate(quadrics), dict(first=ends - np.concatenate(counts), end=ends, colour=np.concatenate(colours),
                                        opacity=np.concatenate(opacities), flat=np.concatenate(flats).astype(np.uint32)),
            ink_quadrics, ink_regions, marked, (np.concatenate(centres), np.concatenate(radii)))


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
        weight = mv.w & seen
        return np.stack([((mv.x & seen) / weight).to_array(), ((mv.y & seen) / weight).to_array()], axis=-1)


def view(points: list[Point], reach: float, elevation: float, azimuth: float, margin: float, pixels: int) -> View:
    """A view framing batches of points, of any weight, and `reach` around each, from an elevation and azimuth, in
    radians, z up and azimuth zero looking along +y, with a margin as a share of the larger side, `pixels` across."""
    turn = (mv.xy * (-azimuth / 2)).exp() * (mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()     # [] Motor
    # Across and up the image: the pairings with the camera's planes through the origin, over the weights.
    right, up = (np.concatenate([((turn >> plane) & batch / (mv.w & batch)).to_array().reshape(-1) for batch in points])
                 for plane in (mv.x, mv.y))
    middle = np.array([right.max() + right.min(), up.max() + up.min()]) / 2
    half = np.array([np.ptp(right), np.ptp(up)]) / 2 + reach
    half = half + margin * half.max()
    return View(aim(turn >> (origin + mv.x.dual() * middle[0] + mv.y.dual() * middle[1]), elevation, azimuth), half, pixels)


def aim(centre: Point, elevation: float, azimuth: float) -> Motor:
    """A camera looking at a point, of any weight, from an elevation and azimuth, in radians, z up and azimuth zero
    looking along +y."""
    turn = (mv.xy * (-azimuth / 2)).exp() * (mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()     # [] Motor
    return translator(centre / (mv.w & centre) - origin) * turn


def lamp(camera: Motor) -> Point:
    """The lamp's direction for a camera: up and to the right of the viewer, and a little behind, so that strokes
    running up the image catch it as well as the faces turned up."""
    toward = camera >> mv.direction([0.35, 0.5, 0.8])
    return toward / (-(toward.dual() & toward)).square_root()


def layer(parts: list[tuple[Quadric, np.ndarray]], view: View, inks: list[Ink] = (),
          translucent: list[tuple[Quadric, np.ndarray, float]] = (),
          meshes: list[tuple[Samples, float]] = ()) -> tuple[np.ndarray, np.ndarray]:
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
    light pass: over black, the layer itself. Where the CPU draws, the picture is traced with fewer pixels, as
    `drawn` gives them, and enlarged to the view's."""
    seen = view._replace(pixels=drawn(view.pixels))                           # the view as it is traced
    quadrics, solids, ink_quadrics, ink_regions, marked, holding = scene(parts, translucent, inks)
    tracer = shader()
    encoder = tracer.device.create_command_encoder()
    # A mesh that shows nothing stands for none, so that the opacities are never empty.
    mesh_opacities = np.array([opacity for _, opacity in meshes] or [0.0], dtype=np.float32)
    outputs = tracer.encode(encoder, seen.shape, (), dict(
        quadrics=quadrics, solids=solids, inks=marked, ink_quadrics=ink_quadrics, ink_regions=ink_regions,
        **sampled(tracer, encoder, [mesh for mesh, _ in meshes], seen), mesh_opacities=mesh_opacities,
        n_meshes=len(meshes), **tiles(*holding, seen), pixel=view.pixel, camera=view.camera, back=4 * view.half.max(),
        half_height=view.half[1], lamp=lamp(view.camera), ambient=ambient, fill_tilt=fill_tilt, headlight=headlight,
        gloss=gloss, shininess=shininess, background=background))
    image = tracer.read(encoder, outputs, seen.shape)["image"]
    return enlarged(image["colour"], view.shape), enlarged(image["depth"][..., :1], view.shape)[..., 0]


def enlarged(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """An image `[rows, columns, channels]` resized to `shape`, each channel interpolated linearly."""
    rows, columns = shape
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(channel, dtype=np.float32), "F")
                                .resize((columns, rows), Image.Resampling.BILINEAR))
                     for channel in np.moveaxis(image, -1, 0)], axis=-1)


def sampled(tracer: Shader, encoder: Any, drawings: list[Samples], view: View) -> dict:
    """The meshes drawn at every sample of the view's pixels, recorded on an encoder, and copied there, on the GPU,
    mesh after mesh, into the buffers the tracer keeps for them: their colours and depths, and the cells a row of
    samples takes, padded as a copy pads it."""
    import wgpu

    rows, columns = view.shape
    shape = (samples_per_side * rows, samples_per_side * columns)
    # A copy's rows start at multiples of 256 bytes; a sample's colour and its depth are four bytes each.
    row_bytes = -(-shape[1] * 4 // 256) * 256
    size = max(len(drawings) * shape[0] * row_bytes, 16)
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    colours, depths = (tracer.buffer_kept(("meshes", name), size, usage) for name in ("colours", "depths"))
    for index, drawing in enumerate(drawings):
        # Each mesh's values in buffers of its own: all are written before the encoder runs.
        targets = drawing.shader.encode_draw(encoder, shape, drawing.n_vertices, ("mesh", index), drawing.values)
        for target, buffer in zip(targets, (colours, depths)):
            encoder.copy_texture_to_buffer(
                {"texture": target},
                {"buffer": buffer, "offset": index * shape[0] * row_bytes, "bytes_per_row": row_bytes, "rows_per_image": shape[0]},
                (shape[1], shape[0], 1))
    return dict(mesh_colours=Resident(colours, size), mesh_depths=Resident(depths, size), mesh_row=row_bytes // 4)


def draw(parts: list[tuple[Quadric, np.ndarray]], view: View, background: np.ndarray, inks: list[Ink] = (),
         translucent: list[tuple[Quadric, np.ndarray, float]] = (),
         meshes: list[tuple[Samples, float]] = ()) -> np.ndarray:
    """The solids as the view sees them on the background, with the lines inked on them, the translucent solids over
    and among them, and the meshes among them, as `layer`, `[rows, columns, 3]`: the background blended in on the GPU,
    sample by sample."""
    colour, _ = traced(parts, view, inks, translucent, meshes, np.asarray(background, dtype=float))
    return np.round(np.clip(colour[..., :3], 0, 1) * 255).astype(np.uint8)
