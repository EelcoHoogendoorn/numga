"""Solids bounded by intersections of quadrics in `PGA3D` are ray-traced directly on the GPU along
parallel orthographic rays and composited sample-by-sample against rasterized surfaces and tubes.

Each quadric is a symmetric linear map `Plane <- Point` whose regressive pairing
`quadric(point) & point` is negative inside the solid and zero on its boundary, where `quadric(point)`
is the outward tangent plane. Balls and ellipsoids require a single quadric, rods combine an
infinite cylinder with two end half-spaces, and boxes, plates, and convex polyhedra are intersections
of oriented planes alone (`arity == 0`), whose ray intervals evaluate in time linear in face count
rather than quadratic. Curves inked onto solid surfaces (`Ink`) evaluate zero sets of extrinsic or
intrinsic quadrics within regional bounds, normalizing algebraic values by their tangent-plane
directional derivatives so screen-space line width stays uniform across curved geometry.

Before tracing, spectral analysis of each quadric's directional growth form bounds ellipsoids by
their centres and cylinders or cones by axial cuts from sibling quadrics, binning bounded solids
into 16-pixel screen tiles while broadcasting unbounded half-space solids to all tiles. Along each
supersampled ray, opaque and translucent quadric hits interleave in depth order (breaking sub-pixel
ties deterministically by scene index) with pre-rasterized sample slots from triangulated surfaces
and 3D tubes.

Surface shading operates entirely in the camera's three-axis screen frame (`lamp_shading` and
`tangent_seen`), pairing tangent planes against the camera's preimage steps across, up, and toward
the viewer. Because illumination depends only on screen-space normal slopes, quadrics in `PGA3D` and
surfaces projected from arbitrary algebras share identical key-lamp, sky-ground fill, headlight, and
specular glint shading.
"""

from __future__ import annotations

from functools import cache
from typing import Any, NamedTuple

import numpy as np

from numga import NumpyContext, concatenate, stack, where
from numga.algebras import PGA3D as ga
from numga.extensor import Extensor
from numga.topology import Topology
from numga.wgsl import Resident, Shader, supersampled

from .camera import Camera, preimage, read
from .drawing import Draw, Slots, sampled, slots

context = NumpyContext(ga)
mv = context.multivector
Point = ga.gatype.antivector()
Scalar = ga.gatype.scalar()
Plane = ga.gatype.vector()
Motor = ga.gatype.rotor()
Line = ga.gatype.bivector()
# Each face's tangent plane is the join of its corners.
Patch = Plane.derive.flipped_orientation                                      # Vector-
Quadric = ga.gatype((Plane, Point))                                          # Plane <- Point
DualQuadric = ga.gatype((Point, Plane))                                      # Point <- Plane
# Intrinsic quadric form mapping points to joined planes, used for join-oriented surface inks.
Joined = ga.gatype((Patch, Point))                                           # Patch- <- Point-
# Coordinate planes of the scene and camera frames (right, up, and toward the viewer as the camera
# looks along -z).
axes = stack((mv.x, mv.y, mv.z))                                             # [3] Plane
origin = mv.w.dual()                                                         # [] Point
# Multiplying an intrinsic point join by `flip` reorients it into the extrinsic plane or line through those points.
flip = mv.antipseudoscalar([1.0])                                            # [] Antipseudoscalar-
# Maps planes through the origin to their orthogonal ideal directions under the Euclidean metric.
normal_direction = Plane.derive.nondegenerate.dual()                         # Point <- Plane
ambient, gloss, shininess = 0.22, 0.45, 32.0
# Sky-ground vertical fill bias brightening upward-facing surfaces relative to downward-facing ones.
fill_tilt = 0.5
# Fraction of illumination aligned with the viewing ray so visible faces retain their hue away from the key lamp.
headlight = 0.25
# Key-lamp direction in screen coordinates (right, up, and toward the viewer).
lamp = np.array([0.35, 0.5, 0.8]) / np.linalg.norm([0.35, 0.5, 0.8])


def lamp_shading() -> str:
    """WGSL source lights screen-space normals with key lamp, sky-ground fill, headlight, and
    specular glint."""
    return """
// The lamp's direction in the picture's frame, across, up and toward the viewer, a unit; the light that falls
// everywhere, how much more of it falls on faces turned up the picture, the share of the light that comes from the
// viewer, and the glint's share and sharpness.
var<uniform> lamp: vec3<f32>;
var<uniform> ambient: f32;
var<uniform> fill_tilt: f32;
var<uniform> headlight: f32;
var<uniform> gloss: f32;
var<uniform> shininess: f32;

// A normal in the picture's frame as a unit, turned to the viewer: a surface is lit on the side the camera sees.
fn turned(seen: vec3<f32>) -> vec3<f32> {
    let size = sqrt(seen.x * seen.x + seen.y * seen.y + seen.z * seen.z);
    return select(seen, -seen, seen.z < 0) / max(size, 1e-30);
}

// How brightly a surface with this unit normal is lit: by the lamp, by the sky above the picture and the ground
// below it, and by the viewer.
fn brightness(normal: vec3<f32>) -> f32 {
    let light = max(normal.x * lamp.x + normal.y * lamp.y + normal.z * lamp.z, 0.0);
    let fill = ambient * (1 + fill_tilt * normal.y) + headlight * normal.z;
    return fill + (1 - ambient - headlight) * light;
}

// The glint where the view ray, into the picture, reflected in the surface, meets the lamp, on a surface it lights.
fn glint(normal: vec3<f32>) -> f32 {
    let reflected = vec3<f32>(0.0, 0.0, -1.0) + normal * (2 * normal.z);
    let facing_lamp = normal.x * lamp.x + normal.y * lamp.y + normal.z * lamp.z > 0;
    return pow(max(reflected.x * lamp.x + reflected.y * lamp.y + reflected.z * lamp.z, 0.0), shininess)
        * select(0.0, 1.0, facing_lamp);
}

// A colour on a surface with this normal in the picture's frame, lit, with its glint.
fn lit_seen(colour: vec3<f32>, seen: vec3<f32>) -> vec3<f32> {
    let normal = turned(seen);
    return colour * brightness(normal) + gloss * glint(normal);
}
"""


def tangent_seen() -> str:
    """WGSL source pairs a `Tangent` plane against the camera's preimage steps to obtain its
    screen-space normal slopes."""
    return """
// The steps the camera reads as one across, one up and one toward the viewer.
var<uniform> step_across: Point;
var<uniform> step_up: Point;
var<uniform> step_toward: Point;

// A tangent plane's slope along the picture's frame: the surface's normal there, as the picture sees it.
fn slopes(tangent: Tangent) -> vec3<f32> {
    return vec3<f32>(tangent & step_across, tangent & step_up, tangent & step_toward);
}

fn lit(colour: vec3<f32>, tangent: Tangent) -> vec3<f32> {
    return lit_seen(colour, slopes(tangent));
}
"""


def quadric_crossings() -> str:
    """WGSL source evaluates a quadric along a ray as a scalar quadratic and solves for its surface
    crossings without catastrophic cancellation."""
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
    """WGSL source intersects a ray with an intersection of quadrics by testing each root against
    all sibling quadrics within a distance interval."""
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
    """WGSL source finds the nearest quadric-solid hit per screen tile after the previously
    composited surface, specializing plane-only solids to linear interval intersection."""
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


def ink_pass(name: str) -> str:
    """WGSL source paints surface curves from `{name}` where a hit lies within half the ink's pixel
    width of its quadric zero set."""
    return f"""
fn {name}_painted(hit: Hit, point: Point, slide_across: Point, slide_up: Point, painted: Painted) -> Painted {{
    var result = painted;
    for (var index = 0u; index < arrayLength(&{name}); index++) {{
        let ink = {name}[index];
        // The cheap tests first: the ink's solids, then its region.
        if (hit.solid < ink.first || hit.solid >= ink.end || ({name}_regions[index](point) & point) > tolerance) {{
            continue;
        }}
        // The form's plane at the point, its value there, and its change per reading across and up on the surface.
        let plane = {name}_quadrics[index](point);
        let rate = vec2<f32>(plane & slide_across, plane & slide_up) * 2;
        let distance = abs(plane & point) / sqrt(rate.x * rate.x + rate.y * rate.y);
        let covers = distance < ink.width / 2 * pixel;
        result = Painted(select(result.colour, vec4<f32>(ink.colour, ink.opacity), covers), result.inked || covers);
    }}
    return result;
}}
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

// The inks whose quadrics give planes and those whose quadrics give joins, each kind in arrays of its own type, a zero
// set the same whatever the sign of its form; where each ink shows, a side: where its region's quadric pairs negatively
// with the point.
var<storage, read> inks: array<Ink>;
var<storage, read> inks_quadrics: array<Quadric>;
var<storage, read> inks_regions: array<Quadric>;
var<storage, read> joined_inks: array<Ink>;
var<storage, read> joined_inks_quadrics: array<Joined>;
var<storage, read> joined_inks_regions: array<Quadric>;
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
// The camera's preimage: the point it reads at the middle of the view, the rays parallel along the step it reads as one
// toward the viewer, starting `back` toward the viewer from there; the image half as high as this.
var<uniform> middle: Point;
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

""" + "".join(ink_pass(name) for name in ("inks", "joined_inks")) + """
// A hit's colour and opacity with the lines inked on its solid, and whether one covers it. A line is as wide as the
// picture sees it: an ink's distance to its zero set, in the camera's readings, is its form's value there over how fast
// the form changes as the picture moves one reading across and one up on the surface, each step slid along the ray back
// onto the tangent plane.
fn inked(hit: Hit, point: Point, direction: Point) -> Painted {
    let look = looks[hit.solid];
    let slide_across = step_across - direction * ((hit.tangent & step_across) / (hit.tangent & direction));
    let slide_up = step_up - direction * ((hit.tangent & step_up) / (hit.tangent & direction));
    let plain = Painted(vec4<f32>(look.colour, look.opacity), false);
    return joined_inks_painted(hit, point, slide_across, slide_up, inks_painted(hit, point, slide_across, slide_up, plain));
}

// A hit's colour, lit by the lamp, the sky and the viewer, and its opacity; an ink keeps a share of its own colour.
fn shade(hit: Hit, start: Point, direction: Point) -> vec4<f32> {
    let normal = turned(slopes(hit.tangent));
    let surface = inked(hit, start + direction * hit.distance, direction);
    let own = select(0.0, ink_own, surface.inked);
    let shaded = mix(surface.colour.rgb * brightness(normal), surface.colour.rgb, own);
    return vec4<f32>(shaded + gloss * glint(normal), surface.colour.a);
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
    // The ray's start: across and up the image, and `back` toward the viewer, read back into the scene.
    let across = screen(place) * half_height;
    let start = middle + step_across * across.x + step_up * across.y + step_toward * back;
    let direction = -step_toward;
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
def translator(offset: Extensor) -> Extensor:
    """Exponentiating `(mv.w ^ offset.dual()) / 2` translates every point along ideal point
    `offset`."""
    return ((mv.w ^ offset.dual()) / 2).exp()


def placed(quadric: Extensor, motor: Extensor) -> Extensor:
    """Sandwiching `quadric` by `motor` (`motor >> quadric(motor << Point)`) transforms a solid
    rigidly in `PGA3D`."""
    return motor >> quadric(motor << Point)


def ellipsoid(semi_axes: np.ndarray) -> Extensor:
    """Origin-centred ellipsoid quadrics scale each coordinate plane pairing inversely by the
    squared `semi_axes` along `x`, `y`, and `z`."""
    return (axes * (axes & Point) / semi_axes**2).sum(axis=-1) - mv.w * (mv.w & Point)


def solid_of_dual_quadric(shape: Extensor) -> Extensor:
    """Subtracting the origin's rank-one dyad from an origin-centred dual quadric embedded from
    Euclidean 3D space and inverting yields its primal solid quadric."""
    return (shape - origin * (Plane & origin)).inverse()


def ball(centres: Extensor, radius: Extensor | float) -> Extensor:
    """Sphere quadrics of `radius` translate from the origin to arbitrary-weight `centres`."""
    round_form = (axes * (axes & Point)).sum(axis=-1) / radius**2 - mv.w * (mv.w & Point)
    return placed(round_form, translator(centres / (mv.w & centres) - origin))


# Quadric pairing every unit-weight point to -1, enclosing all of space.
everywhere = -mv.w * (mv.w & Point)


def half_space(plane: Extensor) -> Extensor:
    """Symmetrizing `plane` with the ideal plane `mv.w` embeds `plane`'s negative half-space as a
    rank-two quadric."""
    return (plane * (mv.w & Point) + mv.w * (plane & Point)) / 2


def aligned(starts: Extensor, ends: Extensor) -> tuple[Extensor, np.ndarray, Extensor]:
    """Each segment between unit-weight endpoints defines a unit normal plane through the origin,
    its sign along `+z`, and the non-inverting rotor carrying `+z` onto that signed axis."""
    across = (-(ends - starts).dual()).normalized()                           # [...] Plane
    sense = np.where((across & mv.z.dual()) >= 0, 1.0, -1.0)
    return across, sense, (1 + across * sense * mv.z).normalized()


def rods(starts: Extensor, ends: Extensor, radius: Extensor | float) -> Extensor:
    """Build capped cylindrical rods `[..., 3] Quadric` of `radius` between `starts` and `ends` by
    combining one cylinder quadric with two orthogonal end-cap half-space quadrics."""
    starts, ends = starts / (mv.w & starts), ends / (mv.w & ends)
    across, _, turn = aligned(starts, ends)
    # Transform the canonical z-axis cylinder onto each segment.
    cylinder = (mv.x * (mv.x & Point) + mv.y * (mv.y & Point)) / radius**2 - mv.w * (mv.w & Point)
    tube = placed(cylinder, translator(starts - origin) * turn)              # [...] Quadric
    # Cap both ends with outward-facing planes orthogonal to the segment.
    beyond_end = half_space(across - mv.w * (across & ends))
    before_start = half_space(-across + mv.w * (across & starts))
    return stack([tube, beyond_end, before_start], axis=-1)


def boxes(motors: Extensor, half_extents: np.ndarray) -> Extensor:
    """Build six outward-facing bounding planes `[..., 6] Plane` displaced by `half_extents`
    (`[..., 3]`) and transformed by `motors`."""
    reach = mv.w * half_extents                                               # [..., 3] Plane
    faces = concatenate([axes - reach, -axes - reach], axis=-1)               # [..., 6] Plane
    return motors[..., None] >> faces


def outward(planes: Extensor, middles: Extensor) -> Extensor:
    """Flipping `planes` where needed places unit-weight interior points `middles` on their
    negative sides."""
    side = planes & middles
    return planes * (np.asarray(side < 0, dtype=int) - np.asarray(side > 0, dtype=int))


def plates(corners: Extensor, thickness: float) -> Extensor:
    """Extrude ordered planar convex polygons `corners` (`[..., k] Point`) by `thickness` into
    outward-facing top, bottom, and perimeter planes `[..., 2 + k] Plane`."""
    corners = corners / (mv.w & corners)
    middle = corners.mean(axis=-1)                                            # [...] Point
    # Join the first three corners to obtain the unit polygon plane passing through the centroid.
    joined = ((corners[..., 0] & corners[..., 1] & corners[..., 2]) * flip).normalized()  # [...] Plane
    face = joined - mv.w * (joined & middle)
    # Span each perimeter plane from an edge step and the polygon's normal direction.
    square = (face - mv.w * (face & origin)).dual()                           # [...] Point
    steps = (Topology.loops([corners.shape[-1]]).d[0] * corners.field()).batch()   # [..., corners] Point
    sides = outward((corners & steps & square[..., None]) * flip, middle[..., None])   # [..., corners] Plane
    return concatenate([stack([face, -face], axis=-1) - mv.w * (thickness / 2), sides], axis=-1)


def polyhedra(corners: Extensor, faces: np.ndarray) -> Extensor:
    """Join the first three vertices of each row in `faces` (`[F, >=3]`) over `corners`
    (`[..., V] Point`) and orient away from the vertex centroid, returning `[..., F] Plane`."""
    corners = corners / (mv.w & corners)
    joined = (corners[..., faces[:, 0]] & corners[..., faces[:, 1]] & corners[..., faces[:, 2]]) * flip   # [..., faces] Plane
    return outward(joined, corners.mean(axis=-1)[..., None])


def edges(starts: Extensor, ends: Extensor, radius: Extensor | float) -> tuple[Extensor, Extensor]:
    """Cylindrical `rods` paired with spheres of equal `radius` at `starts` and `ends` join
    smoothly across shared vertices."""
    return rods(starts, ends, radius), ball(stack([starts, ends], axis=-1).reshape(-1), radius)[:, None]


# --- plumbing -------------------------------------------------------------------------
# Screen tile side length in pixels, matching `tile` in WGSL.
TILE = 16


@cache
def shader() -> Shader:
    """The supersampled `PGA3D` ray-tracing shader composites quadric solids alongside rasterized
    surface slots."""
    return Shader(supersampled() + quadric_tracing() + lamp_shading() + tangent_seen() + source, ga,
                  dict(Point=Point, Plane=Plane, Tangent=Plane, Quadric=Quadric, Joined=Joined), {})


class Ink(NamedTuple):
    """Surface curves inked onto the solids of `part` (indexing opaque parts before translucent
    parts) follow the zero sets of `quadrics` within the negative side of extrinsic `regions`
    (`everywhere` when unbounded)."""
    part: int
    quadrics: Extensor
    regions: Extensor
    colours: np.ndarray
    width: float
    opacity: float


def spheres(solid_quadrics: Extensor) -> tuple[Extensor, Extensor]:
    """Each solid in `solid_quadrics` is enclosed by the tightest bounding sphere derived from its
    quadrics' directional growth forms, with infinite radius when unbounded."""
    # Diagonalize each quadric's directional growth form to classify ellipsoids, cylinders, and cones.
    rates, normals = (solid_quadrics(normal_direction) & normal_direction).eigh()   # [solids, quadrics, 3] Scalar, Plane
    directions = normals.dual()                                               # [solids, quadrics, 3] Point
    zero, one, infinity = mv.scalar([0]), mv.scalar([1]), mv.scalar([np.inf])
    size = solid_quadrics.svdvals()[..., 0]                                   # [solids, quadrics] Scalar+
    growing = rates > 1e-9 * size[..., None]
    positive_rates = where(growing, rates, infinity)
    slowest = positive_rates.min(axis=-1)
    count = growing.sum(axis=-1)
    # Displace the origin along every growing eigendirection to its stationary point (the pole of
    # the ideal plane for an ellipsoid).
    slopes = ((solid_quadrics(origin)[..., None] & directions) + (solid_quadrics[..., None](directions) & origin)) / 2
    steps = where(growing, -slopes / where(growing, rates, one), zero)       # [solids, quadrics, 3] Scalar+
    centres = origin + (directions * steps).sum(axis=-1)                      # [solids, quadrics] Point

    def reach(points: Extensor) -> Extensor:
        return (-(solid_quadrics(points) & points).clip(-np.inf, 0) / slowest).square_root()

    # Intersect each two-growth (cylinder or cone) axis with sibling quadrics that vary linearly along it.
    axis = directions[..., 0]                                                 # [solids, quadrics] Point
    others = solid_quadrics[..., None, :]                                     # [solids, 1, quadrics] Quadric
    along = (others(centres[..., None]) & axis[..., None]) + (others(axis[..., None]) & centres[..., None])
    offset = others(centres[..., None]) & centres[..., None]                  # [solids, quadrics, quadrics] Scalar+
    straight = (others(axis[..., None]) & axis[..., None]).abs() <= 1e-9 * size[:, None, :]
    cuts = straight & ((along > 0) | (along < 0)) & ~np.eye(solid_quadrics.shape[-1], dtype=bool)
    crossing = -offset / where(cuts, along, one)
    upper_cuts = where(cuts & (along > 0), crossing, infinity)
    lower_cuts = where(cuts & (along < 0), crossing, -infinity)
    upper = upper_cuts.min(axis=-1)
    lower = lower_cuts.max(axis=-1)
    cylinder = (count == 2) & upper.isfinite() & lower.isfinite()
    upper, lower = where(cylinder, upper, zero), where(cylinder, lower, zero)
    # Enclose a capped cylinder or cone from the midpoint between its axial cuts across its wider end.
    lower_reach, upper_reach = reach(centres + axis * lower), reach(centres + axis * upper)
    widest = where(lower_reach > upper_reach, lower_reach, upper_reach)
    capped_radius = (((upper - lower) / 2).squared() + widest.squared()).square_root()
    radius = where(count == 3, reach(centres), where(cylinder, capped_radius, infinity))
    centres = centres + axis * ((upper + lower) / 2)
    # Select the tightest bounding sphere yielded by any quadric of each solid.
    tightest = radius.argmin(axis=-1)                                         # [solids]
    picked = np.arange(len(tightest))
    return centres[picked, tightest], radius[picked, tightest]


def tiles(centres: Extensor, radii: Extensor, view: View) -> dict:
    """Bin solids into `TILE`-pixel screen tiles from their bounding spheres `(centres, radii)`,
    placing unbounded solids into every tile."""
    rows, columns = view.shape
    across, down = -(-columns // TILE), -(-rows // TILE)
    seen = view.across(centres)                                                # [solids, 2]
    reach = (radii / view.pixel + 1).to_array()                                # [solids]
    column = (seen[:, 0] + view.half[0]) / view.pixel
    row = (view.half[1] - seen[:, 1]) / view.pixel
    bounded = np.isfinite(reach)
    first = np.where(bounded, np.floor((np.stack([column, row]) - reach) / TILE), 0).clip(0, [[across], [down]])
    last = np.where(bounded, np.floor((np.stack([column, row]) + reach) / TILE), [[across - 1], [down - 1]]).clip(-1, [[across - 1], [down - 1]])
    widths, heights = np.maximum(last - first + 1, 0).astype(int)              # [solids], [solids]
    sizes = widths * heights
    solid = np.repeat(np.arange(len(radii)), sizes)
    within = np.arange(sizes.sum()) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    tile = ((first[1][solid] + within // np.maximum(widths[solid], 1)) * across + first[0][solid]
            + within % np.maximum(widths[solid], 1)).astype(int)
    order = np.lexsort((solid, tile))
    starts = np.concatenate([[0], np.cumsum(np.bincount(tile, minlength=across * down))])
    # Pad `tile_solids` with a trailing entry so the shader storage buffer is never empty.
    return dict(tile_starts=starts.astype(np.uint32), tile_solids=np.append(solid[order], 0).astype(np.uint32),
                tiles_across=across)


def scene(parts: list[tuple], translucent: list[tuple],
          inks: list[Ink]) -> tuple:
    """Pack opaque `parts`, `translucent` parts, and surface `inks` into flat GPU buffers, solid
    ranges, material tables, and bounding spheres, promoting plane-only parts (`arity == 0`) to
    half-space quadrics."""
    all_parts = [(part_solids, colour, 1.0) for part_solids, colour in parts] + list(translucent)
    quadrics, counts, colours, opacities, flats, centres, radii = [], [], [], [], [], [], []
    for part_solids, colour, opacity in all_parts:
        # Nullary plane batches bound convex polyhedra directly as half-spaces with infinite sphere bounds.
        flat = part_solids.arity == 0
        if flat:
            centre = origin[None].broadcast_to(part_solids.shape[:1])
            radius = mv.scalar(np.full((part_solids.shape[0], 1), np.inf))
            part_solids = half_space(part_solids)
        else:
            centre, radius = spheres(part_solids)
        quadrics.append(part_solids.reshape(-1))
        centres.append(centre)
        radii.append(radius)
        counts.append(np.full(part_solids.shape[0], part_solids.shape[1]))
        colours.append(np.broadcast_to(colour, (part_solids.shape[0], 3)))
        opacities.append(np.full(part_solids.shape[0], opacity))
        flats.append(np.full(part_solids.shape[0], flat))
    # Append a sentinel solid with zero quadrics and negative-infinity radius so GPU buffers stay non-empty.
    quadrics.append(everywhere[None])
    counts.append(np.zeros(1, dtype=int))
    colours.append(np.zeros((1, 3)))
    opacities.append(np.zeros(1))
    flats.append(np.zeros(1, dtype=bool))
    centres.append(origin[None])
    radii.append(mv.scalar([[-np.inf]]))
    ends = np.cumsum(np.concatenate(counts))
    part_ends = np.cumsum([part_solids.shape[0] for part_solids, _, _ in all_parts])
    part_starts = part_ends - [part_solids.shape[0] for part_solids, _, _ in all_parts]
    # Partition extrinsic (`Quadric`) and intrinsic (`Joined`) inks into separate typed storage arrays,
    # appending a zero-range sentinel to each so neither buffer is empty.
    inked = {}
    for name, kind in (("inks", Quadric), ("joined_inks", Joined)):
        matching = [ink for ink in inks if ink.quadrics.gatype.derive.output.orientation is kind.derive.output.orientation]
        nothing = context.extensor(kind, np.zeros((1,) + kind.structural_shape))
        matching += [Ink(0, nothing, everywhere[None], np.zeros(3), 0.0, 0.0)]
        sizes = [int(np.prod(ink.quadrics.shape)) for ink in matching]
        inked |= {name: dict(first=np.repeat([part_starts[ink.part] for ink in matching[:-1]] + [0], sizes),
                             end=np.repeat([part_ends[ink.part] for ink in matching[:-1]] + [0], sizes),
                             colour=np.concatenate([np.broadcast_to(ink.colours, (size, 3))
                                                    for ink, size in zip(matching, sizes)]),
                             width=np.repeat([ink.width for ink in matching], sizes),
                             opacity=np.repeat([ink.opacity for ink in matching], sizes)),
                  f"{name}_quadrics": concatenate([ink.quadrics.reshape(-1) for ink in matching]),
                  f"{name}_regions": concatenate([ink.regions.broadcast_to(ink.quadrics.shape).reshape(-1)
                                                  for ink in matching])}
    return (concatenate(quadrics),
            dict(first=ends - np.concatenate(counts), end=ends, flat=np.concatenate(flats).astype(np.uint32)),
            dict(colour=np.concatenate(colours), opacity=np.concatenate(opacities)),
            inked, (concatenate(centres), concatenate(radii)))


class View(NamedTuple):
    """An orthographic camera pose in `PGA3D` looks along its local `-z` axis with scene-unit
    half-extents `half` and horizontal resolution `pixels`."""
    camera: Extensor
    half: np.ndarray
    pixels: int

    @property
    def pixel(self) -> float:
        """Each pixel spans `2 * half[0] / pixels` scene units."""
        return 2 * self.half[0] / self.pixels

    @property
    def shape(self) -> tuple[int, int]:
        """Image dimensions `(rows, columns)` preserve the aspect ratio of `half`."""
        return round(self.pixels * self.half[1] / self.half[0]), self.pixels

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Symmetric `(left, right, bottom, top)` scene bounds align with `across` coordinates."""
        return -self.half[0], self.half[0], -self.half[1], self.half[1]

    def squared(self) -> View:
        """Widening the shorter half-extent makes both image axes match the larger side."""
        return self._replace(half=np.full(2, self.half.max()))

    def across(self, points: Extensor) -> np.ndarray:
        """Pulling `points` back into the camera frame reads horizontal and vertical scene-unit
        offsets from the view centre."""
        seen = self.camera << points
        return ((axes[:2] & seen[..., None]) / (mv.w & seen)[..., None]).to_array()

    def to_camera(self) -> Camera:
        """Pushing `axes` into the scene yields the equivalent `Camera` weighing points against the
        ideal plane `mv.w`."""
        across, up, toward = self.camera >> axes
        return Camera(across, up, toward, mv.w, 0.0, np.zeros(2), self.half, self.pixels)


def view(points: list[Extensor], reach: float, elevation: float, azimuth: float, margin: float, pixels: int) -> View:
    """Support planes along the rotated image axes frame `points` padded by `reach` and fractional
    `margin` from spherical angles `(elevation, azimuth)` (`+z` up, zero azimuth looking along `+y`)."""
    turn = (mv.xy * (-azimuth / 2)).exp() * (mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()     # [] Motor
    screen = turn >> axes                                                    # [3] Plane
    seen = concatenate([(batch / (mv.w & batch)).reshape(-1) for batch in points])   # [points] Point
    # Measure tight horizontal and vertical support planes across all normalized points.
    heights = screen[:2, None] & seen                                         # [2, points] Scalar
    low, high = heights.min(axis=-1), heights.max(axis=-1)                    # [2] Scalar+, Scalar+
    # Intersect the mid-planes between opposing support bounds with the origin depth plane.
    middles = screen[:2] - mv.w * (low + high) / 2                             # [2] Plane
    centre = (middles[0] ^ middles[1] ^ screen[2]) * flip                    # [] Point
    half = (high - low) / 2 + reach
    half = half + margin * half.max()
    return View(aim(centre, elevation, azimuth), half.to_array(), pixels)


def aim(centre: Extensor, elevation: float, azimuth: float) -> Extensor:
    """Spherical angles `(elevation, azimuth)` (`+z` up, zero azimuth looking along `+y`) orient a
    camera motor translated to `centre`."""
    turn = (mv.xy * (-azimuth / 2)).exp() * (mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()     # [] Motor
    return translator(centre / (mv.w & centre) - origin) * turn


def lighting(glint: float) -> dict:
    """Uniform bindings for `lamp_shading` pair default lamp parameters with specular weight `glint`."""
    return dict(lamp=lamp, ambient=ambient, fill_tilt=fill_tilt, headlight=headlight, gloss=glint, shininess=shininess)


def seen_lamp(view: View, lamp_direction: Extensor) -> np.ndarray:
    """Pairing `lamp_direction` with `view`'s camera planes expresses a scene lamp as a unit vector
    in the screen frame."""
    camera = view.to_camera()
    seen = np.array([float(read(plane, lamp_direction).to_array()) for plane in (camera.across, camera.up, camera.toward)])
    return seen / np.linalg.norm(seen)


def steps(view: View) -> dict:
    """Uniform bindings for `tangent_seen` hold the camera's preimage unit steps across, up, and
    toward the viewer."""
    _, across, up, toward = preimage(view.to_camera(), Point)
    return dict(step_across=across, step_up=up, step_toward=toward)


def layer(parts: list[tuple], view: View, inks: list[Ink],
          translucent: list[tuple],
          meshes: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Trace `parts`, `inks`, and `translucent` solids interleaved with supersampled `meshes` over
    transparent black, returning coverage-weighted RGBA colour `[rows, columns, 4]` and view depth
    `[rows, columns]`."""
    return traced(parts, view, inks, translucent, meshes, np.zeros(3))


def traced(parts: list[tuple], view: View, inks: list[Ink],
           translucent: list[tuple], surface_draws: list[tuple],
           background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize supersampled `surface_draws` into temporary sample slots and composite the combined
    scene over `background` in a single GPU submission, returning `[rows, columns, 4]` colour and
    `[rows, columns]` depth."""
    tracer = shader()
    encoder = tracer.device.create_command_encoder()
    held = slots(view.shape, len(surface_draws), lambda name, size, usage: tracer.buffer_kept(("meshes", name), size, usage))
    sampled(encoder, [surface for surface, _ in surface_draws], 0, held)
    return trace(tracer, encoder, view, scene(parts, translucent, inks), held,
                 [opacity for _, opacity in surface_draws], background)


def trace(tracer: Shader, encoder: Any, view: View, packed_scene: tuple, held_slots: Slots,
          opacities: list[float], background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode `tracer` on `encoder` to composite `packed_scene` with pre-rasterized surface slots
    `held_slots` over `background`, reading back `[rows, columns, 4]` colour and `[rows, columns]`
    depth."""
    quadrics, solid_ranges, looks, inked, holding = packed_scene
    # Provide a dummy zero opacity when no rasterized surface slots are active so the storage buffer stays non-empty.
    outputs = tracer.encode(encoder, view.shape, (), dict(
        quadrics=quadrics, solids=solid_ranges, looks=looks, **inked,
        mesh_colours=Resident(held_slots.colours, held_slots.size), mesh_depths=Resident(held_slots.depths, held_slots.size),
        mesh_row=held_slots.row_bytes // 4, mesh_opacities=np.array(opacities or [0.0], dtype=np.float32),
        n_meshes=len(opacities), **tiles(*holding, view), pixel=view.pixel, middle=preimage(view.to_camera(), Point)[0],
        back=4 * view.half.max(), half_height=view.half[1], background=background, **lighting(gloss), **steps(view)))
    image = tracer.read(encoder, outputs, view.shape)["image"]
    return image["colour"], image["depth"][..., 0]


def draw(parts: list[tuple], view: View, background: np.ndarray, inks: list[Ink],
         translucent: list[tuple], meshes: list[tuple]) -> np.ndarray:
    """Trace `parts`, `inks`, `translucent` solids, and `meshes` blended sample-by-sample over
    `background`, returning `uint8` RGB `[rows, columns, 3]`."""
    colour, _ = traced(parts, view, inks, translucent, meshes, np.asarray(background, dtype=float))
    return np.round(np.clip(colour[..., :3], 0, 1) * 255).astype(np.uint8)
