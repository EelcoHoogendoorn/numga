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

import numpy as np

from numga import NumpyContext, concatenate, stack
from numga.algebras import PGA3D as ga
from examples.wgsl import Shader, supersampled

mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Motor = ga.gatype.rotor()
Quadric = ga.gatype((Plane, Point))                                          # Plane <- Point
axes = stack((mv.x, mv.y, mv.z))                                             # [3] Plane
origin = mv.w.dual()                                                         # [] Point
# The light that falls everywhere, the glint's share and sharpness, and the lamp: from up and to the right
# of a viewer looking along -y.
ambient, gloss, shininess = 0.3, 0.3, 40.0

source = supersampled(2) + """
// A solid: the intersection of its quadrics, `quadrics[first]` up to `quadrics[end]`, and its colour.
struct Solid {
    first: u32,
    end: u32,
    colour: vec3<f32>,
}

var<storage, read> solids: array<Solid>;
var<storage, read> quadrics: array<Quadric>;
// A camera looking along its own -z, its rays parallel, starting `back` behind the middle of the view; the
// image half as high as this.
var<uniform> camera: Motor;
var<uniform> back: f32;
var<uniform> half_height: f32;
var<uniform> lamp: Point;
var<uniform> background: vec3<f32>;
var<uniform> ambient: f32;
var<uniform> gloss: f32;
var<uniform> shininess: f32;

alias Sample = vec3<f32>;

const origin = mv.w.dual();
const far = 1e30;
// How far a point may lie outside a quadric and still count as inside, for points on two surfaces.
const tolerance = 1e-5;

struct Hit {
    distance: f32,
    tangent: Plane,
    colour: vec3<f32>,
}

fn inside_others(solid: Solid, own: u32, point: Point) -> bool {
    for (var index = solid.first; index < solid.end; index++) {
        if (index != own && (quadrics[index](point) & point) > tolerance) {
            return false;
        }
    }
    return true;
}

// The first point a ray meets on a solid's surface: on one of its quadrics, and inside all the others.
fn first_hit(start: Point, direction: Point) -> Hit {
    var hit = Hit(far, mv.z, background);
    for (var index = 0u; index < arrayLength(&solids); index++) {
        let solid = solids[index];
        for (var own = solid.first; own < solid.end; own++) {
            let quadric = quadrics[own];
            // Along the ray the quadric is a quadratic in the distance, a t^2 + 2 b t + c.
            let a = quadric(direction) & direction;
            let b = ((quadric(direction) & start) + (quadric(start) & direction)) / 2;
            let c = quadric(start) & start;
            let discriminant = b * b - a * c;
            if (discriminant < 0) {
                continue;
            }
            // Both roots without cancellation; a half-space, with no square term, has only the second.
            let q = -(b + select(-1.0, 1.0, b >= 0) * sqrt(discriminant));
            let roots = vec2<f32>(select(q / a, far, a == 0), c / q);
            for (var which = 0u; which < 2u; which++) {
                let distance = roots[which];
                let point = start + direction * distance;
                if (distance < hit.distance && inside_others(solid, own, point)) {
                    // The quadric's plane at a point of its surface is the tangent plane there.
                    hit = Hit(distance, quadric(point), solid.colour);
                }
            }
        }
    }
    return hit;
}

fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place) * half_height;
    let start = camera >> (origin + mv.z.dual() * back + mv.x.dual() * across.x + mv.y.dual() * across.y);
    let direction = camera >> -mv.z.dual();
    let hit = first_hit(start, direction);
    if (hit.distance >= far) {
        return background;
    }
    // Facing the viewer: a tangent plane is lit on the side the ray comes from.
    let tangent = hit.tangent * select(1.0, -1.0, (hit.tangent & direction) > 0);
    let normal = tangent / (tangent | tangent).square_root();
    let light = max(normal & lamp, 0.0);
    // The view ray reflected in the tangent plane, against the lamp: `-d.dual()` is the plane square to d.
    let glint = pow(max(-((normal >> direction).dual() & lamp), 0.0), shininess) * select(0.0, 1.0, light > 0);
    return hit.colour * (ambient + (1 - ambient) * light) + gloss * glint;
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


def half_space(plane: Plane) -> Quadric:
    """The points on a plane's negative side, as the quadric that pairs them with the plane and with the
    plane at infinity, symmetrized."""
    return (plane * (mv.w & Point) + mv.w * (plane & Point)) / 2


def rods(starts: Point, ends: Point, radius: float) -> Quadric:
    """Solid rods of one radius between points, `[..., 3]` quadrics each: a cylinder about the line through the
    ends, and the half-spaces past each end, square to the line."""
    # The plane through the origin square to the rod, unit.
    starts, ends = starts / (mv.w & starts), ends / (mv.w & ends)
    across = -(ends - starts).dual()                                          # [...] Plane
    across = across / (across | across).square_root()
    # The cylinder about z, turned onto the rod and moved to its start: whichever way along the rod lies nearer
    # z, so the turn never reverses z.
    sense = np.where((across | mv.z).to_array() >= 0, 1.0, -1.0)
    turn = (1 + across * sense * mv.z).normalized()                           # [...] Motor
    cylinder = (mv.x * (mv.x & Point) + mv.y * (mv.y & Point)) / radius**2 - mv.w * (mv.w & Point)
    tube = placed(cylinder, translator(starts - origin) * turn)              # [...] Quadric
    # Past each end: the plane square to the rod through the end, facing out.
    beyond_end = half_space(across - mv.w * (across & ends))
    before_start = half_space(-across + mv.w * (across & starts))
    return stack([tube, beyond_end, before_start], axis=-1)


def boxes(motors: Motor, half_extents: np.ndarray) -> Quadric:
    """Solid boxes placed by motors, their half extents along their own x, y and z on the last axis, `[..., 6]`
    quadrics each: the half-spaces inside each face."""
    reach = mv.w * half_extents                                               # [..., 3] Plane
    faces = concatenate([axes - reach, -axes - reach], axis=-1)               # [..., 6] Plane
    return placed(half_space(faces), motors[..., None])


# --- plumbing -------------------------------------------------------------------------
@cache
def shader() -> Shader:
    return Shader(source, ga, dict(Point=Point, Plane=Plane, Motor=Motor, Quadric=Quadric))


def scene(parts: list[tuple[Quadric, np.ndarray]]) -> tuple[Quadric, dict]:
    """Every solid's quadrics, solid after solid, and each solid's range of them and colour, from parts: each a
    batch of solids, `[solids, quadrics]` Quadric, all of one kind, and their colours, `[solids, 3]` or one
    for all."""
    quadrics, counts, colours = [], [], []
    for solids, colour in parts:
        quadrics.append(solids.reshape(-1))
        counts.append(np.full(solids.shape[0], solids.shape[1]))
        colours.append(np.broadcast_to(colour, (solids.shape[0], 3)))
    counts = np.concatenate(counts)
    ends = np.cumsum(counts)
    return concatenate(quadrics), dict(first=ends - counts, end=ends, colour=np.concatenate(colours))


def view(points: list[Point], reach: float, elevation: float, azimuth: float, margin: float) -> tuple[Motor, np.ndarray]:
    """A camera framing batches of points, of any weight, and `reach` around each, from an elevation and
    azimuth, in radians, z up and azimuth zero looking along +y, with a margin as a share of the larger side:
    its motor, and half the image's width and height."""
    turn = (mv.xy * (-azimuth / 2)).exp() * (mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()     # [] Motor
    # Across and up the image: the pairings with the camera's planes through the origin, over the weights.
    right, up = (np.concatenate([((turn >> plane) & batch / (mv.w & batch)).to_array().reshape(-1) for batch in points])
                 for plane in (mv.x, mv.y))
    middle = np.array([right.max() + right.min(), up.max() + up.min()]) / 2
    half = np.array([np.ptp(right), np.ptp(up)]) / 2 + reach
    half = half + margin * half.max()
    centre = turn >> (origin + mv.x.dual() * middle[0] + mv.y.dual() * middle[1])
    return translator(centre - origin) * turn, half


def on_screen(points: Point, elevation: float, azimuth: float) -> np.ndarray:
    """Points, of any weight, across and up the image of a view from an elevation and azimuth, about the
    origin, `[..., 2]`: for drawing over a traced image whose middle is the camera's eye seen the same way."""
    turn = (mv.xy * (-azimuth / 2)).exp() * (mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()
    unit = points / (mv.w & points)
    return np.stack([((turn >> plane) & unit).to_array() for plane in (mv.x, mv.y)], axis=-1)


def lamp(elevation: float, azimuth: float) -> Point:
    """The lamp's direction for a view: up and to the right of the viewer, and a little behind."""
    turn = (mv.xy * (-azimuth / 2)).exp() * (mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()
    toward = turn >> (mv.x.dual() * 0.4 + mv.y.dual() * 0.7 + mv.z.dual() * 0.6)
    return toward / (-(toward.dual() & toward)).square_root()


def draw(parts: list[tuple[Quadric, np.ndarray]], camera: Motor, half: np.ndarray, elevation: float,
         azimuth: float, pixels: int, background: np.ndarray) -> np.ndarray:
    """The solids as the camera sees them, `[rows, columns, 3]`, `pixels` across."""
    quadrics, solids = scene(parts)
    shape = (int(round(pixels * half[1] / half[0])), pixels)
    image = shader()(shape, quadrics=quadrics, solids=solids, camera=camera, back=4 * half.max(), half_height=half[1],
                     lamp=lamp(elevation, azimuth), background=background, ambient=ambient, gloss=gloss,
                     shininess=shininess)["image"]
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
