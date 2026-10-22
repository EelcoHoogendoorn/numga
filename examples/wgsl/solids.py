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
from typing import NamedTuple

import numpy as np

from numga import NumpyContext, concatenate, stack
from numga.algebras import PGA3D as ga
from examples.wgsl import Shader, supersampled
from examples.wgsl.library import compose

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

source = supersampled(3) + """
// A solid: the intersection of its quadrics, `quadrics[first]` up to `quadrics[end]`, and its colour.
struct Solid {
    first: u32,
    end: u32,
    colour: vec3<f32>,
}

var<storage, read> solids: array<Solid>;
var<storage, read> quadrics: array<Quadric>;
// Lines inked on the surfaces of the solids `first` up to `end`, where the ink's quadric pairs to zero with the point,
// `width` pixels across, each pixel `pixel` across in the scene.
struct Ink {
    first: u32,
    end: u32,
    colour: vec3<f32>,
    width: f32,
}

var<storage, read> inks: array<Ink>;
var<storage, read> ink_quadrics: array<Quadric>;
// Where each ink shows: where its region's quadric pairs negatively with the point.
var<storage, read> ink_regions: array<Quadric>;
var<uniform> pixel: f32;
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

// A sample's colour and its depth along the view, in the first channel, each with its coverage.
struct Sample {
    colour: vec4<f32>,
    depth: vec4<f32>,
}

const origin = mv.w.dual();
const far = 1e30;
// How far a point may lie outside a quadric and still count as inside, for points on two surfaces.
const tolerance = 1e-5;

struct Hit {
    distance: f32,
    tangent: Plane,
    colour: vec3<f32>,
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

// The first point a ray meets on a solid's surface: on one of its quadrics, and inside all the others.
fn first_hit(start: Point, direction: Point) -> Hit {
    var hit = Hit(far, mv.z, vec3<f32>(0.0), 0u);
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
                    hit = Hit(distance, quadric(point), solid.colour, index);
                }
            }
        }
    }
    return hit;
}

// A hit's colour with the lines inked on its solid: the distance along the surface to an ink's zero set is its
// pairing with the point over the part of its gradient along the surface, the quadric's plane there less its part
// along the normal.
fn inked(hit: Hit, point: Point, normal: Plane) -> vec3<f32> {
    var colour = hit.colour;
    for (var index = 0u; index < arrayLength(&inks); index++) {
        let ink = inks[index];
        let gradient = ink_quadrics[index](point) * 2;
        let along = gradient - normal * (gradient | normal);
        let distance = abs(ink_quadrics[index](point) & point) / (along | along).square_root();
        let within = (ink_regions[index](point) & point) <= tolerance;
        let covers = hit.solid >= ink.first && hit.solid < ink.end && within && distance < ink.width / 2 * pixel;
        colour = select(colour, ink.colour, covers);
    }
    return colour;
}

fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place) * half_height;
    let start = camera >> (origin + mv.z.dual() * back + mv.x.dual() * across.x + mv.y.dual() * across.y);
    let direction = camera >> -mv.z.dual();
    let hit = first_hit(start, direction);
    if (hit.distance >= far) {
        return Sample(vec4<f32>(0.0), vec4<f32>(0.0));
    }
    // Facing the viewer: a tangent plane is lit on the side the ray comes from.
    let tangent = hit.tangent * select(1.0, -1.0, (hit.tangent & direction) > 0);
    let normal = tangent / (tangent | tangent).square_root();
    let light = max(normal & lamp, 0.0);
    // The view ray reflected in the tangent plane, against the lamp: `-d.dual()` is the plane square to d.
    let glint = pow(max(-((normal >> direction).dual() & lamp), 0.0), shininess) * select(0.0, 1.0, light > 0);
    let fill = ambient * (1 + fill_tilt * (normal | mv.z)) + headlight * -(normal & direction);
    let colour = inked(hit, start + direction * hit.distance, normal) * (fill + (1 - ambient - headlight) * light) + gloss * glint;
    return Sample(vec4<f32>(colour, 1.0), vec4<f32>(hit.distance - back, 0.0, 0.0, 1.0));
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


def boxes(motors: Motor, half_extents: np.ndarray) -> Quadric:
    """Solid boxes placed by motors, their half extents along their own x, y and z on the last axis, `[..., 6]`
    quadrics each: the half-spaces inside each face."""
    reach = mv.w * half_extents                                               # [..., 3] Plane
    faces = concatenate([axes - reach, -axes - reach], axis=-1)               # [..., 6] Plane
    return placed(half_space(faces), motors[..., None])


def edges(starts: Point, ends: Point, radius: float) -> tuple[Quadric, Quadric]:
    """Edges between points: rods `[..., 3]` and balls of the same radius at both ends `[ends, 1]`, so that edges
    meeting at a point join round."""
    return rods(starts, ends, radius), ball(stack([starts, ends], axis=-1).reshape(-1), radius)[:, None]


# --- plumbing -------------------------------------------------------------------------
@cache
def shader() -> Shader:
    return Shader(source, ga, dict(Point=Point, Plane=Plane, Motor=Motor, Quadric=Quadric))


class Ink(NamedTuple):
    """Lines inked on the surfaces of a part's solids, where each quadric `[lines]` pairs to zero with the point,
    within where each region `[lines]` pairs negatively with it (`everywhere` for no bound): their colours,
    `[lines, 3]` or one for all, and their width in pixels."""
    part: int
    quadrics: Quadric
    regions: Quadric
    colours: np.ndarray
    width: float


def scene(parts: list[tuple[Quadric, np.ndarray]], inks: list[Ink]) -> tuple[Quadric, dict, Quadric, Quadric, dict]:
    """Every solid's quadrics, solid after solid, and each solid's range of them and colour, from parts: each a
    batch of solids, `[solids, quadrics]` Quadric, all of one kind, and their colours, `[solids, 3]` or one
    for all; and every ink's quadric, with its part's range of solids, colour and width."""
    quadrics, counts, colours = [], [], []
    for solids, colour in parts:
        quadrics.append(solids.reshape(-1))
        counts.append(np.full(solids.shape[0], solids.shape[1]))
        colours.append(np.broadcast_to(colour, (solids.shape[0], 3)))
    ends = np.cumsum(np.concatenate(counts))
    # The solids each part spans, part after part.
    part_ends = np.cumsum([solids.shape[0] for solids, _ in parts])
    part_starts = part_ends - [solids.shape[0] for solids, _ in parts]
    # An ink over no solids stands for none, so that the inks' arrays are never empty.
    inks = list(inks) + [Ink(0, everywhere[None], everywhere[None], np.zeros(3), 0.0)]
    ink_quadrics = concatenate([ink.quadrics.reshape(-1) for ink in inks])
    ink_regions = concatenate([ink.regions.reshape(-1) for ink in inks])
    sizes = [int(np.prod(ink.quadrics.shape)) for ink in inks]
    marked = dict(first=np.repeat([part_starts[ink.part] for ink in inks[:-1]] + [0], sizes),
                  end=np.repeat([part_ends[ink.part] for ink in inks[:-1]] + [0], sizes),
                  colour=np.concatenate([np.broadcast_to(ink.colours, (size, 3)) for ink, size in zip(inks, sizes)]),
                  width=np.repeat([ink.width for ink in inks], sizes))
    return (concatenate(quadrics), dict(first=ends - np.concatenate(counts), end=ends, colour=np.concatenate(colours)),
            ink_quadrics, ink_regions, marked)


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


def layer(parts: list[tuple[Quadric, np.ndarray]], view: View, inks: list[Ink] = ()) -> tuple[np.ndarray, np.ndarray]:
    """The solids as the view sees them, with the lines inked on them: their colour, `[rows, columns, 4]`, and their
    depth along the view, `[rows, columns]`, each weighted by coverage."""
    quadrics, solids, ink_quadrics, ink_regions, marked = scene(parts, inks)
    image = shader()(view.shape, quadrics=quadrics, solids=solids, inks=marked, ink_quadrics=ink_quadrics,
                     ink_regions=ink_regions,
                     pixel=view.pixel, camera=view.camera, back=4 * view.half.max(),
                     half_height=view.half[1], lamp=lamp(view.camera), ambient=ambient, fill_tilt=fill_tilt,
                     headlight=headlight, gloss=gloss, shininess=shininess)["image"]
    return image["colour"], image["depth"][..., 0]


def draw(parts: list[tuple[Quadric, np.ndarray]], view: View, background: np.ndarray, inks: list[Ink] = ()) -> np.ndarray:
    """The solids as the view sees them on the background, with the lines inked on them, `[rows, columns, 3]`."""
    return compose([(*layer(parts, view, inks), 1.0)], background)
