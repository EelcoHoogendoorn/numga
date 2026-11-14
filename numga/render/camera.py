"""A camera reads the points of a scene directly in the scene's own algebra through three planes: `across & point`,
`up & point`, and `toward & point` measure horizontal position, vertical position, and depth toward the viewer, each
divided by the affine weight `weight & point + unit`. In a projective algebra `weight` pairs with the plane at infinity;
in a vector algebra or a chart whose generators are all degenerate, points are offsets from the origin with unit weight
(`weight = 0`, `unit = 1`); choosing another plane gives a central projection, such as viewing the three-sphere along
`w`.

Because the regressive product requires no metric, the same shader source compiles for every signature and dimension,
reading projective spaces, vector spaces, and spacetime diagrams alike without moving geometry into another algebra.
Callers declare every camera plane explicitly, including `weight`, because neither generator names nor metric signature
determine which plane weighs points when a degenerate generator instead carries an auxiliary quantity such as entropy.

Orientation governs factor order in every camera pairing: the extrinsic factor stands on the left and the intrinsic
factor on the right (`extrinsic_first`), matching `mv.x & point` and `wavevector & direction`. Reconstructing scene
points from picture readings (`preimage`) inverts the four camera pairings on `Point`'s basis via least squares when a
view leaves directions along the sightline unconstrained. Orientation rules for regressive pairings are detailed in
`docs/orientation.md`.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from numga import Algebra, NumpyContext
from numga.extensor import Extensor
from numga.gatype import GAType, Orientation


class Camera(NamedTuple):
    """Camera planes (`across`, `up`, `toward`), affine weight (`weight`, `unit`), picture centre and half-extents in
    camera readings (`middle`, `half`), and horizontal resolution (`pixels`) determine a view in `across.algebra`."""
    across: Extensor
    up: Extensor
    toward: Extensor
    weight: Extensor
    unit: float
    middle: np.ndarray
    half: np.ndarray
    pixels: int

    @property
    def algebra(self) -> Algebra:
        return self.across.algebra

    @property
    def pixel(self) -> float:
        """One horizontal pixel spans `2 * half[0] / pixels` in camera readings."""
        return 2 * self.half[0] / self.pixels

    @property
    def shape(self) -> tuple[int, int]:
        """Row and column counts preserve the aspect ratio of `half` at `pixels` columns across."""
        return round(self.pixels * self.half[1] / self.half[0]), self.pixels

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Horizontal and vertical bounds `(left, right, bottom, top)` span `middle - half` to `middle + half`."""
        left, bottom = self.middle - self.half
        right, top = self.middle + self.half
        return left, right, bottom, top

    def uniforms(self, depth: float) -> dict[str, Extensor | np.ndarray | float]:
        """Bind camera planes, framing readings, and symmetric depth range `depth` to the `placing` uniform names."""
        return dict(screen_across=self.across, screen_up=self.up, screen_toward=self.toward, screen_weight=self.weight,
                    screen_unit=self.unit, screen_middle=np.asarray(self.middle, dtype=float),
                    screen_half=np.asarray(self.half, dtype=float), screen_depth=depth)


def whole(element: Extensor, algebra: Algebra) -> GAType:
    """Expanding `element`'s support to all blades of its occupied grades in `algebra` while keeping its orientation
    gives the full-grade shader type that fits narrower subspaces and multi-grade values alike."""
    # Count set bits in each blade mask to collect every occupied grade.
    grades = sorted({bin(int(mask)).count("1") for mask in element.gatype.output_subspace.masks})
    return algebra.gatype.from_grades(grades).derive.with_orientations(element.gatype.orientations)


def reading(points: Extensor, camera: Camera) -> dict[str, GAType]:
    """Full-grade `Point` and `Plane` shader types pair `points` against `camera`'s planes in `points.algebra`."""
    return dict(Point=whole(points, points.algebra), Plane=whole(camera.across, points.algebra))


def preimage(camera: Camera, Point: GAType) -> tuple[Extensor, Extensor, Extensor, Extensor]:
    """Solving the camera's four pairings on `Point`'s basis in least squares yields `(origin, across, up, toward)` at
    unit weight for `origin` and zero weight for the three steps, so readings `(x, y, z)` map back to
    `origin + x * across + y * up + z * toward`."""
    context = NumpyContext(Point.algebra)
    n_coefficients = len(Point.output_subspace.masks)
    basis = context.extensor(Point, np.eye(n_coefficients))                     # [coefficients] Point
    readers = (camera.across, camera.up, camera.toward, camera.weight)
    readings = np.stack([read(reader, basis).to_array() for reader in readers])
    # Columns specify target (across, up, toward, weight) readings for origin and unit steps along each axis.
    wanted = np.zeros((4, 4))
    wanted[[0, 1, 2], [1, 2, 3]] = 1.0
    wanted[3, 0] = 1.0 - camera.unit
    coefficients = np.linalg.lstsq(readings, wanted, rcond=None)[0]             # [coefficients, 4]
    origin, across, up, toward = (context.extensor(Point, coefficients[:, index]) for index in range(4))
    return origin, across, up, toward


def extrinsic_first(Plane: GAType) -> bool:
    """Non-intrinsic camera planes stand on the left of `&` when paired against intrinsic scene elements."""
    return Plane.orientation is not Orientation.intrinsic


def read(plane: Extensor, point: Extensor) -> Extensor:
    """Pairing `plane` with `point` via `&` places the extrinsic factor on the left."""
    return plane & point if extrinsic_first(plane.gatype) else point & plane


def placing(Plane: GAType) -> str:
    """The WGSL camera block defines `paired(plane, point)`, `weight_of(point)`, `clip(point)`, and
    `screen_of(direction)` with factor order determined by `Plane`'s orientation."""
    pairing = "plane & point" if extrinsic_first(Plane) else "point & plane"
    return f"""
// Camera plane pairing with the extrinsic factor on the left.
fn paired(plane: Plane, point: Point) -> f32 {{
    return {pairing};
}}
""" + """
var<uniform> screen_across: Plane;
var<uniform> screen_up: Plane;
var<uniform> screen_toward: Plane;
var<uniform> screen_weight: Plane;
var<uniform> screen_unit: f32;
var<uniform> screen_middle: vec2<f32>;
var<uniform> screen_half: vec2<f32>;
var<uniform> screen_depth: f32;

// Affine point weight from pairing with the weight plane plus the constant offset.
fn weight_of(point: Point) -> f32 {
    return paired(screen_weight, point) + screen_unit;
}

// Clip-space position and normalized depth from weight-divided camera readings.
fn clip(point: Point) -> vec4<f32> {
    let weight = weight_of(point);
    let framed = vec2<f32>(paired(screen_across, point), paired(screen_up, point)) / weight - screen_middle;
    return vec4<f32>(framed / screen_half, 0.5 - paired(screen_toward, point) / weight / screen_depth, 1.0);
}

// Unweighted horizontal and vertical camera readings of a displacement or direction.
fn screen_of(direction: Point) -> vec2<f32> {
    return vec2<f32>(paired(screen_across, direction), paired(screen_up, direction));
}
"""

