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

from numga import Algebra, GATypeError, NumpyContext, Unit, UnitGrid, dim
from numga.extensor import Extensor
from numga.gatype import GAType, Orientation


def _pixel_pair(value: Extensor | np.ndarray, *, checked: bool, name: str) -> np.ndarray:
    """Read a 2-element framing pair (`middle` or `half`), requiring `dim.pixel` when checked."""
    if isinstance(value, Extensor):
        if checked and value.has_units and not value.units.is_zero and value.unit != dim.pixel:
            raise GATypeError(f"Camera.{name} must carry pixel units; got {value.gatype.signature}")
        return np.asarray(value.to_array(), dtype=float).reshape(2)
    if checked:
        raise GATypeError(f"checked Camera.{name} must be a pixel-valued Scalar Extensor; got {type(value).__name__}")
    return np.asarray(value, dtype=float).reshape(2)


class Camera(NamedTuple):
    """Camera planes (`across`, `up`, `toward`), affine weight (`weight`, `unit`), picture centre and half-extents in
    camera readings (`middle`, `half`), and optional horizontal resolution (`pixels`) determine a view in
    `across.algebra`. When `across` carries checked units, `across & point` and `up & point` map scene points to
    `dim.pixel`, `weight & point` is dimensionless, and `middle` and `half` carry `dim.pixel`."""
    across: Extensor
    up: Extensor
    toward: Extensor
    weight: Extensor
    unit: float
    middle: Extensor | np.ndarray
    half: Extensor | np.ndarray
    pixels: int | None = None

    @property
    def algebra(self) -> Algebra:
        return self.across.algebra

    @property
    def has_units(self) -> bool:
        return self.across.has_units

    @property
    def _middle_array(self) -> np.ndarray:
        return _pixel_pair(self.middle, checked=self.has_units, name="middle")

    @property
    def _half_array(self) -> np.ndarray:
        return _pixel_pair(self.half, checked=self.has_units, name="half")

    @property
    def _columns(self) -> int:
        if self.pixels is not None:
            return int(self.pixels)
        return round(float(2 * self._half_array[0]))

    @property
    def pixel(self) -> float:
        """One horizontal pixel spans `2 * half[0] / columns` in camera readings (1.0 for pixel-mapped cameras)."""
        return float(2 * self._half_array[0] / self._columns)

    @property
    def shape(self) -> tuple[int, int]:
        """Row and column counts preserve the aspect ratio of `half` at `_columns` columns across."""
        half = self._half_array
        if self.pixels is None:
            return round(float(2 * half[1])), round(float(2 * half[0]))
        return round(float(self.pixels * half[1] / half[0])), int(self.pixels)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Horizontal and vertical bounds `(left, right, bottom, top)` span `middle - half` to `middle + half`."""
        middle, half = self._middle_array, self._half_array
        left, bottom = middle - half
        right, top = middle + half
        return float(left), float(right), float(bottom), float(top)

    def uniforms(self, depth: float) -> dict[str, Extensor | np.ndarray | float]:
        """Bind camera planes, framing readings, and symmetric depth range `depth` to the `placing` uniform names."""
        return dict(screen_across=self.across, screen_up=self.up, screen_toward=self.toward, screen_weight=self.weight,
                    screen_unit=self.unit, screen_middle=self._middle_array,
                    screen_half=self._half_array, screen_depth=depth)


def whole(element: Extensor, algebra: Algebra) -> GAType:
    """Expanding `element`'s support to all blades of its occupied grades in `algebra` while keeping its orientation
    gives the full-grade shader type that fits narrower subspaces and multi-grade values alike."""
    # Count set bits in each blade mask to collect every occupied grade.
    grades = sorted({bin(int(mask)).count("1") for mask in element.gatype.output_subspace.masks})
    return algebra.gatype.from_grades(grades).derive.with_orientations(element.gatype.orientations)


def check_projection(points: Extensor, camera: Camera) -> None:
    """Verify that checked `camera` planes map checked `points` into `dim.pixel` with dimensionless weight."""
    if not (points.has_units and camera.has_units):
        return
    # Validate framing pairs carry pixel units.
    _ = camera._middle_array, camera._half_array
    for name, plane in (("across", camera.across), ("up", camera.up)):
        if plane.units.is_zero:
            continue
        projected = read(plane, points)
        if not projected.units.is_zero and projected.unit != dim.pixel:
            raise GATypeError(
                f"camera.{name} must map points to pixel units; got {projected.gatype.signature}"
            )
    if not camera.weight.units.is_zero and np.any(camera.weight.kernel):
        weighted = read(camera.weight, points)
        if not weighted.units.is_zero and weighted.unit != dim.none:
            raise GATypeError(
                f"camera.weight must map points to dimensionless weight; got {weighted.gatype.signature}"
            )


def reading(points: Extensor, camera: Camera) -> dict[str, GAType]:
    """Full-grade `Point` and `Plane` shader types pair `points` against `camera`'s planes in `points.algebra`."""
    check_projection(points, camera)
    return dict(Point=whole(points, points.algebra), Plane=whole(camera.across, points.algebra))


def preimage(camera: Camera, Point: GAType) -> tuple[Extensor, Extensor, Extensor, Extensor]:
    """Solving the camera's four pairings on `Point`'s basis in least squares yields `(origin, across, up, toward)` at
    unit weight for `origin` and zero weight for the three steps, so readings `(x, y, z)` map back to
    `origin + x * across + y * up + z * toward`. When `camera` carries checked pixel units, `origin` carries scene
    point units and the three steps carry `scene_unit / pixel`."""
    bare_point = Point.derive.unchecked
    bare_context = NumpyContext(bare_point.algebra)
    n_coefficients = len(bare_point.output_subspace.masks)
    eye = np.eye(n_coefficients)
    basis = bare_context.extensor(bare_point, eye)                               # [coefficients] Point
    readers = (camera.across, camera.up, camera.toward, camera.weight)
    readings = np.stack([read(reader, basis).to_array() for reader in readers])
    # Columns specify target (across, up, toward, weight) readings for origin and unit steps along each axis.
    wanted = np.zeros((4, 4))
    wanted[[0, 1, 2], [1, 2, 3]] = 1.0
    wanted[3, 0] = 1.0 - camera.unit
    coefficients = np.linalg.lstsq(readings, wanted, rcond=None)[0]              # [coefficients, 4]
    if not camera.has_units:
        return tuple(bare_context.extensor(bare_point, coefficients[:, index]) for index in range(4))

    _ = camera._middle_array, camera._half_array
    context = camera.across.context
    targets = (dim.pixel, dim.pixel, dim.pixel, dim.none)
    point_units: dict[int, Unit] = {}
    for reader, target_unit in zip(readers, targets):
        if reader.units.is_zero:
            continue
        for j in range(n_coefficients):
            basis_j = context.extensor(
                bare_point.derive.unit(UnitGrid.from_entries((n_coefficients,), {j: dim.none})),
                eye[j],
            )
            value = read(reader, basis_j)
            if value.units.is_zero or abs(float(value.to_array())) == 0.0:
                continue
            blade_unit = target_unit / value.unit
            prev = point_units.get(j)
            if prev is not None and prev != blade_unit:
                raise GATypeError(
                    f"camera planes assign incompatible units {prev} and {blade_unit} to preimage blade"
                )
            point_units[j] = blade_unit

    step_scales = (dim.none, 1 / dim.pixel, 1 / dim.pixel, 1 / dim.pixel)
    results = []
    for index, scale in enumerate(step_scales):
        column = coefficients[:, index].copy()
        peak = float(np.max(np.abs(column))) if column.size else 0.0
        tol = 1e-12 * max(peak, 1.0)
        column[np.abs(column) <= tol] = 0.0
        entries = {
            j: point_units[j] * scale
            for j in range(n_coefficients)
            if column[j] != 0.0 and j in point_units
        }
        gatype = bare_point.derive.unit(UnitGrid.from_entries((n_coefficients,), entries))
        results.append(context.extensor(gatype, column))
    return tuple(results)


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

