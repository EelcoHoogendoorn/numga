"""How a picture reads the points of a scene, in the scene's own algebra.

A camera is three planes of the scene's algebra: a point's regressive products with them, `across & point`,
`up & point` and `toward & point`, are its place across and up the picture and its depth toward the viewer, divided by
the point's weight, an affine reading of it, `weight & point + unit`. In a projective algebra the weight is the pairing
with the plane at infinity; in a vector algebra, or a chart whose generators are all degenerate, a point is an offset
from the origin, its weight one; another plane gives a central projection, such as a view of the three-sphere along w.
The regressive product needs no metric, so the reading is the same in every signature: one shader source, compiled per
algebra, reads points of a projective plane or space, a vector space or spacetime alike, and the scene is never carried
into another algebra to be drawn. A planar picture reads by the planes x and y; a spacetime diagram by x and t; a view
of space by planes a motor turns.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from matplotlib.axes import Axes

from numga.extensor import Extensor
from numga.gatype import GAType, Orientation
from examples.util.wgsl.library import fitted


class Camera(NamedTuple):
    """The planes of the scene's algebra that read a point across and up the picture and toward the viewer; its weight,
    the plane and the constant whose affine reading divides them; the readings across and up at the picture's middle,
    and half its width and height in them; and its pixels across."""
    across: Extensor
    up: Extensor
    toward: Extensor
    weight: Extensor
    unit: float
    middle: np.ndarray
    half: np.ndarray
    pixels: int

    @property
    def algebra(self):
        return self.across.algebra

    @property
    def pixel(self) -> float:
        """A pixel's size in the readings."""
        return 2 * self.half[0] / self.pixels

    @property
    def shape(self) -> tuple[int, int]:
        """The picture's rows and columns."""
        return round(self.pixels * self.half[1] / self.half[0]), self.pixels

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Left, right, bottom and top, in the readings."""
        left, bottom = self.middle - self.half
        right, top = self.middle + self.half
        return left, right, bottom, top

    def uniforms(self, depth: float) -> dict:
        """The camera as the shader's `placing` block binds it, its depth range `depth` readings either way."""
        return dict(screen_across=self.across, screen_up=self.up, screen_toward=self.toward, screen_weight=self.weight,
                    screen_unit=self.unit, screen_middle=np.asarray(self.middle, dtype=float),
                    screen_half=np.asarray(self.half, dtype=float), screen_depth=depth)


def whole(value: Extensor, algebra) -> GAType:
    """The whole of a value's grade in an algebra, with the value's own orientation: the type a shader of that algebra
    declares for it, so that values of one grade with fewer blades, a point of the plane or a single plane, fit it
    alike."""
    # The grade of the value's blades, the generators each blade holds.
    grade = bin(int(value.gatype.output_subspace.masks[0])).count("1")
    return algebra.gatype.from_grades([grade]).derive.with_orientations(value.gatype.orientations)


def reading(points: Extensor, camera: Camera) -> dict:
    """The types a shader reads points by, each as it is: `Point`, the points' own, a point, a direction, a spin's
    patch or a vector alike; `Plane`, the camera's planes', whose regressive product with a point is a number."""
    return dict(Point=whole(points, points.algebra), Plane=whole(camera.across, points.algebra))


def plane(algebra, name: str) -> Extensor:
    """The plane of a generator of an algebra, by its name, the reading along that generator."""
    from numga import NumpyContext
    return getattr(NumpyContext(algebra).multivector, name)


def weighing(across: Extensor) -> tuple[Extensor, float]:
    """A point's weight, read by planes like `across`, as a plane and a constant: the pairing with the plane at infinity
    where one generator alone of the planes' algebra is degenerate, as in a projective algebra; one where none is, or
    all are, as in a vector algebra or a chart."""
    algebra = across.algebra
    degenerate = [name for name, square in zip(algebra.basis_names, algebra.signature) if square == 0]
    if len(degenerate) == 1 and len(algebra.signature) > 1:
        return plane(algebra, degenerate[0]), 0.0
    return 0 * across, 1.0


def panel(ax: Axes, across: Extensor, up: Extensor) -> Camera:
    """The camera of a matplotlib panel, reading its points by the planes `across` and `up`, weighed as their algebra
    weighs points, its limits the readings it spans, as many pixels across as the panel's room holds."""
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    weight, unit = weighing(across)
    camera = Camera(across, up, 0 * across, weight, unit, np.array([left + right, bottom + top]) / 2,
                    np.array([right - left, top - bottom]) / 2, 300)
    return fitted(camera, ax)


def flat_panel(ax: Axes, algebra) -> Camera:
    """The camera of a panel looking down on the plane of an algebra's generators x and y."""
    return panel(ax, plane(algebra, "x"), plane(algebra, "y"))


def extrinsic_first(Plane: GAType) -> bool:
    """Whether a camera's planes of the type `Plane` come first in their pairing with what they read: a reading pairs the
    extrinsic with the intrinsic, the extrinsic first, `mv.x & point` as `wavevector & direction`."""
    return Plane.orientation is not Orientation.intrinsic


def read(plane: Extensor, value: Extensor) -> Extensor:
    """A camera plane's reading of a value, their regressive product, the extrinsic first."""
    return plane & value if extrinsic_first(plane.gatype) else value & plane


def placing(Plane: GAType) -> str:
    """WGSL for a camera whose planes are of the type `Plane`, the same for every algebra: `paired(plane, point)`, the
    pairing of a plane with a point, the extrinsic first; `clip(point)`, a point's place in the picture's clip space and
    its depth, read by the camera's planes and divided by its weight; `screen_of(direction)`, a direction's change of
    place."""
    pairing = "plane & point" if extrinsic_first(Plane) else "point & plane"
    return f"""
// A camera plane's reading of a point, their regressive product, the extrinsic first.
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

// A point's weight: an affine reading of it, its pairing with the camera's weight plane and a constant.
fn weight_of(point: Point) -> f32 {
    return paired(screen_weight, point) + screen_unit;
}

// A point's place in the picture, across and up from its middle in half widths and heights, and its depth.
fn clip(point: Point) -> vec4<f32> {
    let weight = weight_of(point);
    let framed = vec2<f32>(paired(screen_across, point), paired(screen_up, point)) / weight - screen_middle;
    return vec4<f32>(framed / screen_half, 0.5 - paired(screen_toward, point) / weight / screen_depth, 1.0);
}

// A direction's change of place in the picture, across and up, in the readings: no weight to divide by.
fn screen_of(direction: Point) -> vec2<f32> {
    return vec2<f32>(paired(screen_across, direction), paired(screen_up, direction));
}
"""
