"""numga's WGSL dialect computes with scalars of either orientation as numga does: in a shader, a quotient, a product
or a reciprocal has numga's value and numga's orientation, which decides where it may be stored, and a function of a
number has numga's orientation, or is refused where numga refuses the number. A value of the algebra of any size crosses
from the vertex stage to the fragment stage whole."""

from itertools import product

import numpy as np
import pytest

pytest.importorskip("wgpu")

from numga import Algebra, GATypeError, NumpyContext, Orientation, stack

from examples.util.wgsl import Shader

ga = Algebra("x+y+z+w+")
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()                      # Antivector-
Plane = ga.gatype.vector()                          # Vector+
# The join of three points, and an amount, what is integrated over the space.
Facet = ga.gatype.antitrivector()                   # Antitrivector-
Amount = ga.gatype.antiscalar()                     # Antiscalar-
types = dict(Point=Point, Plane=Plane, Facet=Facet, Amount=Amount)
# Single precision against numga's double, relative to each result's largest size.
SINGLE_PRECISION = 3e-4
n_cases = 64

# A tetrahedron's corners, a plane and an amount for each case; each expression's result stored as the type numga
# gives it.
source = """
var<storage, read> corners: array<Point>;
var<storage, read> planes: array<Plane>;
var<storage, read> amounts: array<Amount>;
var<storage, read_write> ramps: array<Plane>;
var<storage, read_write> facets_over_readings: array<Facet>;
var<storage, read_write> scaled_planes: array<Plane>;
var<storage, read_write> facets_over_volumes: array<Plane>;
var<storage, read_write> reciprocals: array<Amount>;

@compute @workgroup_size(8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    if (k >= arrayLength(&planes)) {
        return;
    }
    let corner = corners[4u * k];
    let facet = corners[4u * k + 1u] & corners[4u * k + 2u] & corners[4u * k + 3u];
    let volume = corner & facet;
    ramps[k] = facet / (facet & corner);
    facets_over_readings[k] = facet / (planes[k] & corner);
    scaled_planes[k] = (volume * volume) * planes[k];
    facets_over_volumes[k] = (1.0 / volume) * facet;
    reciprocals[k] = amounts[k].inverse();
}
"""


def cases() -> tuple:
    random = np.random.default_rng(3)
    corners = mv.vector(random.normal(size=(n_cases, 4, 4))).dual()            # [cases, corners] Point-
    planes = mv.vector(random.normal(size=(n_cases, 4)))                         # [cases] Plane+
    amounts = mv(Amount, random.normal(size=(n_cases, 1)))                     # [cases] Amount-
    return corners, planes, amounts


def test_quotients_products_and_reciprocals_are_numgas():
    corners, planes, amounts = cases()
    traced = Shader(source, ga, types, {})((1, n_cases), corners=corners.reshape(-1), planes=planes,
                                            amounts=amounts)
    corner = corners[:, 0]                                                        # [cases] Point-
    facet = corners[:, 1] & corners[:, 2] & corners[:, 3]                        # [cases] Facet-
    volume = corner & facet                                                       # [cases] Scalar-
    expected = dict(ramps=facet / (facet & corner), facets_over_readings=facet / (planes & corner),
                    scaled_planes=(volume * volume) * planes, facets_over_volumes=volume.inverse() * facet,
                    reciprocals=amounts.inverse())
    for name, value in expected.items():
        # The difference is numga's, which takes only values of one orientation.
        error = np.asarray((traced[name][0] - value).kernel)
        assert np.abs(error).max() <= SINGLE_PRECISION * np.abs(np.asarray(value.kernel)).max(), name


def test_a_quotient_of_the_other_orientation_is_not_stored():
    # A facet over a reading of a point by a plane stays a facet, intrinsic: no extrinsic plane holds it.
    wrong = source.replace("var<storage, read_write> facets_over_readings: array<Facet>;",
                           "var<storage, read_write> facets_over_readings: array<Plane>;")
    with pytest.raises(SyntaxError):
        Shader(wrong, ga, types, {})


# The product of two planes, a scalar and a bivector, seven coefficients, carried from the vertices of a triangle over
# the whole picture to its fragments with a point after it, and the point turned by it there, read by the planes x, y,
# z and w.
crossing_source = """
var<uniform> first: Plane;
var<uniform> second: Plane;
var<uniform> place: Point;

struct Corner {
    @builtin(position) position: vec4<f32>,
    turn: Turn,
    place: Point,
}

@vertex
fn spread(@builtin(vertex_index) index: u32) -> Corner {
    let corner = array<vec2<f32>, 3>(vec2<f32>(-1, -1), vec2<f32>(3, -1), vec2<f32>(-1, 3))[index];
    return Corner(vec4<f32>(corner, 0.5, 1.0), first * second, place);
}

@fragment
fn shade(corner: Corner) -> @location(0) vec4<f32> {
    let moved = corner.turn >> corner.place;
    return vec4<f32>(mv.x & moved, mv.y & moved, mv.z & moved, mv.w & moved);
}
"""


def test_a_value_of_seven_coefficients_crosses_between_stages():
    random = np.random.default_rng(5)
    first, second = mv.vector(random.normal(size=(2, 4)))                      # [] Plane+, Plane+
    place = mv.vector(random.normal(size=4)).dual()                             # [] Point-
    turn = first * second                                                       # [] Even+
    shader = Shader(crossing_source, ga, types | dict(Turn=turn.gatype), {}, targets=("rgba32float",))
    image, = shader.draw((4, 4), 3, first=first, second=second, place=place)
    expected = (stack((mv.x, mv.y, mv.z, mv.w)) & (turn >> place)).to_array()   # [4]
    assert np.abs(image - expected).max() <= SINGLE_PRECISION * np.abs(expected).max()


# numga's functions of a number, with the bounds of the one that takes them, and two that numga has not.
functions = dict(square_root=(), inverse_square_root=(), exp=(), log=(), sin=(), cos=(), tan=(), arcsin=(), arccos=(),
                 arctan=(), sinh=(), cosh=(), tanh=(), abs=(), inverse=(), clip=(0.0, 1.0), sign=(), floor=())
# A number of either orientation, by the name of a uniform of it.
numbers = dict(extrinsic=ga.gatype.scalar(), intrinsic=ga.gatype.antipseudoscalar())


def test_functions_of_numbers_have_numgas_orientation():
    disagreements = []
    for (name, bounds), orientation, given in product(functions.items(), numbers, (False, True)):
        number = mv(numbers[orientation], [0.5])                                # [] Scalar+ or Scalar-
        try:
            result = getattr(number, name)(*bounds).gatype.orientation.value
        except (GATypeError, AttributeError):
            result = None
        # The function of a number read at run time, or given when the shader compiles, added to a number of the
        # orientation numga gives its result: numga adds only numbers of one orientation.
        source = f"""
var<uniform> extrinsic: f32;
var<uniform> intrinsic: Intrinsic;
{"" if given else "var<uniform> number: Number;"}
var<storage, read_write> results: array<f32>;

@compute @workgroup_size(1)
fn main() {{
    results[0] = number.{name}({", ".join(map(str, bounds))}) + {result or "extrinsic"};
}}
"""
        try:
            Shader(source, ga, dict(Intrinsic=numbers["intrinsic"], Number=numbers[orientation]),
                   dict(number=number) if given else {})
            taken = True
        except SyntaxError:
            taken = False
        if taken != (result is not None):
            disagreements.append((name, orientation, "given" if given else "read"))
    assert not disagreements
