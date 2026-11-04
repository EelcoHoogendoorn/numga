"""numga's WGSL dialect computes with scalars of either orientation as numga does: in a shader, a quotient, a product
or a reciprocal has numga's value and numga's orientation, which decides where it may be stored."""

import numpy as np
import pytest

pytest.importorskip("wgpu")

from numga import Algebra, NumpyContext

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
