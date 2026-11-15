"""Shaders in the numga dialect run over arrays of extensors: products, sandwiches and exponentials in several
algebras, a local variable starting at zero, and the antinorm, as numga computes them."""

from __future__ import annotations

import numpy as np
import pytest

from numga import NumpyContext, stack
from numga.algebras import PGA3D, STA, VGA3D
from numga.wgsl import Shader

pytest.importorskip("wgpu")

ATOL = 2e-5


def test_vga3d_products_sandwich_exp_and_loop_var_zero_init():
    ctx = NumpyContext(VGA3D)
    mv = ctx.multivector
    Vector = VGA3D.gatype.vector()
    Bivector = VGA3D.gatype.bivector()
    Rotor = VGA3D.gatype.rotor()

    bivectors = stack((0.4 * mv.xy - 0.25 * mv.yz, 1e-5 * mv.zx, 0.0 * mv.xy))
    vectors = mv.vector(np.array([[1.0, 2.0, -0.5], [-0.4, 0.7, 1.3], [0.2, -1.1, 0.8]]))

    source = """
var<storage, read> bivectors: array<Bivector>;
var<storage, read> vectors: array<Vector>;
var<storage, read_write> rotated: array<Vector>;
var<storage, read_write> sums: array<Vector>;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&vectors)) {
        return;
    }
    var acc: Vector;
    for (var k = 0u; k < 3u; k++) {
        acc = acc + vectors[k];
    }
    sums[i] = acc;
    let r = bivectors[i].exp();
    rotated[i] = r >> vectors[i];
}
"""
    shader = Shader(source, VGA3D, dict(Vector=Vector, Bivector=Bivector, Rotor=Rotor), {})
    traced = shader((3,), bivectors=bivectors, vectors=vectors)
    assert np.allclose(traced["rotated"].kernel, (bivectors.exp() >> vectors).kernel, atol=ATOL)
    assert np.allclose(traced["sums"][0].kernel, (vectors[0] + vectors[1] + vectors[2]).kernel, atol=ATOL)


def test_pga3d_antinorm_and_sta_boost_exp():
    pga = NumpyContext(PGA3D).multivector
    Point = PGA3D.gatype.antivector()
    pts = pga.point(np.array([[1.2, -0.4, 2.0], [-0.5, 0.9, 0.3], [0.0, 0.0, 0.0]]))

    pga_shader = Shader(
        """
var<storage, read> pts: array<Point>;
var<storage, read_write> weights: array<f32>;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&pts)) {
        return;
    }
    weights[id.x] = pts[id.x].antinorm();
}
""",
        PGA3D,
        dict(Point=Point),
        {},
    )
    out = pga_shader((3,), pts=pts)
    assert np.allclose(out["weights"], pts.antinorm().kernel, atol=ATOL)

    sta = NumpyContext(STA).multivector
    Boost = STA.gatype.bivector().derive.containing("t")
    Vector = STA.gatype.vector()
    boosts = stack((0.6 * sta.tx - 0.2 * sta.ty, 0.0 * sta.tz))
    vec = 1.5 * sta.t + 0.4 * sta.x - 0.3 * sta.y
    sta_shader = Shader(
        """
var<storage, read> boosts: array<Boost>;
var<uniform> v: Vector;
var<storage, read_write> moved: array<Vector>;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&boosts)) {
        return;
    }
    moved[id.x] = boosts[id.x].exp() >> v;
}
""",
        STA,
        dict(Boost=Boost, Vector=Vector),
        {},
    )
    moved = sta_shader((2,), boosts=boosts, v=vec)["moved"]
    assert np.allclose(moved.kernel, (boosts.exp() >> vec).kernel, atol=ATOL)
