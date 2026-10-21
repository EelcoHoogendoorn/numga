"""Unit and integration tests for WebGPU / Metal compute raytracing and WGSL transpilation."""

from __future__ import annotations

import numpy as np
import pytest

from numga import NumpyContext, stack
from numga.algebras import PGA3D as ga
from examples import raytrace as rt

context = NumpyContext(ga)
mv = context.multivector
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Quadric = ga.gatype((Plane, Point))
Line = ga.gatype.bivector()


def test_transpiler_struct_assignment_resolution():
    """Variable declarations with initializers correctly resolve struct types."""
    shader = """
    struct HitRecord {
        t: f32,
        normal: Plane,
        pos: Point,
    };
    fn test_fn(h: HitRecord) -> f32 {
        let hit: HitRecord = h;
        let n_sq: f32 = hit.normal | hit.normal;
        return n_sq;
    }
    """
    transpiler = rt.NumgaWGSLTranspiler(ga)
    transpiler.register_type("Point", Point)
    transpiler.register_type("Plane", Plane)
    wgsl = transpiler.transpile(shader)

    assert "ga_inner_plane_plane(hit.normal, hit.normal)" in wgsl or "ga_dot_plane_plane(hit.normal, hit.normal)" in wgsl


def test_transpiler_bivector_line_blade_accessors():
    """Higher-dimensional GA types like 6D bivectors produce matching struct definitions and blade accessors."""
    transpiler = rt.NumgaWGSLTranspiler(ga)
    transpiler.register_type("Line", Line)
    transpiler.register_type("Plane", Plane)
    transpiler.register_type("Point", Point)

    shader = """
    fn line_plane(l: Line, p: Plane) -> Point {
        return l & p;
    }
    """
    wgsl = transpiler.transpile(shader)
    assert "struct Line {" in wgsl
    # Verify that clean blade names are used (wx, wy, wz, yz, zx, xy) rather than raw generic indices
    for blade in ["wx", "wy", "wz", "yz", "zx", "xy"]:
        assert f"{blade}: f32," in wgsl


def test_wgpu_pack_struct_matrix_column_major():
    """Arity-1 Extensor linear maps are packed column-major matching WGSL mat4x4 layout."""
    tracer = rt.WGPUCompute(
        """
        struct TestUniforms {
            m: Quadric,
            scale: f32,
        };
        @group(0) @binding(0) var<uniform> u: TestUniforms;
        @group(0) @binding(1) var<storage, read_write> out: array<f32>;
        @compute @workgroup_size(1)
        fn main() { out[0] = u.scale; }
        """,
        algebra=ga,
        types={"Point": Point, "Plane": Plane, "Quadric": Quadric}
    )

    # An asymmetric test map
    mat_np = np.arange(16, dtype=np.float32).reshape(4, 4)
    q_map = context.extensor(Quadric, mat_np)
    packed = tracer._pack_struct("TestUniforms", {"m": q_map, "scale": 3.5})

    # The first 64 bytes are the 4x4 matrix, which must be transposed (column-major)
    unpacked_mat = np.frombuffer(packed[:64], dtype=np.float32).reshape(4, 4)
    np.testing.assert_array_equal(unpacked_mat, mat_np.T)
    # The float scale follows immediately at offset 64
    unpacked_scale = np.frombuffer(packed[64:68], dtype=np.float32)[0]
    assert unpacked_scale == 3.5


def test_wgpu_quadric_raycasting_parity_with_cpu():
    """WGPUCompute executing explicit WGSL GA ray-quadric hits matches CPU rt.hits."""
    pytest.importorskip("wgpu")

    shader = """
    struct Ray {
        ro: Point,
        rd: Point,
    };
    @group(0) @binding(0) var<uniform> ray: Ray;
    @group(0) @binding(1) var<storage, read> surface: array<Quadric>;
    @group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

    @compute @workgroup_size(1)
    fn main() {
        let q: Quadric = surface[0];
        let a: f32 = q(ray.rd) & ray.rd;
        let b: f32 = q(ray.rd) & ray.ro;
        let c: f32 = q(ray.ro) & ray.ro;
        let disc: f32 = b * b - a * c;
        var t: f32 = -1.0;
        if (disc >= 0.0 && a > 0.0) {
            t = (-b - sqrt(disc)) / a;
        }
        output[0] = vec4<f32>(t, a, b, c);
    }
    """
    compute = rt.WGPUCompute(
        shader,
        algebra=ga,
        types={"Point": Point, "Plane": Plane, "Quadric": Quadric}
    )

    q = rt.ellipsoid(np.array([1.5, 0.8, 1.2]))
    origin = mv.point([0.0, 0.0, 5.0])
    heading = mv.direction([0.0, 0.0, -1.0])

    # CPU hits
    dist_cpu, hit_cpu = rt.hits(q, origin, heading)

    # GPU compute
    res = compute(width=1, height=1, ray={"ro": origin, "rd": heading}, surface=q[None])
    dist_gpu = res[0, 0, 0]

    np.testing.assert_allclose(dist_gpu, dist_cpu.item(), atol=1e-5)

