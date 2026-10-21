"""Rasterise and raytrace spherical quadrics on S² using WebGPU compute and in-shader supersampling."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.quadrics.cga_spherical_quadrics.core import Plane, Point, Quadric, ga, point
from examples.quadrics.elliptic_physics.render import hemisphere, paint
from examples.raytrace import WGPUCompute

BACKGROUND = np.array([0.039, 0.059, 0.114, 1.0], dtype=np.float32)  # #0a0f1d
DISK = np.array([0.067, 0.094, 0.153, 1.0], dtype=np.float32)        # #111827
RIM = np.array([0.200, 0.255, 0.333, 1.0], dtype=np.float32)         # #334155

CGA_SPHERICAL_QUADRIC_WGSL = """
struct Uniforms {
    resolution: vec2<f32>,
    dual_map: mat4x4<f32>,
    bg_color: vec4<f32>,
    disk_color: vec4<f32>,
    rim_color: vec4<f32>,
    mode: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> scene: Uniforms;
@group(0) @binding(1) var<storage, read> quadrics: array<Quadric>;
@group(0) @binding(2) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f32>>;

fn sample_point(uv: vec2<f32>) -> vec3<f32> {
    let r2 = uv.x * uv.x + uv.y * uv.y;
    if (r2 > 1.0) {
        return scene.bg_color.xyz;
    }
    if (r2 > 0.985) {
        return scene.rim_color.xyz;
    }

    let z = sqrt(max(0.0, 1.0 - r2));
    let v_null = vec4<f32>(uv.x, uv.y, z, 1.0);
    let p: Point = scene.dual_map * v_null;

    var pixel_color = scene.disk_color.xyz;
    let n_quadrics = arrayLength(&quadrics);
    for (var i = 0u; i < n_quadrics; i = i + 1u) {
        let q: Quadric = quadrics[i];
        let potential: f32 = q(p) & p;
        if (potential < 0.0) {
            pixel_color = colors[i].xyz;
        }
    }

    if (scene.mode > 0.5) {
        let normal = vec3<f32>(uv.x, uv.y, z);
        let lamp = normalize(vec3<f32>(-0.3, 0.4, 0.86));
        let diff = max(0.0, dot(normal, lamp));
        let refl = reflect(-lamp, normal);
        let spec = pow(max(0.0, dot(refl, vec3<f32>(0.0, 0.0, 1.0))), 35.0);
        pixel_color = pixel_color * (0.35 + 0.65 * diff) + vec3<f32>(0.3 * spec);
    }

    return pixel_color;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let px = id.x;
    let py = id.y;
    let w = u32(scene.resolution.x);
    let h = u32(scene.resolution.y);
    if (px >= w || py >= h) { return; }

    var acc = vec3<f32>(0.0);
    let sub_offsets = array<vec2<f32>, 4>(
        vec2<f32>(-0.25, -0.25),
        vec2<f32>(0.25, -0.25),
        vec2<f32>(-0.25, 0.25),
        vec2<f32>(0.25, 0.25)
    );

    for (var s = 0u; s < 4u; s = s + 1u) {
        let uv = vec2<f32>(
            (f32(px) + 0.5 + sub_offsets[s].x) / f32(w) * 2.0 - 1.0,
            1.0 - (f32(py) + 0.5 + sub_offsets[s].y) / f32(h) * 2.0
        );
        acc = acc + sample_point(uv);
    }

    output[py * w + px] = vec4<f32>(acc * 0.25, 1.0);
}
"""

_tracer: WGPUCompute | None = None
_dual_op = None


def get_tracer() -> WGPUCompute:
    global _tracer
    if _tracer is None:
        _tracer = WGPUCompute(
            CGA_SPHERICAL_QUADRIC_WGSL,
            algebra=ga,
            types={"Point": Point, "Plane": Plane, "Quadric": Quadric}
        )
    return _tracer


def get_dual_op():
    global _dual_op
    if _dual_op is None:
        _dual_op = Plane.dual()
    return _dual_op


def render_gpu_frame(quadrics: Quadric, color_rgbs: np.ndarray, resolution: int, mode: float = 0.0) -> np.ndarray:
    """Render a single frame of spherical quadrics on S² via WebGPU compute with in-shader 2x2 supersampling."""
    tracer = get_tracer()
    dual_op = get_dual_op()

    uniforms = {
        "resolution": np.array([resolution, resolution], dtype=np.float32),
        "dual_map": dual_op,
        "bg_color": BACKGROUND,
        "disk_color": DISK,
        "rim_color": RIM,
        "mode": float(mode),
        "_pad0": 0.0,
        "_pad1": 0.0,
        "_pad2": 0.0,
    }

    if getattr(quadrics, "shape", ()) == ():
        quadrics = stack([quadrics])
    cols = color_rgbs
    if cols.ndim == 1:
        cols = cols[None, :]
    if cols.shape[-1] == 3:
        cols = np.pad(cols, [(0, 0), (0, 1)], constant_values=1.0)

    raw = tracer(width=resolution, height=resolution, scene=uniforms, quadrics=quadrics, colors=cols)
    rgb = np.clip(raw[..., :3] * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return rgb


def vortex_frames(world: Quadric, colors: list[str], resolution: int, supersample: int = 2, mode: float = 0.0) -> list[np.ndarray]:
    """One frame per row of world quadrics: a pixel takes a quadric's colour where quadrics(pixels) & pixels < 0.
    Accelerated via WebGPU compute with in-shader supersampling, falling back to CPU if unavailable."""
    rgb = np.array([to_rgb(color) for color in colors], dtype=np.float32)
    try:
        return [render_gpu_frame(quadrics, rgb, resolution, mode=mode) for quadrics in world]
    except Exception:
        # CPU fallback
        coordinates, inside, rim = hemisphere(resolution * supersample)
        pixels = point(coordinates)
        return [paint(((quadrics[:, None](pixels) & pixels) < 0.0), rgb, inside, rim, supersample) for quadrics in world]

