"""Headlight images and WebGPU compute raytracing of Dupin cyclides on the 3-sphere."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.quadrics.cyclides.core import (
    Direction, Point, Quadric, antipode, mv, origin, ray_linear,
)
from examples.raytrace import WGPUCompute

BACKGROUND = np.array([0.013, 0.02, 0.036])
TEAL = np.array([0.055, 0.42, 0.39])
GOLD = np.array([0.6, 0.42, 0.08])
# One colour per body, in turn.
PALETTE = np.array([TEAL, [0.65, 0.38, 0.08], [0.35, 0.13, 0.40], [0.12, 0.35, 0.60]])

CYCLIDES_WGSL = """
struct Uniforms {
    resolution: vec2<f32>,
    fov: f32,
    n_surfaces: f32,
    both_sides: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    bg_color: vec4<f32>,
};

struct SurfaceParams {
    v1: vec3<f32>, c0: f32,
    v3: vec3<f32>, c4: f32,
    m2_row0: vec3<f32>, k2: f32,
    m2_row1: vec3<f32>, _pad0: f32,
    m2_row2: vec3<f32>, _pad1: f32,
};

@group(0) @binding(0) var<uniform> scene: Uniforms;
@group(0) @binding(1) var<storage, read> surface_params: array<SurfaceParams>;
@group(0) @binding(2) var<storage, read> surface_matrices: array<f32>;
@group(0) @binding(3) var<storage, read> ray_basis: array<f32>;
@group(0) @binding(4) var<storage, read> surface_colors: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;

fn cbrt_val(x: f32) -> f32 {
    if (x < 0.0) {
        return -pow(-x, 1.0 / 3.0);
    }
    return pow(x, 1.0 / 3.0);
}

fn solve_quartic_angle(c0: f32, c1: f32, c2: f32, c3: f32, c4: f32) -> f32 {
    let inv_c0 = 1.0 / c0;
    let a3 = c1 * inv_c0;
    let a2 = c2 * inv_c0;
    let a1 = c3 * inv_c0;
    let a0 = c4 * inv_c0;

    let a3_sq = a3 * a3;
    let p = a2 - 0.375 * a3_sq;
    let q = a1 - 0.5 * a3 * a2 + 0.125 * a3 * a3_sq;
    let s = a0 - 0.25 * a3 * a1 + 0.0625 * a3_sq * a2 - 0.01171875 * a3_sq * a3_sq;

    var min_t = 1e30;
    let shift = -0.25 * a3;

    if (abs(q) < 1e-6) {
        let disc = p * p - 4.0 * s;
        if (disc >= 0.0) {
            let sq_disc = sqrt(disc);
            let y2_1 = 0.5 * (-p + sq_disc);
            let y2_2 = 0.5 * (-p - sq_disc);
            if (y2_1 >= 0.0) {
                let y = sqrt(y2_1);
                for (var sgn = -1.0; sgn <= 1.0; sgn = sgn + 2.0) {
                    let r = sgn * y + shift;
                    var t = 2.0 * atan2(1.0, r);
                    if (t < 0.0) { t = t + 6.28318530718; }
                    if (t > 1e-4 && t < min_t) { min_t = t; }
                }
            }
            if (y2_2 >= 0.0) {
                let y = sqrt(y2_2);
                for (var sgn = -1.0; sgn <= 1.0; sgn = sgn + 2.0) {
                    let r = sgn * y + shift;
                    var t = 2.0 * atan2(1.0, r);
                    if (t < 0.0) { t = t + 6.28318530718; }
                    if (t > 1e-4 && t < min_t) { min_t = t; }
                }
            }
        }
        return min_t;
    }

    let A = 2.0 * p;
    let B = p * p - 4.0 * s;
    let C = -q * q;

    let A_sq = A * A;
    let P = B - A_sq / 3.0;
    let Q = C - A * B / 3.0 + 2.0 * A * A_sq / 27.0;

    let half_Q = 0.5 * Q;
    let delta = half_Q * half_Q + (P * P * P) / 27.0;

    var z = 0.0;
    if (delta > 0.0) {
        let sq_delta = sqrt(delta);
        let u1 = cbrt_val(-half_Q + sq_delta);
        let u2 = cbrt_val(-half_Q - sq_delta);
        z = (u1 + u2) - A / 3.0;
    } else {
        let r_trig = sqrt(-P / 3.0);
        let cos_val = clamp(-half_Q / (r_trig * r_trig * r_trig), -1.0, 1.0);
        let phi = acos(cos_val) / 3.0;
        let z1 = 2.0 * r_trig * cos(phi) - A / 3.0;
        let z2 = 2.0 * r_trig * cos(phi + 2.09439510239) - A / 3.0;
        let z3 = 2.0 * r_trig * cos(phi + 4.18879020479) - A / 3.0;
        z = max(z1, max(z2, z3));
    }

    if (z <= 0.0) {
        return min_t;
    }

    let g = sqrt(z);
    let term = (p + z);
    let q_over_g = q / g;

    let disc1 = g * g - 2.0 * (term - q_over_g);
    if (disc1 >= 0.0) {
        let sq1 = sqrt(disc1);
        let y1 = 0.5 * (-g - sq1);
        let y2 = 0.5 * (-g + sq1);
        
        var t1 = 2.0 * atan2(1.0, y1 + shift);
        if (t1 < 0.0) { t1 = t1 + 6.28318530718; }
        if (t1 > 1e-4 && t1 < min_t) { min_t = t1; }

        var t2 = 2.0 * atan2(1.0, y2 + shift);
        if (t2 < 0.0) { t2 = t2 + 6.28318530718; }
        if (t2 > 1e-4 && t2 < min_t) { min_t = t2; }
    }

    let disc2 = g * g - 2.0 * (term + q_over_g);
    if (disc2 >= 0.0) {
        let sq2 = sqrt(disc2);
        let y3 = 0.5 * (g - sq2);
        let y4 = 0.5 * (g + sq2);

        var t3 = 2.0 * atan2(1.0, y3 + shift);
        if (t3 < 0.0) { t3 = t3 + 6.28318530718; }
        if (t3 > 1e-4 && t3 < min_t) { min_t = t3; }

        var t4 = 2.0 * atan2(1.0, y4 + shift);
        if (t4 < 0.0) { t4 = t4 + 6.28318530718; }
        if (t4 > 1e-4 && t4 < min_t) { min_t = t4; }
    }

    return min_t;
}

fn trace_ray(d: vec3<f32>) -> vec3<f32> {
    var nearest_t = 1e30;
    var nearest_idx = 0u;
    let n_surf = u32(scene.n_surfaces);

    for (var k = 0u; k < n_surf; k = k + 1u) {
        let p = surface_params[k];
        let c0 = p.c0;
        let c4 = p.c4;
        let c1 = dot(p.v1, d);
        let c3 = dot(p.v3, d);
        let m2_d = vec3<f32>(
            dot(p.m2_row0, d),
            dot(p.m2_row1, d),
            dot(p.m2_row2, d)
        );
        let c2 = dot(d, m2_d) + p.k2;

        let t = solve_quartic_angle(c0, c1, c2, c3, c4);
        if (t < nearest_t) {
            nearest_t = t;
            nearest_idx = k;
        }
    }

    if (nearest_t >= 1e20) {
        return pow(scene.bg_color.xyz, vec3<f32>(1.0 / 2.2));
    }

    let t = nearest_t;
    let sin_t = sin(t);
    let cos_t = cos(t);
    let omc = 1.0 - cos_t;

    var hit = array<f32, 5>();
    var vel = array<f32, 5>();

    for (var i = 0u; i < 5u; i = i + 1u) {
        let l_val = d.x * ray_basis[10u + i] + d.y * ray_basis[15u + i] + d.z * ray_basis[20u + i];
        let orig = ray_basis[i];
        let q_val = ray_basis[5u + i];
        hit[i] = orig + l_val * sin_t + q_val * omc;
        vel[i] = l_val * cos_t + q_val * sin_t;
    }

    let k = nearest_idx;
    var polar = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
    for (var r = 0u; r < 5u; r = r + 1u) {
        var sum = 0.0;
        for (var c = 0u; c < 5u; c = c + 1u) {
            sum = sum + surface_matrices[k * 25u + r * 5u + c] * hit[c];
        }
        polar[r] = sum;
    }

    var polar_dot_vel = 0.0;
    for (var r = 0u; r < 5u; r = r + 1u) {
        for (var c = 0u; c < 5u; c = c + 1u) {
            polar_dot_vel = polar_dot_vel + polar[r] * ray_basis[25u + r * 5u + c] * vel[c];
        }
    }

    let polar_sq = polar[0] * polar[0] + polar[1] * polar[1] + polar[2] * polar[2] + polar[3] * polar[3] - polar[4] * polar[4];
    let polar_norm = sqrt(max(1e-12, abs(polar_sq)));
    var facing = -polar_dot_vel / polar_norm;
    if (scene.both_sides > 0.5) {
        facing = abs(facing);
    }

    let col = surface_colors[k].xyz;
    let f_clip = clamp(facing, 0.0, 1.0);
    let lit = col * (0.2 + 0.8 * f_clip) + vec3<f32>(0.12 * pow(f_clip, 16.0));
    return pow(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
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

    let tan_half = tan(scene.fov * 0.5);

    for (var s = 0u; s < 4u; s = s + 1u) {
        let sx = (f32(px) + 0.5 + sub_offsets[s].x) / f32(w);
        let sy = (f32(py) + 0.5 + sub_offsets[s].y) / f32(h);
        
        let u = 2.0 * sx - 1.0;
        let v = (1.0 - 2.0 * sy) * f32(h) / f32(w);
        let d = normalize(vec3<f32>(-1.0, u * tan_half, v * tan_half));

        acc = acc + trace_ray(d);
    }

    output[py * w + px] = vec4<f32>(acc * 0.25, 1.0);
}
"""

_tracer: WGPUCompute | None = None
_ray_basis_data: np.ndarray | None = None


def get_tracer() -> WGPUCompute:
    global _tracer
    if _tracer is None:
        _tracer = WGPUCompute(CYCLIDES_WGSL, algebra=ga)
    return _tracer


def get_ray_basis_data() -> np.ndarray:
    global _ray_basis_data
    if _ray_basis_data is None:
        d_basis = mv(Direction, np.eye(3))
        L = ray_linear(d_basis)
        q0 = (antipode - origin) * 0.5
        v_basis = stack([mv.x, mv.y, mv.z, mv.w, mv.e])
        av_basis = mv(ga.gatype.antivector(), np.eye(5))
        pairing = (v_basis[:, None] & av_basis[None, :]).to_array()

        orig_arr = np.asarray(origin.kernel, dtype=np.float32)
        q0_arr = np.asarray(q0.kernel, dtype=np.float32)
        L_arr = np.asarray(L.kernel, dtype=np.float32).reshape(15)
        pair_arr = np.asarray(pairing, dtype=np.float32).reshape(25)

        _ray_basis_data = np.concatenate([orig_arr, q0_arr, L_arr, pair_arr]).astype(np.float32)
    return _ray_basis_data


def prepare_surface_data(surfaces: Quadric) -> tuple[np.ndarray, np.ndarray]:
    """Prepare GPU surface params and 5x5 matrices at the visualization boundary."""
    if getattr(surfaces, "shape", ()) == ():
        surfaces = stack([surfaces])

    n_surfaces = surfaces.shape[0]
    form = Point & surfaces

    d_basis = mv(Direction, np.eye(3))
    L = ray_linear(d_basis)

    c0 = form(origin, origin).to_array().reshape(n_surfaces)
    c4 = form(antipode, antipode).to_array().reshape(n_surfaces)
    k2 = (2 * form(origin, antipode)).to_array().reshape(n_surfaces)
    v1 = (4 * form[:, None](origin, L)).to_array().reshape(n_surfaces, 3)
    v3 = (4 * form[:, None](antipode, L)).to_array().reshape(n_surfaces, 3)
    M2 = (4 * form[:, None, None](L[:, None], L[None, :])).to_array().reshape(n_surfaces, 3, 3)


    params = np.zeros((n_surfaces, 20), dtype=np.float32)
    for i in range(n_surfaces):
        params[i, 0:3] = v1[i]
        params[i, 3] = c0[i]
        params[i, 4:7] = v3[i]
        params[i, 7] = c4[i]
        params[i, 8:11] = M2[i, 0]
        params[i, 11] = k2[i]
        params[i, 12:15] = M2[i, 1]
        params[i, 15] = 0.0
        params[i, 16:19] = M2[i, 2]
        params[i, 19] = 0.0

    matrices = np.asarray(surfaces.kernel, dtype=np.float32).reshape(n_surfaces, 25)
    return params, matrices


def render_gpu_frame(
    surfaces: Quadric,
    shape: tuple[int, int] = (180, 240),
    fov: float = np.radians(90),
    colors: np.ndarray | None = None,
    both_sides: bool = False,
) -> np.ndarray:
    """Raytrace quadrics on S³ in 5D CGA Cl(4, 1) using WebGPU compute and in-shader 2x2 supersampling."""
    height, width = shape
    if getattr(surfaces, "shape", ()) == ():
        surfaces = stack([surfaces])
    n_surfaces = surfaces.shape[0]

    if colors is None:
        colors = PALETTE[:n_surfaces]
    cols = np.asarray(colors, dtype=np.float32)
    if cols.ndim == 1:
        cols = cols[None, :]
    if cols.shape[0] < n_surfaces:
        cols = np.pad(cols, [(0, n_surfaces - cols.shape[0]), (0, 0)], mode="wrap")
    if cols.shape[-1] == 3:
        cols = np.pad(cols, [(0, 0), (0, 1)], constant_values=1.0)

    params, matrices = prepare_surface_data(surfaces)
    ray_basis = get_ray_basis_data()

    uniforms = {
        "resolution": np.array([width, height], dtype=np.float32),
        "fov": float(fov),
        "n_surfaces": float(n_surfaces),
        "both_sides": 1.0 if both_sides else 0.0,
        "_pad0": 0.0,
        "_pad1": 0.0,
        "_pad2": 0.0,
        "bg_color": np.append(BACKGROUND, 1.0).astype(np.float32),
    }

    tracer = get_tracer()
    raw = tracer(
        width=width,
        height=height,
        scene=uniforms,
        surface_params=params,
        surface_matrices=matrices,
        ray_basis=ray_basis,
        surface_colors=cols,
    )
    return np.clip(raw[..., :3] * 255.0, 0.0, 255.0).round().astype(np.uint8)


def render_surfaces_frames(
    surfaces_seq: Iterable[Quadric],
    shape: tuple[int, int] = (180, 240),
    fov: float = np.radians(90),
    colors: np.ndarray | None = None,
    both_sides: bool = False,
) -> Iterator[np.ndarray]:
    """Render a sequence of Quadric surfaces via WebGPU compute with in-shader supersampling."""
    for s in surfaces_seq:
        yield render_gpu_frame(s, shape=shape, fov=fov, colors=colors, both_sides=both_sides)


def headlight(facing: np.ndarray, visible: np.ndarray, color: np.ndarray) -> np.ndarray:

    """Surface colour lit from the eye: facing is the cosine between the ray and the normal."""
    lit = color * (0.2 + 0.8 * np.clip(facing, 0, 1))[..., None] + 0.12 * np.clip(facing, 0, 1)[..., None] ** 16
    return np.where(visible[..., None], np.clip(lit, 0, 1) ** (1 / 2.2), BACKGROUND ** (1 / 2.2))


def facing_frames(traced: Iterable, shape: tuple[int, int]) -> Iterator[np.ndarray]:
    """Headlight images as RGB frames, one per traced (facing, angle) pair of one surface, as they are consumed."""
    for facing, angle in traced:
        rgb = headlight(facing.to_array(), np.isfinite(angle), TEAL).reshape(*shape, 3)
        yield (rgb * 255).round().astype(np.uint8)


def scene_frames(traced: Iterable, colors: np.ndarray, shape: tuple[int, int]) -> Iterator[np.ndarray]:
    """RGB frames of several surfaces, one per traced (facing, angle) pair over the bodies: each pixel shows the
    nearest hit among them, lit in that body's colour."""
    for facing, angle in traced:
        nearest = angle.argmin(axis=0)
        pixels = np.arange(angle.shape[1])
        facing = facing.to_array()[nearest, pixels]
        rgb = headlight(facing, np.isfinite(angle[nearest, pixels]), colors[nearest % len(colors)]).reshape(*shape, 3)
        yield (rgb * 255).round().astype(np.uint8)


def draw_facing(facing, angle: np.ndarray, shape: tuple[int, int]) -> plt.Figure:
    """Headlight images, one panel per surface along the leading axis of a Scalar facing; angle is inf on a miss."""
    rgb = headlight(facing.to_array(), np.isfinite(angle), TEAL).reshape(-1, *shape, 3)
    figure, axes = plt.subplots(1, len(rgb), figsize=(4.5 * len(rgb), 3.6), squeeze=False)
    for ax, image in zip(axes[0], rgb):
        ax.imshow(image)
        ax.axis("off")
    figure.tight_layout()
    return figure


def draw_images(images: list[np.ndarray], columns: int) -> plt.Figure:
    """RGB images in a grid of the given number of columns."""
    rows = -(-len(images) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.0 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image in zip(axes.flat, images):
        ax.imshow(image)
    figure.tight_layout()
    return figure

