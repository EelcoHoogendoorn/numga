"""GPU-accelerated CSG raytracing on Metal/WebGPU: quadric intersections, CSG intervals, and recursive reflections."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga.algebras import PGA3D as ga
from examples.raytrace import WGPURaytracer
from examples.quadrics.csg_raytracer.core import Plane, Point, Quadric

SHADER_SOURCE = """
struct Uniforms {
    hyp: Quadric,
    hyp_top: Plane,
    hyp_bot: Plane,
    bowl_out: Quadric,
    bowl_in: Quadric,
    bowl_clip: Plane,
    mirror: Quadric,
    floor: Plane,
    lamp: Plane,
    toward_viewer: Plane,
    cam_pos: Point,
    cam_right: Point,
    cam_up: Point,
    cam_dir: Point,
    resolution: vec2<f32>,
};

@group(0) @binding(0) var<uniform> scene: Uniforms;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;

struct HitRecord {
    t: f32,
    normal: Plane,
    color: vec3<f32>,
    material: u32,
};

fn trace_scene(ro: Point, rd: Point, is_secondary: bool) -> HitRecord {
    var rec: HitRecord;
    rec.t = 1e9;
    rec.color = vec3<f32>(0.96, 0.96, 0.95);
    rec.material = 0u;

    // 1. Ruled Hyperboloid of One Sheet:
    let a_h: f32 = scene.hyp(rd) & rd;
    let b_h: f32 = scene.hyp(rd) & ro;
    let c_h: f32 = scene.hyp(ro) & ro;
    let disc_h: f32 = b_h * b_h - a_h * c_h;

    if (disc_h >= 0.0) {
        let s_root = sqrt(disc_h);
        let t1 = (-b_h - s_root) / a_h;
        let t2 = (-b_h + s_root) / a_h;
        let near_t = min(t1, t2);
        let far_t = max(t1, t2);

        if (near_t > 0.001 && near_t < rec.t) {
            let p: Point = ro + rd * near_t;
            if ((scene.hyp_top & p) <= 0.0 && (scene.hyp_bot & p) <= 0.0) {
                rec.t = near_t;
                rec.normal = scene.hyp(p);
                let strip: f32 = sin(p.x * 25.0 + p.z * 18.0) * sin(p.y * 25.0 - p.z * 18.0);
                var copper = vec3<f32>(0.84, 0.48, 0.24);
                if (strip > 0.4) { copper = copper * 0.75; }
                rec.color = copper;
                rec.material = 0u;
            }
        }
        if (far_t > 0.001 && far_t < rec.t) {
            let p: Point = ro + rd * far_t;
            if ((scene.hyp_top & p) <= 0.0 && (scene.hyp_bot & p) <= 0.0) {
                rec.t = far_t;
                rec.normal = -scene.hyp(p);
                rec.color = vec3<f32>(0.84, 0.48, 0.24) * 0.7;
                rec.material = 0u;
            }
        }
    }

    // 2. Hollow Clipped Bowl (CSG):
    let a_bo: f32 = scene.bowl_out(rd) & rd;
    let b_bo: f32 = scene.bowl_out(rd) & ro;
    let c_bo: f32 = scene.bowl_out(ro) & ro;
    let disc_bo: f32 = b_bo * b_bo - a_bo * c_bo;

    let a_bi: f32 = scene.bowl_in(rd) & rd;
    let b_bi: f32 = scene.bowl_in(rd) & ro;
    let c_bi: f32 = scene.bowl_in(ro) & ro;
    let disc_bi: f32 = b_bi * b_bi - a_bi * c_bi;

    if (disc_bo >= 0.0 && a_bo > 0.0) {
        let s_out = sqrt(disc_bo);
        let t_o1 = (-b_bo - s_out) / a_bo;
        let t_o2 = (-b_bo + s_out) / a_bo;

        var t_i1: f32 = -1e9;
        var t_i2: f32 = -1e9;
        var has_inner: bool = false;
        if (disc_bi >= 0.0 && a_bi > 0.0) {
            let s_in = sqrt(disc_bi);
            t_i1 = (-b_bi - s_in) / a_bi;
            t_i2 = (-b_bi + s_in) / a_bi;
            has_inner = true;
        }

        if (t_o1 > 0.001 && t_o1 < rec.t) {
            let p: Point = ro + rd * t_o1;
            let inside_inner: bool = has_inner && (t_o1 > t_i1 && t_o1 < t_i2);
            if (!inside_inner && (scene.bowl_clip & p) <= 0.0) {
                rec.t = t_o1;
                rec.normal = scene.bowl_out(p);
                rec.color = vec3<f32>(0.15, 0.45, 0.55);
                rec.material = 0u;
            }
        }

        if (has_inner && t_i2 > 0.001 && t_i2 < rec.t) {
            let p: Point = ro + rd * t_i2;
            let inside_outer: bool = (t_i2 >= t_o1 && t_i2 <= t_o2);
            if (inside_outer && (scene.bowl_clip & p) <= 0.0) {
                rec.t = t_i2;
                rec.normal = -scene.bowl_in(p);
                rec.color = vec3<f32>(0.92, 0.72, 0.22);
                rec.material = 0u;
            }
        }

        let denom_clip: f32 = scene.bowl_clip & rd;
        if (abs(denom_clip) > 1e-5) {
            let t_cut = -(scene.bowl_clip & ro) / denom_clip;
            if (t_cut > 0.001 && t_cut < rec.t) {
                let p_cut: Point = ro + rd * t_cut;
                let on_solid: bool = (t_cut >= t_o1 && t_cut <= t_o2) && (!has_inner || t_cut <= t_i1 || t_cut >= t_i2);
                if (on_solid) {
                    rec.t = t_cut;
                    rec.normal = scene.bowl_clip;
                    rec.color = vec3<f32>(0.85, 0.85, 0.88);
                    rec.material = 0u;
                }
            }
        }
    }

    // 3. Mirror Sphere:
    let a_m: f32 = scene.mirror(rd) & rd;
    let b_m: f32 = scene.mirror(rd) & ro;
    let c_m: f32 = scene.mirror(ro) & ro;
    let disc_m: f32 = b_m * b_m - a_m * c_m;

    if (disc_m >= 0.0 && a_m > 0.0) {
        let t_m = (-b_m - sqrt(disc_m)) / a_m;
        if (t_m > 0.001 && t_m < rec.t) {
            rec.t = t_m;
            let p_m: Point = ro + rd * t_m;
            rec.normal = scene.mirror(p_m);
            rec.color = vec3<f32>(0.95, 0.95, 0.96);
            rec.material = 1u;
        }
    }

    // 4. Floor Plane:
    let denom_f: f32 = scene.floor & rd;
    if (abs(denom_f) > 1e-5) {
        let t_f = -(scene.floor & ro) / denom_f;
        if (t_f > 0.001 && t_f < rec.t) {
            rec.t = t_f;
            let p_f: Point = ro + rd * t_f;
            rec.normal = scene.floor;
            let tile: f32 = (floor(p_f.x * 1.4) + floor(p_f.y * 1.4)) % 2.0;
            var f_col = vec3<f32>(0.88, 0.87, 0.85);
            if (abs(tile) > 0.5) { f_col = vec3<f32>(0.75, 0.74, 0.72); }
            rec.color = f_col;
            rec.material = 0u;
        }
    }

    return rec;
}

fn shadow_test(hit_pt: Point, lamp_dir: Point) -> f32 {
    let s_rec: HitRecord = trace_scene(hit_pt, lamp_dir, true);
    if (s_rec.t < 1e8) {
        return 0.28;
    }
    return 1.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let px = id.x;
    let py = id.y;
    let width = u32(scene.resolution.x);
    let height = u32(scene.resolution.y);
    if (px >= width || py >= height) { return; }

    let uv = vec2<f32>(
        (f32(px) + 0.5) / f32(width) * 2.0 - 1.0,
        1.0 - (f32(py) + 0.5) / f32(height) * 2.0
    );
    let aspect = scene.resolution.x / scene.resolution.y;
    let origin: Point = scene.cam_pos;
    let ray_dir: Point = scene.cam_dir + scene.cam_right * (uv.x * aspect * 0.6) + scene.cam_up * (uv.y * 0.6);

    let hit: HitRecord = trace_scene(origin, ray_dir, false);
    var final_color = hit.color;

    if (hit.t < 1e8) {
        let hit_pos: Point = origin + ray_dir * hit.t;
        let n_len: f32 = sqrt(hit.normal | hit.normal);
        let n_unit: Plane = hit.normal * (1.0 / n_len);
        let lamp_pt: Point = Point(scene.lamp.x, scene.lamp.y, scene.lamp.z, 0.0);

        if (hit.material == 1u) {
            let refl_plane: Plane = n_unit * (2.0 * (n_unit | scene.lamp)) - scene.lamp;
            let refl_dir: Point = ray_dir - Point(n_unit.x, n_unit.y, n_unit.z, 0.0) * (2.0 * (n_unit.x * ray_dir.x + n_unit.y * ray_dir.y + n_unit.z * ray_dir.z));
            let sec_origin: Point = hit_pos + Point(n_unit.x, n_unit.y, n_unit.z, 0.0) * 0.005;
            let sec_hit: HitRecord = trace_scene(sec_origin, refl_dir, true);

            var reflected_color = sec_hit.color;
            if (sec_hit.t < 1e8) {
                let sec_pos: Point = sec_origin + refl_dir * sec_hit.t;
                let s_n_len: f32 = sqrt(sec_hit.normal | sec_hit.normal);
                let s_n_unit: Plane = sec_hit.normal * (1.0 / s_n_len);
                let s_diff: f32 = max(0.0, s_n_unit | scene.lamp);
                let s_shadow: f32 = shadow_test(sec_pos + Point(s_n_unit.x, s_n_unit.y, s_n_unit.z, 0.0) * 0.005, lamp_pt);
                reflected_color = sec_hit.color * (0.25 + 0.75 * s_diff * s_shadow);
            }
            let spec: f32 = pow(max(0.0, refl_plane | scene.toward_viewer), 60.0);
            final_color = reflected_color * 0.88 + vec3<f32>(spec * 0.4);
        } else {
            let shadow: f32 = shadow_test(hit_pos + Point(n_unit.x, n_unit.y, n_unit.z, 0.0) * 0.005, lamp_pt);
            let diff: f32 = max(0.0, n_unit | scene.lamp);
            let refl_plane: Plane = n_unit * (2.0 * diff) - scene.lamp;
            let spec: f32 = pow(max(0.0, refl_plane | scene.toward_viewer), 35.0);

            final_color = hit.color * (0.28 + 0.72 * diff * shadow) + vec3<f32>(0.3 * spec * shadow);
        }
    }

    output[py * width + px] = vec4<f32>(final_color, 1.0);
}
"""

_tracer: WGPURaytracer | None = None

def get_tracer() -> WGPURaytracer:
    """Lazily compile and cache the CSG raytracer pipeline."""
    global _tracer
    if _tracer is None:
        _tracer = WGPURaytracer(
            SHADER_SOURCE,
            ga,
            {"Point": Point, "Plane": Plane, "Quadric": Quadric}
        )
    return _tracer


# --- plumbing: JIT array extraction at the GPU render boundary ------------------------
def render_frame(scene_objects: dict, cam_pos: np.ndarray, cam_target: np.ndarray,
                 width: int = 640, height: int = 640) -> np.ndarray:
    """Render a single frame of the CSG scene by packing uniform buffers and dispatching to GPU."""
    tracer = get_tracer()

    forward = cam_target - cam_pos[:3]
    forward /= np.linalg.norm(forward)
    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, up_world)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    toward_viewer = np.array([-forward[0], -forward[1], -forward[2], 0.0], dtype=np.float32)
    cam_right = np.array([*right, 0.0], dtype=np.float32)
    cam_up = np.array([*up, 0.0], dtype=np.float32)
    cam_forward = np.array([*forward, 0.0], dtype=np.float32)

    # Array extraction happens strictly JIT here at the hardware uniform buffer boundary:
    uniform_data = np.concatenate([
        scene_objects["hyp"].kernel.T.flatten(),
        scene_objects["hyp_top"].kernel,
        scene_objects["hyp_bot"].kernel,
        scene_objects["bowl_out"].kernel.T.flatten(),
        scene_objects["bowl_in"].kernel.T.flatten(),
        scene_objects["bowl_clip"].kernel,
        scene_objects["mirror"].kernel.T.flatten(),
        scene_objects["floor"].kernel if hasattr(scene_objects["floor"], "kernel") else scene_objects["floor"],
        scene_objects["lamp"].kernel if hasattr(scene_objects["lamp"], "kernel") else scene_objects["lamp"],
        toward_viewer,
        cam_pos,
        cam_right,
        cam_up,
        cam_forward,
        np.array([width, height, 0.0, 0.0], dtype=np.float32)
    ]).astype(np.float32)

    raw = tracer.render(uniform_data, width, height)
    return np.clip(raw[..., :3], 0.0, 1.0)


def draw_scene(scene_objects: dict, cam_pos: np.ndarray, cam_target: np.ndarray,
               width: int = 640, height: int = 640) -> plt.Figure:
    """Draw the CSG scene and return a Matplotlib figure."""
    image = render_frame(scene_objects, cam_pos, cam_target, width, height)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    ax.imshow(image)
    ax.axis("off")
    fig.tight_layout(pad=0.0)
    return fig


def frames(scene_objects: dict, cam_positions: np.ndarray, cam_target: np.ndarray,
           width: int = 480, height: int = 480) -> list[np.ndarray]:
    """Render a sequence of animation frames along an orbit trajectory."""
    return [(render_frame(scene_objects, pos, cam_target, width, height) * 255).astype(np.uint8)
            for pos in cam_positions]
