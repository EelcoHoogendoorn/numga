"""Drawing for the XPBD chain: every pixel's ray meets the nearest link, a solid ellipsoid placed by its motor,
over a floor that catches the links' shadows."""

from __future__ import annotations

import numpy as np

from examples import raytrace as rt
from examples.mechanics.xpbd.core import Motor

# Two alternating link colours, the pin's colour, the floor's two tiles and the background.
LINK_COLOURS = np.array([[0.86, 0.42, 0.20], [0.20, 0.52, 0.66]])
PIN_COLOUR = np.array([0.25, 0.25, 0.28])
FLOOR_COLOURS = np.array([[0.86, 0.85, 0.82], [0.76, 0.75, 0.72]])
BACKGROUND = np.array([0.97, 0.97, 0.96])
# The darker tone of a link's alternate octants, which shows it turning about its own axes.
OCTANT_CONTRAST = 0.22
# The view: its angles and its margin around the whole swing and its shadows, as a fraction of the larger side;
# the floor's depth below the swing's lowest point and its reach beyond the swing, and a floor tile's side,
# in link spacings; the pin's radius as a fraction of the spacing, and the pixels across.
ELEVATION, AZIMUTH = 14, -58
MARGIN = 0.08
FLOOR_DROP, FLOOR_REACH, TILE = 1.0, 12.0, 3.0
PIN_RADIUS = 0.18
PIXELS = 360


from numga import Extensor, stack
from numga.algebras import PGA3D as ga

Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Quadric = ga.gatype((Plane, Point))
MotorMap = ga.gatype((Point, Point))

CHAIN_RAYTRACE_WGSL = """
struct Uniforms {
    cam_pos: Point,
    cam_right: Point,
    cam_up: Point,
    cam_dir: Point,
    lamp_dir: Point,
    lamp_plane: Plane,
    toward_viewer: Plane,
    floor_plane: Plane,
    plane_x: Plane,
    plane_y: Plane,
    plane_z: Plane,
    plane_w: Plane,
    half_size: vec2<f32>,
    resolution: vec2<f32>,
    floor_bounds_low: vec2<f32>,
    floor_bounds_high: vec2<f32>,
    floor_z: f32,
    tile_size: f32,
    has_floor: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> scene: Uniforms;
@group(0) @binding(1) var<storage, read> bodies: array<Quadric>;
@group(0) @binding(2) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> inv_motors: array<MotorMap>;
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;

fn is_shadowed(pt: Point, lamp_pt: Point, ignore_idx: u32) -> f32 {
    let s_ro: Point = pt + lamp_pt * 0.003;
    let s_rd: Point = lamp_pt;
    let n_bodies = arrayLength(&bodies);
    for (var j = 0u; j < n_bodies; j = j + 1u) {
        if (j == ignore_idx) { continue; }
        let sq: Quadric = bodies[j];
        let sa: f32 = sq(s_rd) & s_rd;
        let sb: f32 = sq(s_rd) & s_ro;
        let sc: f32 = sq(s_ro) & s_ro;
        let sdisc: f32 = sb * sb - sa * sc;
        if (sdisc >= 0.0 && sa > 0.0) {
            let st = (-sb - sqrt(sdisc)) / sa;
            if (st > 0.001) {
                return 0.25;
            }
        }
    }
    return 1.0;
}

fn trace_pixel(uv: vec2<f32>) -> vec3<f32> {
    // Parallel camera rays:
    let ro: Point = scene.cam_pos + scene.cam_right * (uv.x * scene.half_size.x) + scene.cam_up * (uv.y * scene.half_size.y);
    let rd: Point = scene.cam_dir;

    var best_t: f32 = 1e9;
    var best_idx: u32 = 0u;

    // 1. Raytrace Quadrics (pin + links) in GA:
    let n_bodies = arrayLength(&bodies);
    for (var i = 0u; i < n_bodies; i = i + 1u) {
        let q: Quadric = bodies[i];
        let a: f32 = q(rd) & rd;
        let b: f32 = q(rd) & ro;
        let c: f32 = q(ro) & ro;
        let disc: f32 = b * b - a * c;
        if (disc >= 0.0 && a > 0.0) {
            let t = (-b - sqrt(disc)) / a;
            if (t > 0.001 && t < best_t) {
                best_t = t;
                best_idx = i;
            }
        }
    }

    // 2. Raytrace Floor Plane in GA:
    var t_floor: f32 = 1e9;
    var on_floor: bool = false;
    if (scene.has_floor > 0.5) {
        let denom: f32 = scene.floor_plane & rd;
        if (abs(denom) > 1e-5) {
            let tf = -(scene.floor_plane & ro) / denom;
            if (tf > 0.001) {
                let f_pt: Point = ro + rd * tf;
                let f_w: f32 = scene.plane_w & f_pt;
                let f_x: f32 = (scene.plane_x & f_pt) / f_w;
                let f_y: f32 = (scene.plane_y & f_pt) / f_w;
                if (f_x >= scene.floor_bounds_low.x && f_x <= scene.floor_bounds_high.x &&
                    f_y >= scene.floor_bounds_low.y && f_y <= scene.floor_bounds_high.y) {
                    t_floor = tf;
                    on_floor = true;
                }
            }
        }
    }

    var col = vec3<f32>(0.97, 0.97, 0.96); // Background

    if (best_t < t_floor && best_t < 1e8) {
        // Hit a quadric:
        let hit: Point = ro + rd * best_t;
        let norm_plane: Plane = bodies[best_idx](hit);
        let n_unit: Plane = norm_plane * (1.0 / sqrt(max(norm_plane | norm_plane, 1e-12)));

        let lit: f32 = is_shadowed(hit, scene.lamp_dir, best_idx);
        let diffuse: f32 = max(0.0, n_unit | scene.lamp_plane) * lit;
        let refl: Plane = n_unit * (2.0 * (n_unit | scene.lamp_plane)) - scene.lamp_plane;
        let spec: f32 = pow(max(0.0, refl | scene.toward_viewer), 40.0) * f32(diffuse > 0.0);

        // Octant contrast in local body frame:
        let p_local: Point = inv_motors[best_idx](hit);
        let local_x: f32 = scene.plane_x & p_local;
        let local_y: f32 = scene.plane_y & p_local;
        let local_z: f32 = scene.plane_z & p_local;
        let oct_count: u32 = u32(local_x > 0.0) + u32(local_y > 0.0) + u32(local_z > 0.0);
        let oct: f32 = f32(oct_count % 2u);
        let oct_contrast: f32 = colors[best_idx].w;
        let base_col: vec3<f32> = colors[best_idx].xyz * (1.0 - oct_contrast * oct);

        col = base_col * (0.3 + 0.7 * diffuse) + vec3<f32>(0.3 * spec);

    } else if (on_floor && t_floor < 1e8) {
        // Hit the checkered floor:
        let floor_hit: Point = ro + rd * t_floor;
        let lit: f32 = is_shadowed(floor_hit, scene.lamp_dir, 999999u);
        let diffuse: f32 = max(0.0, scene.floor_plane | scene.lamp_plane) * lit;
        let refl: Plane = scene.floor_plane * (2.0 * (scene.floor_plane | scene.lamp_plane)) - scene.lamp_plane;
        let spec: f32 = pow(max(0.0, refl | scene.toward_viewer), 40.0) * f32(diffuse > 0.0);

        let hit_w: f32 = scene.plane_w & floor_hit;
        let hit_x: f32 = (scene.plane_x & floor_hit) / hit_w;
        let hit_y: f32 = (scene.plane_y & floor_hit) / hit_w;
        let tx = i32(floor(hit_x / scene.tile_size));
        let ty = i32(floor(hit_y / scene.tile_size));
        let tile_idx = u32(((tx + ty) % 2 + 2) % 2);
        var tile_col = vec3<f32>(0.86, 0.85, 0.82);
        if (tile_idx == 1u) {
            tile_col = vec3<f32>(0.76, 0.75, 0.72);
        }
        col = tile_col * (0.3 + 0.7 * diffuse) + vec3<f32>(0.3 * spec);
    }

    return col;
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
        acc = acc + trace_pixel(uv);
    }

    output[py * w + px] = vec4<f32>(acc * 0.25, 1.0);
}
"""

_tracer: rt.WGPUCompute | None = None

def _get_tracer() -> rt.WGPUCompute:
    global _tracer
    if _tracer is None:
        _tracer = rt.WGPUCompute(
            CHAIN_RAYTRACE_WGSL,
            algebra=ga,
            types={"Point": Point, "Plane": Plane, "Quadric": Quadric, "MotorMap": MotorMap}
        )
    return _tracer


# --- plumbing -------------------------------------------------------------------------
def frame(links: Motor, shape: rt.Quadric, pin: rt.Quadric, floor: tuple[np.ndarray, np.ndarray], tile: float, view) -> np.ndarray:
    """One RGB image: the links and the pin over the floor, each pixel the nearest hit, shaded by the lamp;
    floor pixels whose path to the lamp passes through a link lie in its shadow."""
    try:
        tracer = _get_tracer()
        elevation, azimuth, middle, half, pixels = view
        right, up, back = rt.view_axes(elevation, azimuth)
        lamp = rt.lamp(elevation, azimuth)

        rows = max(1, int(round(pixels * half[1] / half[0])))
        width, height = pixels, rows

        placed_links = links >> shape(links << rt.Point)
        all_bodies = Extensor.stack([pin, *placed_links])

        n_total = len(all_bodies)
        colors = np.zeros((n_total, 4), dtype=np.float32)
        colors[0, :3] = PIN_COLOUR
        colors[0, 3] = 0.0
        for i in range(len(placed_links)):
            colors[1 + i, :3] = LINK_COLOURS[i % 2]
            colors[1 + i, 3] = OCTANT_CONTRAST

        pin_map = ga.operator.identity(Point)
        link_maps = links << Point
        all_inv_maps = Extensor.stack([pin_map, *link_maps])

        low, high = floor
        floor_z = float(low[2])
        floor_plane = rt.mv.z - rt.mv.w * floor_z

        img = tracer(
            width=width,
            height=height,
            scene=dict(
                cam_pos=rt.mv.point(middle + 20.0 * max(half) * back),
                cam_right=rt.mv.direction(right),
                cam_up=rt.mv.direction(up),
                cam_dir=rt.mv.direction(-back),
                lamp_dir=rt.mv.direction(lamp),
                lamp_plane=rt.mv.vector([*lamp, 0.0]),
                toward_viewer=rt.mv.vector([*back, 0.0]),
                floor_plane=floor_plane,
                plane_x=rt.mv.x,
                plane_y=rt.mv.y,
                plane_z=rt.mv.z,
                plane_w=rt.mv.w,
                half_size=np.array([half[0], half[1]], dtype=np.float32),
                resolution=np.array([width, height], dtype=np.float32),
                floor_bounds_low=low[:2].astype(np.float32),
                floor_bounds_high=high[:2].astype(np.float32),
                floor_z=floor_z,
                tile_size=float(tile),
                has_floor=1.0,
                _pad=0.0,
            ),
            bodies=all_bodies,
            colors=colors,
            inv_motors=all_inv_maps,
        )
        downsampled = rt.downsample(img[..., :3], rt.SUPERSAMPLE)
        return (np.clip(downsampled, 0, 1) * 255).astype(np.uint8)
    except Exception:
        # Fallback to CPU raytracing if WGPU device is unavailable
        origins, heading, lamp = rt.camera(*view)
        rays = origins[..., None]
        placed = links >> shape(links << rt.Point)
        distance, hit = rt.hits(placed, rays, heading)
        nearest = distance.argmin(axis=-1)
        depth = np.take_along_axis(distance, nearest[..., None], axis=-1)[..., 0]
        toward_viewer = -rt.coordinates(heading + rt.mv.w.dual())

        normal = np.take_along_axis(rt.normals(placed, hit), nearest[..., None, None], axis=-2)[..., 0, :]
        local = np.take_along_axis(rt.coordinates(links << hit), nearest[..., None, None], axis=-2)[..., 0, :]
        octant = (local > 0).sum(axis=-1) % 2
        colour = LINK_COLOURS[nearest % 2] * (1 - OCTANT_CONTRAST * octant)[..., None]
        image = np.where(np.isfinite(depth)[..., None], rt.shade(colour, normal, lamp, toward_viewer, 1.0), BACKGROUND)

        pin_distance, pin_hit = rt.hits(pin, origins, heading)
        pin_rgb = rt.shade(PIN_COLOUR, rt.normals(pin, pin_hit), lamp, toward_viewer, 1.0)
        image = np.where((pin_distance < depth)[..., None], pin_rgb, image)
        depth = np.minimum(depth, pin_distance)

        low, high = floor
        start, direction = rt.coordinates(origins), -toward_viewer
        floor_distance = (low[2] - start[..., 2]) / direction[2]
        floor_xyz = start + floor_distance[..., None] * direction
        on_floor = np.all((floor_xyz[..., :2] > low[:2]) & (floor_xyz[..., :2] < high[:2]), axis=-1)
        tiles = (np.floor(floor_xyz[..., 0] / tile) + np.floor(floor_xyz[..., 1] / tile)) % 2
        shadow, _ = rt.hits(placed, rt.mv.point(floor_xyz)[..., None], rt.mv.direction(lamp))
        lit = 1.0 - 0.75 * np.isfinite(shadow).any(axis=-1)
        floor_rgb = rt.shade(FLOOR_COLOURS[tiles.astype(int)], np.array([0.0, 0.0, 1.0]), lamp, toward_viewer, lit)
        image = np.where((on_floor & (floor_distance < depth))[..., None], floor_rgb, image)
        return (np.clip(rt.downsample(image, rt.SUPERSAMPLE), 0, 1) * 255).astype(np.uint8)


def animate_chain(motors: Motor, semi_axes: np.ndarray, spacing: float) -> list[np.ndarray]:
    """One frame per step: the links swinging from the pin, the view framed by the whole swing."""
    centres = rt.coordinates(motors >> rt.mv.w.dual()).reshape(-1, 3)         # [steps + 1 * bodies, 3]
    low, high = centres.min(axis=0), centres.max(axis=0)
    # The floor: a square below the swing, reaching well beyond the frame.
    floor = (np.append(low[:2] - FLOOR_REACH * spacing, low[2] - FLOOR_DROP * spacing), high + FLOOR_REACH * spacing)
    # Frame the swing, each link's whole extent, and its shadow on the floor, cast along the lamp.
    toward_lamp = rt.lamp(ELEVATION, AZIMUTH)
    shadows = centres - toward_lamp * ((centres[:, 2] - floor[0][2]) / toward_lamp[2])[:, None]
    middle, half = rt.framing(np.concatenate([centres, shadows]), ELEVATION, AZIMUTH, MARGIN)
    view = (ELEVATION, AZIMUTH, middle, half + semi_axes.max(), PIXELS * rt.SUPERSAMPLE)
    # The pin sits at the joint of the fixed body and the first link.
    pin = rt.quadric(np.eye(3) / (PIN_RADIUS * spacing) ** 2, centres[0] + [spacing / 2, 0, 0])
    shape = rt.ellipsoid(semi_axes)
    return [frame(links, shape, pin, floor, TILE * spacing, view) for links in motors[:, 1:]]
