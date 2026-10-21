"""Rasterising triangle meshes into images, as a renderer's shader sees them: for every pixel, the nearest
triangle covering it, the pixel centre's barycentric weights in that triangle, and its depth. Anything
given at the corners of the triangles, a colour, a normal, a phase, interpolates per pixel from these."""

from __future__ import annotations

import numpy as np

# Triangles rasterised together, bounding the memory one pass takes.
CHUNK = 4096


def view_frame(elevation: float, azimuth: float) -> np.ndarray:
    """The rows screen right, screen up and toward the viewer `[3, 3]`, for a view from the given elevation
    and azimuth in degrees, with z up."""
    elevation, azimuth = np.radians(elevation), np.radians(azimuth)
    toward = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    right = np.cross([0.0, 0.0, 1.0], toward)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(toward, right), toward])


def screen(points: np.ndarray, frame: np.ndarray, centre: np.ndarray, reach: float, size: int) -> np.ndarray:
    """Points `[..., 3]` in pixel coordinates and depth toward the viewer, for an image of the given side
    in pixels showing the given reach around the centre."""
    viewed = (points - centre) @ frame.T
    return viewed * (size / (2 * reach)) + [size / 2, size / 2, 0.0]


def rasterise(screen: np.ndarray, faces: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The nearest triangle at each covered pixel, with the pixel centre's barycentric weights in it.

    `screen` holds each vertex's pixel coordinates and its depth toward the viewer. Returns the flat pixel
    indices, the triangle covering each, the weights of its three corners, and the depth.
    """
    pixels, owners, depths, weights = [], [], [], []
    for start in range(0, len(faces), CHUNK):
        chunk = np.arange(start, min(start + CHUNK, len(faces)))
        corners = screen[faces[chunk]]                                       # [T, 3, 3]
        # the pixels whose centres fall within each triangle's bounds
        low = np.clip(np.ceil(corners[..., :2].min(axis=1) - 0.5), 0, size - 1).astype(int)
        high = np.clip(np.floor(corners[..., :2].max(axis=1) - 0.5), 0, size - 1).astype(int)
        span = np.maximum(high - low + 1, 0)
        counts = span[:, 0] * span[:, 1]
        owner = np.repeat(np.arange(len(chunk)), counts)
        local = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        x = low[owner, 0] + local % span[owner, 0]
        y = low[owner, 1] + local // span[owner, 0]
        a, b, c = (corners[owner, k] for k in range(3))
        area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        centre_x, centre_y = x + 0.5, y + 0.5
        first = ((b[:, 0] - centre_x) * (c[:, 1] - centre_y) - (b[:, 1] - centre_y) * (c[:, 0] - centre_x)) / area
        second = ((c[:, 0] - centre_x) * (a[:, 1] - centre_y) - (c[:, 1] - centre_y) * (a[:, 0] - centre_x)) / area
        weight = np.stack([first, second, 1 - first - second], axis=-1)
        inside = (weight >= 0).all(axis=-1) & (area != 0)
        pixels.append((y * size + x)[inside])
        owners.append(chunk[owner[inside]])
        depths.append(np.einsum("pk,pk->p", weight[inside], np.stack([a[inside, 2], b[inside, 2], c[inside, 2]], -1)))
        weights.append(weight[inside])
    pixel, owner, depth, weight = (np.concatenate(parts) for parts in (pixels, owners, depths, weights))
    order = np.lexsort((-depth, pixel))
    nearest = order[np.r_[True, pixel[order][1:] != pixel[order][:-1]]]
    return pixel[nearest], owner[nearest], weight[nearest], depth[nearest]


# --- WebGPU Hardware Mesh Rasterizer --------------------------------------------------

DEFAULT_MESH_SHADER = """
struct Uniforms {
    view_proj: mat4x4<f32>,
    light: Point,
    base_color: vec3<f32>,
    ambient: Scalar,
};

@group(0) @binding(0) var<uniform> scene: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) normal: Plane,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = scene.view_proj * vec4<f32>(in.position, 1.0);
    out.normal = Plane(in.normal.x, in.normal.y, in.normal.z, 0.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal: Plane = in.normal;
    let light: Point = scene.light;

    let n_len: Scalar = length(normal);
    let cosine: Scalar = (normal & light) / n_len;
    let lambert: Scalar = max(0.0, cosine);

    let lit_color = in.color * (scene.ambient + (1.0 - scene.ambient) * lambert);
    return vec4<f32>(lit_color, 1.0);
}
"""


class WGPURasterizer:
    """Hardware-accelerated triangle mesh rasterizer using WebGPU RenderPipeline and NumgaWGSLTranspiler.

    Executes vertex transformations and fragment shading written in the Numga WGSL dialect,
    backed by hardware triangle rasterization and early-Z depth testing.
    """

    def __init__(self, shader_code: str = DEFAULT_MESH_SHADER, algebra=None, gatypes=None):
        import wgpu
        from examples.raytrace import NumgaWGSLTranspiler, get_wgpu_device, ga as default_ga, Point as DefaultPoint, Plane as DefaultPlane, Scalar as DefaultScalar

        self.device = get_wgpu_device()
        used_algebra = algebra or default_ga
        self.transpiler = NumgaWGSLTranspiler(used_algebra)
        if gatypes is None:
            gatypes = {"Point": DefaultPoint, "Plane": DefaultPlane, "Scalar": DefaultScalar}
        for name, gatype in gatypes.items():
            self.transpiler.register_type(name, gatype)

        self.generated_wgsl = self.transpiler.transpile(shader_code)
        self.shader_module = self.device.create_shader_module(code=self.generated_wgsl)

        self.pipeline = self.device.create_render_pipeline(
            layout="auto",
            vertex={
                "module": self.shader_module,
                "entry_point": "vs_main",
                "buffers": [
                    {
                        "array_stride": 9 * 4,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x3", "offset": 3 * 4, "shader_location": 1},
                            {"format": "float32x3", "offset": 6 * 4, "shader_location": 2},
                        ],
                    }
                ],
            },
            primitive={
                "topology": "triangle-list",
                "front_face": "ccw",
                "cull_mode": "none",
            },
            depth_stencil={
                "format": "depth24plus",
                "depth_write_enabled": True,
                "depth_compare": "less",
            },
            fragment={
                "module": self.shader_module,
                "entry_point": "fs_main",
                "targets": [{"format": "rgba8unorm"}],
            },
        )

        self.color_texture = None
        self.depth_texture = None
        self.readback_buffer = None
        self.current_shape = None

    def _ensure_targets(self, width: int, height: int):
        import wgpu

        if self.current_shape != (width, height):
            self.color_texture = self.device.create_texture(
                size=(width, height, 1),
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
                format="rgba8unorm",
            )
            self.depth_texture = self.device.create_texture(
                size=(width, height, 1),
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
                format="depth24plus",
            )
            bytes_per_row = width * 4
            self.readback_buffer = self.device.create_buffer(
                size=bytes_per_row * height,
                usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
            )
            self.current_shape = (width, height)

    def render(self, vertex_data: np.ndarray, index_data: np.ndarray, uniform_data: np.ndarray, width: int, height: int) -> np.ndarray:
        import wgpu

        self._ensure_targets(width, height)

        v_buf = self.device.create_buffer_with_data(data=vertex_data.astype(np.float32), usage=wgpu.BufferUsage.VERTEX)
        i_buf = self.device.create_buffer_with_data(data=index_data.astype(np.uint32), usage=wgpu.BufferUsage.INDEX)
        u_buf = self.device.create_buffer_with_data(data=uniform_data.astype(np.float32), usage=wgpu.BufferUsage.UNIFORM)

        bind_group = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": u_buf, "offset": 0, "size": u_buf.size}}],
        )

        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[{
                "view": self.color_texture.create_view(),
                "resolve_target": None,
                "clear_value": (0.02, 0.02, 0.03, 1.0),
                "load_op": "clear",
                "store_op": "store",
            }],
            depth_stencil_attachment={
                "view": self.depth_texture.create_view(),
                "depth_clear_value": 1.0,
                "depth_load_op": "clear",
                "depth_store_op": "store",
            },
        )
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, bind_group)
        render_pass.set_vertex_buffer(0, v_buf)
        render_pass.set_index_buffer(i_buf, wgpu.IndexFormat.uint32)
        render_pass.draw_indexed(len(index_data), 1, 0, 0, 0)
        render_pass.end()

        bytes_per_row = width * 4
        encoder.copy_texture_to_buffer(
            {"texture": self.color_texture, "origin": (0, 0, 0)},
            {"buffer": self.readback_buffer, "offset": 0, "bytes_per_row": bytes_per_row, "rows_per_image": height},
            (width, height, 1),
        )
        self.device.queue.submit([encoder.finish()])

        self.readback_buffer.map_sync(wgpu.MapMode.READ)
        data = np.frombuffer(self.readback_buffer.read_mapped(), dtype=np.uint8).reshape((height, width, 4)).copy()
        self.readback_buffer.unmap()
        return data
