"""Drawing on the GPU: shaders of the numga dialect drawn as triangles into textures that stay on the GPU, each draw laid
over the ones before it or kept where it is nearest, and averaged over each pixel's samples.

A `Draw` is a shader's triangles over a picture: the shader, the picture's pixels, how many corners its vertex stage
places, and the values the shader reads. Where draws go is a `Target`: the formats of the textures the fragment stage
writes, and whether each fragment is laid over what is drawn before it, its colour weighted by its coverage, or hidden
where something nearer is drawn. Draws are drawn `side` times finer across and up than their pixels, and each pixel's
block of samples averaged on the GPU; only the pixels come back. `composited` lays a panel's draws, and pictures made on
the host, over each other in order; `sampled` keeps draws at every sample in slots of buffers on the GPU, for a pass
that reads them, as the tracer of solids reads the meshes among its solids; `compose` lays pictures that carry their
depth over each other on the host, nearest first.

Nothing here reads geometry: the marks of `numga.render` make the draws. A pass of a caller's own, as a window's, records
a draw with `pipeline(draw.shader, target)` and the bind group of the shader's own `bind`.
"""

from __future__ import annotations

from functools import cache
from typing import Any, Callable, NamedTuple

import numpy as np

from numga import Algebra
from numga.wgsl import Shader, device, samples_per_side


class Draw(NamedTuple):
    """A shader's triangles over a picture of `shape` pixels, rows and columns: `n_vertices` invocations of its vertex
    stage, three to a triangle, which read the corners they place, and everything else, from `values`."""
    shader: Shader
    shape: tuple[int, int]
    n_vertices: int
    values: dict


class Target(NamedTuple):
    """Where draws go: the formats of the textures the fragment stage writes, one each, and whether each fragment is laid
    `over` what is drawn before it, its colour weighted by its coverage, rather than hidden where something nearer is
    drawn."""
    formats: tuple[str, ...]
    over: bool


# A panel's layers, each laid over the ones before; a lit scene's meshes, their colour and their depth, the nearest kept.
LAID_OVER = Target(("rgba8unorm",), True)
NEAREST = Target(("rgba8unorm", "r32float"), False)

# The textures and buffers kept between pictures, by their role and size.
kept = {}                                                                     # dict[tuple, Any]


@cache
def pipeline(shader: Shader, target: Target) -> Any:
    """The render pipeline drawing the shader's triangles into the target's textures, one sample a texel."""
    blend = {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"}
    laid_over = {"blend": {"color": blend, "alpha": blend}} if target.over else {}
    return shader.device.create_render_pipeline(
        layout=shader.pipeline_layout,
        vertex={"module": shader.module, "entry_point": shader.program.vertex, "buffers": []},
        fragment={"module": shader.module, "entry_point": shader.program.fragment,
                  "targets": [{"format": format} | laid_over for format in target.formats]},
        primitive={"topology": "triangle-list", "cull_mode": "none"},
        depth_stencil={"format": "depth32float", "depth_write_enabled": not target.over,
                       "depth_compare": "always" if target.over else "less"})


def surface(role: tuple, shape: tuple[int, int], format: str, usage: int) -> Any:
    """The texture kept for a role and a size, rows and columns, made on the first call that needs it."""
    if (role, shape) not in kept:
        kept[role, shape] = device().create_texture(size=(shape[1], shape[0], 1), format=format, usage=usage)
    return kept[role, shape]


def buffer(role: tuple, size: int, usage: int) -> Any:
    """The buffer kept for a role, at least `size` bytes, made again only when a call needs it larger."""
    if role not in kept or kept[role].size < size:
        kept[role] = device().create_buffer(size=max(size, 16), usage=usage)
    return kept[role]


def encode(encoder: Any, draws: list[Draw], target: Target, side: int, key: tuple) -> list[Any]:
    """The draws recorded on an encoder in one pass into the target's textures, in order, `side` samples a pixel across
    and up: the textures, kept for the target and their size, which the encoder's later work may read. Each draw's
    values are bound under `key` and its place in the list, as the values of work recorded on one encoder must be."""
    import wgpu

    shape = (side * draws[0].shape[0], side * draws[0].shape[1])
    usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.TEXTURE_BINDING
    textures = [surface((target, index), shape, format, usage) for index, format in enumerate(target.formats)]
    depth = surface((target, "depth"), shape, "depth32float", wgpu.TextureUsage.RENDER_ATTACHMENT)
    groups = [draw.shader.bind(shape, draw.values, key + (index,))[0] for index, draw in enumerate(draws)]
    drawing = encoder.begin_render_pass(
        color_attachments=[{"view": texture.create_view(), "clear_value": (0, 0, 0, 0), "load_op": "clear",
                            "store_op": "store"} for texture in textures],
        depth_stencil_attachment={"view": depth.create_view(), "depth_clear_value": 1.0,
                                  "depth_load_op": "clear", "depth_store_op": "discard"})
    for draw, group in zip(draws, groups):
        drawing.set_pipeline(pipeline(draw.shader, target))
        drawing.set_bind_group(0, group)
        drawing.draw(draw.n_vertices)
    drawing.end()
    return textures


# Each pixel's average over its block of samples, `size.z` to a side, of a texture drawn that much finer than the image
# of `size.x` columns and `size.y` rows.
AVERAGE = """
@group(0) @binding(0) var samples: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> pixels: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> size: vec4<u32>;

@compute @workgroup_size(8, 8)
fn average(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    var sum = vec4<f32>(0.0);
    for (var down = 0u; down < size.z; down++) {
        for (var across = 0u; across < size.z; across++) {
            sum += textureLoad(samples, vec2<u32>(id.x * size.z + across, id.y * size.z + down), 0);
        }
    }
    pixels[id.y * size.x + id.x] = sum / f32(size.z * size.z);
}
"""


@cache
def averaging() -> Any:
    """The compute pipeline averaging each pixel's samples: it reads them without a sampler, as any float texture
    binds, filterable or not."""
    return device().create_compute_pipeline(layout="auto", compute={
        "module": device().create_shader_module(code=AVERAGE), "entry_point": "average"})


def drawn(draws: list[Draw], target: Target, side: int) -> list[np.ndarray]:
    """The pictures the draws make, one per format of the target, `[rows, columns, 4]` each, in floats: drawn `side`
    times finer across and up, in one pass, each pixel's block of samples averaged on the GPU, and only the pixels read
    back."""
    import wgpu

    rows, columns = draws[0].shape
    n_bytes = rows * columns * 16
    gpu = device()
    encoder = gpu.create_command_encoder()
    textures = encode(encoder, draws, target, side, ("drawn",))
    size = buffer(("size",), 16, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    gpu.queue.write_buffer(size, 0, np.array([columns, rows, side, 0], dtype=np.uint32).tobytes())
    readbacks = []
    for index, texture in enumerate(textures):
        pixels = buffer(("pixels", index), n_bytes, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        averaging_pass = encoder.begin_compute_pass()
        averaging_pass.set_pipeline(averaging())
        averaging_pass.set_bind_group(0, gpu.create_bind_group(layout=averaging().get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": texture.create_view()},
            {"binding": 1, "resource": {"buffer": pixels, "offset": 0, "size": n_bytes}},
            {"binding": 2, "resource": {"buffer": size, "offset": 0, "size": 16}}]))
        averaging_pass.dispatch_workgroups(-(-columns // 8), -(-rows // 8), 1)
        averaging_pass.end()
        readbacks.append(buffer(("readback", index), n_bytes, wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ))
        encoder.copy_buffer_to_buffer(pixels, 0, readbacks[-1], 0, n_bytes)
    gpu.queue.submit([encoder.finish()])
    pictures = []
    for readback in readbacks:
        readback.map_sync(wgpu.MapMode.READ, 0, n_bytes)
        pictures.append(np.frombuffer(readback.read_mapped(0, n_bytes), dtype=np.float32).reshape(rows, columns, 4).copy())
        readback.unmap()
    return pictures


# A picture made on the host, its colour weighted by its coverage, laid over a panel's samples: each sample takes its
# pixel's colour.
image_source = """
var image: texture_2d<f32>;
var<uniform> side: u32;

struct Corner {
    @builtin(position) position: vec4<f32>,
}

// One triangle over the whole picture.
@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let corner = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0))[index];
    return Corner(vec4<f32>(corner, 0.5, 1.0));
}

@fragment
fn shade(corner: Corner) -> @location(0) vec4<f32> {
    return textureLoad(image, vec2<u32>(corner.position.xy) / side, 0);
}
"""


@cache
def image_shader() -> Shader:
    """The shader of a picture made on the host; it reads no values of an algebra."""
    return Shader(image_source, Algebra("x+"), {}, {})


def composited(pictures: list[Draw | np.ndarray]) -> np.ndarray:
    """A panel's pictures, draws or colour and coverage `[rows, columns, 4]` made on the host, each laid over the ones
    before, sample by sample: colour and coverage `[rows, columns, 4]`."""
    side = samples_per_side()
    draws = [picture if isinstance(picture, Draw) else
             Draw(image_shader(), picture.shape[:2], 3, dict(
                 image=np.concatenate((picture[..., :3] * picture[..., 3:], picture[..., 3:]), axis=-1), side=side))
             for picture in pictures]
    weighted, = drawn(draws, LAID_OVER, side)
    # The colour divided back by the coverage where anything covers, both within the unit their rounding may overshoot.
    coverage = weighted[..., 3:]
    with np.errstate(divide="ignore", invalid="ignore"):
        colour = np.where(coverage > 0, weighted[..., :3] / coverage, weighted[..., :3])
    return np.minimum(np.concatenate((colour, coverage), axis=-1), 1.0)


class Slots(NamedTuple):
    """Buffers on the GPU holding draws at every sample of a picture of `shape` pixels, slot after slot: their colours
    and their depths, the bytes each holds, and the bytes a row of samples takes, padded as a copy pads it."""
    colours: Any
    depths: Any
    size: int
    row_bytes: int
    shape: tuple[int, int]


def slots(shape: tuple[int, int], n_slots: int, made: Callable[[str, int, int], Any]) -> Slots:
    """Slots for the samples of `n_slots` draws over a picture of `shape` pixels, each buffer from `made(name, size,
    usage)`."""
    import wgpu

    rows, columns = shape
    # A copy's rows start at multiples of 256 bytes; a sample's colour and its depth are four bytes each.
    row_bytes = -(-samples_per_side() * columns * 4 // 256) * 256
    size = max(n_slots * samples_per_side() * rows * row_bytes, 16)
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    return Slots(made("colours", size, usage), made("depths", size, usage), size, row_bytes, shape)


def sampled(encoder: Any, draws: list[Draw], first: int, kept_slots: Slots) -> None:
    """The draws drawn at every sample of their pixels, each the nearest of its own triangles, recorded on an encoder
    and copied there, on the GPU, draw after draw, into the slots from `first` on, the slots before it left as they
    are."""
    side = samples_per_side()
    rows, columns = side * kept_slots.shape[0], side * kept_slots.shape[1]
    slot_bytes = rows * kept_slots.row_bytes
    for index, draw in enumerate(draws, start=first):
        # A draw of nothing, such as a trail not yet begun, shows nothing: its slot cleared.
        if draw.n_vertices == 0:
            for held in (kept_slots.colours, kept_slots.depths):
                encoder.clear_buffer(held, index * slot_bytes, slot_bytes)
            continue
        textures = encode(encoder, [draw], NEAREST, side, ("slot", index))
        for texture, held in zip(textures, (kept_slots.colours, kept_slots.depths)):
            encoder.copy_texture_to_buffer(
                {"texture": texture},
                {"buffer": held, "offset": index * slot_bytes, "bytes_per_row": kept_slots.row_bytes, "rows_per_image": rows},
                (columns, rows, 1))


def compose(layers: list[tuple[np.ndarray, np.ndarray, float]], background: np.ndarray) -> np.ndarray:
    """Layers blended over a background, nearest first at each pixel, `[rows, columns, 3]` bytes: each layer its
    colour and its depth along the view, `[rows, columns, 4]` and `[rows, columns]`, both weighted by how much of
    the pixel it covers, and its opacity."""
    colours = np.stack([colour * opacity for colour, _, opacity in layers])          # [layers, rows, columns, 4]
    with np.errstate(divide="ignore", invalid="ignore"):
        depths = np.stack([np.where(colour[..., 3] > 0, depth / colour[..., 3], np.inf) for colour, depth, _ in layers])
    order = np.argsort(depths, axis=0)
    shown = np.zeros(colours.shape[1:3] + (3,))
    passing = np.ones(colours.shape[1:3] + (1,))
    for nearest in np.take_along_axis(colours, order[..., None], axis=0):
        shown += passing * nearest[..., :3]
        passing *= 1 - nearest[..., 3:]
    return np.round(np.clip(shown + passing * background, 0, 1) * 255).astype(np.uint8)
