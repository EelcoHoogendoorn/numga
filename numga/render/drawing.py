"""GPU triangle rasterization, supersampled sample-block averaging, and layer compositing underpin
the geometric marks in `numga.render` without inspecting algebraic values directly.

A `Draw` pairs a compiled `Shader` with pixel dimensions `shape` (`(rows, columns)`), vertex-stage
invocation count `n_vertices` (three per triangle), and uniform or storage bindings `values`. A
`Target` specifies attachment texture formats and whether fragments blend premultiplied `over` prior
draws (`LAID_OVER`) or resolve nearest depth (`NEAREST`). `drawn` rasterizes `side` samples per
pixel axis in a single pass and averages each pixel's `side x side` sample block on the GPU via
`textureLoad` (which binds any float texture regardless of filterability), reading back only
`[rows, columns, 4]` pixel arrays.

`composited` blends `Draw` passes and host `[rows, columns, 4]` RGBA images in order and divides out
premultiplied coverage. For 3D scenes mixing rasterized surfaces with ray-traced quadrics, `sampled`
rasterizes each `Draw` into padded per-sample GPU `Slots` buffers read directly by the `solids` ray
tracer, while `compose` depth-sorts coverage-weighted `[rows, columns, 4]` colour and
`[rows, columns]` depth layers on the host. Custom window or render passes can record any `Draw`
directly through `pipeline(draw.shader, target)` and `draw.shader.bind(...)`.
"""

from __future__ import annotations

from functools import cache
from typing import Any, Callable, NamedTuple

import numpy as np

from numga import Algebra
from numga.wgsl import Shader, device, samples_per_side


class Draw(NamedTuple):
    """A triangle-list raster pass running `n_vertices` vertex invocations (three per triangle) of
    `shader` over an output grid of `shape` (`(rows, columns)`) pixels with resource bindings
    `values`."""
    shader: Shader
    shape: tuple[int, int]
    n_vertices: int
    values: dict


class Target(NamedTuple):
    """Attachment `formats` written by a fragment stage and blending mode `over` (`True` for
    premultiplied alpha-over blending without depth writes, `False` for nearest-depth testing)."""
    formats: tuple[str, ...]
    over: bool


# A panel's layers, each laid over the ones before; a lit scene's meshes, their colour and their depth, the nearest kept.
LAID_OVER = Target(("rgba8unorm",), True)
NEAREST = Target(("rgba8unorm", "r32float"), False)

# The textures and buffers kept between pictures, by their role and size.
kept = {}                                                                     # dict[tuple, Any]


@cache
def pipeline(shader: Shader, target: Target) -> Any:
    """Build and cache the triangle-list render pipeline for `shader` writing into `target`."""
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
    """Return the cached 2D GPU texture for `(role, shape)` (`(rows, columns)`), creating it on first use."""
    if (role, shape) not in kept:
        kept[role, shape] = device().create_texture(size=(shape[1], shape[0], 1), format=format, usage=usage)
    return kept[role, shape]


def buffer(role: tuple, size: int, usage: int) -> Any:
    """Return the cached GPU buffer for `role` of at least `size` bytes, reallocating only when needed."""
    if role not in kept or kept[role].size < size:
        kept[role] = device().create_buffer(size=max(size, 16), usage=usage)
    return kept[role]


def encode(encoder: Any, draws: list[Draw], target: Target, side: int, key: tuple) -> list[Any]:
    """Record `draws` in order onto `encoder` in a single render pass at `side` samples per pixel
    axis, binding each draw's resources under `key + (index,)` and returning the target attachment
    textures."""
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
    """Build and cache the compute pipeline that averages `side x side` sample blocks via
    `textureLoad` without requiring a filtering sampler."""
    return device().create_compute_pipeline(layout="auto", compute={
        "module": device().create_shader_module(code=AVERAGE), "entry_point": "average"})


def drawn(draws: list[Draw], target: Target, side: int) -> list[np.ndarray]:
    """Rasterize `draws` into `target` at `side` samples per pixel axis, average each pixel's sample
    block on the GPU, and return one `float32` `[rows, columns, 4]` array per attachment format."""
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
    """Compile and cache the full-screen triangle shader that upsamples a host premultiplied RGBA
    texture onto subpixel samples."""
    return Shader(image_source, Algebra("x+"), {}, {})


def composited(pictures: list[Draw | np.ndarray]) -> np.ndarray:
    """Composite `Draw` passes and host `[rows, columns, 4]` straight-RGBA arrays in order at
    subpixel resolution and return straight RGBA `[rows, columns, 4]`."""
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
    """GPU storage buffers holding per-sample packed RGBA `colours` and float `depths` across
    sequential draw slots for an output grid of `shape` (`(rows, columns)`) pixels, with total byte
    capacity `size` and 256-byte-aligned `row_bytes` per sample row."""
    colours: Any
    depths: Any
    size: int
    row_bytes: int
    shape: tuple[int, int]


def slots(shape: tuple[int, int], n_slots: int, made: Callable[[str, int, int], Any]) -> Slots:
    """Allocate `Slots` for `n_slots` supersampled draws over `shape` (`(rows, columns)`) pixels
    using buffer factory `made(name, size, usage)`."""
    import wgpu

    rows, columns = shape
    # A copy's rows start at multiples of 256 bytes; a sample's colour and its depth are four bytes each.
    row_bytes = -(-samples_per_side() * columns * 4 // 256) * 256
    size = max(n_slots * samples_per_side() * rows * row_bytes, 16)
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    return Slots(made("colours", size, usage), made("depths", size, usage), size, row_bytes, shape)


def sampled(encoder: Any, draws: list[Draw], first: int, kept_slots: Slots) -> None:
    """Rasterize each draw in `draws` with nearest-depth testing at every subpixel sample and copy
    its colour and depth attachments on `encoder` into `kept_slots` starting at slot `first`, leaving
    earlier slots untouched."""
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
    """Depth-sort and blend `(colour, depth, opacity)` layers (`[rows, columns, 4]` premultiplied
    RGBA and `[rows, columns]` coverage-weighted depth) nearest first over RGB `background`,
    returning `uint8` RGB `[rows, columns, 3]`."""
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
