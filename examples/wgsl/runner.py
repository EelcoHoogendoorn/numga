"""Run a shader in the numga dialect over an image, headless: values of the algebra in, arrays out.

A `Shader` compiles once. Calling it with an image shape and the values its declarations name binds
each one where the compiled shader expects it, runs one invocation per pixel, and returns every
storage array the shader writes and was not given, by name: arrays of the algebra's types come back as
batches of that type, other arrays as arrays of the image's shape. A shader keeps the buffers and textures it
runs with from one call to the next, each made again only when a call needs it larger, so that a frame of an
animation only writes its values into them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np

from numga import NumpyContext
from numga.extensor import Extensor
from numga.gatype import GAType

from examples.wgsl.dialect import Binding, Program, compile, template

_device = None


def software_driver() -> None:
    """In a Colab runtime, Mesa's software Vulkan driver, installed once: the GPU of a runtime without one, and
    harmless beside a real one, which the adapter request prefers."""
    if "google.colab" in sys.modules and not any(Path("/usr/share/vulkan/icd.d").glob("lvp_icd*.json")):
        subprocess.run("apt-get -qq update && apt-get -qq install -y mesa-vulkan-drivers", shell=True, check=True)


def device() -> Any:
    """The default GPU, opened once, with filtering of float textures where the adapter has it, and buffers as large
    as the adapter allows: a picture's samples, sixteen to a pixel, outgrow the default limits."""
    global _device
    if _device is None:
        import wgpu
        software_driver()
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        wanted = [feature for feature in ("float32-filterable",) if feature in adapter.features]
        largest = {name: adapter.limits[name] for name in ("max-buffer-size", "max-storage-buffer-binding-size")}
        _device = adapter.request_device_sync(required_features=wanted, required_limits=largest)
    return _device


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


class Resident(NamedTuple):
    """A buffer already on the GPU, bound to its first `size` bytes as a storage array a shader reads: data another
    pass wrote, which never visits the host."""
    buffer: Any
    size: int


class Layout:
    """Where values lie in a storage buffer, by WGSL's rules for its types."""

    def __init__(self, program: Program, scalar: GAType) -> None:
        self.program = program
        self.scalar = scalar
        self.gatypes = {kind: name for name, kind in program.gatypes.items()}

    def shape(self, kind: Any) -> tuple[str, Any]:
        """A type as one of: scalar, numeric WGSL, algebra, struct or array."""
        if kind == self.scalar or kind == "f32":
            return "number", "f32"
        if isinstance(kind, GAType):
            return "algebra", kind
        if kind in self.program.structs:
            return "struct", kind
        name, arguments = template(kind)
        if name == "array":
            return "array", arguments
        return "number", kind

    def size(self, kind: Any) -> tuple[int, int]:
        """A type's alignment and size in bytes."""
        form, detail = self.shape(kind)
        if form == "algebra":
            return 4, 4 * int(np.prod([len(subspace) for subspace in detail.subspaces]))
        if form == "struct":
            offset, alignment = 0, 1
            for _, member in self.program.structs[detail]:
                member_alignment, member_size = self.size(member)
                offset = -(-offset // member_alignment) * member_alignment + member_size
                alignment = max(alignment, member_alignment)
            return alignment, -(-offset // alignment) * alignment
        if form == "array":
            element_alignment, element_size = self.size(self.resolve(detail[0]))
            stride = -(-element_size // element_alignment) * element_alignment
            return element_alignment, stride * int(detail[1])
        name, arguments = template(detail)
        if name.startswith("vec"):
            count = int(name[3])
            return (8 if count == 2 else 16), 4 * count
        if name.startswith("mat"):
            columns, rows = int(name[3]), int(name[5])
            alignment = 8 if rows == 2 else 16
            return alignment, columns * alignment
        return 4, 4

    def stride(self, kind: Any) -> int:
        """The bytes between an array's elements: the element's size, rounded up to its alignment."""
        alignment, size = self.size(kind)
        return -(-size // alignment) * alignment

    def offsets(self, kind: str) -> list[int]:
        """Where each member of a struct begins."""
        found, offset = [], 0
        for _, member in self.program.structs[kind]:
            alignment, size = self.size(member)
            offset = -(-offset // alignment) * alignment
            found.append(offset)
            offset += size
        return found

    def resolve(self, text: str) -> Any:
        if text in self.program.gatypes:
            return self.program.gatypes[text]
        return self.scalar if text == "f32" else text

    def dtype(self, kind: Any) -> type:
        name, arguments = template(kind if isinstance(kind, str) else "f32")
        scalar = arguments[0] if arguments else name
        return {"u32": np.uint32, "i32": np.int32}.get(scalar, np.float32)

    def pack(self, kind: Any, value: Any) -> bytes:
        """A value's bytes, laid out as the type is in a storage buffer."""
        form, detail = self.shape(kind)
        alignment, size = self.size(kind)
        if isinstance(value, Extensor):
            data = coefficients(value, detail if form == "algebra" else self.scalar)
        elif form == "struct":
            data = bytearray()
            for member, member_type in self.program.structs[detail]:
                member_alignment, _ = self.size(member_type)
                data += bytes(-len(data) % member_alignment) + self.pack(member_type, value[member])
            data = bytes(data)
        elif form == "array":
            element = self.resolve(detail[0])
            data = b"".join(self.pack(element, entry) for entry in value)
        else:
            data = np.asarray(value, dtype=self.dtype(detail)).tobytes()
        return data + bytes(size - len(data)) if size > len(data) else data

    def array(self, kind: Any, value: Any) -> bytes:
        """A runtime-sized array's bytes: a batch of the algebra's type, or an array of numbers whose
        last axis is the element's components."""
        if isinstance(value, Extensor):
            return coefficients(value, kind)
        if kind in self.program.structs:
            # A struct's members by name, each a batch along the array, laid column by column at their offsets.
            stride = self.stride(kind)
            columns = [np.frombuffer(self.array(member_type, value[member]), dtype=np.uint8)
                       for member, member_type in self.program.structs[kind]]
            count = len(columns[0]) // self.stride(self.program.structs[kind][0][1])
            data = np.zeros((count, stride), dtype=np.uint8)
            for column, (_, member_type), offset in zip(columns, self.program.structs[kind], self.offsets(kind)):
                _, size = self.size(member_type)
                data[:, offset:offset + size] = column.reshape(count, -1)[:, :size]
            return data.tobytes()
        # The element's components, by its type, each padded to the array's stride.
        _, size = self.size(kind)
        values = np.asarray(value, dtype=self.dtype(kind)).reshape(-1, size // 4)
        padded = np.zeros((values.shape[0], self.stride(kind) // 4), dtype=values.dtype)
        padded[:, :values.shape[1]] = values
        return padded.tobytes()


def coefficients(value: Extensor, kind: GAType) -> bytes:
    """An Extensor's coefficients in the type the shader declares, flattened in kernel order."""
    if value.gatype != kind:
        value = value.cast(kind)
    structural = len(kind.subspaces)
    kernel = np.asarray(value.kernel, dtype=np.float32)
    return kernel.reshape(kernel.shape[:kernel.ndim - structural] + (-1,)).tobytes()


# The texture formats the raster stage may draw into: their bytes and channels a pixel, and how their numbers are kept.
FORMATS = {"rgba32float": (16, 4, np.float32), "r32float": (4, 1, np.float32), "rgba8unorm": (4, 4, np.uint8)}


class Shader:
    """A compiled shader in the numga dialect, ready to run over images; where it draws, the formats of the targets its
    fragment stage writes, one per target."""

    def __init__(self, source: str, algebra: Any, types: Mapping[str, GAType], targets: tuple[str, ...] = ()) -> None:
        self.targets = targets
        self.program = compile(source, algebra, types)
        self.context = NumpyContext(algebra)
        self.layout = Layout(self.program, algebra.gatype.scalar())
        self.device = device()
        module = self.device.create_shader_module(code=self.program.code)
        # Float textures filter where the device allows it, and store at half precision where it does not.
        self.filterable = "float32-filterable" in self.device.features
        self.bind_layout = self.device.create_bind_group_layout(entries=[self.layout_entry(binding) for binding in self.program.bindings])
        pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[self.bind_layout])
        self.pipelines = [(self.device.create_compute_pipeline(layout=pipeline_layout, compute={"module": module, "entry_point": name}), size)
                          for name, size in self.program.entry_points]
        # Triangles drawn with depth, one sample a pixel, into the targets' formats.
        if self.program.vertex is not None:
            self.raster = self.device.create_render_pipeline(
                layout=pipeline_layout,
                vertex={"module": module, "entry_point": self.program.vertex, "buffers": []},
                fragment={"module": module, "entry_point": self.program.fragment,
                          "targets": [{"format": target} for target in targets]},
                primitive={"topology": "triangle-list", "cull_mode": "none"},
                depth_stencil={"format": "depth32float", "depth_write_enabled": True, "depth_compare": "less"})
        self.sampler = self.device.create_sampler(mag_filter="linear", min_filter="linear",
                                                  address_mode_u="repeat", address_mode_v="clamp-to-edge")
        # The buffers and textures kept between calls, by their role.
        self.kept = {}                                                          # dict[tuple, Any]

    def layout_entry(self, binding: Binding) -> dict:
        """How the shader sees a binding, for every entry point alike."""
        import wgpu
        writable = binding.kind == "storage" and binding.access == "read_write"
        # A vertex stage may not write to storage; every stage reads.
        stages = wgpu.ShaderStage.COMPUTE | wgpu.ShaderStage.FRAGMENT | (0 if writable else wgpu.ShaderStage.VERTEX)
        entry = {"binding": binding.index, "visibility": stages}
        if binding.kind == "texture":
            return entry | {"texture": {"sample_type": "float", "view_dimension": "2d"}}
        if binding.kind == "sampler":
            return entry | {"sampler": {"type": "filtering"}}
        return entry | {"buffer": {"type": "storage" if writable else "read-only-storage"}}

    @property
    def code(self) -> str:
        return self.program.code

    def __call__(self, shape: tuple[int, int], **values: Any) -> dict[str, Any]:
        """Every written storage array, after each compute entry point in turn, in the order the shader
        declares them, with one invocation per pixel of an image of `shape`, rows first; the shader reads
        its pixel as `id.xy`, column then row, and the image's size, columns then rows, from a uniform
        `pixels` where it declares one. Written arrays start at zero."""
        encoder = self.device.create_command_encoder()
        outputs = self.encode(encoder, shape, (), values)
        return self.read(encoder, outputs, shape)

    def encode(self, encoder: Any, shape: tuple[int, int], key: tuple, values: Mapping[str, Any]) -> dict:
        """Each compute entry point in turn over an image of `shape`, recorded on an encoder, as a call runs them, with
        the values bound in the buffers kept under `key`: the arrays the shader writes, for `read`."""
        group, outputs = self.bind(shape, values, key)
        # The written arrays start at zero, whatever the last call left in them.
        for buffer, _, element_size in outputs.values():
            encoder.clear_buffer(buffer, 0, int(np.prod(shape)) * element_size)
        for pipeline, size in self.pipelines:
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(pipeline)
            compute.set_bind_group(0, group)
            compute.dispatch_workgroups(-(-shape[1] // size[0]), -(-shape[0] // size[1]), 1)
            compute.end()
        return outputs

    def draw(self, shape: tuple[int, int], n_vertices: int, **values: Any) -> list[np.ndarray]:
        """The images the shader's vertex and fragment stages draw, one per target the fragment stage writes,
        `[rows, columns, channels]` each in its format's numbers, zero where no triangle covers a pixel: `n_vertices`
        invocations of the vertex stage, three to a triangle, which read the vertices they place from storage, with the
        nearest surface kept by depth, one sample at each pixel's centre."""
        import wgpu

        rows, columns = shape
        encoder = self.device.create_command_encoder()
        targets = self.encode_draw(encoder, shape, n_vertices, (), values)
        readbacks = []
        for index, (target, format) in enumerate(zip(targets, self.targets)):
            # Rows of a texture copy start at multiples of 256 bytes.
            row_bytes = -(-columns * FORMATS[format][0] // 256) * 256
            readback = self.buffer_kept(("readback", index), row_bytes * rows, wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
            encoder.copy_texture_to_buffer({"texture": target}, {"buffer": readback, "bytes_per_row": row_bytes, "rows_per_image": rows},
                                           (columns, rows, 1))
            readbacks.append((readback, row_bytes))
        self.device.queue.submit([encoder.finish()])
        images = []
        for (readback, row_bytes), format in zip(readbacks, self.targets):
            size, channels, kind = FORMATS[format]
            readback.map_sync(wgpu.MapMode.READ, 0, row_bytes * rows)
            data = np.frombuffer(readback.read_mapped(0, row_bytes * rows), dtype=np.uint8).reshape(rows, row_bytes)[:, :columns * size]
            images.append(data.copy().view(kind).reshape(rows, columns, channels))
            readback.unmap()
        return images

    def draw_averaged(self, shape: tuple[int, int], samples_per_side: int, n_vertices: int, **values: Any) -> list[np.ndarray]:
        """The images `draw` draws, drawn `samples_per_side` times finer across and up and each pixel's block of samples
        averaged on the GPU: `[rows, columns, 4]` each, in the targets' numbers, only the pixels read back."""
        import wgpu

        rows, columns = shape
        encoder = self.device.create_command_encoder()
        targets = self.encode_draw(encoder, (samples_per_side * rows, samples_per_side * columns), n_vertices, (), values)
        pipeline = self.averaging()
        size = self.buffer(("average", "size"), np.array([columns, rows, samples_per_side, 0], dtype=np.uint32).tobytes())
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        n_bytes = rows * columns * 16
        readbacks = []
        for index, target in enumerate(targets):
            pixels = self.buffer_kept(("average", index), n_bytes, usage)
            group = self.device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
                {"binding": 0, "resource": target.create_view()},
                {"binding": 1, "resource": {"buffer": pixels, "offset": 0, "size": n_bytes}},
                {"binding": 2, "resource": size}])
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(pipeline)
            compute.set_bind_group(0, group)
            compute.dispatch_workgroups(-(-columns // 8), -(-rows // 8), 1)
            compute.end()
            readback = self.buffer_kept(("readback", "average", index), n_bytes, wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
            encoder.copy_buffer_to_buffer(pixels, 0, readback, 0, n_bytes)
            readbacks.append(readback)
        self.device.queue.submit([encoder.finish()])
        images = []
        for readback in readbacks:
            readback.map_sync(wgpu.MapMode.READ, 0, n_bytes)
            images.append(np.frombuffer(readback.read_mapped(0, n_bytes), dtype=np.float32).reshape(rows, columns, 4).copy())
            readback.unmap()
        return images

    def averaging(self) -> Any:
        """The compute pipeline averaging each pixel's samples, made once for the shader: it reads the samples without a
        sampler, as any float texture binds, filterable or not."""
        import wgpu
        if ("pipeline", "average") not in self.kept:
            module = self.device.create_shader_module(code=AVERAGE)
            entries = [{"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
                        "texture": {"sample_type": "unfilterable-float", "view_dimension": "2d"}},
                       {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
                       {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}}]
            layout = self.device.create_pipeline_layout(bind_group_layouts=[self.device.create_bind_group_layout(entries=entries)])
            self.kept[("pipeline", "average")] = self.device.create_compute_pipeline(
                layout=layout, compute={"module": module, "entry_point": "average"})
        return self.kept[("pipeline", "average")]

    def encode_draw(self, encoder: Any, shape: tuple[int, int], n_vertices: int, key: tuple,
                    values: Mapping[str, Any]) -> list[Any]:
        """The triangles drawn into the targets, as `draw` draws them, recorded on an encoder, with the values bound in
        the buffers kept under `key`: the targets, textures of `shape` the encoder's later work may copy from. The
        targets are kept by shape, so a later draw of the same shape on the encoder draws over them."""
        import wgpu

        group, _ = self.bind(shape, values, key)
        depth = self.surface(("depth", shape), shape, "depth32float", wgpu.TextureUsage.RENDER_ATTACHMENT)
        targets = [self.surface(("target", index, shape), shape, target,
                                wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.TEXTURE_BINDING)
                   for index, target in enumerate(self.targets)]
        drawing = encoder.begin_render_pass(
            color_attachments=[{"view": target.create_view(), "clear_value": (0, 0, 0, 0), "load_op": "clear",
                                "store_op": "store"} for target in targets],
            depth_stencil_attachment={"view": depth.create_view(), "depth_clear_value": 1.0,
                                      "depth_load_op": "clear", "depth_store_op": "discard"})
        drawing.set_pipeline(self.raster)
        drawing.set_bind_group(0, group)
        drawing.draw(n_vertices)
        drawing.end()
        return targets

    def bind(self, shape: tuple[int, int], values: Mapping[str, Any], key: tuple) -> tuple[Any, dict]:
        """The bind group of the values the shader declares, each written into the buffer kept for it under `key`, a
        `Resident` buffer bound as it is, and the arrays it writes, one element per pixel, in the buffers kept for
        them. Draws recorded on one encoder take keys of their own: values are written before the encoder runs."""
        import wgpu

        count = int(np.prod(shape))
        entries, outputs = [], {}
        unused = set(values)
        # A shader that declares the image's size reads it from the shape it runs over.
        if any(name == "pixels" for name, _ in self.program.uniforms):
            values = values | dict(pixels=[shape[1], shape[0]])
        for binding in self.program.bindings:
            if binding.kind == "uniforms":
                missing = [name for name, _ in self.program.uniforms if name not in values]
                if missing:
                    raise KeyError(f"no values given for uniforms {missing}")
                data = bytearray()
                for name, kind in self.program.uniforms:
                    alignment, _ = self.layout.size(kind)
                    data += bytes(-len(data) % alignment) + self.layout.pack(kind, values[name])
                    unused.discard(name)
                resource = self.buffer(key + ("uniforms",), bytes(data) + bytes(-len(data) % 16))
            elif binding.kind == "storage" and isinstance(values.get(binding.name), Resident):
                resident = values[binding.name]
                resource = {"buffer": resident.buffer, "offset": 0, "size": resident.size}
                unused.discard(binding.name)
            elif binding.kind == "storage" and binding.name in values:
                element = self.layout.resolve(template(binding.type)[1][0])
                resource = self.buffer(key + ("storage", binding.name), self.layout.array(element, values[binding.name]))
                unused.discard(binding.name)
            elif binding.kind == "storage":
                element = self.layout.resolve(template(binding.type)[1][0])
                size = self.layout.stride(element)
                buffer = self.buffer_kept(key + ("output", binding.name), count * size,
                                          wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
                outputs[binding.name] = (buffer, element, size)
                resource = {"buffer": buffer, "offset": 0, "size": count * size}
            elif binding.kind == "texture":
                resource = self.texture(key, binding, values[binding.name]).create_view()
                unused.discard(binding.name)
            else:
                resource = self.sampler
            entries.append({"binding": binding.index, "resource": resource})
        if unused:
            raise KeyError(f"the shader declares no {sorted(unused)}")

        group = self.device.create_bind_group(layout=self.bind_layout, entries=entries)
        return group, outputs

    def read(self, encoder: Any, outputs: dict, shape: tuple[int, int]) -> dict[str, Any]:
        """The written arrays, once the encoded work is done."""
        import wgpu

        count = int(np.prod(shape))
        readbacks = {}
        for name, (buffer, _, element_size) in outputs.items():
            readback = self.buffer_kept(("readback", name), count * element_size, wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
            encoder.copy_buffer_to_buffer(buffer, 0, readback, 0, count * element_size)
            readbacks[name] = readback
        self.device.queue.submit([encoder.finish()])

        results = {}
        for name, readback in readbacks.items():
            _, element, element_size = outputs[name]
            readback.map_sync(wgpu.MapMode.READ, 0, count * element_size)
            raw = np.frombuffer(readback.read_mapped(0, count * element_size), dtype=np.uint8).reshape(count, element_size).copy()
            readback.unmap()
            results[name] = self.unpack(element, raw, tuple(shape))
        return results

    def unpack(self, kind: Any, raw: np.ndarray, shape: tuple[int, ...]) -> Any:
        """Elements' bytes, `[elements, bytes]`, as the batch of a type of the algebra, a dict of a struct's
        members, or an array of numbers whose last axis is the element's components."""
        if kind in self.program.structs:
            return {member: self.unpack(member_type, raw[:, offset:offset + self.layout.size(member_type)[1]], shape)
                    for (member, member_type), offset in zip(self.program.structs[kind], self.layout.offsets(kind))}
        _, size = self.layout.size(kind)
        dtype = self.layout.dtype(kind) if not isinstance(kind, GAType) or kind == self.layout.scalar else np.float32
        data = np.ascontiguousarray(raw[:, :size]).view(dtype)
        if isinstance(kind, GAType) and kind != self.layout.scalar:
            structural = tuple(len(subspace) for subspace in kind.subspaces)
            return self.context.extensor(kind, data.astype(np.float64).reshape(shape + structural))
        return data.reshape(shape + (size // 4,))

    def buffer_kept(self, role: tuple, size: int, usage: int) -> Any:
        """The buffer kept for a role, at least `size` bytes: made on the first call that needs it, and again only when
        a call needs it larger."""
        buffer = self.kept.get(role)
        if buffer is None or buffer.size < size:
            buffer = self.device.create_buffer(size=max(size, 16), usage=usage)
            self.kept[role] = buffer
        return buffer

    def buffer(self, role: tuple, data: bytes) -> dict:
        """The storage buffer kept for a role, holding data, bound to just its length."""
        import wgpu
        buffer = self.buffer_kept(role, len(data), wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        self.device.queue.write_buffer(buffer, 0, data)
        return {"buffer": buffer, "offset": 0, "size": len(data)}

    def surface(self, role: tuple, shape: tuple[int, int], format: str, usage: int) -> Any:
        """The texture kept for a role and an image's shape, rows and columns, made on the first call that needs it."""
        if role not in self.kept:
            rows, columns = shape
            self.kept[role] = self.device.create_texture(size=(columns, rows, 1), format=format, usage=usage)
        return self.kept[role]

    def texture(self, key: tuple, binding: Binding, image: np.ndarray) -> Any:
        """A float texture holding an image, `[rows, columns, channels]`; channels pad to four, the last
        with one."""
        import wgpu
        image = np.asarray(image, dtype=np.float32)
        padded = np.ones(image.shape[:2] + (4,), dtype=np.float32 if self.filterable else np.float16)
        padded[..., :image.shape[2]] = image
        height, width = padded.shape[:2]
        texture = self.surface(key + ("texture", binding.name, (height, width)), (height, width),
                               "rgba32float" if self.filterable else "rgba16float",
                               wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.device.queue.write_texture({"texture": texture}, padded.tobytes(),
                                        {"bytes_per_row": width * padded.itemsize * 4, "rows_per_image": height}, (width, height, 1))
        return texture
