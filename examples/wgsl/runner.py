"""Run a shader in the numga dialect over an image, headless: values of the algebra in, arrays out.

A `Shader` compiles once. Calling it with an image shape and the values its declarations name binds
each one where the compiled shader expects it, runs one invocation per pixel, and returns every
storage array the shader writes and was not given, by name: arrays of the algebra's types come back as
batches of that type, other arrays as arrays of the image's shape.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from numga import NumpyContext
from numga.extensor import Extensor
from numga.gatype import GAType

from examples.wgsl.dialect import Binding, Program, compile, template

_device = None


def device() -> Any:
    """The default GPU, opened once, with filtering of float textures where the adapter has it."""
    global _device
    if _device is None:
        import wgpu
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        wanted = [feature for feature in ("float32-filterable",) if feature in adapter.features]
        _device = adapter.request_device_sync(required_features=wanted)
    return _device


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


class Shader:
    """A compiled shader in the numga dialect, ready to run over images."""

    def __init__(self, source: str, algebra: Any, types: Mapping[str, GAType]) -> None:
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
        # Triangles drawn with depth, their edges resolved over several samples a pixel.
        self.multisample = 4
        if self.program.vertex is not None:
            self.raster = self.device.create_render_pipeline(
                layout=pipeline_layout,
                vertex={"module": module, "entry_point": self.program.vertex, "buffers": []},
                fragment={"module": module, "entry_point": self.program.fragment,
                          "targets": [{"format": "rgba16float"}] * self.program.targets},
                primitive={"topology": "triangle-list", "cull_mode": "none"},
                depth_stencil={"format": "depth32float", "depth_write_enabled": True, "depth_compare": "less"},
                multisample={"count": self.multisample})
        self.sampler = self.device.create_sampler(mag_filter="linear", min_filter="linear",
                                                  address_mode_u="repeat", address_mode_v="clamp-to-edge")

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
        import wgpu

        import wgpu

        group, outputs = self.bind(shape, values)
        encoder = self.device.create_command_encoder()
        for pipeline, size in self.pipelines:
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(pipeline)
            compute.set_bind_group(0, group)
            compute.dispatch_workgroups(-(-shape[1] // size[0]), -(-shape[0] // size[1]), 1)
            compute.end()
        return self.read(encoder, outputs, shape)

    def draw(self, shape: tuple[int, int], n_vertices: int, **values: Any) -> list[np.ndarray]:
        """The images the shader's vertex and fragment stages draw, one per target the fragment stage writes,
        `[rows, columns, 4]` each, premultiplied by how much of each pixel the triangles cover: `n_vertices`
        invocations of the vertex stage, three to a triangle, which read the vertices they place from storage,
        with the nearest surface kept by depth and edges resolved over several samples a pixel."""
        import wgpu

        group, _ = self.bind(shape, values)
        rows, columns = shape
        attachment = dict(size=(columns, rows, 1), sample_count=self.multisample, usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
        depth = self.device.create_texture(format="depth32float", **attachment)
        targets = []
        for _ in range(self.program.targets):
            sampled = self.device.create_texture(format="rgba16float", **attachment)
            resolved = self.device.create_texture(size=(columns, rows, 1), format="rgba16float",
                                                  usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
            targets.append((sampled, resolved))
        encoder = self.device.create_command_encoder()
        drawing = encoder.begin_render_pass(
            color_attachments=[{"view": sampled.create_view(), "resolve_target": resolved.create_view(),
                                "clear_value": (0, 0, 0, 0), "load_op": "clear", "store_op": "discard"}
                               for sampled, resolved in targets],
            depth_stencil_attachment={"view": depth.create_view(), "depth_clear_value": 1.0,
                                      "depth_load_op": "clear", "depth_store_op": "discard"})
        drawing.set_pipeline(self.raster)
        drawing.set_bind_group(0, group)
        drawing.draw(n_vertices)
        drawing.end()
        # Rows of a texture copy start at multiples of 256 bytes.
        row_bytes = -(-columns * 8 // 256) * 256
        readbacks = []
        for _, resolved in targets:
            readback = self.device.create_buffer(size=row_bytes * rows, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
            encoder.copy_texture_to_buffer({"texture": resolved}, {"buffer": readback, "bytes_per_row": row_bytes, "rows_per_image": rows},
                                           (columns, rows, 1))
            readbacks.append(readback)
        self.device.queue.submit([encoder.finish()])
        images = []
        for readback in readbacks:
            readback.map_sync(wgpu.MapMode.READ)
            data = np.frombuffer(readback.read_mapped(), dtype=np.float16).reshape(rows, row_bytes // 2)
            images.append(data[:, :columns * 4].reshape(rows, columns, 4).astype(np.float32))
            readback.unmap()
        return images

    def bind(self, shape: tuple[int, int], values: dict) -> tuple[Any, dict]:
        """The bind group of the values the shader declares, and the arrays it writes, one element per
        pixel, made for the purpose and zeroed."""
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
                resource = self.buffer(bytes(data) + bytes(-len(data) % 16), wgpu.BufferUsage.STORAGE)
            elif binding.kind == "storage" and binding.name in values:
                element = self.layout.resolve(template(binding.type)[1][0])
                resource = self.buffer(self.layout.array(element, values[binding.name]), wgpu.BufferUsage.STORAGE)
                unused.discard(binding.name)
            elif binding.kind == "storage":
                element = self.layout.resolve(template(binding.type)[1][0])
                size = self.layout.stride(element)
                buffer = self.device.create_buffer(size=count * size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
                outputs[binding.name] = (buffer, element, size)
                resource = {"buffer": buffer, "offset": 0, "size": count * size}
            elif binding.kind == "texture":
                resource = self.texture(binding, values[binding.name]).create_view()
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
            readback = self.device.create_buffer(size=count * element_size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
            encoder.copy_buffer_to_buffer(buffer, 0, readback, 0, count * element_size)
            readbacks[name] = readback
        self.device.queue.submit([encoder.finish()])

        results = {}
        for name, readback in readbacks.items():
            _, element, element_size = outputs[name]
            readback.map_sync(wgpu.MapMode.READ)
            raw = np.frombuffer(readback.read_mapped(), dtype=np.uint8).reshape(count, element_size).copy()
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

    def buffer(self, data: bytes, usage: int) -> dict:
        import wgpu
        buffer = self.device.create_buffer_with_data(data=data, usage=usage | wgpu.BufferUsage.COPY_DST)
        return {"buffer": buffer, "offset": 0, "size": len(data)}

    def texture(self, binding: Binding, image: np.ndarray) -> Any:
        """A float texture holding an image, `[rows, columns, channels]`; channels pad to four, the last
        with one."""
        import wgpu
        image = np.asarray(image, dtype=np.float32)
        padded = np.ones(image.shape[:2] + (4,), dtype=np.float32 if self.filterable else np.float16)
        padded[..., :image.shape[2]] = image
        height, width = padded.shape[:2]
        texture = self.device.create_texture(
            size=(width, height, 1), format="rgba32float" if self.filterable else "rgba16float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.device.queue.write_texture({"texture": texture}, padded.tobytes(),
                                        {"bytes_per_row": width * padded.itemsize * 4, "rows_per_image": height}, (width, height, 1))
        return texture
