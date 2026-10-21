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
        if isinstance(kind, GAType):
            return coefficients(value, kind)
        alignment, size = self.size(kind)
        values = np.asarray(value, dtype=self.dtype(kind))
        width = size // 4
        values = values.reshape(-1, values.shape[-1])
        padded = np.zeros((values.shape[0], width), dtype=values.dtype)
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
        self.pipeline = self.device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "main"})
        self.sampler = self.device.create_sampler(mag_filter="linear", min_filter="linear",
                                                  address_mode_u="repeat", address_mode_v="clamp-to-edge")

    @property
    def code(self) -> str:
        return self.program.code

    def __call__(self, shape: tuple[int, int], **values: Any) -> dict[str, Any]:
        """Every written storage array, after one invocation per pixel of an image of `shape`, rows
        first; the shader reads its pixel as `id.xy`, column then row."""
        import wgpu

        count = int(np.prod(shape))
        entries, outputs = [], {}
        unused = set(values)
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
                _, size = self.layout.size(element)
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

        group = self.device.create_bind_group(layout=self.pipeline.get_bind_group_layout(0), entries=entries)
        encoder = self.device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(self.pipeline)
        compute.set_bind_group(0, group)
        size = self.program.workgroup_size
        compute.dispatch_workgroups(-(-shape[1] // size[0]), -(-shape[0] // size[1]), 1)
        compute.end()
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
            data = np.frombuffer(readback.read_mapped(), dtype=self.layout.dtype(element)).copy()
            readback.unmap()
            if isinstance(element, GAType):
                structural = tuple(len(subspace) for subspace in element.subspaces)
                results[name] = self.context.extensor(element, data.astype(np.float64).reshape(tuple(shape) + structural))
            else:
                results[name] = data.reshape(tuple(shape) + (element_size // 4,))
        return results

    def buffer(self, data: bytes, usage: int) -> dict:
        import wgpu
        buffer = self.device.create_buffer_with_data(data=data, usage=usage | wgpu.BufferUsage.COPY_DST)
        return {"buffer": buffer, "offset": 0, "size": len(data)}

    def texture(self, binding: Binding, image: np.ndarray) -> Any:
        """A float texture holding an image, `[rows, columns, channels]`, or a table, `[entries,
        channels]`, for a one-dimensional texture; channels pad to four, the last with one."""
        import wgpu
        image = np.asarray(image, dtype=np.float32)
        one_dimensional = binding.type.startswith("texture_1d")
        rows = image[None] if one_dimensional else image
        padded = np.ones(rows.shape[:2] + (4,), dtype=np.float32)
        padded[..., :rows.shape[2]] = rows
        height, width = padded.shape[:2]
        texture = self.device.create_texture(
            size=(width, height, 1), format="rgba32float",
            dimension="1d" if one_dimensional else "2d",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.device.queue.write_texture({"texture": texture}, padded.tobytes(),
                                        {"bytes_per_row": width * 16, "rows_per_image": height}, (width, height, 1))
        return texture
