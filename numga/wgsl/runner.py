"""Headless WebGPU execution compiles shaders written in the `numga` WGSL dialect once against an
`Algebra` and dispatches compute pipelines over grids of up to three axes, while exposing vertex and
fragment entry points for rasterization passes. Calling a `Shader` with a grid shape and named
Python inputs binds uniform values, storage arrays, `Resident` GPU buffers, and 2D float textures in
declaration order, clears unprovided `read_write` storage buffers to zero, runs each `@compute` entry
point in sequence, and reads back every unprovided storage output.

`Layout` maps between `numga` `Extensor` batches, Python dicts of struct columns, NumPy arrays, and
WGSL storage-buffer byte layouts. Non-scalar nullary and unary `GAType` elements are packed as
contiguous `float32` blade coefficients cast to the shader's declared `GAType` and unpacked back into
`float64` `Extensor` batches on readback, while composite structs, vectors, matrices, and fixed-length
arrays follow WGSL alignment and stride padding rules.

`device` opens the shared high-performance WebGPU adapter once at its maximum buffer size limits,
enabling `float32-filterable` textures where supported and falling back to `rgba16float` storage
otherwise (after installing Mesa's software Vulkan driver on headless Colab runtimes). Each `Shader`
caches its GPU buffers and textures across calls by role key, reallocating buffers only when a larger
byte capacity is needed and skipping texture uploads when passed the identical NumPy array object
across frames.
"""

from __future__ import annotations

from collections.abc import Mapping
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from numga import Algebra, NumpyContext
from numga.extensor import Extensor
from numga.gatype import GAType

from .dialect import Binding, Program, compile, template

_device = None


def software_driver() -> None:
    """Colab runtimes without a Vulkan ICD install Mesa's `lvp` software driver once before adapter discovery."""
    if "google.colab" in sys.modules and not any(Path("/usr/share/vulkan/icd.d").glob("lvp_icd*.json")):
        subprocess.run("apt-get -qq update && apt-get -qq install -y mesa-vulkan-drivers", shell=True, check=True)


def device() -> Any:
    """Opening the shared high-performance WebGPU device once requests `float32-filterable` when
    available and raises buffer size limits to the adapter's maximums."""
    global _device
    if _device is None:
        import wgpu
        software_driver()
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        wanted = [feature for feature in ("float32-filterable",) if feature in adapter.features]
        largest = {name: adapter.limits[name] for name in ("max-buffer-size", "max-storage-buffer-binding-size")}
        _device = adapter.request_device_sync(required_features=wanted, required_limits=largest)
    return _device


class Resident(NamedTuple):
    """Passing a `Resident` handle binds the first `size` bytes of an existing GPU `buffer` directly
    to a storage array without copying through host memory."""
    buffer: Any
    size: int


class Layout:
    """Byte alignments, strides, member offsets, and binary packing follow WGSL storage-buffer layout
    rules across algebraic structs, composite structs, and numeric types."""

    def __init__(self, program: Program, scalar: GAType) -> None:
        self.program = program
        self.scalar = scalar
        # Reversing `program.gatypes` keeps each `GAType`'s primary struct name ahead of later aliases.
        self.gatypes = {kind: name for name, kind in reversed(program.gatypes.items())}

    def shape(self, kind: GAType | str) -> tuple[str, Any]:
        """Classifying a type distinguishes scalar numbers, multi-blade algebraic `GAType` structs,
        composite WGSL structs, fixed-length arrays, and native WGSL primitives."""
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

    def size(self, kind: GAType | str) -> tuple[int, int]:
        """Alignment and byte size follow 4-byte `f32` blade packing for `GAType` structs and WGSL
        member-alignment rules for vectors, matrices, arrays, and structs."""
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

    def stride(self, kind: GAType | str) -> int:
        """Element byte size rounds up to the element alignment to give the byte stride between
        consecutive array entries."""
        alignment, size = self.size(kind)
        return -(-size // alignment) * alignment

    def offsets(self, struct_name: str) -> list[int]:
        """Struct members align sequentially to their respective byte alignments, yielding each
        member's starting byte offset."""
        found, offset = [], 0
        for _, member in self.program.structs[struct_name]:
            alignment, size = self.size(member)
            offset = -(-offset // alignment) * alignment
            found.append(offset)
            offset += size
        return found

    def resolve(self, text: str) -> GAType | str:
        if text in self.program.gatypes:
            return self.program.gatypes[text]
        return self.scalar if text == "f32" else text

    def dtype(self, kind: GAType | str) -> type:
        name, arguments = template(kind if isinstance(kind, str) else "f32")
        scalar = arguments[0] if arguments else name
        return {"u32": np.uint32, "i32": np.int32}.get(scalar, np.float32)

    def pack(self, kind: GAType | str, value: Any) -> bytes:
        """Serialize a single value to padded storage-buffer bytes, casting `Extensor` inputs to the
        declared `GAType` and aligning struct members in declaration order."""
        form, detail = self.shape(kind)
        alignment, size = self.size(kind)
        if form == "array":
            # Fixed-length arrays pack each entry sequentially along the leading axis.
            element = self.resolve(detail[0])
            data = b"".join(self.pack(element, entry) for entry in value)
        elif isinstance(value, Extensor):
            data = coefficients(value, detail if form == "algebra" else self.scalar)
        elif form == "struct":
            data = bytearray()
            for member, member_type in self.program.structs[detail]:
                member_alignment, _ = self.size(member_type)
                data += bytes(-len(data) % member_alignment) + self.pack(member_type, value[member])
            data = bytes(data)
        else:
            data = np.asarray(value, dtype=self.dtype(detail)).tobytes()
        return data + bytes(size - len(data)) if size > len(data) else data

    def array(self, kind: GAType | str, value: Any) -> bytes:
        """Serialize a batched input to contiguous element strides for a runtime-sized storage array,
        interleaving dict-of-batches columns across struct members."""
        if isinstance(value, Extensor):
            form, detail = self.shape(kind)
            return coefficients(value, self.resolve(detail[0]) if form == "array" else kind)
        if kind in self.program.structs:
            # Struct arrays pack each member batch as a byte column at its aligned struct offset.
            stride = self.stride(kind)
            columns = [np.frombuffer(self.array(member_type, value[member]), dtype=np.uint8)
                       for member, member_type in self.program.structs[kind]]
            count = len(columns[0]) // self.stride(self.program.structs[kind][0][1])
            data = np.zeros((count, stride), dtype=np.uint8)
            for column, (_, member_type), offset in zip(columns, self.program.structs[kind], self.offsets(kind)):
                _, size = self.size(member_type)
                data[:, offset:offset + size] = column.reshape(count, self.stride(member_type))[:, :size]
            return data.tobytes()
        # Numeric elements pad each row out to the aligned array stride.
        _, size = self.size(kind)
        values = np.asarray(value, dtype=self.dtype(kind)).reshape(-1, size // 4)
        padded = np.zeros((values.shape[0], self.stride(kind) // 4), dtype=values.dtype)
        padded[:, :values.shape[1]] = values
        return padded.tobytes()


def coefficients(extensor: Extensor, target: GAType) -> bytes:
    """Casting `extensor` to `target` when needed flattens its blade coefficients into `float32`
    bytes in kernel order."""
    if extensor.gatype != target:
        extensor = extensor.cast(target)
    return np.asarray(extensor.kernel, dtype=np.float32).tobytes()


class Shader:
    """Compiled WGSL pipelines pair a dialect `Program` with its `Layout`, bind-group layout, and
    cached GPU buffers and textures across dispatches."""

    def __init__(self, source: str, algebra: Algebra, types: Mapping[str, GAType], constants: Mapping[str, Any]) -> None:
        self.program = compile(source, algebra, types, constants)
        self.context = NumpyContext(algebra)
        self.layout = Layout(self.program, algebra.gatype.scalar())
        self.device = device()
        self.module = self.device.create_shader_module(code=self.program.code)
        # Float textures filter where the device allows it, and store at half precision where it does not.
        self.filterable = "float32-filterable" in self.device.features
        self.bind_layout = self.device.create_bind_group_layout(entries=[self.layout_entry(binding) for binding in self.program.bindings])
        self.pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[self.bind_layout])
        self.pipelines = [(self.device.create_compute_pipeline(layout=self.pipeline_layout, compute={"module": self.module, "entry_point": name}), size)
                          for name, size in self.program.entry_points]
        self.sampler = self.device.create_sampler(mag_filter="linear", min_filter="linear",
                                                  address_mode_u="repeat", address_mode_v="clamp-to-edge")
        self.kept = {}

    def layout_entry(self, binding: Binding) -> dict[str, Any]:
        """Bind-group layout entries expose resources across compute, fragment, and (when read-only)
        vertex stages."""
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

    def __call__(self, shape: tuple[int, ...], **values: Any) -> dict[str, Any]:
        """Dispatching all `@compute` entry points over `shape` maps the last axis of `shape` to `id.x`
        (so `(rows, columns)` addresses `(column, row)` in `id.xy` and populates `pixels` as
        `(columns, rows)`), returning every unprovided storage array by name."""
        encoder = self.device.create_command_encoder()
        outputs = self.encode(encoder, shape, (), values)
        return self.read(encoder, outputs, shape)

    def encode(self, encoder: Any, shape: tuple[int, ...], key: tuple, values: Mapping[str, Any]) -> dict[str, tuple]:
        """Recording zero-clears and compute dispatches onto `encoder` binds inputs into buffers
        cached under `key` and returns the output buffer descriptors for `read`."""
        group, outputs = self.bind(shape, values, key)
        # The written arrays start at zero, whatever the last call left in them.
        for buffer, _, element_size in outputs.values():
            encoder.clear_buffer(buffer, 0, int(np.prod(shape)) * element_size)
        # Reversing `shape` maps the last grid axis (image columns) to workgroup axis `x`.
        threads = (*shape[::-1], *(1,) * (3 - len(shape)))
        for pipeline, size in self.pipelines:
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(pipeline)
            compute.set_bind_group(0, group)
            compute.dispatch_workgroups(-(-threads[0] // size[0]), -(-threads[1] // size[1]), -(-threads[2] // size[2]))
            compute.end()
        return outputs

    def bind(self, shape: tuple[int, ...], values: Mapping[str, Any], key: tuple) -> tuple[Any, dict]:
        """Uploading declared uniforms, storage arrays, and textures into buffers cached under `key`
        builds the bind group alongside output storage buffers sized to `shape`. Distinct passes
        recorded on the same command encoder require distinct `key` prefixes because queue writes
        execute before the encoder submits."""
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
                # An empty array binds one element of nothing, which the shader, told it holds nothing, never reads.
                data = self.layout.array(element, values[binding.name]) or bytes(self.layout.stride(element))
                resource = self.buffer(key + ("storage", binding.name), data)
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

    def read(self, encoder: Any, outputs: Mapping[str, tuple], shape: tuple[int, ...]) -> dict[str, Any]:
        """Submitting `encoder` copies each output storage buffer to a host-mappable staging buffer
        and unpacks the results to `shape`."""
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

    def unpack(self, kind: GAType | str, raw: np.ndarray, shape: tuple[int, ...]) -> Any:
        """Decode raw `[n_elements, stride]` bytes into a `float64` `Extensor` batch for non-scalar
        `GAType` outputs, a dict of unpacked fields for structs, or a shaped numeric array."""
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
        """Return the cached GPU buffer for `role`, allocating on first use and reallocating only
        when `size` exceeds the existing capacity."""
        buffer = self.kept.get(role)
        if buffer is None or buffer.size < size:
            buffer = self.device.create_buffer(size=max(size, 16), usage=usage)
            self.kept[role] = buffer
        return buffer

    def buffer(self, role: tuple, data: bytes) -> dict[str, Any]:
        """Writing `data` into the storage buffer cached for `role` returns a binding descriptor
        spanning `len(data)` bytes."""
        import wgpu
        buffer = self.buffer_kept(role, len(data), wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        self.device.queue.write_buffer(buffer, 0, data)
        return {"buffer": buffer, "offset": 0, "size": len(data)}

    def surface(self, role: tuple, shape: tuple[int, int], texture_format: str, usage: int) -> Any:
        """Return the cached 2D GPU texture for `role` with width `shape[1]` and height `shape[0]`."""
        if role not in self.kept:
            rows, columns = shape
            self.kept[role] = self.device.create_texture(size=(columns, rows, 1), format=texture_format, usage=usage)
        return self.kept[role]

    def texture(self, key: tuple, binding: Binding, image: np.ndarray) -> Any:
        """Upload an image array `[rows, columns, channels]` as `rgba32float` or `rgba16float`,
        padding missing channels up to four (with alpha `1.0`) and reusing the GPU texture when
        passed the identical `image` array object on subsequent calls."""
        import wgpu
        written = self.kept.get(key + ("written", binding.name))
        if written is not None and written[0] is image:
            return written[1]
        source = image
        image = np.asarray(image, dtype=np.float32)
        padded = np.ones(image.shape[:2] + (4,), dtype=np.float32 if self.filterable else np.float16)
        padded[..., :image.shape[2]] = image
        height, width = padded.shape[:2]
        texture = self.surface(key + ("texture", binding.name, (height, width)), (height, width),
                               "rgba32float" if self.filterable else "rgba16float",
                               wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.device.queue.write_texture({"texture": texture}, padded.tobytes(),
                                        {"bytes_per_row": width * padded.itemsize * 4, "rows_per_image": height}, (width, height, 1))
        self.kept[key + ("written", binding.name)] = (source, texture)
        return texture
