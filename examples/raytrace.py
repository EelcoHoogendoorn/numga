"""Ray tracing solid quadrics in PGA3D, for the examples' renders: parallel camera rays, where each ray enters
a quadric, the normal there from the polar plane, and shading by a lamp.

A quadric is a map from points to planes, `Plane <- Point`, zero on its surface where `quadric(point) & point`
vanishes and negative inside. Bound to a ray in both slots it is a quadratic in the distance along the ray, so a
ray meets it where that quadratic has a root.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple
import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA3D as ga
from numga.gatype import GAType

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Quadric = ga.gatype((Plane, Point))                                   # Plane <- Point
# The coordinate planes.
AXES = stack((mv.x, mv.y, mv.z))                                      # [3] Plane
# The lamp's ambient share, its specular share and the specular exponent.
AMBIENT, SPECULAR, SHININESS = 0.3, 0.3, 40
# Traced pixels along each side of a shown one.
SUPERSAMPLE = 2


def ellipsoid(semi_axes: np.ndarray) -> Quadric:
    """The solid ellipsoid about the origin with the given semi-axes along x, y and z on the last axis: each
    coordinate plane weighted by the inverse square of its semi-axis, less the plane at infinity."""
    weights = 1 / np.asarray(semi_axes, dtype=float) ** 2
    return (AXES * (AXES & Point) * weights).sum(axis=-1) - mv.w * (mv.w & Point)


def quadric(matrix: np.ndarray, centre: np.ndarray) -> Quadric:
    """The solid quadric `(x - centre) @ matrix @ (x - centre) <= 1` for symmetric matrices `[..., 3, 3]` and
    centres `[..., 3]`, as a map from points to planes."""
    at_origin = (AXES[:, None] * (AXES & Point)[None, :] * matrix).sum(axis=(-1, -2)) - mv.w * (mv.w & Point)
    shift = ((stack((mv.xw, mv.yw, mv.zw)) * np.asarray(centre, dtype=float)).sum(axis=-1) / 2).exp()
    return shift >> at_origin(shift << Point)


def view_axes(elevation: float, azimuth: float) -> np.ndarray:
    """The unit directions screen right, screen up and toward the viewer `[3, 3]`, for a view from the given
    elevation and azimuth in degrees with z up."""
    tilt, turn = np.radians(elevation), np.radians(azimuth)
    back = np.array([np.cos(tilt) * np.cos(turn), np.cos(tilt) * np.sin(turn), np.sin(tilt)])
    right = np.array([-np.sin(turn), np.cos(turn), 0.0])
    return np.stack([right, np.cross(back, right), back])


def framing(points: np.ndarray, elevation: float, azimuth: float, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """The centre and the half width and half height on the screen that frame points `[..., 3]` seen from the
    given elevation and azimuth, with a margin as a fraction of the larger side."""
    right, up, _ = view_axes(elevation, azimuth)
    screen = np.reshape(points, (-1, 3)) @ np.stack([right, up]).T
    low, high = screen.min(axis=0), screen.max(axis=0)
    half = (high - low) / 2 + margin * (high - low).max()
    return np.stack([right, up]).T @ ((low + high) / 2), half


def camera(elevation: float, azimuth: float, centre: np.ndarray, half: np.ndarray, pixels: int):
    """Parallel rays for an image of the given half width and half height about the centre, `pixels` across,
    seen from the given elevation and azimuth in degrees with z up: one origin per pixel on a screen far
    behind the scene, the shared heading, a direction, and the unit direction toward the lamp, above the
    viewer's left shoulder."""
    right, up, back = view_axes(elevation, azimuth)
    rows = max(1, int(round(pixels * half[1] / half[0])))
    across = np.linspace(-half[0], half[0], pixels)
    down = np.linspace(half[1], -half[1], rows)
    screen = centre + across[None, :, None] * right + down[:, None, None] * up + 20 * max(half) * back
    return mv.point(screen), mv.direction(-back), lamp(elevation, azimuth)


def lamp(elevation: float, azimuth: float) -> np.ndarray:
    """The unit direction toward the lamp, above the left shoulder of a viewer at the given elevation and
    azimuth."""
    right, up, back = view_axes(elevation, azimuth)
    toward = back - 0.4 * right + 0.9 * up
    return toward / np.linalg.norm(toward)


def ellipsoid_points(matrix: np.ndarray, centre: np.ndarray, n_samples: int) -> np.ndarray:
    """Points `[..., samples, 3]` on the surfaces of the solid quadrics `(x - centre) @ matrix @ (x - centre) <= 1`
    of positive definite matrices, for framing: the unit sphere, sampled on a spiral, carried by the inverse
    square root of each matrix."""
    index = np.arange(n_samples) + 0.5
    height = 1 - 2 * index / n_samples
    turn = np.pi * (1 + 5**0.5) * index
    sphere = np.stack([np.sqrt(1 - height**2) * np.cos(turn), np.sqrt(1 - height**2) * np.sin(turn), height], axis=-1)
    values, vectors = np.linalg.eigh(matrix)
    root = vectors @ (vectors / np.sqrt(values)[..., None, :]).swapaxes(-1, -2)
    return np.asarray(centre)[..., None, :] + sphere @ root.swapaxes(-1, -2)


def hits(surface: Quadric, origins: Point, heading: Point) -> tuple[np.ndarray, Point]:
    """The distance along each ray to where it enters the solid quadric, infinite where it misses or where the
    entry lies behind the origin, and the points hit. At the entering root the quadratic falls through zero."""
    a = (surface(heading) & heading).to_array()
    b = (surface(heading) & origins).to_array()
    c = (surface(origins) & origins).to_array()
    discriminant = b * b - a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = (-b - np.sqrt(np.clip(discriminant, 0, None))) / a
    distance = np.where((discriminant >= 0) & (a > 0) & (distance > 0), distance, np.inf)
    return distance, origins + heading * mv.scalar(np.where(np.isfinite(distance), distance, 0.0)[..., None])


def coordinates(points: Point) -> np.ndarray:
    """Euclidean coordinates of points `[..., 3]`: their pairings with the coordinate planes, over their weights."""
    return ((AXES & points[..., None]) / (mv.w & points)[..., None]).to_array()


def normals(surface: Quadric, points: Point) -> np.ndarray:
    """Unit outward normals `[..., 3]` of the quadric at points on it: the normals of their polar planes, read
    by the metric against the coordinate planes."""
    normal = (surface(points)[..., None] | AXES).to_array()
    return normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)


def shade(colour: np.ndarray, normal: np.ndarray, lamp: np.ndarray, toward_viewer: np.ndarray, lit: np.ndarray) -> np.ndarray:
    """Colours `[..., 3]` lit by the lamp: an ambient share everywhere, diffuse light and a highlight where the
    surface is lit and faces the lamp."""
    facing = np.clip(normal @ lamp, 0, None) * lit
    mirrored = 2 * (normal @ lamp)[..., None] * normal - lamp
    highlight = np.clip(mirrored @ toward_viewer, 0, None) ** SHININESS * (facing > 0)
    return colour * (AMBIENT + (1 - AMBIENT) * facing)[..., None] + SPECULAR * highlight[..., None]


def scene(surfaces: Quadric, colours: np.ndarray, view, background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An RGB image `[pixels, pixels, 3]` of solid quadrics on the last axis, each pixel the nearest one hit,
    in its colour `[surfaces, 3]`, shaded by the lamp, and the depth of that hit, infinite on the background."""
    origins, heading, lamp = camera(*view)
    distance, hit = hits(surfaces, origins[..., None], heading)              # [pixels, pixels, surfaces]
    nearest = distance.argmin(axis=-1)
    depth = np.take_along_axis(distance, nearest[..., None], axis=-1)[..., 0]
    normal = np.take_along_axis(normals(surfaces, hit), nearest[..., None, None], axis=-2)[..., 0, :]
    toward_viewer = view_axes(*view[:2])[2]
    rgb = shade(np.asarray(colours)[nearest], normal, lamp, toward_viewer, 1.0)
    return np.where(np.isfinite(depth)[..., None], rgb, background), depth


def segments(starts: np.ndarray, ends: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Thin solid ellipsoids along segments between points `[..., 3]`, reaching from end to end with the given
    radius across: their matrices and centres, for `quadric`."""
    starts, ends = np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)
    length = np.linalg.norm(ends - starts, axis=-1)[..., None, None]
    unit = (ends - starts) / length[..., 0]
    along = unit[..., :, None] * unit[..., None, :]
    return along / (length / 2) ** 2 + (np.eye(3) - along) / radius**2, (starts + ends) / 2


def downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """An image `[rows, columns, channels]` traced at `factor` times the pixels it is shown at, each shown
    pixel the mean of its block of traced ones: supersampling, which smooths the edges."""
    rows, columns = (np.array(image.shape[:2]) // factor) * factor
    blocks = image[:rows, :columns].reshape(rows // factor, factor, columns // factor, factor, -1)
    return blocks.mean(axis=(1, 3))


def spheres(centres: np.ndarray, radius: float, colours: np.ndarray, view) -> tuple[np.ndarray, np.ndarray]:
    """An RGB image of many small spheres about centres `[n, 3]`, in colours `[n, 3]`, shaded by the lamp,
    and the depth along the rays of each pixel's nearest hit, infinite where none is: for parallel rays a
    sphere covers a disc on the screen, where its depth and normal have a closed form, so each sphere is
    drawn into its own pixels only."""
    elevation, azimuth, centre, half, pixels = view
    right, up, back = view_axes(elevation, azimuth)
    rows = max(1, int(round(pixels * half[1] / half[0])))
    scale = pixels / (2 * half[0])
    # Pixel coordinates, and the distance from the screen, of each centre.
    offset = np.asarray(centres) - centre
    column = (offset @ right + half[0]) * scale
    row = (half[1] - offset @ up) * scale
    distance = 20 * max(half) - offset @ back
    depth = np.full((rows, pixels), np.inf)
    normal = np.zeros((rows, pixels, 3))
    owner = np.zeros((rows, pixels), dtype=int)
    reach = int(np.ceil(radius * scale)) + 1
    for index in np.argsort(distance)[::-1]:
        low_row, low_column = max(int(row[index]) - reach, 0), max(int(column[index]) - reach, 0)
        rr, cc = np.mgrid[low_row:min(int(row[index]) + reach + 1, rows), low_column:min(int(column[index]) + reach + 1, pixels)]
        across = (cc - column[index]) / scale, (row[index] - rr) / scale
        inside = across[0] ** 2 + across[1] ** 2 < radius**2
        towards = np.sqrt(np.clip(radius**2 - across[0] ** 2 - across[1] ** 2, 0, None))
        hit = distance[index] - towards
        nearer = inside & (hit < depth[rr, cc])
        depth[rr[nearer], cc[nearer]] = hit[nearer]
        owner[rr[nearer], cc[nearer]] = index
        normal[rr[nearer], cc[nearer]] = ((across[0][..., None] * right + across[1][..., None] * up + towards[..., None] * back) / radius)[nearer]
    hit = np.isfinite(depth)
    rgb = np.zeros((rows, pixels, 3))
    rgb[hit] = shade(np.asarray(colours).reshape(-1, 3)[owner[hit]], normal[hit], lamp(elevation, azimuth), back, 1.0)
    return rgb, depth


# --- WebGPU / Metal Compute Raytracer & Transpiler ------------------------------------

BLADE_ALIASES = {
    "ywz": "x", "wxz": "y", "xwy": "z", "xyz": "w",
    "1": "s",
}

def clean_blade(blade: str) -> str:
    return BLADE_ALIASES.get(blade, blade)


class NumgaWGSLTranspiler:
    """Translates a Numga GA shader dialect into standard WGSL via type inference.

    Nullary 4D types (Point, Plane) map to vec4<f32>, providing native SIMD for linear
    combinations (a + b, a - b, a * s) and swizzles (.x, .y, .z, .w).
    Unary maps (Quadric) map to mat4x4<f32>.
    Bilinear GA products (&, |, ^) and map applications are unrolled via Numga's SymbolicKernel.
    """

    def __init__(self, algebra=ga):
        self.algebra = algebra
        self.types: Dict[str, GAType] = {}
        self.helper_functions: Dict[str, str] = {}
        self.function_return_types: Dict[str, str] = {}
        # Automatically register Scalar (0-grade) in the GA type system
        self.register_type("Scalar", self.algebra.gatype.scalar())

    def register_type(self, name: str, gatype: GAType):
        self.types[name] = gatype

    def emit_structs(self) -> str:
        code_lines = []
        for name, gatype in self.types.items():
            if gatype.arity == 0:
                dim = len(gatype.output_subspace)
                if dim == 1:
                    code_lines.append(f"alias {name} = f32;")
                elif dim <= 4:
                    code_lines.append(f"alias {name} = vec{dim}<f32>;")
                else:
                    fields = [f"    {clean_blade(b)}: f32," for b in gatype.output_subspace.blade_names]
                    code_lines.append(f"struct {name} {{\n" + "\n".join(fields) + "\n};")
            elif gatype.arity == 1:
                out_sub, in_sub = gatype.subspaces
                code_lines.append(f"struct {name} {{\n    matrix: mat{len(in_sub)}x{len(out_sub)}<f32>,\n}};")
        return "\n".join(code_lines) + "\n"

    def get_or_create_unary_apply(self, map_type_name: str, in_type_name: str) -> Tuple[str, str]:
        fn_name = f"ga_apply_{map_type_name.lower()}_{in_type_name.lower()}"
        map_type = self.types[map_type_name]
        out_sub, in_sub = map_type.subspaces
        out_name = [k for k, v in self.types.items() if v.output_subspace == out_sub and v.arity == 0][0]
        self.function_return_types[fn_name] = out_name

        if fn_name not in self.helper_functions:
            self.helper_functions[fn_name] = (
                f"fn {fn_name}(m: {map_type_name}, p: {in_type_name}) -> {out_name} {{\n"
                f"    return m.matrix * p;\n"
                f"}}\n"
            )
        return fn_name, out_name

    def get_or_create_binary_op(self, op_symbol: str, in1_type_name: str, in2_type_name: str) -> Tuple[str, str]:
        op_map = {
            "&": ("meet", self.algebra.operator.regressive),
            "|": ("dot", self.algebra.operator.inner),
            "^": ("wedge", self.algebra.operator.wedge),
        }
        op_prefix, op_factory = op_map[op_symbol]
        fn_name = f"ga_{op_prefix}_{in1_type_name.lower()}_{in2_type_name.lower()}"
        
        in1_type = self.types[in1_type_name]
        in2_type = self.types[in2_type_name]
        op_extensor = op_factory(in1_type, in2_type)

        kernel = op_extensor.kernel._values
        out_sub = op_extensor.gatype.output_subspace
        in1_sub = op_extensor.gatype.subspaces[1]
        in2_sub = op_extensor.gatype.subspaces[2]

        out_dim, in1_dim, in2_dim = kernel.shape

        def accessor(var_name: str, idx: int, sub) -> str:
            if len(sub) <= 4:
                return f"{var_name}.{('x', 'y', 'z', 'w')[idx]}"
            return f"{var_name}.{clean_blade(sub.blade_names[idx])}"

        if out_dim == 1:
            out_type_name = "Scalar" if "Scalar" in self.types else "f32"
            self.function_return_types[fn_name] = out_type_name
            if fn_name not in self.helper_functions:
                terms = []
                for j in range(in1_dim):
                    for k in range(in2_dim):
                        coeff = kernel[0, j, k]
                        if coeff != 0:
                            sign = " + " if coeff > 0 else " - "
                            mag = f"{abs(coeff)} * " if abs(coeff) != 1 else ""
                            a_comp = accessor("a", j, in1_sub)
                            b_comp = accessor("b", k, in2_sub)
                            terms.append(f"{sign}{mag}{a_comp} * {b_comp}")
                expr = "".join(terms).lstrip(" + ") if terms else "0.0"
                self.helper_functions[fn_name] = (
                    f"fn {fn_name}(a: {in1_type_name}, b: {in2_type_name}) -> {out_type_name} {{\n"
                    f"    return {expr};\n"
                    f"}}\n"
                )
        else:
            matching = [k for k, v in self.types.items() if v.output_subspace == out_sub and v.arity == 0]
            if matching:
                out_type_name = matching[0]
            elif out_dim <= 4:
                out_type_name = f"vec{out_dim}<f32>"
            else:
                same_dim = [k for k, v in self.types.items() if len(v.output_subspace) == out_dim and v.arity == 0]
                out_type_name = same_dim[0] if same_dim else f"vec{out_dim}<f32>"
            self.function_return_types[fn_name] = out_type_name
            if fn_name not in self.helper_functions:
                expr_components = []
                for i in range(out_dim):
                    terms = []
                    for j in range(in1_dim):
                        for k in range(in2_dim):
                            coeff = kernel[i, j, k]
                            if coeff != 0:
                                sign = " + " if coeff > 0 else " - "
                                mag = f"{abs(coeff)} * " if abs(coeff) != 1 else ""
                                a_comp = accessor("a", j, in1_sub)
                                b_comp = accessor("b", k, in2_sub)
                                terms.append(f"{sign}{mag}{a_comp} * {b_comp}")
                    expr = "".join(terms).lstrip(" + ") if terms else "0.0"
                    expr_components.append(expr)
                joined = ", ".join(expr_components)
                self.helper_functions[fn_name] = (
                    f"fn {fn_name}(a: {in1_type_name}, b: {in2_type_name}) -> {out_type_name} {{\n"
                    f"    return {out_type_name}({joined});\n"
                    f"}}\n"
                )

        return fn_name, out_type_name

    def transpile(self, source_wgsl: str) -> str:
        """Infer types and rewrite GA expressions without macro wrappers or coordinate unpacking."""
        env: Dict[str, str] = {}

        struct_defs: Dict[str, Dict[str, str]] = {}
        for sm in re.finditer(r"struct\s+(\w+)\s*\{([^}]+)\}", source_wgsl):
            sname, sbody = sm.group(1), sm.group(2)
            struct_defs[sname] = {}
            for fm in re.finditer(r"(\w+)\s*:\s*(\w+)", sbody):
                struct_defs[sname][fm.group(1)] = fm.group(2)

        for m in re.finditer(r"(?:var(?:<[^>]+>)?|let)\s+(\w+)\s*:\s*([^=;,\n]+)", source_wgsl):
            vname, vtype = m.group(1), m.group(2).strip()
            arr_m = re.match(r"array<(\w+)>", vtype)
            if arr_m and arr_m.group(1) in self.types:
                env[vname] = f"array<{arr_m.group(1)}>"
            elif vtype in self.types:
                env[vname] = vtype
            elif vtype in struct_defs:
                for fname, ftype in struct_defs[vtype].items():
                    if ftype in self.types:
                        env[f"{vname}.{fname}"] = ftype

        for fn_m in re.finditer(r"fn\s+\w+\s*\(([^)]*)\)", source_wgsl):
            for param in fn_m.group(1).split(","):
                if ":" in param:
                    pname, ptype = param.split(":")[:2]
                    pname, ptype = pname.strip(), ptype.strip()
                    if ptype in self.types:
                        env[pname] = ptype

        def resolve_expr_type(expr: str) -> str | None:
            expr = expr.strip()
            if expr.startswith("(") and expr.endswith(")"):
                return resolve_expr_type(expr[1:-1])
            idx_m = re.match(r"^([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)\[.+\]$", expr)
            if idx_m:
                base_var = idx_m.group(1)
                if base_var in env and env[base_var].startswith("array<"):
                    return env[base_var][6:-1]
            call_m = re.match(r"^([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)\s*\((.*)\)$", expr)
            if call_m:
                callee = call_m.group(1)
                if callee in self.function_return_types:
                    return self.function_return_types[callee]
                if callee in env and env[callee] in self.types and self.types[env[callee]].arity == 1:
                    out_sub = self.types[env[callee]].subspaces[0]
                    return [k for k, v in self.types.items() if v.output_subspace == out_sub and v.arity == 0][0]
            return env.get(expr)

        transformed_lines = []
        atom_regex = r"(?:(?:[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)(?:\[[^\]]+\])?(?:\([^\(\)]*\))?|\([^\(\)]+\))"
        binary_op_pattern = re.compile(rf"({atom_regex})\s*([&|^])\s*({atom_regex})")

        for line in source_wgsl.splitlines():
            let_match = re.search(r"\b(?:let|var)\s+(\w+)\s*(?::\s*([^=;]+))?\s*=", line)
            if let_match:
                var_name, explicit_type = let_match.group(1), let_match.group(2)
                if explicit_type:
                    explicit_type = explicit_type.strip()
                    if explicit_type in self.types:
                        env[var_name] = explicit_type

            for var_path, var_type in sorted(env.items(), key=lambda x: len(x[0]), reverse=True):
                escaped_path = re.escape(var_path)
                if var_type.startswith("array<"):
                    elem_type = var_type[6:-1]
                    if elem_type in self.types and self.types[elem_type].arity == 1:
                        arr_call_pattern = re.compile(rf"(?<![\w.]){escaped_path}\s*\[([^\]]+)\]\s*\(\s*([\w.]+)\s*\)")
                        def repl_arr_call(cm):
                            idx_expr, arg_path = cm.group(1), cm.group(2)
                            arg_type = resolve_expr_type(arg_path)
                            if arg_type:
                                fn_name, out_type = self.get_or_create_unary_apply(elem_type, arg_type)
                                return f"{fn_name}({var_path}[{idx_expr}], {arg_path})"
                            return cm.group(0)
                        line = arr_call_pattern.sub(repl_arr_call, line)
                elif var_type in self.types and self.types[var_type].arity == 1:
                    call_pattern = re.compile(rf"(?<![\w.]){escaped_path}\s*\(\s*([\w.]+)\s*\)")
                    def repl_call(cm):
                        arg_path = cm.group(1)
                        arg_type = resolve_expr_type(arg_path)
                        if arg_type:
                            fn_name, out_type = self.get_or_create_unary_apply(var_type, arg_type)
                            return f"{fn_name}({var_path}, {arg_path})"
                        return cm.group(0)
                    line = call_pattern.sub(repl_call, line)

            while True:
                m = binary_op_pattern.search(line)
                if not m:
                    break
                left_expr, op_char, right_expr = m.group(1), m.group(2), m.group(3)
                t_left = resolve_expr_type(left_expr)
                t_right = resolve_expr_type(right_expr)
                if t_left in self.types and t_right in self.types:
                    fn_name, out_type = self.get_or_create_binary_op(op_char, t_left, t_right)
                    span_start, span_end = m.span()
                    line = line[:span_start] + f"{fn_name}({left_expr}, {right_expr})" + line[span_end:]
                else:
                    break

            transformed_lines.append(line)

        structs_code = self.emit_structs()
        helpers_code = "\n".join(self.helper_functions.values())
        body_code = "\n".join(transformed_lines)

        return (
            "// --- Generated Numga WGSL Header ---\n"
            f"{structs_code}\n"
            "// --- Generated Numga Operator Functions ---\n"
            f"{helpers_code}\n"
            "// --- Shader Body ---\n"
            f"{body_code}"
        )


_wgpu_device = None

def get_wgpu_device():
    """Lazily initialize and cache the default WebGPU device on Apple Silicon Metal or Vulkan."""
    global _wgpu_device
    if _wgpu_device is None:
        import wgpu
        adapter = wgpu.gpu.request_adapter_sync()
        _wgpu_device = adapter.request_device_sync()
    return _wgpu_device


class WGPUCompute:
    """Headless WebGPU compute runner for shaders written in the Numga GA dialect.

    Reflects automatically on shader bindings (@group(0) @binding(...)) and structs,
    converting Extensors, matrices, and uniform dicts directly into GPU buffers
    and returning the rendered array without any manual buffer or dispatch plumbing.
    """

    def __init__(self, shader_source: str, algebra=ga, types: Dict[str, GAType] = None):
        import wgpu
        self.algebra = algebra
        self.transpiler = NumgaWGSLTranspiler(algebra)
        if types:
            for name, gatype in types.items():
                self.transpiler.register_type(name, gatype)
        self.wgsl_code = self.transpiler.transpile(shader_source)
        self.device = get_wgpu_device()
        self.shader_module = self.device.create_shader_module(code=self.wgsl_code)
        self.pipeline = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": self.shader_module, "entry_point": "main"}
        )
        self._buffers: dict[str, wgpu.GPUBuffer] = {}
        self._readback: dict[tuple[int, int], wgpu.GPUBuffer] = {}
        self._parse_shader()

    def _parse_shader(self):
        """Reflect on struct definitions, binding locations, and workgroup size."""
        # Struct definitions
        self.structs: dict[str, list[tuple[str, str]]] = {}
        for sm in re.finditer(r"struct\s+(\w+)\s*\{([^}]+)\}", self.wgsl_code):
            sname = sm.group(1)
            fields = []
            for line in sm.group(2).splitlines():
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                fm = re.match(r"(\w+)\s*:\s*([^,;]+)", line)
                if fm:
                    fields.append((fm.group(1), fm.group(2).strip()))
            self.structs[sname] = fields

        # Bindings: @group(0) @binding(i) var<usage> name: type;
        self.bindings: list[dict] = []
        b_pat = re.compile(r"@group\((\d+)\)\s*@binding\((\d+)\)\s*var(?:<([^>]+)>)?\s+(\w+)\s*:\s*([^;]+);")
        for m in b_pat.finditer(self.wgsl_code):
            self.bindings.append({
                "group": int(m.group(1)),
                "binding": int(m.group(2)),
                "usage": (m.group(3) or "").strip(),
                "name": m.group(4).strip(),
                "type": m.group(5).strip(),
            })
        self.bindings.sort(key=lambda b: (b["group"], b["binding"]))

        # Workgroup size
        wg_m = re.search(r"@workgroup_size\((\d+)(?:,\s*(\d+))?(?:,\s*(\d+))?\)", self.wgsl_code)
        if wg_m:
            gx = int(wg_m.group(1))
            gy = int(wg_m.group(2)) if wg_m.group(2) else 1
            gz = int(wg_m.group(3)) if wg_m.group(3) else 1
            self.workgroup_size = (gx, gy, gz)
        else:
            self.workgroup_size = (8, 8, 1)

    def _align(self, offset: int, alignment: int) -> int:
        return (offset + alignment - 1) & ~(alignment - 1)

    def _pack_struct(self, struct_name: str, values: dict) -> bytes:
        """Pack dictionary values into a binary buffer matching WGSL std430 layout."""
        fields = self.structs.get(struct_name, [])
        buf = bytearray()
        offset = 0

        for fname, ftype in fields:
            val = values[fname]
            if isinstance(val, (int, float, np.floating, np.integer)):
                val_arr = np.array([val], dtype=np.float32)
                align = 4
            elif hasattr(val, "kernel"):
                arr = np.asarray(val.kernel).astype(np.float32)
                if getattr(val, "gatype", None) and val.gatype.arity == 1:
                    arr = arr.swapaxes(-1, -2)
                val_arr = arr.flatten()
                align = 16 if len(val_arr) > 2 else (8 if len(val_arr) == 2 else 4)
            elif isinstance(val, (tuple, list, np.ndarray)):
                arr = np.asarray(val, dtype=np.float32)
                if arr.ndim >= 2 and arr.shape[-1] == 4 and arr.shape[-2] == 4 and ("mat4x4" in ftype or "Quadric" in ftype):
                    arr = arr.swapaxes(-1, -2)
                val_arr = arr.flatten()
                align = 16 if len(val_arr) > 2 else (8 if len(val_arr) == 2 else 4)
            else:
                val_arr = np.array([val], dtype=np.float32)
                align = 4

            aligned_offset = self._align(offset, align)
            if aligned_offset > offset:
                buf.extend(b"\x00" * (aligned_offset - offset))
                offset = aligned_offset

            raw_bytes = val_arr.tobytes()
            buf.extend(raw_bytes)
            offset += len(raw_bytes)

        # Pad struct size to multiple of 16 bytes per WGSL rules
        final_size = self._align(offset, 16)
        if final_size > offset:
            buf.extend(b"\x00" * (final_size - offset))
        return bytes(buf)

    def _value_to_bytes(self, b_desc: dict, val) -> bytes:
        """Serialize an Extensor, array, struct dict, or scalar into GPU bytes."""
        usage = b_desc["usage"]
        type_str = b_desc["type"]

        if isinstance(val, bytes):
            return val
        if "uniform" in usage and isinstance(val, dict):
            return self._pack_struct(type_str, val)
        if hasattr(val, "kernel"):
            arr = np.asarray(val.kernel)
            if val.gatype.arity == 1:
                # Transpose 2D slices to column-major order for WGSL matMxN
                arr = arr.swapaxes(-1, -2)
            elif val.gatype.arity == 0 and arr.shape[-1] < 4 and ("vec4" in type_str or "Point" in type_str or "Plane" in type_str):
                pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, 4 - arr.shape[-1])]
                arr = np.pad(arr, pad_width)
            return arr.astype(np.float32).tobytes()
        if isinstance(val, np.ndarray):
            if val.ndim >= 2 and val.shape[-1] == 4 and val.shape[-2] == 4 and ("mat4x4" in type_str or "Quadric" in type_str):
                val = val.swapaxes(-1, -2)
            return val.astype(np.float32).tobytes()
        if isinstance(val, (tuple, list)):
            return np.asarray(val, dtype=np.float32).tobytes()
        if isinstance(val, (int, float, np.floating, np.integer)):
            return np.array([val], dtype=np.float32).tobytes()
        raise TypeError(f"Cannot serialize argument for binding {b_desc['name']}: {type(val)}")

    def __call__(self, width: int = None, height: int = None, **kwargs) -> np.ndarray:
        """Execute compute shader and return (H, W, 4) float32 RGBA array."""
        import wgpu

        # Infer dimensions from kwargs if not explicit
        if width is None or height is None:
            for v in kwargs.values():
                if isinstance(v, dict) and "resolution" in v:
                    width = int(v["resolution"][0])
                    height = int(v["resolution"][1])
                    break
        if width is None or height is None:
            width, height = 512, 512

        out_bytes = width * height * 4 * 4
        entries = []
        output_buffer = None

        for b in self.bindings:
            b_idx = b["binding"]
            b_name = b["name"]
            usage = b["usage"]

            if "storage" in usage and "read_write" in usage and b_name not in kwargs:
                # Output storage buffer
                buf_key = f"out_{width}_{height}"
                if buf_key not in self._buffers:
                    self._buffers[buf_key] = self.device.create_buffer(
                        size=out_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
                    )
                output_buffer = self._buffers[buf_key]
                entries.append({"binding": b_idx, "resource": {"buffer": output_buffer, "offset": 0, "size": out_bytes}})
            elif b_name in kwargs:
                data_bytes = self._value_to_bytes(b, kwargs[b_name])
                buf_usage = wgpu.BufferUsage.UNIFORM if "uniform" in usage else wgpu.BufferUsage.STORAGE
                buf_usage |= wgpu.BufferUsage.COPY_DST
                buf_key = f"in_{b_idx}_{len(data_bytes)}"
                if buf_key not in self._buffers:
                    self._buffers[buf_key] = self.device.create_buffer_with_data(data=data_bytes, usage=buf_usage)
                else:
                    self.device.queue.write_buffer(self._buffers[buf_key], 0, data_bytes)
                entries.append({"binding": b_idx, "resource": {"buffer": self._buffers[buf_key], "offset": 0, "size": len(data_bytes)}})
            else:
                raise KeyError(f"Missing input for shader binding '{b_name}' (@binding({b_idx}))")

        bind_group = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=entries,
        )

        rb_key = (width, height)
        if rb_key not in self._readback:
            self._readback[rb_key] = self.device.create_buffer(
                size=out_bytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
            )
        read_buf = self._readback[rb_key]

        encoder = self.device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.pipeline)
        cpass.set_bind_group(0, bind_group)
        cpass.dispatch_workgroups(
            (width + self.workgroup_size[0] - 1) // self.workgroup_size[0],
            (height + self.workgroup_size[1] - 1) // self.workgroup_size[1],
            1,
        )
        cpass.end()

        encoder.copy_buffer_to_buffer(output_buffer, 0, read_buf, 0, out_bytes)
        self.device.queue.submit([encoder.finish()])

        read_buf.map_sync(wgpu.MapMode.READ)
        raw = np.frombuffer(read_buf.read_mapped(), dtype=np.float32).reshape((height, width, 4)).copy()
        read_buf.unmap()
        return raw

    def render(self, uniform_data: np.ndarray, width: int, height: int, workgroup_size: tuple[int, int] = (8, 8)) -> np.ndarray:
        """Legacy render method for backward compatibility."""
        return self(width=width, height=height, scene=uniform_data)


WGPURaytracer = WGPUCompute





