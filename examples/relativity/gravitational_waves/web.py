"""Compiles a standalone WebGPU HTML page that flies a relativistic camera through the 600-cell triangulation of the
closed three-sphere as the exact lowest quadrupole gravitational wave stretches and squeezes three spring-bound
crystals.

The camera moves through the Einstein static universe rest frame along its local forward axis. Spacetime boosts
`((ahead & time).dual() * (rapidity / 2)).exp()` at the eye carry rest-frame pixel sightlines into the lattice rest
frame, swinging sideward guardrail decagons forward by relativistic aberration while dilating lattice time per camera
clock tick. Backward ray legs march in retarded time across the tetrahedral charts and blend crystal atom positions and
strain colours across adjacent ring-buffer moments.

Unsigned 16-bit indices pack the 600-cell's corners, facet neighbours, and guardrail memberships into the HTML file,
while a startup compute pass builds each tetrahedron's simplex gradients and guardrail bead and strut styles on the GPU
within WebGPU's guaranteed eight-storage-buffer limit. Keyboard, mouse, and touch controls steer yaw, pitch, roll,
relativistic thrust, time hold, wave amplitude, and ray leg length:

    python -c "from examples.relativity.gravitational_waves.web import page; page()"
    python -m examples.relativity.gravitational_waves.web
    python -c "from examples.relativity.gravitational_waves.web import bounded_crystals; bounded_crystals()"
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from functools import reduce
from operator import and_
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numga import stack

from examples.relativity.gravitational_waves import core, gpu, numerical, render, scenarios
from numga.topology import Topology
from numga.wgsl import Shader
from numga.wgsl.dialect import template
from numga.wgsl.page import encoded, packed, write_page

mv = core.mv

# Canvas size in CSS points; device resolution scales down dynamically when frame rate drops.
image_points = (480, 320)
start_leg, finest_leg, coarsest_leg = 0.35, 0.0125, 0.4
thrust_per_second, speed_per_scroll = 0.4, 0.0005
turn_per_second, turn_per_point = 1.2, 0.004
# Quadrupole amplitude matching the 600-cell's stepped wave strain at the eye for `scenarios.wave_swing`.
wave_swing = 0.112
# Forward drift speed and `xy`-plane spin per wave period for the soft, tuned, and stiff crystals.
crystal_drifts = np.array([0.1, 0, 0])                                           # [3]
crystal_spins = np.array([0, 0, 0.5])                                            # [3]
page_path = Path(__file__).parent / "gravitational_waves_webgpu.html"

display_source = '''
struct View {
    columns: u32,
    stride: u32,
    unused: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> samples: array<f32>;
@group(0) @binding(1) var<uniform> view: View;

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fragment(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let first = (u32(at.y) * view.columns + u32(at.x)) * view.stride;
    let colour = vec3<f32>(samples[first], samples[first + 1u], samples[first + 2u]);
    return vec4<f32>(clamp(colour, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
'''


# Startup compute pass building each tetrahedron's simplex gradients `Tet` and guardrail bead and strut styles
# `TetVisual` from the packed corner, neighbour, and guardrail tables.
setup_source = '''
struct Simplex {
    corners: vec4<u32>,
    neighbours: vec4<u32>,
    entries: vec4<u32>,
}

var<storage, read> simplices: array<Simplex>;
var<storage, read> places: array<Point>;
var<storage, read> rails: array<u32>;
var<storage, read_write> tets: array<Tet>;
var<storage, read_write> visuals: array<TetVisual>;

@compute @workgroup_size(64)
fn setup(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&simplices)) {
        return;
    }
    let simplex = simplices[index];
    var corners: array<Point, 4>;
    var beads: array<vec4<f32>, 4>;
    for (var corner = 0u; corner < 4u; corner++) {
        let vertex = simplex.corners[corner];
        corners[corner] = places[vertex];
        beads[corner] = vec4<f32>(rail_inks[rails[vertex]], bead_radius_squared);
    }
    var struts: array<vec4<f32>, 6>;
    for (var edge = 0u; edge < 6u; edge++) {
        let tail = simplex.corners[tet_edges[edge].x];
        let head = simplex.corners[tet_edges[edge].y];
        let side = vec4<f32>(rail_inks[rails[tail]], rail_radius_squared);
        struts[edge] = select(vec4<f32>(ghost_ink, ghost_radius_squared), side, rails[tail] == rails[head]);
    }
    var gradients: array<Plane, 4>;
    for (var corner = 0u; corner < 4u; corner++) {
        let face = faces[corner];
        let facet = corners[face.x] & corners[face.y] & corners[face.z];
        gradients[corner] = facet / (facet & corners[corner]);
    }
    tets[index] = Tet(simplex.neighbours, simplex.entries, gradients);
    visuals[index] = TetVisual(corners, beads, struts);
}
'''


def tables_source(topology: Topology, n_rails: int) -> str:
    """Opposite-facet corner indices of a tetrahedron and guardrail RGB hues format as constant WGSL `faces` and `rail_inks` arrays."""
    faces = ", ".join(f"vec3<u32>({first}u, {second}u, {third}u)" for first, second, third in topology.corners[3, 2].T)
    inks = ", ".join(f"vec3<f32>({float(red)!r}, {float(green)!r}, {float(blue)!r})"
                     for red, green, blue in render.hues(n_rails))
    return (f"\nconst faces = array<vec3<u32>, 4>({faces});\n"
            f"const rail_inks = array<vec3<f32>, {n_rails}>({inks});\n")


def setup_shader(topology: Topology, vertices: core.Point, n_rails: int, ghosts: float) -> Shader:
    """One compute invocation per tetrahedron builds simplex gradients `Tet` and guardrail bead and strut styles `TetVisual` on the GPU."""
    mean_edge = render.mean_chord(topology, vertices)
    return Shader(core.tet_struct_source + render.visual_source + tables_source(topology, n_rails) + setup_source,
                  core.ga, render.inside_types,
                  dict(ghost_ink=render.ghost_ink, ghost_radius_squared=(ghosts * mean_edge) ** 2,
                       rail_radius_squared=(render.rail_share * mean_edge) ** 2,
                       bead_radius_squared=(render.bead_scale * render.rail_share * mean_edge) ** 2))


def crystal_velocities(frames: core.Motor, atoms: core.Point) -> core.Point:
    """Initial tangent velocities `[V] Point-` combine each crystal's great-circle drift `crystal_drifts` with its `xy`-plane spin `crystal_spins`."""
    n_crystals = frames.shape[0]
    places = atoms.batch().reshape(n_crystals, 4)                               # [n_crystals, 4] Point-
    # Pull atom positions back into each crystal's frame to obtain local corner offsets at the origin.
    offsets = (frames[:, None].inverse() >> places).cast(core.Step)             # [n_crystals, 4] Point-
    drifts = mv.z.dual() * crystal_drifts[:, None]                              # [n_crystals, 1] Point-
    spins = offsets.commutator(mv.xy * (2 * np.pi * crystal_spins / gpu.wave_period)[:, None])   # [n_crystals, 4] Point-
    in_crystal = (drifts + spins).cast(core.Step)                               # [n_crystals, 4] Point-
    velocities = (core.frame(places).inverse() * frames[:, None]) >> in_crystal   # [n_crystals, 4] Point-
    return velocities.cast(core.Step).reshape(-1)


def crystals_start(bonds: Topology, atoms: core.Point, frames: core.Motor, frequency: float) -> dict:
    """Initial atom states, tuned spring constants, unstrained ring-buffer levels, and bounding balls pack the starting crystal buffers."""
    V, E, F, T = bonds.n_elements
    n_crystals = len(scenarios.tunings)
    stiffnesses = frequency ** 2 / 2 * scenarios.tunings                        # [n_crystals]
    dampings = scenarios.critical_share * np.sqrt(2 * stiffnesses)              # [n_crystals]
    round_steps = core.round_steps(bonds, atoms)                                # Point-[E]
    rest = round_steps.norm().batch().to_array()                                # [E, 1]
    unstrained = np.tile(plt.get_cmap(render.strain_map)(0.5), (E, 1))          # [E, 4]
    return dict(atoms=dict(place=atoms.batch(), velocity=crystal_velocities(frames, atoms)),
                springs=dict(rest=rest, stiffness=np.repeat(stiffnesses, E // n_crystals),
                             damping=np.repeat(dampings, E // n_crystals), length=rest),
                history=atoms.batch(), inks=unstrained, balls=render.crystal_balls(bonds, atoms[None]))


# --- plumbing -------------------------------------------------------------------------
# Camera struct fields copied each frame into the tracer and sightline uniform blocks.
copied_members = dict(tracer=(("place", "start_place"), ("tet", "start_tet"), ("right", "right"), ("up", "up"),
                              ("forward", "forward"), ("light", "light"), ("clock", "eye_time")),
                      sight=(("speed", "speed"),))
tiled_state = dict(history=gpu.n_levels, inks=gpu.n_levels)


def element(shader: Shader, name: str) -> object:
    """Storage binding `name` in `shader` resolves to its WGSL element layout descriptor."""
    return shader.layout.resolve(template(next(binding.type for binding in shader.program.bindings
                                               if binding.name == name))[1][0])


def storage_bytes(shaders: dict[str, Shader], name: str, value: object) -> bytes:
    """First shader binding `name` packs `value` into a raw storage-array byte buffer."""
    shader = next(shader for shader in shaders.values()
                  if any(binding.name == name for binding in shader.program.bindings))
    return shader.layout.array(element(shader, name), value)


def packed_table(shaders: dict[str, Shader], name: str, value: object) -> dict:
    """Storage table `name` encodes unsigned 16-bit index columns and 32-bit float columns in base64 with element offsets."""
    shader = next(shader for shader in shaders.values()
                  if any(binding.name == name for binding in shader.program.bindings))
    kind, layout = element(shader, name), shader.layout
    members = (dict(zip([member for member, _ in shader.program.structs[kind]],
                        zip(layout.offsets(kind), [member_type for _, member_type in shader.program.structs[kind]])))
               if kind in shader.program.structs else {name: (0, kind)})
    given = value if isinstance(value, dict) else {name: value}
    columns = []
    for member, numbers in given.items():
        offset, member_type = members[member]
        indices = layout.dtype(member_type) == np.uint32
        words = np.frombuffer(layout.array(member_type, numbers), dtype=np.uint32 if indices else np.float32)
        count = len(words) * 4 // layout.stride(member_type)
        columns.append(dict(offset=offset, width=len(words) // count, kind="u16" if indices else "f32",
                            data=encoded(words.astype(np.uint16 if indices else np.float32).tobytes())))
    return dict(count=count, stride=layout.stride(kind), columns=columns)


def unpacked_bytes(table: dict) -> bytes:
    """Base64 columns in `table` widen to 32-bit words at their element offsets just as the browser unpacks them."""
    words = np.zeros((table["count"], table["stride"] // 4), dtype=np.uint32)
    for column in table["columns"]:
        numbers = np.frombuffer(base64.b64decode(column["data"]), dtype=np.uint16 if column["kind"] == "u16" else np.float32)
        target = words if column["kind"] == "u16" else words.view(np.float32)
        target[:, column["offset"] // 4:column["offset"] // 4 + column["width"]] = numbers.reshape(table["count"], -1)
    return words.tobytes()


def bindings(shader: Shader, block: bytes) -> list[dict]:
    """Ordered buffer binding descriptors record read/write access and minimum byte sizes for `shader`."""
    return [dict(index=binding.index, kind=binding.kind, name=binding.name,
                 writable=binding.kind == "storage" and binding.access == "read_write",
                 least=len(block) if binding.kind == "uniforms" else shader.layout.stride(element(shader, binding.name)))
            for binding in shader.program.bindings]


def copies(advance: Shader, offsets: dict, pairs: tuple) -> list[list[int]]:
    """Byte offsets and sizes map `Camera` struct members in `advance` to uniform destinations in `offsets`."""
    members = dict(zip([member for member, _ in advance.program.structs["Camera"]], advance.layout.offsets("Camera")))
    kinds = dict(advance.program.structs["Camera"])
    return [[members[member], offsets[uniform], advance.layout.size(kinds[member])[1]] for member, uniform in pairs]


def scene(ghosts: float, eye: core.Motor, swing: float) -> dict:
    """Shaders, uniform dictionaries, 600-cell tables, initial camera and crystal state, and GPU output sizes assemble the WebGPU scene."""
    topology, vertices = scenarios.six_hundred_cell()
    V, E, F, T = topology.n_elements
    rails = scenarios.guardrails(vertices, scenarios.n_guardrail_sides)        # [12, n_guardrail_sides]
    frames = scenarios.crystal_places()                                         # [3] Motor+
    bonds, atoms = scenarios.crystals(frames)
    V_bonds, E_bonds, F_bonds, T_bonds = bonds.n_elements
    place = eye >> core.origin                                                  # [] Point-
    axes = stack((mv.x, mv.y, mv.z)).dual()                                     # [3] Point-
    view = (core.frame(place).inverse() * eye) >> axes                          # [3] Point-
    lamp = render.lamp_placement >> core.origin                                 # [] Point-
    start_tet = int(numerical.picking(topology, vertices)(place[None])[0])
    walk = dict(origin=core.origin, translating=core.translating, moving=core.moving, turning=core.turning, step=start_leg,
                n_rolls=core.n_rolls, swing=swing)
    tracer = walk | dict(
        n_particles=V_bonds, n_bonds=E_bonds, n_crystals=T_bonds,
        particle_ink=render.particle_ink,
        particle_radius_squared=render.particle_radius ** 2, bond_radius_squared=render.bond_radius ** 2,
        max_legs=render.max_legs, eye_time=0.0, pixels=list(image_points), start_place=place, start_tet=start_tet,
        right=view[0], up=view[1], forward=view[2], light=eye >> lamp)
    sight = dict(pixels=list(image_points), speed=0.0,
                 half_height=float(np.tan(render.fov_radians / 2) * image_points[1] / image_points[0]))
    advance = walk | dict(dt=0.0, thrust=0.0, roll=0.0, yaw=0.0, pitch=0.0, braking=0.0, fastest=gpu.fastest,
                          lamp=lamp, n_crystals=T_bonds, longest_substep=scenarios.crystal_step,
                          strain_reach=render.strain_reach)
    neighbours, entries = topology.neighbours
    on_rail = np.zeros(V, dtype=np.uint32)                                     # [V]
    on_rail[rails] = np.arange(len(rails))[:, None]
    tables = dict(simplices=dict(corners=topology.incidence_indices[3, 0], neighbours=neighbours, entries=entries),
                  places=vertices.batch(), rails=on_rail)
    camera = dict(motor=eye[None], speed=[0.0], clock=[0.0], tet=[start_tet], place=place[None], right=view[0:1],
                  up=view[1:2], forward=view[2:3], light=(eye >> lamp)[None])
    state = dict(camera=camera) | crystals_start(bonds, atoms, frames, core.quadrupole_frequency)
    shaders = dict(setup=setup_shader(topology, vertices, len(rails), ghosts), advance=gpu.advance_shader(bonds),
                   sight=gpu.sight_shader(), tracer=gpu.tracer_shader(bonds))
    outputs = {name: T * shaders["setup"].layout.stride(element(shaders["setup"], name))
               for name in ("tets", "visuals")}
    return dict(setup={}, advance=advance, sight=sight, tracer=tracer, tables=tables, state=state, outputs=outputs,
                shaders=shaders, n_setup=T, bonds=bonds)


def state_bytes(shaders: dict[str, Shader], state: dict) -> dict[str, bytes]:
    """Initial `state` buffers repeat ring-buffer entries across all `gpu.n_levels` time levels."""
    return {name: storage_bytes(shaders, name, value) * tiled_state.get(name, 1) for name, value in state.items()}


def contents() -> dict:
    """Compiled WGSL shaders, base64 tables, ring-buffer states, uniform copy ranges, and control limits pack the page payload."""
    values = scene(render.ghost_share, scenarios.eye, wave_swing)
    shaders = values["shaders"]
    shader_specs = {}
    for name, shader in shaders.items():
        uniforms, offsets = packed(shader, values[name])
        shader_specs[name] = dict(code=shader.code, uniforms=encoded(uniforms), offsets=offsets,
                                  entry_points=[[entry, list(size[:2])] for entry, size in shader.program.entry_points],
                                  bindings=bindings(shader, uniforms))
    advance = shaders["advance"]
    members = dict(zip([member for member, _ in advance.program.structs["Camera"]], advance.layout.offsets("Camera")))
    return dict(shaders=shader_specs, display=display_source,
                tables={name: packed_table(shaders, name, value) for name, value in values["tables"].items()},
                outputs=values["outputs"],
                state={name: dict(data=encoded(storage_bytes(shaders, name, value)), tiles=tiled_state.get(name, 1))
                       for name, value in values["state"].items()},
                copies={name: copies(advance, shader_specs[name]["offsets"], pairs)
                        for name, pairs in copied_members.items()},
                speed_offset=members["speed"],
                stride=shaders["tracer"].layout.stride(element(shaders["tracer"], "image")),
                sight_stride=shaders["sight"].layout.stride(element(shaders["sight"], "sightlines")),
                size=list(image_points),
                start=dict(leg=start_leg, period=gpu.wave_period, seconds_per_period=gpu.seconds_per_period,
                           longest_frame_seconds=gpu.longest_frame_seconds, setup_invocations=values["n_setup"]),
                limits=dict(finest=finest_leg, coarsest=coarsest_leg, thrust=thrust_per_second, scroll=speed_per_scroll,
                            turn_per_second=turn_per_second, turn_per_point=turn_per_point))


def page() -> None:
    """Writes the standalone WebGPU HTML page to `page_path`."""
    write_page(Path(__file__).parent / "web_template.html", page_path, contents())


def flight(controls: Iterator[dict], shape: tuple[int, int], ghosts: float, eye: core.Motor, swing: float, leg: float,
           n_rolls: int) -> Iterator[tuple[np.ndarray, core.Point, dict]]:
    """Yields RGB frames, atom positions `[V] Point-`, and unpacked `Camera` states stepped and ray-traced on the GPU across `controls`."""
    import wgpu

    values = scene(ghosts, eye, swing)
    shaders = values["shaders"]
    device = shaders["tracer"].device
    rows, columns = shape
    values["tracer"] = values["tracer"] | dict(pixels=[columns, rows], step=leg, n_rolls=n_rolls)
    values["sight"] = values["sight"] | dict(pixels=[columns, rows])
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    stride = shaders["tracer"].layout.stride(element(shaders["tracer"], "image"))
    blocks, offsets, uniforms, buffers = {}, {}, {}, {}
    for name, shader in shaders.items():
        blocks[name], offsets[name] = packed(shader, values[name])
        if shader.program.uniforms:
            uniforms[name] = device.create_buffer(size=len(blocks[name]), usage=usage)
            device.queue.write_buffer(uniforms[name], 0, blocks[name])
    tables = {name: unpacked_bytes(packed_table(shaders, name, value)) for name, value in values["tables"].items()}
    for name, data in (tables | state_bytes(shaders, values["state"])).items():
        buffers[name] = device.create_buffer(size=len(data), usage=usage)
        device.queue.write_buffer(buffers[name], 0, data)
    for name, size in values["outputs"].items():
        buffers[name] = device.create_buffer(size=size, usage=usage)
    buffers["sightlines"] = device.create_buffer(
        size=rows * columns * shaders["sight"].layout.stride(element(shaders["sight"], "sightlines")), usage=usage)
    buffers["image"] = device.create_buffer(size=rows * columns * stride, usage=usage)
    groups = {name: device.create_bind_group(layout=shader.bind_layout, entries=[
        {"binding": binding.index, "resource": {"buffer": buffer, "offset": 0, "size": buffer.size}}
        for binding, buffer in ((binding, uniforms[name] if binding.kind == "uniforms" else buffers[binding.name])
                                for binding in shader.program.bindings)]) for name, shader in shaders.items()}
    copied = {name: copies(shaders["advance"], offsets[name], pairs) for name, pairs in copied_members.items()}
    readback = device.create_buffer(size=rows * columns * stride, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
    V_bonds, E_bonds, F_bonds, T_bonds = values["bonds"].n_elements
    n_atoms = V_bonds
    atom_bytes = buffers["atoms"].size
    atoms_back = device.create_buffer(size=atom_bytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
    camera_bytes = buffers["camera"].size
    camera_back = device.create_buffer(size=camera_bytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)

    def dispatch(encoder, name: str, size: tuple[int, int]) -> None:
        for pipeline, workgroup in shaders[name].pipelines:
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(pipeline)
            compute.set_bind_group(0, groups[name])
            compute.dispatch_workgroups(-(-size[0] // workgroup[0]), -(-size[1] // workgroup[1]), 1)
            compute.end()

    def buffer_bytes(buffer, n_bytes: int) -> np.ndarray:
        """Maps `buffer` after GPU submission completes and copies out `n_bytes` unsigned bytes."""
        buffer.map_sync(wgpu.MapMode.READ, 0, n_bytes)
        data = np.frombuffer(buffer.read_mapped(0, n_bytes), dtype=np.uint8).copy()
        buffer.unmap()
        return data

    encoder = device.create_command_encoder()
    dispatch(encoder, "setup", (values["n_setup"], 1))
    device.queue.submit([encoder.finish()])
    block = bytearray(blocks["advance"])
    for control in controls:
        for name, value in control.items():
            block[offsets["advance"][name]:offsets["advance"][name] + 4] = np.float32(value).tobytes()
        device.queue.write_buffer(uniforms["advance"], 0, bytes(block))
        encoder = device.create_command_encoder()
        dispatch(encoder, "advance", (1, 1))
        for name, pairs in copied.items():
            for state_offset, uniform_offset, n_bytes in pairs:
                encoder.copy_buffer_to_buffer(buffers["camera"], state_offset, uniforms[name], uniform_offset, n_bytes)
        dispatch(encoder, "sight", (columns, rows))
        dispatch(encoder, "tracer", (columns, rows))
        encoder.copy_buffer_to_buffer(buffers["image"], 0, readback, 0, rows * columns * stride)
        encoder.copy_buffer_to_buffer(buffers["atoms"], 0, atoms_back, 0, atom_bytes)
        encoder.copy_buffer_to_buffer(buffers["camera"], 0, camera_back, 0, camera_bytes)
        device.queue.submit([encoder.finish()])
        image = buffer_bytes(readback, rows * columns * stride).view(np.float32).reshape(rows, columns, stride // 4)[..., :3]
        atoms = shaders["advance"].unpack("Atom", buffer_bytes(atoms_back, atom_bytes).reshape(n_atoms, -1), (n_atoms,))
        camera = shaders["advance"].unpack("Camera", buffer_bytes(camera_back, camera_bytes).reshape(1, -1), ())
        yield np.round(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), atoms["place"], camera


# Scripted flight steering past the soft crystal and turning right to keep the remaining crystals in view.
n_flight_frames = 100
flight_shape = (213, 320)
flight_aim_frames = (0, 8)
flight_aim_per_second = -1.0
flight_thrust_frames = (5, 30)
flight_yaw_frames = (35, 95)
flight_yaw_per_second = 0.5


def scripted_controls() -> Iterator[dict]:
    """Yields per-frame thrust and yaw controls for the scripted flight animation."""
    seconds = scenarios.duration_ms / 1000
    dt = gpu.wave_period * seconds / gpu.seconds_per_period
    for frame in range(n_flight_frames):
        aiming = flight_aim_per_second * (flight_aim_frames[0] <= frame < flight_aim_frames[1])
        turning = flight_yaw_per_second * (flight_yaw_frames[0] <= frame < flight_yaw_frames[1])
        yield dict(dt=dt,
                   thrust=thrust_per_second * seconds * (flight_thrust_frames[0] <= frame < flight_thrust_frames[1]),
                   roll=0.0, yaw=(aiming + turning) * seconds, pitch=0.0, braking=0.0)


# Stationary-camera run watching the three crystals evolve over about one hundred seconds of page time.
n_crystal_frames = 2500
crystal_shape = (80, 120)


def still_controls() -> Iterator[dict]:
    """Yields stationary camera controls advancing only the wave clock over `n_crystal_frames` frames."""
    seconds = scenarios.duration_ms / 1000
    dt = gpu.wave_period * seconds / gpu.seconds_per_period
    for _ in range(n_crystal_frames):
        yield dict(dt=dt, thrust=0.0, roll=0.0, yaw=0.0, pitch=0.0, braking=0.0)


def crystals_in_the_wave() -> None:
    """Renders the stationary-camera crystal animation and plots each crystal's signed volume ratio across time."""
    from examples.util.animation import save_animation, save_figure

    bonds, _ = scenarios.crystals(scenarios.crystal_places())
    states = []

    def frames() -> Iterator[np.ndarray]:
        for image, places, _ in flight(still_controls(), crystal_shape, render.wave_ghost_share, scenarios.eye,
                                       wave_swing, start_leg, core.n_rolls):
            states.append(places)
            yield image

    save_animation(frames(), "gravitational_waves_web_gauges", scenarios.duration_ms)
    # Joining a crystal's four corners gives its signed volume; negative values mark an inverted tetrahedron.
    volumes = stack([reduce(and_, bonds.incidence[3, 0] * places.field()).batch() for places in states])   # [n_crystal_frames, 3] Amount-
    times = np.cumsum([control["dt"] for control in still_controls()])          # [n_crystal_frames]
    save_figure(render.crystal_volumes(times, volumes, ["soft", "tuned", "stiff"]), "gravitational_waves_web_volumes")


aberration_speed = 0.8


def aberration() -> None:
    """Renders a side-by-side comparison of the view at rest and boosted forward at `aberration_speed`."""
    from examples.util.animation import save_figure
    from examples.util.panels import image_figure

    frozen_controls = dict(dt=0.0, roll=0.0, yaw=0.0, pitch=0.0, braking=0.0)
    rest_image, boosted_image = [image for image, _, _ in flight(
        iter([frozen_controls | dict(thrust=0.0), frozen_controls | dict(thrust=aberration_speed)]),
        flight_shape, render.ghost_share, scenarios.eye, wave_swing, start_leg, core.n_rolls)]
    gap = np.full((flight_shape[0], 4, 3), 128, dtype=np.uint8)
    save_figure(image_figure(np.concatenate([rest_image, gap, boosted_image], axis=1)),
                "gravitational_waves_web_aberration")


n_close_up_frames = 75
close_up_crystals = (0, 2)
close_up_names = ("drifting", "spinning")


def close_ups() -> None:
    """Renders close-up animations following the drifting and spinning crystals from one bond length behind each."""
    from examples.util.animation import save_animation, save_figure
    from examples.util.panels import image_figure

    frames = scenarios.crystal_places()                                         # [3] Motor+
    bonds, atoms = scenarios.crystals(frames)
    bond_length = core.round_steps(bonds, atoms).norm().batch()[0]              # [] Scalar+
    behind = (core.moving(mv.z.dual()) * (bond_length / 2)).exp()               # [] Motor+
    seconds = scenarios.duration_ms / 1000
    dt = gpu.wave_period * seconds / gpu.seconds_per_period
    images = []
    for crystal, label in zip(close_up_crystals, close_up_names):
        controls = [dict(dt=dt, thrust=crystal_drifts[crystal] * (frame == 0), roll=0.0, yaw=0.0, pitch=0.0,
                         braking=0.0) for frame in range(n_close_up_frames)]
        run = [image for image, _, _ in flight(iter(controls), flight_shape, render.ghost_share,
                                               frames[crystal] * behind, wave_swing, start_leg, core.n_rolls)]
        save_animation(run, f"gravitational_waves_web_close_up_{label}", scenarios.duration_ms)
        images.append(run[-1])
    gap = np.full((flight_shape[0], 4, 3), 128, dtype=np.uint8)
    save_figure(image_figure(np.concatenate([images[0], gap, images[1]], axis=1)), "gravitational_waves_web_close_ups")


n_compared_views = 4
compared_pixels = (2 * image_points[1], 2 * image_points[0])
compared_yaws = np.pi / 2 * np.arange(n_compared_views)                         # [n_compared_views]
n_timed = 4


def held_views(cameras: core.Motor, eye_times: np.ndarray, changed: dict) -> np.ndarray:
    """Net GPU ray-tracing times `[n_compared_views]` subtract zero-leg upload overhead from minimum trace times across `cameras`."""
    import time

    values = scene(render.ghost_share, scenarios.eye, wave_swing)
    shaders = values["shaders"]
    topology, vertices = scenarios.six_hundred_cell()
    setup_buffers = shaders["setup"]((1, values["n_setup"]), **values["tables"])
    sightlines = shaders["sight"](compared_pixels, **values["sight"])["sightlines"]
    pick = numerical.picking(topology, vertices)
    axes = stack((mv.x, mv.y, mv.z)).dual()                                     # [3] Point-
    lamp = render.lamp_placement >> core.origin                                 # [] Point-
    ring_buffers = dict(history=stack([values["state"]["history"]] * gpu.n_levels).reshape(-1),
                        inks=np.tile(values["state"]["inks"], (gpu.n_levels, 1)), balls=values["state"]["balls"])
    seconds = []
    for camera, eye_time in zip(cameras, eye_times):
        place = camera >> core.origin                                           # [] Point-
        view = (core.frame(place).inverse() * camera) >> axes                   # [3] Point-
        bound = values["tracer"] | setup_buffers | ring_buffers | dict(
            sightlines=sightlines, eye_time=float(eye_time), start_place=place, start_tet=int(pick(place[None])[0]),
            right=view[0], up=view[1], forward=view[2], light=camera >> lamp) | changed

        def least(timed: dict) -> float:
            """Minimum elapsed seconds across `n_timed` tracer dispatches with overrides `timed`."""
            times = []
            for _ in range(n_timed):
                begun = time.perf_counter()
                shaders["tracer"](compared_pixels, **(bound | timed))
                times.append(time.perf_counter() - begun)
            return min(times)

        seconds.append(least({}) - least(dict(max_legs=0)))
    return np.array(seconds)


def bounded_crystals() -> None:
    """Benchmarks per-frame ray-tracing times with crystal bounding balls, with full-sphere bounds, and without crystals."""
    from examples.util.animation import save_animation, save_figure

    cameras = scenarios.eye * ((mv.zx * compared_yaws) * -0.5).exp()            # [n_compared_views] Motor+
    eye_times = np.arange(n_compared_views) * gpu.wave_period / n_compared_views   # [n_compared_views]
    bonds, atoms = scenarios.crystals(scenarios.crystal_places())
    balls = render.crystal_balls(bonds, atoms[None])
    everywhere = dict(centre=balls["centre"], reach=np.full_like(balls["reach"], np.pi))
    seconds = [held_views(cameras, eye_times, changed)
               for changed in ({}, dict(balls=everywhere), dict(n_particles=0, n_bonds=0, n_crystals=0))]
    milliseconds = np.array([1000 * view_times.mean() for view_times in seconds])   # [3]
    save_figure(render.bounded_crystals(milliseconds, compared_pixels), "gravitational_waves_web_bounds")
    save_animation((image for image, _, _ in flight(scripted_controls(), flight_shape, render.wave_ghost_share,
                                                    scenarios.eye, wave_swing, start_leg, core.n_rolls)),
                   "gravitational_waves_web_flight", scenarios.duration_ms)


def main() -> None:
    """Writes the standalone WebGPU page and renders the scripted flight, aberration, close-up, and crystal-volume runs."""
    from examples.util.animation import save_animation

    page()
    save_animation((image for image, _, _ in flight(scripted_controls(), flight_shape, render.wave_ghost_share,
                                                    scenarios.eye, wave_swing, start_leg, core.n_rolls)),
                   "gravitational_waves_web_flight", scenarios.duration_ms)
    aberration()
    close_ups()
    crystals_in_the_wave()


if __name__ == "__main__":
    main()
