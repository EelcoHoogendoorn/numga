"""A page that flies through the closed universe in the browser, on the reader's own GPU: the view from inside the
600-cell traced by the shader of the notebooks, compiled once here with the mesh it walks and the crystals it draws,
written into one HTML file. The universe carries the exact slowest gravitational wave of the round three-sphere, the
squash swinging at the root of eight, evaluated by the shader at any place and time; the camera flies in it along its
own forward axis, its motor stepped on the GPU every frame; and three crystals of atoms on springs, soft, tuned and
stiff, are stepped there too, as `core.massive` steps them, their bonds coloured by their strain, the soft one drifting
and the stiff one spinning along great circles so that gauges fly past as well as hang in the wave. Every edge of the
600-cell is a strut, the guardrail decagons in colour over the grey of the rest.

The lattice's frame is the universe's own, the rest frame of the Einstein static universe, and the camera moves through
it. In its own frame the camera is an eye and a sensor behind it, each pixel a point of the sensor, and the light that
reaches a pixel ran along the null line from the eye's event to the pixel's. Those lines are formed in the spacetime
algebra at the eye and carried into the lattice's frame by the boost of the camera's rapidity along its forward axis,
`((ahead & time).dual() * (rapidity / 2)).exp()`, so that the sky ahead is drawn in and what lies beside swings forward
as the camera speeds up, the field of view being the one it has at rest. A frame is a tick of the camera's own clock,
over which the lattice's time runs on by the boost's time component. That physics is `gpu`'s.

Light takes time through the wave, as in the notebooks: each ray runs back from the eye along its lattice-frame
direction, leg by leg, its time running back as its length grows, and sees the wave as it was when its light passed. The
gauges are seen so too: their atoms' places and their bonds' inks are kept at each level of the wave's step, the way the
notebooks keep the wave, as far back as light takes to cross the universe, and a leg meets them as they were at its
time, the two nearest moments blended. Nothing here has a spectrum, so the light's Doppler shift is left out, and the
crystals, slow, keep springs without contraction or dilation of their own.

The file holds the shaders, the 600-cell's vertices, its tetrahedra's corners and neighbours, and the rail each vertex
lies on, the indices as unsigned 16-bit integers; the tetrahedra's ramps, and the places, beads and struts the view
draws in each, are made on the GPU when the page loads. Nothing is computed on a server, and no picture is sent; the
image traces at fewer pixels when frames come slower than the GPU can keep up with. Each shader binds at most eight
storage buffers in a stage, its uniforms' block among them, as many as WebGPU guarantees, so the page asks for no
limit.

The camera flies as an airplane does, along its own forward axis: the arrow keys turn it up, down, left and right, or a
drag does, `a` and `d` roll it, `w` and `s` speed it up and slow it down, scrolling adds speed, space stops it, `p`
holds time, `r` returns to the start, and `[` and `]` halve and double a ray's legs.

GitHub Pages serves the page from the repository, so it is written only when asked; the module's own run traces the
page's passes here, as the page runs them, into animations in the plots; and a comparison times the tracer with each
crystal's atoms and bonds met only by the legs that come within its ball, met by every leg, and with no crystals, into
the plots too:

    python -c "from examples.relativity.gravitational_waves.web import page; page()"   # writes the page beside this file
    python -m examples.relativity.gravitational_waves.web                                # the page's runs, into the plots
    python -c "from examples.relativity.gravitational_waves.web import bounded_crystals; bounded_crystals()"   # the bound
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

from examples.relativity.gravitational_waves import core, gpu, render, scenarios
from examples.util.topology import Topology
from examples.util.wgsl import Shader
from examples.util.wgsl.dialect import template
from examples.util.wgsl.page import encoded, packed, write_page

mv = core.mv

# The page's image, in points, traced at the screen's pixels behind them, or fewer when frames come slowly.
image_points = (480, 320)
# A ray's legs at the start, in arcs of the sphere, each rolled in `core.n_rolls` rolls a leg that long, and the lengths
# `[` and `]` move between.
start_leg, finest_leg, coarsest_leg = 0.35, 0.0125, 0.4
# The camera's speeds, as shares of the light's: how fast thrust adds speed a second held, and how much a scroll's notch
# adds; how fast the arrow keys turn and roll the view a second, and a drag a point.
thrust_per_second, speed_per_scroll = 0.4, 0.0005
turn_per_second, turn_per_point = 1.2, 0.004
# The swing of the page's exact wave: the strain the notebooks' stepped squash has at the eye, launched with the swing
# `scenarios.wave_swing`, which the 600-cell's waves carry a little short of the continuum's.
wave_swing = 0.112
# Each crystal's drift along the great circle ahead of it, as a share of the light's speed, and its turns a wave
# period in its own plane `xy`: the soft one drifts, the tuned one hangs in the wave and the stiff one spins.
crystal_drifts = np.array([0.1, 0, 0])                                           # [crystals]
crystal_spins = np.array([0, 0, 0.5])                                            # [crystals]
page_path = Path(__file__).parent / "gravitational_waves_webgpu.html"

# The traced light drawn over the page's canvas by one triangle, each pixel reading its sample, `stride` numbers to a
# pixel; the view's block padded to sixteen bytes, as a uniform buffer's binding is sized.
display_source = """
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
"""


# The tetrahedra's ramps, and the places, beads and struts the view draws in each, made on the GPU when the page loads,
# one invocation to a tetrahedron, from the file's simplices, the vertices' places and the rails, as `scene` decides
# them for the notebooks' renderer: every vertex lies on one of the guardrail decagons, and has a bead in its colour;
# an edge whose ends lie on one decagon is a side of it, a strut in its colour, and every other edge a grey ghost. Each
# corner's ramp is made as `core.coframe` makes it: the plane through the face across from the corner, the join of that
# face's corners, over its pairing with the corner, so that it reads one at the corner and nothing on the face; the
# quotient of the intrinsic join by the intrinsic pairing is extrinsic.
setup_source = """
// A tetrahedron as the file holds it: its corners, and across the face facing each corner the neighbour and the face
// it is entered by there.
struct Simplex {
    corners: vec4<u32>,
    neighbours: vec4<u32>,
    entries: vec4<u32>,
}

var<storage, read> simplices: array<Simplex>;
var<storage, read> places: array<Point>;
var<storage, read> rails: array<u32>;
var<storage, read_write> tets: array<Tet>;
var<storage, read_write> cells: array<Cell>;

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
        let tail = simplex.corners[cell_edges[edge].x];
        let head = simplex.corners[cell_edges[edge].y];
        let side = vec4<f32>(rail_inks[rails[tail]], rail_radius_squared);
        struts[edge] = select(vec4<f32>(ghost_ink, ghost_radius_squared), side, rails[tail] == rails[head]);
    }
    var ramps: array<Plane, 4>;
    for (var corner = 0u; corner < 4u; corner++) {
        let face = faces[corner];
        let facet = corners[face.x] & corners[face.y] & corners[face.z];
        ramps[corner] = facet / (facet & corners[corner]);
    }
    tets[index] = Tet(simplex.neighbours, simplex.entries, ramps);
    cells[index] = Cell(corners, beads, struts);
}
"""


def tables_source(mesh: Topology, n_rails: int) -> str:
    """The local corners of the face across from each corner of a tetrahedron, in the topology's order, `faces`, and
    the guardrails' colours round the colour wheel, `rail_inks`, as tables in WGSL."""
    faces = ", ".join(f"vec3<u32>({first}u, {second}u, {third}u)" for first, second, third in mesh.corners[3, 2].T)
    inks = ", ".join(f"vec3<f32>({float(red)!r}, {float(green)!r}, {float(blue)!r})"
                     for red, green, blue in render.hues(n_rails))
    return (f"\nconst faces = array<vec3<u32>, 4>({faces});\n"
            f"const rail_inks = array<vec3<f32>, {n_rails}>({inks});\n")


def setup_shader(mesh: Topology, vertices: core.Point, n_rails: int, ghosts: float) -> Shader:
    """The tetrahedra's ramps and what the view draws in each, made from the file's simplices, the vertices' places
    and the rails: struts on the rails and beads at their corners as thick as `render` draws them, every other edge a
    ghost `ghosts` of the mean edge thick."""
    mean_edge = render.mean_chord(mesh, vertices)
    return Shader(core.tet_struct_source + render.cell_source + tables_source(mesh, n_rails) + setup_source, core.ga,
                  render.inside_types,
                  dict(ghost_ink=render.ghost_ink, ghost_radius_squared=(ghosts * mean_edge) ** 2,
                       rail_radius_squared=(render.rail_share * mean_edge) ** 2,
                       bead_radius_squared=(render.bead_scale * render.rail_share * mean_edge) ** 2))


def crystal_velocities(frames: core.Motor, atoms: core.Point) -> core.Point:
    """The crystals' atoms' velocities at the start, `[atoms] Point-`, steps of the origin in the frame at each atom:
    each crystal drifting along the great circle ahead of it by `crystal_drifts` and turning in its own plane `xy`
    by `crystal_spins` a wave period, each atom carried round its crystal's middle."""
    n_crystals = frames.shape[0]
    places = atoms.batch().reshape(n_crystals, -1)                              # [crystals, corners] Point-
    # Each atom's offset from its crystal's middle, a step of the origin in the crystal's frame.
    offsets = (frames[:, None].inverse() >> places).cast(core.Step)             # [crystals, corners] Point-
    drifts = mv.z.dual() * crystal_drifts[:, None]                              # [crystals, 1] Point-
    spins = offsets.commutator(mv.xy * (2 * np.pi * crystal_spins / gpu.wave_period)[:, None])   # [crystals, corners] Point-
    # The velocities in the crystals' frames, read in the frame at each atom.
    in_crystal = (drifts + spins).cast(core.Step)                               # [crystals, corners] Point-
    velocities = (core.frame(places).inverse() * frames[:, None]) >> in_crystal   # [crystals, corners] Point-
    return velocities.cast(core.Step).reshape(-1)


def crystals_start(bonds: Topology, atoms: core.Point, frames: core.Motor, frequency: float) -> dict:
    """The crystals at the start, as the advance pass holds them: the atoms, their places at rest in their crystals and
    their velocities; their springs, the rest lengths, the stiffnesses and dampings, the tuned crystal's stiffness putting
    the swing the squash drives at the wave's frequency, as `crystal_states` tunes them, and the lengths, which start
    at rest; a level of the ring of kept moments, the atoms' places and the bonds' inks, at no strain, which every
    level holds at the start; and each crystal's ball about those moments."""
    n_crystals = len(scenarios.tunings)
    stiffnesses = frequency ** 2 / 2 * scenarios.tunings                        # [crystals]
    dampings = scenarios.critical_share * np.sqrt(2 * stiffnesses)              # [crystals]
    n_bonds = bonds.n_elements[1]
    round_steps = core.round_steps(bonds, atoms)                                # Point-[bonds]
    rest = round_steps.dot(round_steps).square_root().batch().to_array()        # [bonds, 1]
    unstrained = np.tile(plt.get_cmap(render.strain_map)(0.5), (n_bonds, 1))    # [bonds, 4]
    return dict(atoms=dict(place=atoms.batch(), velocity=crystal_velocities(frames, atoms)),
                springs=dict(rest=rest, stiffness=np.repeat(stiffnesses, n_bonds // n_crystals),
                             damping=np.repeat(dampings, n_bonds // n_crystals), length=rest),
                history=atoms.batch(), inks=unstrained, balls=render.crystal_balls(bonds, atoms[None]))


# --- plumbing -------------------------------------------------------------------------
# The camera's members copied into the uniforms of the shaders that read them each frame: the tracer's eye, its axes,
# the lamp and the lattice's time, and the sightlines' speed.
copied_members = dict(tracer=(("place", "start_place"), ("tet", "start_tet"), ("right", "right"), ("up", "up"),
                              ("forward", "forward"), ("light", "light"), ("clock", "eye_time")),
                      sight=(("speed", "speed"),))
# The levels of the ring every array of kept moments holds, filled alike at the start.
tiled_state = dict(history=gpu.n_levels, inks=gpu.n_levels)


def element(shader: Shader, name: str) -> object:
    """The type of the elements of a shader's storage array."""
    return shader.layout.resolve(template(next(binding.type for binding in shader.program.bindings
                                               if binding.name == name))[1][0])


def laid_out(shaders: dict[str, Shader], name: str, value: object) -> bytes:
    """A storage array's bytes, laid out as the first of the shaders that binds it declares its elements."""
    shader = next(shader for shader in shaders.values()
                  if any(binding.name == name for binding in shader.program.bindings))
    return shader.layout.array(element(shader, name), value)


def compact(shaders: dict[str, Shader], name: str, value: object) -> dict:
    """A table the file holds, as the page unpacks it: its elements' count and stride, and each member the table
    gives, at its offset in the element and as many numbers to an element as it takes there, in base64, indices as
    unsigned 16-bit integers and every other number as a 32-bit float. A member it leaves out the setup makes."""
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


def unpacked(table: dict) -> bytes:
    """A table's bytes as the page unpacks them: each member's numbers widened to 32 bits at its offset in every
    element, the members the table leaves out nothing."""
    words = np.zeros((table["count"], table["stride"] // 4), dtype=np.uint32)
    for column in table["columns"]:
        numbers = np.frombuffer(base64.b64decode(column["data"]), dtype=np.uint16 if column["kind"] == "u16" else np.float32)
        target = words if column["kind"] == "u16" else words.view(np.float32)
        target[:, column["offset"] // 4:column["offset"] // 4 + column["width"]] = numbers.reshape(table["count"], -1)
    return words.tobytes()


def bindings(shader: Shader, block: bytes) -> list[dict]:
    """How a shader binds its buffers, in order: the uniform block or a storage array by name, read only or read and
    written, and the fewest bytes a binding of it holds, the block's own or an element's."""
    return [dict(index=binding.index, kind=binding.kind, name=binding.name,
                 writable=binding.kind == "storage" and binding.access == "read_write",
                 least=len(block) if binding.kind == "uniforms" else shader.layout.stride(element(shader, binding.name)))
            for binding in shader.program.bindings]


def copies(advance: Shader, offsets: dict, pairs: tuple) -> list[list[int]]:
    """Where each copied member of the camera's state begins, where its uniform begins in a shader's block, and its
    bytes."""
    members = dict(zip([member for member, _ in advance.program.structs["Camera"]], advance.layout.offsets("Camera")))
    kinds = dict(advance.program.structs["Camera"])
    return [[members[member], offsets[uniform], advance.layout.size(kinds[member])[1]] for member, uniform in pairs]


def scene(ghosts: float, eye: core.Motor, swing: float) -> dict:
    """Everything the four shaders read: the setup's nothing besides its tables; the advance pass's uniforms, the keys
    at rest; the sightlines', the view's height and the camera's speed; the tracer's, the exact wave's swing `swing`
    and the eye; the tables the file holds, the 600-cell's simplices, its vertices' places, and the guardrail decagon
    each vertex lies on; the state the advance pass carries and `r` restores, the camera and the crystals at the start,
    the camera at `eye`; the arrays the setup alone writes, with their sizes; and the shaders, the setup's making every
    edge off the decagons a grey ghost `ghosts` of the mesh's mean edge thick, none where that is nothing."""
    mesh, vertices = scenarios.six_hundred_cell()
    rails = scenarios.guardrails(vertices, scenarios.n_guardrail_sides)        # [decagons, sides]
    frames = scenarios.crystal_places()                                         # [crystals] Motor+
    bonds, atoms = scenarios.crystals(frames)
    place = eye >> core.origin                                                  # [] Point-
    axes = stack((mv.x, mv.y, mv.z)).dual()                                     # [3] Point-
    view = (core.frame(place).inverse() * eye) >> axes                          # [3] Point-
    lamp = render.lamp_placement >> core.origin                                 # [] Point-
    start_tet = int(core.picking(mesh, vertices)(place[None])[0])
    walk = dict(origin=core.origin, translating=core.translating, moving=core.moving, turning=core.turning, step=start_leg,
                n_rolls=core.n_rolls, swing=swing)
    tracer = walk | dict(
        n_particles=bonds.n_vertices, n_bonds=bonds.n_elements[1], n_crystals=frames.shape[0],
        particle_ink=render.particle_ink,
        particle_radius_squared=render.particle_radius ** 2, bond_radius_squared=render.bond_radius ** 2,
        max_legs=render.max_legs, eye_time=0.0, pixels=list(image_points), start_place=place, start_tet=start_tet,
        right=view[0], up=view[1], forward=view[2], light=eye >> lamp)
    sight = dict(pixels=list(image_points), speed=0.0,
                 half_height=float(np.tan(render.fov_radians / 2) * image_points[1] / image_points[0]))
    advance = walk | dict(dt=0.0, thrust=0.0, roll=0.0, yaw=0.0, pitch=0.0, braking=0.0, fastest=gpu.fastest,
                          lamp=lamp, n_crystals=frames.shape[0], longest_substep=scenarios.crystal_step,
                          saturating_strain=render.saturating_strain)
    neighbours, entries = mesh.neighbours
    # The decagon each vertex lies on: every vertex of the 600-cell lies on one.
    on_rail = np.zeros(mesh.n_vertices, dtype=np.uint32)                       # [vertices]
    on_rail[rails] = np.arange(len(rails))[:, None]
    tables = dict(simplices=dict(corners=mesh.incidence_indices[3, 0], neighbours=neighbours, entries=entries),
                  places=vertices.batch(), rails=on_rail)
    # The camera at the eye and at rest, the one element of the array the advance pass steps, at the lattice's time
    # nought.
    camera = dict(motor=eye[None], speed=[0.0], clock=[0.0], tet=[start_tet], place=place[None], right=view[0:1],
                  up=view[1:2], forward=view[2:3], light=(eye >> lamp)[None])
    state = dict(camera=camera) | crystals_start(bonds, atoms, frames, core.quadrupole_frequency)
    shaders = dict(setup=setup_shader(mesh, vertices, len(rails), ghosts), advance=gpu.advance_shader(bonds),
                   sight=gpu.sight_shader(), tracer=gpu.tracer_shader(bonds))
    outputs = {name: mesh.n_elements[3] * shaders["setup"].layout.stride(element(shaders["setup"], name))
               for name in ("tets", "cells")}
    return dict(setup={}, advance=advance, sight=sight, tracer=tracer, tables=tables, state=state, outputs=outputs,
                shaders=shaders, n_setup=mesh.n_elements[3], bonds=bonds)


def state_bytes(shaders: dict[str, Shader], state: dict) -> dict[str, bytes]:
    """The state the advance pass carries, as the page writes it into its arrays: each ring of kept moments its first
    level as many times as it has levels."""
    return {name: laid_out(shaders, name, value) * tiled_state.get(name, 1) for name, value in state.items()}


def contents() -> dict:
    """Everything the page reads: the four compiled shaders, their bindings and uniform blocks, the tables they read,
    the arrays the GPU alone writes with their sizes, the state the advance pass carries and `r` restores, each with
    the levels it repeats, where the camera's state is copied into the other shaders' uniforms each frame, and where
    the camera's speed lies for the page's line."""
    values = scene(render.ghost_share, scenarios.eye, wave_swing)
    shaders = values["shaders"]
    described = {}
    for name, shader in shaders.items():
        uniforms, offsets = packed(shader, values[name])
        described[name] = dict(code=shader.code, uniforms=encoded(uniforms), offsets=offsets,
                               entry_points=[[entry, list(size[:2])] for entry, size in shader.program.entry_points],
                               bindings=bindings(shader, uniforms))
    advance = shaders["advance"]
    members = dict(zip([member for member, _ in advance.program.structs["Camera"]], advance.layout.offsets("Camera")))
    return dict(shaders=described, display=display_source,
                tables={name: compact(shaders, name, value) for name, value in values["tables"].items()},
                outputs=values["outputs"],
                state={name: dict(data=encoded(laid_out(shaders, name, value)), tiles=tiled_state.get(name, 1))
                       for name, value in values["state"].items()},
                copies={name: copies(advance, described[name]["offsets"], pairs)
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
    """Write the page beside this file, where GitHub Pages serves it."""
    write_page(Path(__file__).parent / "web_template.html", page_path, contents())


def flight(controls: Iterator[dict], shape: tuple[int, int], ghosts: float, eye: core.Motor, swing: float, leg: float,
           n_rolls: int) -> Iterator[tuple[np.ndarray, core.Point, dict]]:
    """Frames of the page's view as the page traces them, `[rows, columns, 3]` bytes each, the crystals' atoms then,
    `[atoms] Point-`, and the camera's state then, by member of its `Camera`, one for each frame's `controls`, the keys
    as the page turns them into the advance pass's uniforms: `dt`, `thrust`, `roll`, `yaw`, `pitch` and `braking`, from
    the camera `eye` through the exact wave at the swing `swing`, with the edges off the guardrails ghosts `ghosts` of
    the mean edge thick, and a ray's legs at most `leg` long, rolled in `n_rolls` rolls a leg that long. The passes run
    on the GPU as the page runs them, on the same buffers: the mesh's styles and ramps are made once, then each frame
    the camera and the crystals advance, the camera's state is copied into the other shaders' uniforms, the sightlines
    are boosted, and the tracer draws."""
    import wgpu

    values = scene(ghosts, eye, swing)
    shaders = values["shaders"]
    device = shaders["tracer"].device
    rows, columns = shape
    values["tracer"] = values["tracer"] | dict(pixels=[columns, rows], step=leg, n_rolls=n_rolls)
    values["sight"] = values["sight"] | dict(pixels=[columns, rows])
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    stride = shaders["tracer"].layout.stride(element(shaders["tracer"], "image"))
    # Each shader's uniform block, where it has uniforms, and the storage arrays, each made once and shared by name
    # between the shaders.
    blocks, offsets, uniforms, buffers = {}, {}, {}, {}
    for name, shader in shaders.items():
        blocks[name], offsets[name] = packed(shader, values[name])
        if shader.program.uniforms:
            uniforms[name] = device.create_buffer(size=len(blocks[name]), usage=usage)
            device.queue.write_buffer(uniforms[name], 0, blocks[name])
    tables = {name: unpacked(compact(shaders, name, value)) for name, value in values["tables"].items()}
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
    n_atoms = values["bonds"].n_vertices
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

    def read(buffer, n_bytes: int) -> np.ndarray:
        """A readback buffer's bytes, once the work before it is done."""
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
        image = read(readback, rows * columns * stride).view(np.float32).reshape(rows, columns, stride // 4)[..., :3]
        atoms = shaders["advance"].unpack("Atom", read(atoms_back, atom_bytes).reshape(n_atoms, -1), (n_atoms,))
        camera = shaders["advance"].unpack("Camera", read(camera_back, camera_bytes).reshape(1, -1), ())
        yield np.round(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), atoms["place"], camera


# The flight the module draws into the plots, as the page would trace it but with the guardrails alone, for the
# file's size: so many frames at the animations' pace, the wave's time running as on the page; the keys held over
# spans of frames, a turn to the left and thrust ahead through the first second, so that the camera passes close by
# the soft crystal, and a slow turn to the right to keep the other crystals in view as it flies on.
n_flight_frames = 100
flight_shape = (213, 320)
flight_aim_frames = (0, 8)
flight_aim_per_second = -1.0
flight_thrust_frames = (5, 30)
flight_yaw_frames = (35, 95)
flight_yaw_per_second = 0.5


def scripted_controls() -> Iterator[dict]:
    """The flight's controls at each frame, as the page's keys would give them."""
    seconds = scenarios.duration_ms / 1000
    dt = gpu.wave_period * seconds / gpu.seconds_per_period
    for frame in range(n_flight_frames):
        aiming = flight_aim_per_second * (flight_aim_frames[0] <= frame < flight_aim_frames[1])
        turning = flight_yaw_per_second * (flight_yaw_frames[0] <= frame < flight_yaw_frames[1])
        yield dict(dt=dt,
                   thrust=thrust_per_second * seconds * (flight_thrust_frames[0] <= frame < flight_thrust_frames[1]),
                   roll=0.0, yaw=(aiming + turning) * seconds, pitch=0.0, braking=0.0)


# The crystals in the wave alone, the camera still before them, as long as a reader might watch, about a hundred
# seconds of the page's clock; drawn small, for the file's size. The crystals being stepped in the lattice's time, how
# the camera flies changes nothing for them but how far the frames' ticks carry it.
n_crystal_frames = 2500
crystal_shape = (80, 120)


def still_controls() -> Iterator[dict]:
    """The controls of a camera left alone, at each frame."""
    seconds = scenarios.duration_ms / 1000
    dt = gpu.wave_period * seconds / gpu.seconds_per_period
    for frame in range(n_crystal_frames):
        yield dict(dt=dt, thrust=0.0, roll=0.0, yaw=0.0, pitch=0.0, braking=0.0)


def crystals_in_the_wave() -> None:
    """The crystals in the wave alone, as the page steps them: the animation, and each crystal's signed volume over it,
    which a crystal turned inside out takes below nothing."""
    from examples.util.animation import save_animation, save_figure

    bonds, _ = scenarios.crystals(scenarios.crystal_places())
    states = []

    def frames() -> Iterator[np.ndarray]:
        for image, places, _ in flight(still_controls(), crystal_shape, render.wave_ghost_share, scenarios.eye,
                                       wave_swing, start_leg, core.n_rolls):
            states.append(places)
            yield image

    save_animation(frames(), "gravitational_waves_web_gauges", scenarios.duration_ms)
    # Each crystal's signed volume, the join of its corners.
    volumes = stack([reduce(and_, bonds.incidence[3, 0] * places.field()).batch() for places in states])   # [frames, crystals] Scalar-
    # The lattice's time at each frame, the camera's own at rest.
    times = np.cumsum([control["dt"] for control in still_controls()])          # [frames]
    save_figure(render.crystal_volumes(times, volumes, ["soft", "tuned", "stiff"]), "gravitational_waves_web_volumes")


# One frame at rest beside one at a speed, from the same place at the same time, looking the same way: time held, and
# all the speed given at once.
aberration_speed = 0.8


def aberration() -> None:
    """The view at rest beside the view moving along it at `aberration_speed` of the light's: the sky ahead drawn in
    and the rails beside swung forward, the light's aberration."""
    from examples.util.animation import save_figure
    from examples.util.wgsl import image_figure

    held = dict(dt=0.0, roll=0.0, yaw=0.0, pitch=0.0, braking=0.0)
    at_rest, moving = [image for image, _, _ in flight(iter([held | dict(thrust=0.0), held | dict(thrust=aberration_speed)]),
                                                       flight_shape, render.ghost_share, scenarios.eye, wave_swing,
                                                       start_leg, core.n_rolls)]
    gap = np.full((flight_shape[0], 4, 3), 128, dtype=np.uint8)
    save_figure(image_figure(np.concatenate([at_rest, gap, moving], axis=1)), "gravitational_waves_web_aberration")


# Close-ups of the moving crystals, a few seconds into the page: the camera parked a bond's length behind the drifting
# crystal, following it at its speed, and behind the spinning one, at rest, each looking along the crystal's own axis.
n_close_up_frames = 75
close_up_crystals = (0, 2)


def close_ups() -> None:
    """The drifting and the spinning crystal from a bond's length behind each, as the page draws them: each run's
    animation, and the last frames side by side."""
    from examples.util.animation import save_animation, save_figure
    from examples.util.wgsl import image_figure

    frames = scenarios.crystal_places()                                         # [crystals] Motor+
    bonds, atoms = scenarios.crystals(frames)
    bond_length = core.round_steps(bonds, atoms).dot(core.round_steps(bonds, atoms)).square_root().batch()[0]   # [] Scalar+
    # The motion back along a crystal's own axis by a bond's length.
    behind = (core.moving(mv.z.dual()) * (bond_length.to_array().item() / 2)).exp()   # [] Motor+
    seconds = scenarios.duration_ms / 1000
    dt = gpu.wave_period * seconds / gpu.seconds_per_period
    images = []
    for crystal in close_up_crystals:
        controls = [dict(dt=dt, thrust=crystal_drifts[crystal] * (frame == 0), roll=0.0, yaw=0.0, pitch=0.0,
                         braking=0.0) for frame in range(n_close_up_frames)]
        run = [image for image, _, _ in flight(iter(controls), flight_shape, render.ghost_share,
                                               frames[crystal] * behind, wave_swing, start_leg, core.n_rolls)]
        save_animation(run, "gravitational_waves_web_close_up", scenarios.duration_ms)
        images.append(run[-1])
    gap = np.full((flight_shape[0], 4, 3), 128, dtype=np.uint8)
    save_figure(image_figure(np.concatenate([images[0], gap, images[1]], axis=1)), "gravitational_waves_web_close_ups")


# The bound compared: the page's view held still from views ahead of the eye, to its right, behind it and to its left,
# each a quarter of the wave's period later, traced at the page's leg and full pixels on a screen of two device pixels
# a point, the least time of so many traces taken.
compared_pixels = (2 * image_points[1], 2 * image_points[0])
compared_yaws = np.pi / 2 * np.arange(4)                                         # [views]
n_timed = 4


def held_views(cameras: core.Motor, eye_times: np.ndarray, changed: dict) -> np.ndarray:
    """The tracer's seconds a frame, `[views]`, for the page's view held still from each of the `cameras`, `[views]
    Motor+`, at each of the `eye_times`, traced at the pixels `compared_pixels`, the crystals at their start and the
    tracer's values changed by `changed`: the least of `n_timed` traces less the least of as many that take no leg,
    which upload and read back alike."""
    import time

    values = scene(render.ghost_share, scenarios.eye, wave_swing)
    shaders = values["shaders"]
    mesh, vertices = scenarios.six_hundred_cell()
    made = shaders["setup"]((1, values["n_setup"]), **values["tables"])
    sightlines = shaders["sight"](compared_pixels, **values["sight"])["sightlines"]
    pick = core.picking(mesh, vertices)
    axes = stack((mv.x, mv.y, mv.z)).dual()                                     # [3] Point-
    lamp = render.lamp_placement >> core.origin                                 # [] Point-
    # Every level of the ring holds the crystals at their start, as the page fills it, and the balls are theirs.
    stored = dict(history=stack([values["state"]["history"]] * gpu.n_levels).reshape(-1),
                  inks=np.tile(values["state"]["inks"], (gpu.n_levels, 1)), balls=values["state"]["balls"])
    seconds = []
    for camera, eye_time in zip(cameras, eye_times):
        place = camera >> core.origin                                           # [] Point-
        view = (core.frame(place).inverse() * camera) >> axes                   # [3] Point-
        bound = values["tracer"] | made | stored | dict(
            sightlines=sightlines, eye_time=float(eye_time), start_place=place, start_tet=int(pick(place[None])[0]),
            right=view[0], up=view[1], forward=view[2], light=camera >> lamp) | changed

        def least(timed: dict) -> float:
            """The least seconds of `n_timed` traces of the view with the values `timed`."""
            times = []
            for _ in range(n_timed):
                begun = time.perf_counter()
                shaders["tracer"](compared_pixels, **(bound | timed))
                times.append(time.perf_counter() - begun)
            return min(times)

        seconds.append(least({}) - least(dict(max_legs=0)))
    return np.array(seconds)


def bounded_crystals() -> None:
    """The tracer's time a frame at the page's leg and full pixels, each crystal's atoms and bonds met only by the legs
    that come within its ball, met by every leg, and no crystals at all, into a chart in the plots; and the page's
    scripted flight, the crystals behind their balls, into the plots."""
    from examples.util.animation import save_animation, save_figure

    cameras = scenarios.eye * ((mv.zx * compared_yaws) * -0.5).exp()            # [views] Motor+
    eye_times = np.arange(len(compared_yaws)) * gpu.wave_period / len(compared_yaws)   # [views]
    bonds, atoms = scenarios.crystals(scenarios.crystal_places())
    balls = render.crystal_balls(bonds, atoms[None])
    # Balls that hold the whole sphere, which every leg meets.
    everywhere = dict(centre=balls["centre"], reach=np.full_like(balls["reach"], np.pi))
    seconds = [held_views(cameras, eye_times, changed)
               for changed in ({}, dict(balls=everywhere), dict(n_particles=0, n_bonds=0, n_crystals=0))]
    milliseconds = np.array([1000 * views.mean() for views in seconds])       # [kinds]
    save_figure(render.bounded_crystals(milliseconds, compared_pixels), "gravitational_waves_web_bounds")
    save_animation((image for image, _, _ in flight(scripted_controls(), flight_shape, render.wave_ghost_share,
                                                    scenarios.eye, wave_swing, start_leg, core.n_rolls)),
                   "gravitational_waves_web_flight", scenarios.duration_ms)


def main() -> None:
    """The page's runs, as the page traces them, into the plots: the scripted flight, the aberration, the close-ups
    and the crystals in the wave."""
    from examples.util.animation import save_animation

    save_animation((image for image, _, _ in flight(scripted_controls(), flight_shape, render.wave_ghost_share,
                                                    scenarios.eye, wave_swing, start_leg, core.n_rolls)),
                   "gravitational_waves_web_flight", scenarios.duration_ms)
    aberration()
    close_ups()
    crystals_in_the_wave()


if __name__ == "__main__":
    main()
