"""A camera above a rotating black hole's disk, traced on the GPU at the window's own pixels, as fast as the GPU
allows: the shader of the GPU notebook, its light exposed and drawn into the window without leaving the GPU.

Drag to fly the camera around the hole, scroll to move it in or out, `[` and `]` to halve or double the integration
step, the fraction of a ray's radius each Runge–Kutta step covers, space to stop or restart the disk's orbit, and `s`
to stop or restart the spin, which sweeps from turning one way to turning the other and back, the disk orbiting with
the hole and then against it. Against the hole the disk is far cooler and dimmer, a few hundredths of its light with
it, so the exposure follows the brightest light of the last frame, as a camera's meter would; `e` holds it at the
scenario's exposure instead, to compare brightness across spins. The terminal reports the frames per second, the step and the spin.

    python -m examples.relativity.kerr.viewer
"""

from __future__ import annotations

import time

import numpy as np

from examples.relativity.kerr import core, gpu, render, scenarios

mv = core.mv
# The window's size at the start, in points; the image is traced at the screen's pixels behind them.
WINDOW = (480, 320)
# The camera's nearest and farthest distance from the hole, in masses, and how fast dragging and scrolling move it.
NEAREST, FARTHEST = 8.0, 90.0
TURN_PER_POINT, ZOOM_PER_SCROLL = 0.006, 0.0015
# The step at the start, a fraction of the ray's radius, and the steps `[` and `]` move between: a step of 0.4
# changes a few pixels by a few levels from one eight times finer.
STEP, FINEST, COARSEST = 0.4, 0.05, 1.6
# The spin's largest size, in masses, and the seconds of one sweep there and back.
LARGEST_SPIN, SPIN_PERIOD = 0.95, 12.0
# Where a frame's brightest light sits, in mid-greys, when the exposure follows it: where it sits in the scenario's
# still; and the share of the way each frame's exposure moves toward it.
BRIGHTEST, FOLLOWING = 3.7, 0.3

# The traced light, exposed as `render.develop` exposes it and added to the sky, drawn over the window by one
# triangle: each pixel reads its sample from the tracer's output, `stride` numbers to a pixel.
DISPLAY = """
struct View {
    columns: u32,
    stride: u32,
    sky: u32,
    count: u32,
    reference: f32,
    mid_grey: f32,
    gamma: f32,
    saturation: f32,
}

@group(0) @binding(0) var<storage, read> samples: array<f32>;
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var<storage, read_write> brightest: atomic<u32>;

const weights = vec3<f32>(0.2126, 0.7152, 0.0722);

// The frame's brightest disk light, by its luminance: a non-negative number's bits order as the number does.
@compute @workgroup_size(64)
fn meter(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= view.count) {
        return;
    }
    let first = id.x * view.stride;
    atomicMax(&brightest, bitcast<u32>(dot(vec3<f32>(samples[first], samples[first + 1u], samples[first + 2u]), weights)));
}

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fragment(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let first = (u32(at.y) * view.columns + u32(at.x)) * view.stride;
    let light = vec3<f32>(samples[first], samples[first + 1u], samples[first + 2u]);
    let sky = vec3<f32>(samples[first + view.sky], samples[first + view.sky + 1u], samples[first + view.sky + 2u]);
    // Mid-grey at the reference blackbody, linear below it and rolled off above, the colour kept.
    let luminance = dot(light, weights);
    let exposure = luminance / view.reference;
    let mapped = select(1.0 - (1.0 - view.mid_grey) * exp(1.0 - exposure), view.mid_grey * exposure, exposure < 1.0);
    let scale = select(0.0, mapped / luminance, luminance > 0.0);
    let shown = pow(light * scale, vec3<f32>(1.0 / view.gamma));
    let grey = dot(shown, weights);
    let vivid = max(vec3<f32>(grey) + view.saturation * (shown - vec3<f32>(grey)), vec3<f32>(0.0));
    let exposed = vivid / max(max(vivid.x, max(vivid.y, vivid.z)), 1.0);
    return vec4<f32>(clamp(exposed + sky, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
"""


# --- math -----------------------------------------------------------------------------
def camera(azimuth: float, inclination: float, distance: float) -> dict:
    """The camera's orientation and distance from the hole, for a camera turned by `inclination` from above the hole
    and by `azimuth` around it; the shader finds its eye and its rest frame."""
    orientation = (mv.xy * (-azimuth / 2)).exp() * (mv.zx * (-inclination / 2)).exp()     # [] Rotor
    return dict(orientation=orientation, distance=distance)


def luminance(temperature: float) -> float:
    """A blackbody's luminance at a temperature, K, in the units of the traced light."""
    return float(render.blackbody(np.array(temperature)) @ render.luminance_weights)


def max_steps(step_fraction: float) -> int:
    """Steps enough for a ray to cover what the scenario's step and count cover."""
    return int(np.ceil(scenarios.camera_step_fraction * scenarios.n_camera_steps / step_fraction))



# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    import wgpu
    from rendercanvas.glfw import RenderCanvas, loop

    tracer = gpu.shader()
    device = tracer.device
    values = gpu.scene()
    state = dict(azimuth=0.0, inclination=scenarios.camera_inclination, distance=scenarios.camera_distance,
                 step_fraction=STEP, flowing=True, elapsed=0.0, spinning=True, sweep=0.0, following=True, reference=luminance(scenarios.exposure_temperature), dragging=None)

    canvas = RenderCanvas(size=WINDOW, title="Kerr", update_mode="continuous", max_fps=240, vsync=False)
    context = canvas.get_context("wgpu")
    context.configure(device=device, format="bgra8unorm")

    module = device.create_shader_module(code=DISPLAY)
    pipeline = device.create_render_pipeline(
        layout="auto", vertex={"module": module, "entry_point": "vertex", "buffers": []},
        fragment={"module": module, "entry_point": "fragment", "targets": [{"format": "bgra8unorm"}]},
        primitive={"topology": "triangle-list"})
    meter = device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "meter"})
    view = device.create_buffer(size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    brightest = device.create_buffer(size=4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
    done = device.create_buffer(size=4, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
    fixed = luminance(scenarios.exposure_temperature)
    sample = gpu_sample_layout(tracer)

    @canvas.add_event_handler("pointer_down", "pointer_up", "pointer_move", "wheel", "key_down")
    def on_event(event: dict) -> None:
        kind = event["event_type"]
        if kind == "pointer_down":
            state["dragging"] = (event["x"], event["y"])
        elif kind == "pointer_up":
            state["dragging"] = None
        elif kind == "pointer_move" and state["dragging"] is not None:
            x, y = state["dragging"]
            state["azimuth"] -= TURN_PER_POINT * (event["x"] - x)
            state["inclination"] = float(np.clip(state["inclination"] - TURN_PER_POINT * (event["y"] - y), 0.02, np.pi - 0.02))
            state["dragging"] = (event["x"], event["y"])
        elif kind == "wheel":
            state["distance"] = float(np.clip(state["distance"] * np.exp(ZOOM_PER_SCROLL * event["dy"]), NEAREST, FARTHEST))
        elif kind == "key_down" and event["key"] == "[":
            state["step_fraction"] = max(state["step_fraction"] / 2, FINEST)
        elif kind == "key_down" and event["key"] == "]":
            state["step_fraction"] = min(state["step_fraction"] * 2, COARSEST)
        elif kind == "key_down" and event["key"] == " ":
            state["flowing"] = not state["flowing"]
        elif kind == "key_down" and event["key"] == "s":
            state["spinning"] = not state["spinning"]
        elif kind == "key_down" and event["key"] == "e":
            state["following"] = not state["following"]

    clock = dict(last=time.perf_counter(), frames=0, reported=time.perf_counter())

    @canvas.request_draw
    def draw() -> None:
        now = time.perf_counter()
        if state["flowing"]:
            state["elapsed"] += (now - clock["last"]) * scenarios.flow_time / 10
        if state["spinning"]:
            state["sweep"] += (now - clock["last"]) * 2 * np.pi / SPIN_PERIOD
        clock["last"] = now
        spin_size = LARGEST_SPIN * np.sin(state["sweep"])
        target = context.get_current_texture()
        columns, rows = target.size[0], target.size[1]
        frame = values | camera(state["azimuth"], state["inclination"], state["distance"]) | dict(spin=mv.xy.dual() * spin_size,
            elapsed=state["elapsed"], fade=0.0, step_fraction=state["step_fraction"], n_steps=max_steps(state["step_fraction"]))
        encoder = device.create_command_encoder()
        outputs = tracer.encode(encoder, (rows, columns), ("viewer",), frame)
        samples, _, element_size = outputs["image"]
        reference = state["reference"] if state["following"] else fixed
        device.queue.write_buffer(view, 0, np.array([columns, element_size // 4, sample["sky"], rows * columns], dtype=np.uint32).tobytes()
                                  + np.array([reference, render.mid_grey, render.display_gamma, render.colour_saturation],
                                             dtype=np.float32).tobytes())
        size = rows * columns * element_size
        encoder.clear_buffer(brightest, 0, 4)
        metering = encoder.begin_compute_pass()
        metering.set_pipeline(meter)
        metering.set_bind_group(0, device.create_bind_group(layout=meter.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": {"buffer": samples, "offset": 0, "size": size}},
            {"binding": 1, "resource": {"buffer": view, "offset": 0, "size": 32}},
            {"binding": 2, "resource": {"buffer": brightest, "offset": 0, "size": 4}}]))
        metering.dispatch_workgroups(-(-rows * columns // 64))
        metering.end()
        group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": {"buffer": samples, "offset": 0, "size": size}},
            {"binding": 1, "resource": {"buffer": view, "offset": 0, "size": 32}}])
        drawing = encoder.begin_render_pass(color_attachments=[{"view": target.create_view(), "load_op": "clear",
                                                                "store_op": "store", "clear_value": (0, 0, 0, 1)}])
        drawing.set_pipeline(pipeline)
        drawing.set_bind_group(0, group)
        drawing.draw(3)
        drawing.end()
        encoder.copy_buffer_to_buffer(brightest, 0, done, 0, 4)
        device.queue.submit([encoder.finish()])
        # One frame on the GPU at a time, waited for by reading back its brightest light: the next is traced for where
        # the camera is when this one is done, frames never queue up behind each other, and the next exposure moves
        # toward the one that puts this frame's brightest light where the scenario's still has it.
        done.map_sync(wgpu.MapMode.READ, 0, 4)
        light = float(np.frombuffer(done.read_mapped(0, 4), dtype=np.float32)[0])
        done.unmap()
        if light > 0:
            state["reference"] *= (light / BRIGHTEST / state["reference"]) ** FOLLOWING
        clock["frames"] += 1
        if now - clock["reported"] > 1:
            print(f"{clock['frames'] / (now - clock['reported']):5.1f} frames a second at {columns} x {rows} pixels,"
                  f" step {state['step_fraction']:.3g} of the radius, spin {spin_size:+.2f}")
            clock["frames"], clock["reported"] = 0, now

    loop.run()


def gpu_sample_layout(tracer) -> dict:
    """Where the sky's colour begins in a traced pixel's sample, in numbers from its start."""
    offsets = dict(zip([member for member, _ in tracer.program.structs["Sample"]], tracer.layout.offsets("Sample")))
    return dict(sky=offsets["sky"] // 4)


if __name__ == "__main__":
    main()
