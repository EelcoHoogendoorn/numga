"""Interactive WebGPU window tracing a camera above a rotating Kerr black hole's accretion disk.

Drag to orbit the camera around the hole, scroll to zoom in or out, press `[` and `]` to halve or double the integration step fraction, press space to pause or resume the disk's orbital flow, press `s` to pause or resume the black-hole spin sweep, and press `e` to toggle automatic exposure metering against the fixed scenario reference temperature.

    python -m examples.relativity.kerr.viewer
"""

from __future__ import annotations

import time

import numpy as np

from examples.relativity.kerr import core, gpu, render, scenarios

mv = core.mv
# The window's size at the start, in points; the image is traced at the screen's pixels behind them.
window_size = (480, 320)
# The camera's nearest and farthest distance from the hole, in masses, and how fast dragging and scrolling move it.
nearest_distance, farthest_distance = 8.0, 90.0
turn_per_point, zoom_per_scroll = 0.006, 0.0015
# The step at the start, a fraction of the ray's radius, and the steps `[` and `]` move between: a step of 0.4
# changes a few pixels by a few levels from one eight times finer.
initial_step, finest_step, coarsest_step = 0.4, 0.05, 1.6
# The spin's largest size, in masses, and the seconds of one sweep there and back.
largest_spin, spin_period = 0.95, 12.0
# Where a frame's brightest light sits, in mid-greys, when the exposure follows it: where it sits in the scenario's
# still; and the share of the way each frame's exposure moves toward it.
brightest_target, exposure_smoothing = 3.7, 0.3

# The traced light, exposed as `render.tone_mapped_frame` exposes it and added to the sky, drawn over the window by one
# triangle: each pixel reads its sample from the tracer's output, `stride` numbers to a pixel.
display_shader = """
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
@compute @workgroup_size(8, 8)
fn meter(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= view.columns || id.y >= view.count / view.columns) {
        return;
    }
    let first = (id.y * view.columns + id.x) * view.stride;
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
def luminance(temperature: float) -> float:
    """Gives a blackbody's linear sRGB luminance at `temperature` K."""
    return float(render.blackbody(np.array(temperature)) @ render.luminance_weights)


def max_steps(step_fraction: float) -> int:
    """Scales the integration step count inversely with `step_fraction` to match the scenario ray budget."""
    return int(np.ceil(scenarios.camera_step_fraction * scenarios.n_camera_steps / step_fraction))


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Opens a live WebGPU window and runs the interactive Kerr ray-tracing loop."""
    import wgpu
    from rendercanvas.glfw import RenderCanvas, loop

    tracer = gpu.live_shader()
    device = tracer.device
    values = gpu.scene()
    state = dict(azimuth=0.0, inclination=scenarios.camera_inclination, distance=scenarios.camera_distance,
                 step_fraction=initial_step, flowing=True, elapsed=0.0, spinning=True, sweep=0.0, following=True,
                 reference=luminance(scenarios.exposure_temperature), dragging=None)

    canvas = RenderCanvas(size=window_size, title="Kerr", update_mode="continuous", max_fps=240, vsync=False)
    context = canvas.get_context("wgpu")
    context.configure(device=device, format="bgra8unorm")

    module = device.create_shader_module(code=display_shader)
    pipeline = device.create_render_pipeline(
        layout="auto", vertex={"module": module, "entry_point": "vertex", "buffers": []},
        fragment={"module": module, "entry_point": "fragment", "targets": [{"format": "bgra8unorm"}]},
        primitive={"topology": "triangle-list"})
    meter = device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "meter"})
    view = device.create_buffer(size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    brightest = device.create_buffer(size=4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
    # Two readbacks of the brightest light, a frame apart, each read while the other's frame runs.
    readback_buffers = [device.create_buffer(size=4, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST) for _ in range(2)]
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
            state["azimuth"] -= turn_per_point * (event["x"] - x)
            state["inclination"] = float(np.clip(state["inclination"] - turn_per_point * (event["y"] - y), 0.02, np.pi - 0.02))
            state["dragging"] = (event["x"], event["y"])
        elif kind == "wheel":
            state["distance"] = float(np.clip(state["distance"] * np.exp(zoom_per_scroll * event["dy"]), nearest_distance, farthest_distance))
        elif kind == "key_down" and event["key"] == "[":
            state["step_fraction"] = max(state["step_fraction"] / 2, finest_step)
        elif kind == "key_down" and event["key"] == "]":
            state["step_fraction"] = min(state["step_fraction"] * 2, coarsest_step)
        elif kind == "key_down" and event["key"] == " ":
            state["flowing"] = not state["flowing"]
        elif kind == "key_down" and event["key"] == "s":
            state["spinning"] = not state["spinning"]
        elif kind == "key_down" and event["key"] == "e":
            state["following"] = not state["following"]

    clock = dict(last=time.perf_counter(), frames=0, drawn=0, reported=time.perf_counter())

    @canvas.request_draw
    def draw() -> None:
        current_time = time.perf_counter()
        if state["flowing"]:
            state["elapsed"] = (state["elapsed"] + (current_time - clock["last"]) * scenarios.flow_time / 10) % scenarios.flow_time
        if state["spinning"]:
            state["sweep"] += (current_time - clock["last"]) * 2 * np.pi / spin_period
        clock["last"] = current_time
        spin_size = largest_spin * np.sin(state["sweep"])
        target = context.get_current_texture()
        columns, rows = target.size[0], target.size[1]
        frame = values | dict(
            azimuth=state["azimuth"], inclination=state["inclination"], distance=state["distance"],
            spin=mv.xy.dual() * spin_size,
            elapsed=state["elapsed"],
            fade=np.pi / 2 * state["elapsed"] / scenarios.flow_time,
            step_fraction=state["step_fraction"], n_steps=max_steps(state["step_fraction"]))
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
        metering.dispatch_workgroups(-(-columns // 8), -(-rows // 8))
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
        encoder.copy_buffer_to_buffer(brightest, 0, readback_buffers[clock["drawn"] % 2], 0, 4)
        device.queue.submit([encoder.finish()])
        previous_buffer = readback_buffers[(clock["drawn"] + 1) % 2]
        previous_buffer.map_sync(wgpu.MapMode.READ, 0, 4)
        light = float(np.frombuffer(previous_buffer.read_mapped(0, 4), dtype=np.float32)[0])
        previous_buffer.unmap()
        if light > 0:
            state["reference"] *= (light / brightest_target / state["reference"]) ** exposure_smoothing
        clock["frames"] += 1
        clock["drawn"] += 1
        if current_time - clock["reported"] > 1:
            print(f"{clock['frames'] / (current_time - clock['reported']):5.1f} frames a second at {columns} x {rows} pixels,"
                  f" step {state['step_fraction']:.3g} of the radius, spin {spin_size:+.2f}")
            clock["frames"], clock["reported"] = 0, current_time

    loop.run()


def gpu_sample_layout(tracer) -> dict:
    """Gives the 32-bit float word offset of the `sky` field inside `Sample`."""
    offsets = dict(zip([member for member, _ in tracer.program.structs["Sample"]], tracer.layout.offsets("Sample")))
    return dict(sky=offsets["sky"] // 4)


if __name__ == "__main__":
    main()
