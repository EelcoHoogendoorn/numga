"""A page that traces the black hole in the browser, on the reader's own GPU: the GPU notebook's shader, compiled once
here, with the data it reads, the sky, the blackbody colours and the disk's waves, written into one HTML file. The page
drives the shader with the spin and the camera's azimuth, inclination and distance, from which the shader forms the
camera's orientation; nothing is computed on a server, and no picture is sent. The sky is drawn on the GPU when the
page loads, so that the file holds only the shader and a few tables, and the image traces at fewer pixels when frames
come slower than the GPU can keep up with.

Drag to fly the camera around the hole, scroll to move it in or out, `[` and `]` to halve or double the step, space to
stop or restart the disk, `s` the spin's sweep, and `e` the exposure's following of the brightest light, as in the
window of `viewer`.

    python -m examples.relativity.kerr.web            # writes kerr_webgpu.html beside this file
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from examples.relativity.kerr import core, gpu, render, scenarios, viewer
from examples.util.wgsl.runner import coefficients

mv = core.mv
# The page's image, in points, traced at the screen's pixels behind them, or fewer when frames come slowly; and the
# sky's size, the scenario's.
SIZE = viewer.WINDOW
SKY = (scenarios.sky_width, scenarios.sky_height)

# The sky of `render.star_texture`, drawn on the GPU into a texture once: a faint dusty band across the background,
# and stars spread evenly over the sphere, each a small Gaussian stamp. Each texel gathers the stars of its own cell and
# the eight around it, a cell holding at most one, at a hashed place, kept by the cosine of its latitude, so that they
# fall evenly over the sphere as the catalogue's do; their brightness and warmth are hashed as the catalogue's are drawn.
SKY_SHADER = """
@group(0) @binding(0) var sky: texture_storage_2d<rgba16float, write>;

const pi = 3.141592653589793;
// A cell's side in texels: one star per 140 texels on average once the cosine has thinned them by 2 / pi.
const cell_side = 9.44;

fn hash(value: u32) -> u32 {
    var state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn uniform(cell: vec2<u32>, which: u32) -> f32 {
    return f32(hash(hash(cell.x * 1973u + cell.y * 9277u + 26699u) + which * 1046527u)) / 4294967296.0;
}

@compute @workgroup_size(8, 8)
fn draw(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(sky);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let longitude = (f32(id.x) + 0.5) * (2 * pi / f32(size.x)) - pi;
    let latitude = (0.5 - (f32(id.y) + 0.5) / f32(size.y)) * pi;
    let direction = vec3<f32>(cos(latitude) * cos(longitude), cos(latitude) * sin(longitude), sin(latitude));
    let band_height = 0.2 * direction.x - 0.6 * direction.y - sqrt(0.6) * direction.z;
    let cloud = sin(13 * direction.x + 9 * direction.y - 7 * direction.z)
        + 0.45 * sin(31 * direction.x - 23 * direction.y + 19 * direction.z)
        + 0.18 * sin(79 * direction.x + 47 * direction.y + 53 * direction.z);
    let band = exp(-pow(band_height / 0.16, 2.0)) * (0.75 + 0.2 * cloud);
    let dust = exp(-pow((band_height + 0.025 * cloud) / 0.035, 2.0));
    var colour = vec3<f32>(0.006, 0.009, 0.019) + band * vec3<f32>(0.065, 0.058, 0.09) * (1 - 0.7 * dust);

    let cells = max(vec2<u32>(round(vec2<f32>(size) / cell_side)), vec2<u32>(1u));
    let cell_size = vec2<f32>(size) / vec2<f32>(cells);
    let home = vec2<i32>(vec2<f32>(f32(id.x) + 0.5, f32(id.y) + 0.5) / cell_size);
    for (var down = -1; down <= 1; down++) {
        for (var across = -1; across <= 1; across++) {
            let row = home.y + down;
            if (row < 0 || row >= i32(cells.y)) {
                continue;
            }
            let cell = vec2<u32>(u32((home.x + across + i32(cells.x)) % i32(cells.x)), u32(row));
            let place = (vec2<f32>(cell) + vec2<f32>(uniform(cell, 0u), uniform(cell, 1u))) * cell_size;
            let star_latitude = (0.5 - place.y / f32(size.y)) * pi;
            if (uniform(cell, 2u) > cos(star_latitude)) {
                continue;
            }
            // The star's place from this texel's centre, across the seam where the sky wraps round.
            var offset = place - vec2<f32>(f32(id.x) + 0.5, f32(id.y) + 0.5);
            offset.x = offset.x - f32(size.x) * round(offset.x / f32(size.x));
            if (abs(offset.x) > 2.5 || abs(offset.y) > 2.5) {
                continue;
            }
            let brightness = 0.35 + 2.5 * pow(uniform(cell, 3u), 6.0);
            let warmth = uniform(cell, 4u);
            let tint = mix(vec3<f32>(0.57, 0.72, 1.0), vec3<f32>(1.0, 0.79, 0.51), warmth);
            colour += exp(-dot(offset, offset) / (2 * 0.6 * 0.6)) * brightness * tint;
        }
    }
    textureStore(sky, id.xy, vec4<f32>(clamp(colour, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
"""
PAGE = Path(__file__).parent / "kerr_webgpu.html"


# --- plumbing -------------------------------------------------------------------------
def packed(tracer, values: dict) -> tuple[bytes, dict]:
    """The uniform block as the runner packs it, and where each uniform begins in it."""
    data, offsets = bytearray(), {}
    for name, kind in tracer.program.uniforms:
        alignment, _ = tracer.layout.size(kind)
        data += bytes(-len(data) % alignment)
        offsets[name] = len(data)
        data += tracer.layout.pack(kind, values[name])
    return bytes(data) + bytes(-len(data) % 16), offsets


def half_floats(image: np.ndarray) -> dict:
    """An image `[rows, columns, channels]` as a half-float RGBA texture, the alpha one."""
    padded = np.ones(image.shape[:2] + (4,), dtype=np.float16)
    padded[..., :image.shape[2]] = image
    return dict(width=padded.shape[1], height=padded.shape[0], data=encoded(padded.tobytes()))


def encoded(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def contents() -> dict:
    """Everything the page reads: the compiled shader, its bindings, its uniforms and the data they hold."""
    tracer = gpu.live_shader()
    program = tracer.program
    values = gpu.scene() | dict(pixels=[SIZE[0], SIZE[1]], step_fraction=viewer.STEP, n_steps=viewer.max_steps(viewer.STEP))
    uniforms, offsets = packed(tracer, values)
    kinds = dict(program.uniforms)
    mode = tracer.layout.resolve("Mode")
    # Half floats hold the blackbody table only scaled into their range, its hottest entry one; the light it gives, and
    # the exposure it is shown at, scale alike.
    unit = 1 / values["blackbodies"].max()
    sample = viewer.gpu_sample_layout(tracer)
    return dict(
        code=program.code, display=viewer.DISPLAY,
        entry_points=[[name, list(size[:2])] for name, size in program.entry_points],
        bindings=[dict(index=binding.index, kind=binding.kind, name=binding.name,
                       writable=binding.kind == "storage" and binding.access == "read_write") for binding in program.bindings],
        uniforms=encoded(uniforms), offsets=offsets,
        spin=np.frombuffer(coefficients(mv.xy.dual(), kinds["spin"]), np.float32).tolist(),
        modes=encoded(tracer.layout.array(mode, values["modes"])),
        blackbodies=half_floats(values["blackbodies"] * unit),
        sky=dict(code=SKY_SHADER, width=SKY[0], height=SKY[1]),
        stride=tracer.layout.stride(tracer.layout.resolve("Sample")), sky_offset=sample["sky"],
        size=list(SIZE),
        start=dict(inclination=scenarios.camera_inclination, distance=scenarios.camera_distance, step=viewer.STEP,
                   flow_time=scenarios.flow_time, scenario_step=scenarios.camera_step_fraction,
                   scenario_steps=scenarios.n_camera_steps),
        limits=dict(nearest=viewer.NEAREST, farthest=viewer.FARTHEST, finest=viewer.FINEST, coarsest=viewer.COARSEST,
                    largest_spin=viewer.LARGEST_SPIN, spin_period=viewer.SPIN_PERIOD, turn=viewer.TURN_PER_POINT,
                    zoom=viewer.ZOOM_PER_SCROLL, brightest=viewer.BRIGHTEST, following=viewer.FOLLOWING),
        exposure=dict(fixed=viewer.luminance(scenarios.exposure_temperature) * unit, mid_grey=render.mid_grey,
                      gamma=render.display_gamma, saturation=render.colour_saturation))


def main() -> None:
    template = (Path(__file__).parent / "web_template.html").read_text()
    PAGE.write_text(template.replace("/*CONTENTS*/null", json.dumps(contents())))
    print(f"wrote {PAGE.name}, {PAGE.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
