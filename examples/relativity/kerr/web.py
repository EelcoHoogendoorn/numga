"""Standalone browser WebGPU page builder for interactive Kerr black-hole ray tracing.

Compiles the live WGSL shader from `gpu`, packs its uniform buffer, blackbody lookup table, clump wave modes, and procedural sky compute shader into `kerr_webgpu.html`, so the browser runs the entire ray tracer locally on the reader's GPU.

    python -m examples.relativity.kerr.web
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from examples.relativity.kerr import core, gpu, render, scenarios, viewer
from numga.wgsl.page import encoded, packed, write_page
from numga.wgsl.runner import coefficients

mv = core.mv
# The page's image, in points, traced at the screen's pixels behind them, or fewer when frames come slowly; and the
# sky's size, the scenario's.
page_size = viewer.window_size
sky_size = (scenarios.n_sky_width, scenarios.n_sky_height)

sky_shader = """
@group(0) @binding(0) var sky: texture_storage_2d<rgba16float, write>;

const pi = 3.141592653589793;
// A grid tile's side in texels: one star per 140 texels on average once the cosine has thinned them by 2 / pi.
const tile_side = 9.44;

fn hash(value: u32) -> u32 {
    var state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn uniform(tile: vec2<u32>, which: u32) -> f32 {
    return f32(hash(hash(tile.x * 1973u + tile.y * 9277u + 26699u) + which * 1046527u)) / 4294967296.0;
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

    let tiles = max(vec2<u32>(round(vec2<f32>(size) / tile_side)), vec2<u32>(1u));
    let tile_size = vec2<f32>(size) / vec2<f32>(tiles);
    let home = vec2<i32>(vec2<f32>(f32(id.x) + 0.5, f32(id.y) + 0.5) / tile_size);
    for (var down = -1; down <= 1; down++) {
        for (var right = -1; right <= 1; right++) {
            let row = home.y + down;
            if (row < 0 || row >= i32(tiles.y)) {
                continue;
            }
            let tile = vec2<u32>(u32((home.x + right + i32(tiles.x)) % i32(tiles.x)), u32(row));
            let place = (vec2<f32>(tile) + vec2<f32>(uniform(tile, 0u), uniform(tile, 1u))) * tile_size;
            let star_latitude = (0.5 - place.y / f32(size.y)) * pi;
            if (uniform(tile, 2u) > cos(star_latitude)) {
                continue;
            }
            var offset = place - vec2<f32>(f32(id.x) + 0.5, f32(id.y) + 0.5);
            offset.x = offset.x - f32(size.x) * round(offset.x / f32(size.x));
            if (abs(offset.x) > 2.5 || abs(offset.y) > 2.5) {
                continue;
            }
            let brightness = 0.35 + 2.5 * pow(uniform(tile, 3u), 6.0);
            let warmth = uniform(tile, 4u);
            let tint = mix(vec3<f32>(0.57, 0.72, 1.0), vec3<f32>(1.0, 0.79, 0.51), warmth);
            colour += exp(-dot(offset, offset) / (2 * 0.6 * 0.6)) * brightness * tint;
        }
    }
    textureStore(sky, id.xy, vec4<f32>(clamp(colour, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
"""
page_path = Path(__file__).parent / "kerr_webgpu.html"


# --- plumbing -------------------------------------------------------------------------
def half_floats(image: np.ndarray) -> dict:
    """Packs `image` (`[n_rows, n_columns, n_channels]`) into a base64-encoded half-float RGBA texture dictionary."""
    padded = np.ones(image.shape[:2] + (4,), dtype=np.float16)
    padded[..., :image.shape[2]] = image
    return dict(width=padded.shape[1], height=padded.shape[0], data=encoded(padded.tobytes()))


def contents() -> dict:
    """Assembles the compiled WGSL shader, bindings, uniforms, lookup tables, and viewer limits for the standalone page."""
    tracer = gpu.live_shader()
    program = tracer.program
    values = gpu.scene() | dict(pixels=[page_size[0], page_size[1]], step_fraction=viewer.initial_step,
                                n_steps=viewer.max_steps(viewer.initial_step))
    uniforms, offsets = packed(tracer, values)
    kinds = dict(program.uniforms)
    mode = tracer.layout.resolve("Mode")
    unit = 1 / values["blackbodies"].max()
    sample = viewer.gpu_sample_layout(tracer)
    return dict(
        code=program.code, display=viewer.display_shader,
        entry_points=[[name, list(size[:2])] for name, size in program.entry_points],
        bindings=[dict(index=binding.index, kind=binding.kind, name=binding.name,
                       writable=binding.kind == "storage" and binding.access == "read_write") for binding in program.bindings],
        uniforms=encoded(uniforms), offsets=offsets,
        spin=np.frombuffer(coefficients(mv.xy.dual(), kinds["spin"]), np.float32).tolist(),
        modes=encoded(tracer.layout.array(mode, values["modes"])),
        blackbodies=half_floats(values["blackbodies"] * unit),
        sky=dict(code=sky_shader, width=sky_size[0], height=sky_size[1]),
        stride=tracer.layout.stride(tracer.layout.resolve("Sample")), sky_offset=sample["sky"],
        size=list(page_size),
        start=dict(inclination=scenarios.camera_inclination, distance=scenarios.camera_distance, step=viewer.initial_step,
                   flow_time=scenarios.flow_time, scenario_step=scenarios.camera_step_fraction,
                   scenario_steps=scenarios.n_camera_steps),
        limits=dict(nearest=viewer.nearest_distance, farthest=viewer.farthest_distance, finest=viewer.finest_step,
                    coarsest=viewer.coarsest_step, largest_spin=viewer.largest_spin, spin_period=viewer.spin_period,
                    turn=viewer.turn_per_point, zoom=viewer.zoom_per_scroll, brightest=viewer.brightest_target,
                    following=viewer.exposure_smoothing),
        exposure=dict(fixed=viewer.luminance(scenarios.exposure_temperature) * unit, mid_grey=render.mid_grey,
                      gamma=render.display_gamma, saturation=render.colour_saturation))


def page() -> None:
    """Writes `kerr_webgpu.html` from `web_template.html` and `contents()`."""
    write_page(Path(__file__).parent / "web_template.html", page_path, contents())


if __name__ == "__main__":
    page()
