"""A page that traces the black hole in the browser, on the reader's own GPU: the GPU notebook's shader, compiled once
here, with the data it reads, the sky, the blackbody colours and the disk's waves, written into one HTML file. The page
drives the shader with the spin and the camera's orientation and distance; nothing is computed on a server, and no
picture is sent.

Drag to fly the camera around the hole, scroll to move it in or out, `[` and `]` to halve or double the step, space to
stop or restart the disk, `s` the spin's sweep, and `e` the exposure's following of the brightest light, as in the
window of `viewer`.

    python -m examples.relativity.kerr.web            # writes build/kerr.html
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from examples.relativity.kerr import core, gpu, render, scenarios, viewer
from examples.util.wgsl.runner import coefficients

mv = core.mv
# The page's image, in points, traced at the screen's pixels behind them: smaller than the window's, for slower GPUs;
# and the sky's size, half the scenario's, so that the page stays a few megabytes.
SIZE = (360, 240)
SKY = (scenarios.sky_width // 2, scenarios.sky_height // 2)
PAGE = Path("build") / "kerr.html"


# --- math -----------------------------------------------------------------------------
def rotor_terms(kind) -> list[list[float]]:
    """The camera's orientation, `(mv.xy * (-azimuth / 2)).exp() * (mv.zx * (-inclination / 2)).exp()`, as four
    coefficient arrays the page weights by the products of the half angles' cosines and sines: `cos * cos`,
    `cos * sin`, `sin * cos` and `sin * sin` of the azimuth and the inclination."""
    one = mv.scalar([1.0])
    return [np.frombuffer(coefficients(term, kind), np.float32).tolist()
            for term in (one * one, one * -mv.zx, -mv.xy * one, -mv.xy * -mv.zx)]


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
    tracer = gpu.shader()
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
        rotor=rotor_terms(kinds["orientation"]),
        modes=encoded(tracer.layout.array(mode, values["modes"])),
        blackbodies=half_floats(values["blackbodies"] * unit),
        stars=half_floats(render.star_texture(SKY[0], SKY[1], scenarios.sky_seed)),
        stride=tracer.layout.stride(tracer.layout.resolve("Sample")), sky=sample["sky"],
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
    PAGE.parent.mkdir(exist_ok=True)
    template = (Path(__file__).parent / "web.html").read_text()
    PAGE.write_text(template.replace("/*CONTENTS*/null", json.dumps(contents())))
    print(f"wrote {PAGE}, {PAGE.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
