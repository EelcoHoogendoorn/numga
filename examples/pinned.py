"""Regenerating the figures and animations tracked in `plots/` for the README and documentation runs
each listed example's `scenarios.pinned()` entry point in its own subprocess. Inside each scenario,
`pin_figure` and `pin_animation` save to `PLOT_DIR` and overwrite the tracked file under `plots/`
with its fixed name.

Running `python -m examples.pinned` rebuilds every entry in gallery order and reports the elapsed
seconds for each package; passing short names such as `python -m examples.pinned dynamo gps`
rebuilds only the matching examples.
"""

from __future__ import annotations

import subprocess
import sys
import time

# Example subpackages with a `scenarios.pinned()` entry point, ordered to match the gallery followed
# by documentation and article figures.
EXAMPLES = (
    "estimation.multiview",
    "relativity.kerr",
    "thermodynamics.snowflake",
    "quantum.condensate",
    "fluids.wing",
    "electromagnetism.dynamo",
    "mechanics.rattleback",
    "relativity.gravitational_waves",
    "quantum.kane_mele",
    "electromagnetism.beltrami_fields",
    "thermodynamics.rubber_band",
    "geometry.scenegraph",
    "surfaces.arap",
    "mechanics.bending",
    "quantum.skyrmions",
    "relativity.twistors",
    "electromagnetism.superconductor",
    "fluids.superfluid",
    "thermodynamics.magnetotransport",
    "fluids.eigen_fluids",
    "electromagnetism.induced_charge",
    "fluids.liquid_crystals",
    "quantum.process_tomography",
    "mechanics.elastica",
    "electromagnetism.piezoelectric",
    "optics.black_sandwich",
    "relativity.ligo",
    "electromagnetism.moving_charge",
    "quantum.magnetic_resonance",
    "fluids.jeffery",
    "mechanics.slip_systems",
    "thermodynamics.phase_change",
    "mechanics.modes",
    "estimation.odometry",
    "mechanics.area_transport",
    "relativity.gps",
    "orientation.normals",
    "orientation.mirror_energy",
    "orientation.spinning_wheel",
    "optics.crystal_diffraction",
)


def main() -> None:
    chosen = set(sys.argv[1:])
    for example in EXAMPLES:
        if chosen and example.rsplit(".", 1)[-1] not in chosen:
            continue
        print(f"Pinning {example}", flush=True)
        start = time.perf_counter()
        subprocess.run([sys.executable, "-c", f"from examples.{example}.scenarios import pinned; pinned()"], check=True)
        print(f"Pinned {example} in {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
