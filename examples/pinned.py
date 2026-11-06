"""Every picture the repository tracks, for its README and its docs, made again. Each example that pins pictures has a
`pinned` in its scenarios that makes exactly those pictures, through `pin_figure` and `pin_animation`, which copy them
over the checkout's `plots/` under their fixed names. `python -m examples.pinned` runs them all, one example at a time,
each in a process of its own, and says how long each took, so that a slow one shows; naming examples,
`python -m examples.pinned dynamo gps`, runs only theirs.
"""

from __future__ import annotations

import subprocess
import sys
import time

# The examples that pin pictures, as their packages under `examples`, in the order the gallery shows them, then the
# docs' other pictures.
EXAMPLES = (
    "geometry.scenegraph",
    "relativity.kerr",
    "estimation.multiview",
    "thermodynamics.snowflake",
    "fluids.wing",
    "quantum.kane_mele",
    "electromagnetism.dynamo",
    "relativity.twistors",
    "electromagnetism.beltrami_fields",
    "quantum.condensate",
    "mechanics.rattleback",
    "relativity.gravitational_waves",
    "mechanics.bending",
    "fluids.eigen_fluids",
    "electromagnetism.superconductor",
    "fluids.superfluid",
    "thermodynamics.magnetotransport",
    "quantum.process_tomography",
    "mechanics.elastica",
    "electromagnetism.piezoelectric",
    "optics.black_sandwich",
    "relativity.ligo",
    "electromagnetism.moving_charge",
    "quantum.magnetic_resonance",
    "surfaces.arap",
    "fluids.jeffery",
    "mechanics.slip_systems",
    "thermodynamics.phase_change",
    "mechanics.modes",
    "estimation.odometry",
    "mechanics.area_transport",
    "relativity.gps",
    "orientation.normals",
    "orientation.mirror_energy",
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
