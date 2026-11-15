"""Scene parameters and animation generation for spinor pulses and standing modes on a triangulated sphere."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import numpy as np

from numga import stack
from numga.sparse import SparseExtensor
from numga.topology import Topology
from examples.surfaces.dirac_waves import core
from examples.util import shapes

mv = core.mv
# Subdivision depth of the unit icosphere, leapfrog step size, and GIF frame duration in milliseconds.
n_subdivisions = 3
dt = 0.04
frame_duration_ms = 40
# Gaussian bump width and step count slightly past the antipodal focusing time `pi`.
bump_width = 0.25
n_steps = 84
# Number of nonzero frequency shells (multiplicity `8 * k` after the 4 zero-frequency constant spinors) and phases per period.
n_levels = 5
n_constant = 4
n_phases = 25


# --- math -----------------------------------------------------------------------------
def sphere_bump_setup(
    subdivisions: int,
) -> tuple[Topology, core.Point, SparseExtensor, SparseExtensor, SparseExtensor, core.Even]:
    """Subdivides the unit sphere (`Point-[V]`), builds its Dirac and area maps, and places an xy-plane Gaussian bump at the north pole."""
    topology, coordinates = shapes.sphere(subdivisions)
    vertices = mv.vector(coordinates).dual().field()                            # Point-[V]
    dirac, face_areas, vertex_areas = core.dirac_system(topology, vertices)
    initial_spinor = ((vertices - mv.z.dual()).norm_squared() * (-0.5 / bump_width**2)).exp() * mv.xy   # Even+[V]
    return topology, vertices, dirac, face_areas, vertex_areas, initial_spinor


def sphere_standing_setup(
    topology: Topology,
    vertices: core.Point,
    dirac: SparseExtensor,
    face_areas: SparseExtensor,
    vertex_areas: SparseExtensor,
    levels: int,
) -> tuple[core.Scalar, core.Even, core.Even, core.Even, np.ndarray]:
    """Solves the first `levels` nonzero Dirac frequency shells on the sphere and aligns one standing mode per shell with an equatorial bump."""
    level_bounds = n_constant + 8 * np.cumsum(np.arange(levels + 1))            # [levels + 1] -> [...]
    probe = ((vertices - mv.x.dual()).norm_squared() * (-0.5 / bump_width**2)).exp() * (mv.yz + mv.zx + mv.xy)   # Even+[V]
    frequencies = np.arange(1, levels + 1)
    squared_frequencies, vertex_modes, face_modes = core.standing_modes(
        dirac, face_areas, vertex_areas, level_bounds, probe, frequencies,
    )
    return squared_frequencies, vertex_modes, face_modes, probe, frequencies


# --- plumbing -------------------------------------------------------------------------
def save_states(
    states: Iterable,
    animate: Callable[..., Iterable[np.ndarray]],
    *arguments,
    name: str,
    duration_ms: int,
) -> list:
    """Streams `states` once through `animate` into `save_animation` and returns the collected states."""
    from examples.util.animation import save_animation

    recorded = []

    def passing() -> Iterator:
        for state in states:
            recorded.append(state)
            yield state

    save_animation(animate(*arguments, passing()), name, duration_ms)
    return recorded


def main() -> None:
    import matplotlib.pyplot as plt
    from examples.surfaces.dirac_waves import render
    from examples.util.animation import save_figure

    topology, vertices, dirac, face_areas, vertex_areas, initial_spinor = sphere_bump_setup(n_subdivisions)
    bump_fig = render.spinor_figure(topology, vertices, initial_spinor, dirac * initial_spinor)
    save_figure(bump_fig, "dirac_waves_bump")
    plt.close(bump_fig)

    recorded_pulse = save_states(
        core.pulse_states(dirac, face_areas, vertex_areas, initial_spinor, dt, n_steps),
        render.pulse_animation,
        topology,
        vertices,
        name="dirac_waves_pulse",
        duration_ms=frame_duration_ms,
    )

    phases = np.linspace(0.0, 2 * np.pi, n_phases, endpoint=False)
    squared_frequencies, vertex_modes, face_modes, probe, frequencies = sphere_standing_setup(
        topology, vertices, dirac, face_areas, vertex_areas, n_levels,
    )
    recorded_modes = save_states(
        core.mode_states(vertex_modes, face_modes, phases),
        render.modes_animation,
        topology,
        vertices,
        n_levels,
        name="dirac_waves_modes",
        duration_ms=frame_duration_ms,
    )

    # checks
    # Area-weighted `dirac` and `~dirac` are mutual adjoints under the scalar product.
    sample_face = recorded_modes[n_phases // 4][1][0]                           # Even-[F]
    forward_pair = (face_areas * sample_face.reverse().scalar_product(dirac * probe)).sites.sum()   # Scalar+
    reverse_pair = (~dirac * (face_areas * sample_face)).reverse().scalar_product(probe).sites.sum()   # Scalar+
    assert (forward_pair - reverse_pair).abs() < 1e-12 * forward_pair.abs()

    # Staggered leapfrog integration conserves the discrete Dirac wave energy across every step.
    energies = stack([
        (vertex_areas * v_after.scalar_norm_squared()).sites.sum()
        + (face_areas * f_before.reverse().scalar_product(f_after)).sites.sum()
        for (v_after, f_after), (_, f_before) in zip(recorded_pulse[1:], recorded_pulse)
    ])                                                                          # [n_steps - 1] Scalar+
    assert ((energies - energies[0]).abs() < 1e-12 * energies[0].abs()).all()

    # At time `pi`, the north-pole pulse focuses onto the south-polar cap (`z < -0.9`).
    antipodal_spinor = recorded_pulse[int(round(np.pi / dt))][0]                # Even+[V]
    vertex_energy = (vertex_areas * antipodal_spinor.scalar_norm_squared()).batch()   # [V] Scalar+
    south_cap = (mv.z & vertices.batch()) < -0.9
    assert bool(vertex_energy[south_cap].sum() > 0.5 * vertex_energy.sum())

    # On the unit sphere, four constant spinors sit at zero frequency followed by integer frequencies `k` of multiplicity `8 * k`.
    assert (squared_frequencies[:n_constant].abs() < 1e-9).all()
    expected = np.repeat(frequencies, 8 * frequencies)
    assert (((squared_frequencies[n_constant:].square_root() - expected) / expected).abs() < 0.03).all()


if __name__ == "__main__":
    main()
