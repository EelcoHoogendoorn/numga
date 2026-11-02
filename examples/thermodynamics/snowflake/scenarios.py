"""A snowflake growing from a seed of ice in undercooled water, and the same seed growing with no anisotropy, which
grows no arms of its own, each until its tips near the dish's rim: on a mesh of staggered rings, with no six-fold
symmetry of its own, the six arms come from the crystal's six-fold surface energy. The total entropy over the
snowflake's growth rises throughout."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack

from examples.thermodynamics.snowflake import core
from examples.util import shapes
from examples.util.topology import Topology

mv = core.mv

# The dish of water's radius, and the radial intervals of its mesh, nearly equilateral triangles in staggered rings.
dish_radius, n_divisions = 4.5, 150
# The seed of ice in its middle: its radius and the width of its edge.
seed_radius, seed_edge = 0.1, 0.02
# The crystal's axis, a plane whose normal, up, one arm grows along.
crystal_axis = mv.y                                                           # [] Plane+
# The phase's relaxation time, its edge's mean width, the six-fold anisotropy of that width, the steepness below
# which the anisotropy fades, where the phase is flat, and the double well's height.
relaxation_time, mean_width, anisotropy, regularization, well_height = 3e-4, 0.0082, 0.025, 1.0, 1 / 6
# The latent heat's warming per unit of ice, the melting temperature, with the cold water at nothing, and absolute
# zero on that scale.
latent_heat, melting_temperature, absolute_zero = 1.8, 1.0, -5.0
# The time step and the steps between frames; the frames each run takes until its tips reach about nine tenths of the
# dish's radius, the bush without anisotropy growing slower; the frames' duration in milliseconds.
dt, stride = 2e-4, 20
n_snowflake_frames, n_isotropic_frames = 75, 111
frame_duration_ms = 40


def dish() -> tuple[Topology, np.ndarray, core.Point]:
    """The dish of water: its topology, its vertices' places across and up `[vertices, 2]`, and its vertices, places of
    unit volume `Point-[vertices]`."""
    topology, coordinates = shapes.concentric_disk(np.array([dish_radius]), n_divisions)
    vertices = (core.volume + core.across * coordinates[:, 0] + core.up * coordinates[:, 1]).field()
    return topology, coordinates, vertices


def seed_phase(coordinates: np.ndarray) -> np.ndarray:
    """The phase at the start at places across and up `[vertices, 2]`, `[vertices]`: ice within the seed's radius of
    the middle, with a smooth edge."""
    return (1 + np.tanh((seed_radius - np.hypot(*coordinates.T)) / seed_edge)) / 2


def seeded(coordinates: np.ndarray) -> tuple[core.Point, core.Scalar]:
    """The water's parcels `Point-[vertices]` and the phase `Scalar+[vertices]` at the start: the seed's phase, and
    parcels of unit volume holding the ice it gives, with no heat, the water at the cold water's temperature."""
    phase = mv.scalar(seed_phase(coordinates)[:, None]).field()               # Scalar+[vertices]
    return core.volume + core.ice * core.held_ice(phase) + 0 * core.heat, phase


def frozen(topology: Topology, vertices: core.Point, parcels: core.Point, phase: core.Scalar, anisotropy: float,
           n_frames: int, stride: int) -> Iterator[tuple[core.Point, core.Scalar]]:
    """The water's parcels `Point-[vertices]` and the phase `Scalar+[vertices]` at each frame, from these, with this
    anisotropy."""
    areas, gradients, cells = core.triangles(topology, vertices)
    return core.growth(topology, areas, gradients, cells, parcels, phase, crystal_axis, relaxation_time, mean_width,
                       anisotropy, regularization, well_height, latent_heat, melting_temperature, absolute_zero, dt,
                       n_frames, stride)


def entropies(topology: Topology, vertices: core.Point, states: list[tuple[core.Point, core.Scalar]]) -> core.Amount:
    """The total entropy at each of these states `[frames] Scalar-`, the parcels' less the edge's free energy over the
    absolute melting temperature."""
    areas, gradients, cells = core.triangles(topology, vertices)
    gradient, spreading, _ = core.operators(topology, areas, gradients)
    return stack([core.total_entropy(areas, cells, gradient, spreading, crystal_axis, mean_width, anisotropy,
                                     regularization, well_height, latent_heat, melting_temperature, absolute_zero,
                                     parcels, phase)
                  for parcels, phase in states])


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.snowflake import render

    topology, coordinates, vertices = dish()
    parcels, phase = seeded(coordinates)
    save_figure(render.draw_seed(topology, vertices, parcels), "snowflake_seed")
    snowflake = list(frozen(topology, vertices, parcels, phase, anisotropy, n_snowflake_frames, stride))
    isotropic = list(frozen(topology, vertices, parcels, phase, 0.0, n_isotropic_frames, stride))
    snowflake_frames = list(render.animate(topology, vertices, snowflake))
    isotropic_frames = list(render.animate(topology, vertices, isotropic))
    save_animation(render.side_by_side(snowflake_frames, isotropic_frames), "snowflake_isotropic", frame_duration_ms)
    save_figure(render.draw_grown(topology, vertices, snowflake[-1][0]), "snowflake_grown")
    save_figure(render.draw_grown(topology, vertices, isotropic[-1][0]), "snowflake_isotropic_grown")
    save_figure(render.draw_entropy(dt * stride * np.arange(n_snowflake_frames),
                                    entropies(topology, vertices, snowflake)), "snowflake_entropy")


def pinned() -> None:
    """The pictures the repository tracks: the snowflake growing from its seed, for the gallery."""
    from examples.util.animation import pin_animation
    from examples.thermodynamics.snowflake import render

    topology, coordinates, vertices = dish()
    parcels, phase = seeded(coordinates)
    snowflake = frozen(topology, vertices, parcels, phase, anisotropy, n_snowflake_frames, stride)
    pin_animation(render.animate(topology, vertices, snowflake), "snowflake", frame_duration_ms)


# A preview of how symmetric the snowflake grows: each setting a change from the scenario's parameters, the edge's
# relaxation time, mean width, anisotropy and double well's height, the latent heat and absolute zero; each grown until
# its ice first reaches the share of the dish's radius, at most the frames given, and shown in about the frames given.
symmetry_settings = {
    "baseline": {},
    "anisotropy": dict(anisotropy=0.04),
    "edge_energy": dict(mean_width=0.01, well_height=1 / 4),
    "slow_relaxation": dict(relaxation_time=6e-4),
    "small_undercooling": dict(latent_heat=2.2, absolute_zero=-6.5),
}
# A second preview, of the anisotropy's step from 0.02 through the scenario's to just under the six-fold limit of
# 1 / 35, above which the edge's stiffness turns negative at some orientations.
anisotropy_settings = {
    "anisotropy_020": dict(anisotropy=0.02),
    "anisotropy_025": dict(anisotropy=0.025),
    "anisotropy_028": dict(anisotropy=0.028),
}
preview_reach, n_preview_frames, n_preview_shown = 0.9, 250, 76


def symmetry_preview(settings: dict[str, dict], name: str) -> None:
    """A preview: the snowflake grown under each of these settings until its ice first reaches the preview's share of
    the dish's radius, saved as an animation each, and the grown crystals side by side under this name."""
    import time
    from examples.util.animation import save_animation, save_figure
    from examples.util.wgsl import image_figure
    from examples.thermodynamics.snowflake import render

    topology, coordinates, vertices = dish()
    parcels, phase = seeded(coordinates)
    areas, gradients, cells = core.triangles(topology, vertices)
    # The vertices beyond the preview's share of the dish's radius: the ice reaching one ends a run.
    outer = np.hypot(*coordinates.T) > preview_reach * dish_radius
    scenario = dict(relaxation_time=relaxation_time, mean_width=mean_width, anisotropy=anisotropy,
                    well_height=well_height, latent_heat=latent_heat, absolute_zero=absolute_zero)
    grown = []
    for setting_name, changes in settings.items():
        setting = scenario | changes
        started = time.monotonic()
        states = []
        for state in core.growth(topology, areas, gradients, cells, parcels, phase, crystal_axis,
                                 setting["relaxation_time"], setting["mean_width"], setting["anisotropy"],
                                 regularization, setting["well_height"], setting["latent_heat"], melting_temperature,
                                 setting["absolute_zero"], dt, n_preview_frames, stride):
            states.append(state)
            if np.asarray(state[1] > 0.5)[outer].any():
                break
        stepped = time.monotonic() - started
        every = max(1, round(len(states) / n_preview_shown))
        frames = list(render.animate(topology, vertices, states[::every] + states[-1:]))
        save_animation(frames, f"snowflake_preview_{setting_name}", frame_duration_ms)
        grown.append(frames[-1])
        print(f"{setting_name}: {(len(states) - 1) * stride} steps, time {(len(states) - 1) * stride * dt:.3f}, "
              f"stepped in {stepped:.0f} s, reached: {np.asarray(states[-1][1] > 0.5)[outer].any()}")
        del states
    row = [grown[0]]
    for picture in grown[1:]:
        row = list(render.side_by_side(row, [picture]))
    save_figure(image_figure(row[0]), name)


if __name__ == "__main__":
    main()
