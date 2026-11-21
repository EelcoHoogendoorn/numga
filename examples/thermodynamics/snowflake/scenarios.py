"""A snowflake grows from a seed of ice in undercooled water, beside the same seed growing without anisotropy into a
bush of fingers. On a triangulated disc of staggered rings with no six-fold symmetry of its own, the six arms come from
the crystal's six-fold surface energy, and the total entropy rises throughout the growth."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import dim, stack

from examples.thermodynamics.snowflake import core
from examples.util import shapes
from numga.topology import Topology

mv = core.mv

# Dish radius and radial ring count for the triangulated disc.
dish_radius_value, n_divisions = 4.5, 150
dish_radius = mv.scalar(dish_radius_value, unit=dim.length)                   # [] Scalar+, length
# Initial ice seed radius and smooth interface width.
seed_radius = mv.scalar(0.1, unit=dim.length)                                 # [] Scalar+, length
seed_edge = mv.scalar(0.02, unit=dim.length)                                  # [] Scalar+, length
# Reference plane for the crystal's six-fold axis, whose normal points upward.
crystal_axis = mv.y                                                           # [] Upright+
# Thermal diffusivity, kinetic relaxation modulus, interface gradient width, six-fold anisotropy,
# steepness regularization, and double-well height.
diffusivity = mv.scalar(1.0, unit=dim.length**2 / dim.time)                   # [] Scalar+, length^2/time
relaxation_time = mv.scalar(3e-4, unit=dim.time * dim.temperature)            # [] Scalar+, temperature*time
mean_width = mv.scalar(0.0082, unit=dim.length * dim.temperature.root(2))     # [] Scalar+, length*temperature^(1/2)
anisotropy = 0.025
regularization = mv.scalar(1.0, unit=1 / dim.length)                          # [] Scalar+, 1/length
well_height = mv.scalar(1 / 6, unit=dim.temperature)                          # [] Scalar+, temperature
# Latent heat per unit ice, melting temperature above the initial water temperature, and absolute zero on that scale.
latent_heat = mv.scalar(1.8, unit=dim.temperature)                            # [] Scalar+, temperature
melting_temperature = mv.scalar(1.0, unit=dim.temperature)                    # [] Scalar+, temperature
absolute_zero = mv.scalar(-5.0, unit=dim.temperature)                         # [] Scalar+, temperature
# Time step, steps per frame, frame counts to reach ~90% of the dish radius, and frame duration in milliseconds.
dt, stride = mv.scalar(2e-4, unit=dim.time), 20
n_snowflake_frames, n_isotropic_frames = 75, 111
frame_duration_ms = 40


def dish(n_divisions: int) -> tuple[Topology, core.Point]:
    """The triangulated disc of staggered rings and its unit-weight vertex places."""
    topology, coordinates = shapes.concentric_disk(np.array([dish_radius_value]), n_divisions)
    vertices = (mv(core.Upright, coordinates) * dim.length + mv.w).dual().field()  # Point-[V], 1 | length
    return topology, vertices


def seed_phase(vertices: core.Point) -> core.Scalar:
    """Initial phase field on the dish, equal to one inside the seed radius with a smooth hyperbolic-tangent rim."""
    radii = ((mv.x & vertices) * mv.x + (mv.y & vertices) * mv.y).norm()        # Scalar+[V], length
    return (1 + ((seed_radius - radii) / seed_edge).tanh()) / 2


def seeded(vertices: core.Point) -> tuple[core.Point, core.Scalar]:
    """Initial parcel and phase fields for the cold water and central ice seed."""
    phase = seed_phase(vertices)                                              # Scalar+[V]
    return core.volume + core.ice * core.ice_content(phase) + mv.scalar(0.0, unit=dim.temperature) * core.heat, phase


def frozen(topology: Topology, vertices: core.Point, parcels: core.Point, phase: core.Scalar, anisotropy: float,
           n_frames: int, stride: int) -> Iterator[tuple[core.Point, core.Scalar]]:
    """Yield parcel and phase fields across frames on the triangulated dish for a given six-fold anisotropy."""
    areas, gradients, dual_areas = core.triangles(topology, vertices)
    return core.growth(topology, areas, gradients, dual_areas, parcels, phase, crystal_axis, diffusivity,
                       relaxation_time, mean_width, anisotropy, regularization, well_height, latent_heat,
                       melting_temperature, absolute_zero, dt, n_frames, stride)


def entropies(topology: Topology, vertices: core.Point,
              states: list[tuple[core.Point, core.Scalar]]) -> core.Antiscalar:
    """Total entropy across a sequence of recorded parcel and phase states."""
    areas, gradients, dual_areas = core.triangles(topology, vertices)
    gradient, spreading, _ = core.operators(topology, areas, gradients, diffusivity)
    return stack([core.total_entropy(areas, dual_areas, gradient, spreading, crystal_axis, mean_width, anisotropy,
                                     regularization, well_height, latent_heat, melting_temperature, absolute_zero,
                                     parcels, phase)
                  for parcels, phase in states])


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.snowflake import render

    topology, vertices = dish(n_divisions)
    parcels, phase = seeded(vertices)
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
    """Save the gallery animation of the snowflake growing from its seed."""
    from examples.util.animation import pin_animation
    from examples.thermodynamics.snowflake import render

    topology, vertices = dish(n_divisions)
    parcels, phase = seeded(vertices)
    snowflake = frozen(topology, vertices, parcels, phase, anisotropy, n_snowflake_frames, stride)
    pin_animation(render.animate(topology, vertices, snowflake), "snowflake", frame_duration_ms)


# Parameter variations around the baseline scenario, each run until its ice reaches nine tenths of the dish radius.
symmetry_settings = {
    "baseline": {},
    "anisotropy": dict(anisotropy=0.04),
    "edge_energy": dict(
        mean_width=mv.scalar(0.01, unit=dim.length * dim.temperature.root(2)),
        well_height=mv.scalar(1 / 4, unit=dim.temperature),
    ),
    "slow_relaxation": dict(relaxation_time=mv.scalar(6e-4, unit=dim.time * dim.temperature)),
    "small_undercooling": dict(
        latent_heat=mv.scalar(2.2, unit=dim.temperature),
        absolute_zero=mv.scalar(-6.5, unit=dim.temperature),
    ),
}
# Anisotropy sweep from 0.02 up to just below the six-fold stiffness limit of 1 / 35.
anisotropy_settings = {
    "anisotropy_020": dict(anisotropy=0.02),
    "anisotropy_025": dict(anisotropy=0.025),
    "anisotropy_028": dict(anisotropy=0.028),
}
preview_reach, n_preview_frames, n_preview_shown = 0.9, 250, 76


def symmetry_preview(settings: dict[str, dict], name: str) -> None:
    """Grow the snowflake under each parameter override, saving each run's animation and a side-by-side comparison
    figure of the final crystals."""
    import time
    from examples.util.animation import save_animation, save_figure
    from examples.util.panels import image_figure
    from examples.thermodynamics.snowflake import render

    topology, vertices = dish(n_divisions)
    parcels, phase = seeded(vertices)
    areas, gradients, dual_areas = core.triangles(topology, vertices)
    # Stop each run once the ice front crosses `preview_reach` of the dish radius.
    radii = ((mv.x & vertices) * mv.x + (mv.y & vertices) * mv.y).norm()        # Scalar+[V]
    outer = radii > preview_reach * dish_radius
    scenario = dict(relaxation_time=relaxation_time, mean_width=mean_width, anisotropy=anisotropy,
                    well_height=well_height, latent_heat=latent_heat, absolute_zero=absolute_zero)
    grown = []
    for setting_name, changes in settings.items():
        setting = scenario | changes
        started = time.monotonic()
        states = []
        for state in core.growth(topology, areas, gradients, dual_areas, parcels, phase, crystal_axis, diffusivity,
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
        print(f"{setting_name}: {(len(states) - 1) * stride} steps, time {float(((len(states) - 1) * stride * dt).to_array()):.3f}, "
              f"stepped in {stepped:.0f} s, reached: {np.asarray(states[-1][1] > 0.5)[outer].any()}")
        del states
    row = [grown[0]]
    for picture in grown[1:]:
        row = list(render.side_by_side(row, [picture]))
    save_figure(image_figure(row[0]), name)


if __name__ == "__main__":
    main()
