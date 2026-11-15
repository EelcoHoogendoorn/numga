"""This module tabulates saturated superfluid helium properties and runs sound-speed, heat-pulse, and fountain-effect scenarios across a cavity divided by a porous powder wall.

Thermodynamic and two-fluid properties at saturated vapour pressure follow the reference tables of Donnelly and Barenghi (1998), covering total density (table 1.2), normal-fluid density (table 2.3), first, second, and fourth sound speeds (tables 3.3, 4.3, and 6.3), molar heat capacity (table 7.4), and specific entropy (table 8.5).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack

from examples.fluids.superfluid import core

# Saturated liquid helium properties from 1.2 K to 1.9 K in SI units (K, kg/m^3, m/s, J/(mol K), kg/mol, J/(kg K)).
temperatures = np.array([1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
densities = np.array([145.1151, 145.1215, 145.1373, 145.1646, 145.2071, 145.2686, 145.3538, 145.4684])
normal_densities = np.array([3.75, 6.52, 10.57, 16.17, 23.58, 33.21, 45.54, 61.03])
first_sounds = np.array([237.3, 236.9, 236.3, 235.5, 234.5, 233.1, 231.3, 229.0])
molar_heat_capacities = np.array([1.297, 2.057, 3.092, 4.468, 6.285, 8.678, 11.81, 15.90])
molar_mass = 4.0026e-3
entropies = np.array([52.04, 85.21, 132.3, 196.5, 282.4, 395.0, 539.9, 725.5])
second_sounds = np.array([18.51, 19.00, 19.56, 20.03, 20.33, 20.33, 19.90, 18.83])
fourth_sounds = np.array([234.5, 231.8, 227.8, 222.2, 214.7, 205.1, 192.6, 175.7])
heat_capacities = molar_heat_capacities / molar_mass
n_temperatures = len(temperatures)

# Cavity geometry in millimetres, operating index (1.8 K), and millikelvin Gaussian pulse amplitude.
width, height = 16.0, 8.0
n_columns, n_rows = 65, 33
wall_middle, wall_half_thickness = 9.0, 0.75
heater_position = np.array([4.0, 4.5])
pulse_width = 0.5
cavity_index = 6
pulse_rise = 1e-3
# Half-Courant time-step factor, step and frame stride counts, 40 ms frame duration, and colour-saturation fractions.
courant = 0.5
n_steps, stride = 1400, 14
frame_duration_ms = 40
n_pinned_colours = 32
pulse_temperature_share, pulse_pressure_share = 0.2, 0.1
warming_rate = 1e-3
heater_share = 1.5


def fluid_components() -> tuple[core.State, np.ndarray]:
    """Tabulated densities and specific entropies define the superfluid and normal-fluid parcels per unit volume (`[n_temperatures, 2] State-`) and component densities (`[n_temperatures, 2]`)."""
    masses = np.stack([densities - normal_densities, normal_densities], axis=-1)  # [n_temperatures, 2]
    held_entropies = np.stack([0 * entropies, densities * entropies], axis=-1)    # [n_temperatures, 2]
    return masses * core.mass + held_entropies * core.entropy, masses


def liquid_stiffnesses() -> core.Extensor:
    """Tabulated densities, first-sound speeds, temperatures, entropies, and heat capacities determine the liquid's thermodynamic stiffness (`[n_temperatures] Potential+ <- State-`)."""
    return core.thermodynamic_stiffness(densities, first_sounds, temperatures, entropies, heat_capacities)


def sound_curves() -> tuple[core.Scalar, core.Scalar]:
    """Two-fluid eigenmodes give the squared sound speeds (`[n_temperatures, 2] Scalar+`) in open liquid and with the normal fluid clamped."""
    parcels, masses = fluid_components()
    stiffnesses = liquid_stiffnesses()
    return (
        core.sound_speeds(stiffnesses, parcels, masses),
        core.sound_speeds(stiffnesses, parcels[:, :1], masses[:, :1]),
    )


def cavity_scene(n_columns: int, n_rows: int) -> tuple[core.Topology, np.ndarray, np.ndarray, np.ndarray]:
    """An `n_columns` by `n_rows` cavity grid defines the triangulation, vertex coordinates (`[V, 2]`), porous-wall triangle mask (`[F]`), and Gaussian temperature-pulse profile (`[V]`)."""
    topology, coordinates = core.cavity(n_columns, n_rows, width, height)
    triangle_centres_x = coordinates[topology.elements[2], 0].mean(axis=-1)
    distances_squared = ((coordinates - heater_position) ** 2).sum(axis=-1)
    powder_triangles = np.abs(triangle_centres_x - wall_middle) < wall_half_thickness
    pulse_profile = pulse_rise * np.exp(-distances_squared / (2 * pulse_width ** 2))
    return topology, coordinates, powder_triangles, pulse_profile


def heat_pulse(
    topology: core.Topology,
    vertices: core.Point,
    powder_triangles: np.ndarray,
    pulse_profile: np.ndarray,
) -> Iterator[core.State]:
    """Yields the cavity states (`State-[V]`) evolving from an isochoric Gaussian entropy pulse with the normal fluid clamped inside the porous wall."""
    parcels, masses = fluid_components()
    stiffness = liquid_stiffnesses()[cavity_index]
    # Entropy per unit volume that raises the initial temperature by pulse_profile at fixed density.
    pulse_entropy = pulse_profile * densities[cavity_index] * heat_capacities[cavity_index] / temperatures[cavity_index]  # [V]
    dt = courant * (width / (n_columns - 1)) / first_sounds[cavity_index]
    normal_clamp = core.mv.scalar(powder_triangles[:, None] * 1.0).field()    # Scalar+[F]
    initial_state = (0 * core.mass + core.entropy * pulse_entropy).field()    # State-[V]
    zero_supply = (0 * core.mass + 0 * core.entropy * pulse_entropy).field()  # State-[V]
    return core.cavity_states(
        topology,
        vertices,
        stiffness,
        parcels[cavity_index],
        masses[cavity_index],
        normal_clamp,
        initial_state,
        zero_supply,
        dt,
        n_steps,
        stride,
    )


def steady_heater(
    topology: core.Topology,
    vertices: core.Point,
    powder_triangles: np.ndarray,
    pulse_profile: np.ndarray,
) -> Iterator[core.State]:
    """Yields the cavity states (`State-[V]`) from rest under a steady isochoric entropy supply warming the heated chamber at `warming_rate` on average."""
    parcels, masses = fluid_components()
    stiffness = liquid_stiffnesses()[cavity_index]
    # Steady entropy supply per unit volume and time shaped like the Gaussian pulse.
    chamber_area = (wall_middle - wall_half_thickness) * height
    cell_area = (width / (n_columns - 1)) * (height / (n_rows - 1))
    heating_rate = pulse_profile / (pulse_profile.sum() * cell_area) * warming_rate * chamber_area  # [V]
    entropy_supply = heating_rate * densities[cavity_index] * heat_capacities[cavity_index] / temperatures[cavity_index]  # [V]
    dt = courant * (width / (n_columns - 1)) / first_sounds[cavity_index]
    normal_clamp = core.mv.scalar(powder_triangles[:, None] * 1.0).field()    # Scalar+[F]
    initial_state = (0 * core.mass + 0 * core.entropy * entropy_supply).field()  # State-[V]
    supply = (0 * core.mass + core.entropy * entropy_supply).field()          # State-[V]
    return core.cavity_states(
        topology,
        vertices,
        stiffness,
        parcels[cavity_index],
        masses[cavity_index],
        normal_clamp,
        initial_state,
        supply,
        dt,
        n_steps,
        stride,
    )


def chamber_weights(coordinates: np.ndarray, dual_areas: core.Scalar) -> core.Scalar:
    """Area-weighted vertex masks form the linear functional (`Scalar+[V]`) measuring the mean difference between the heated chamber and the far chamber."""
    hot_mask = coordinates[:, 0] < wall_middle - wall_half_thickness
    cold_mask = coordinates[:, 0] > wall_middle + wall_half_thickness
    hot_areas = dual_areas * core.mv.scalar(hot_mask[:, None] * 1.0).field()   # Scalar+[V]
    cold_areas = dual_areas * core.mv.scalar(cold_mask[:, None] * 1.0).field() # Scalar+[V]
    return hot_areas / hot_areas.sites.sum() - cold_areas / cold_areas.sites.sum()


def pinned() -> None:
    """Writes the gallery animation of a heat pulse propagating as second sound and driving pressure across the porous wall."""
    from examples.fluids.superfluid import render
    from examples.util.animation import pin_animation

    topology, coordinates, powder_triangles, pulse_profile = cavity_scene(n_columns, n_rows)
    vertices = (core.origin + core.mv(core.Upright, coordinates).dual()).field()  # Point-[V]
    stiffness = liquid_stiffnesses()[cavity_index]
    parcels, masses = fluid_components()
    thermometer = stiffness(core.entropy)                                     # [] Potential+
    barometer = stiffness(parcels[cavity_index].sum(axis=-1))                 # [] Potential+
    fountain_per_kelvin = densities[cavity_index] * entropies[cavity_index]
    states = heat_pulse(topology, vertices, powder_triangles, pulse_profile)
    pin_animation(
        render.cavity_animation(
            topology,
            vertices,
            powder_triangles,
            thermometer,
            barometer,
            pulse_temperature_share * pulse_rise,
            pulse_pressure_share * fountain_per_kelvin * pulse_rise,
            render.n_bands,
            states,
        ),
        "superfluid",
        frame_duration_ms,
        colors=n_pinned_colours,
    )


def main() -> None:
    """Writes the sound-speed comparison, cavity triangulation, heat-pulse animation, steady-heater animation, and fountain-pressure chart."""
    from examples.fluids.superfluid import render
    from examples.util.animation import save_animation, save_figure

    bulk_sounds, powder_sounds = sound_curves()
    save_figure(
        render.sounds_figure(temperatures, bulk_sounds, powder_sounds, first_sounds, second_sounds, fourth_sounds),
        "superfluid_sounds",
    )
    topology, coordinates, powder_triangles, pulse_profile = cavity_scene(n_columns, n_rows)
    vertices = (core.origin + core.mv(core.Upright, coordinates).dual()).field()  # Point-[V]
    save_figure(render.cavity_figure(topology, vertices, powder_triangles), "superfluid_cavity")
    stiffness = liquid_stiffnesses()[cavity_index]
    parcels, masses = fluid_components()
    areas, gradients, dual_areas = core.triangle_geometry(topology, vertices)
    # Maxwell symmetry lets stiffness applied to unit entropy or the total liquid parcel read temperature or Gibbs-Duhem pressure by pairing.
    thermometer = stiffness(core.entropy)                                     # [] Potential+
    barometer = stiffness(parcels[cavity_index].sum(axis=-1))                 # [] Potential+
    fountain_per_kelvin = densities[cavity_index] * entropies[cavity_index]
    pulse_states = list(heat_pulse(topology, vertices, powder_triangles, pulse_profile))
    save_animation(
        render.cavity_animation(
            topology,
            vertices,
            powder_triangles,
            thermometer,
            barometer,
            pulse_temperature_share * pulse_rise,
            pulse_pressure_share * fountain_per_kelvin * pulse_rise,
            render.n_bands,
            pulse_states,
        ),
        "superfluid_pulse",
        frame_duration_ms,
    )
    heater_states = list(steady_heater(topology, vertices, powder_triangles, pulse_profile))
    n_frames = len(heater_states)
    times = np.arange(n_frames) * stride * courant * (width / (n_columns - 1)) / first_sounds[cavity_index]
    temperature_scale = heater_share * warming_rate * times[-1]
    save_animation(
        render.cavity_animation(
            topology,
            vertices,
            powder_triangles,
            thermometer,
            barometer,
            temperature_scale,
            fountain_per_kelvin * temperature_scale,
            None,
            heater_states,
        ),
        "superfluid_heater",
        frame_duration_ms,
    )
    difference = chamber_weights(coordinates, dual_areas)
    pressures = stack([(difference * (barometer & state)).sites.sum() for state in heater_states])   # [n_frames] Scalar+
    fountain = fountain_per_kelvin * stack(
        [(difference * (thermometer & state)).sites.sum() for state in heater_states]
    )                                                                         # [n_frames] Scalar+
    save_figure(render.fountain_figure(times, pressures, fountain), "superfluid_fountain")

    # checks
    assert ((bulk_sounds[:, 1] - first_sounds ** 2).abs() < 1e-8 * first_sounds ** 2).all()
    assert ((bulk_sounds[:, 0] - second_sounds ** 2).abs() < 0.12 * second_sounds ** 2).all()
    assert ((powder_sounds[:, 1] - fourth_sounds ** 2).abs() < 0.02 * fourth_sounds ** 2).all()
    initial_total = (dual_areas * pulse_states[0]).sites.sum()
    final_total = (dual_areas * pulse_states[-1]).sites.sum()
    assert ((stack((core.mv.m, core.mv.s)) & (final_total - initial_total)).abs() < 1e-10).all()
    half_frame = n_frames // 2
    assert ((pressures[half_frame:].mean() - fountain[half_frame:].mean()).abs() < 0.15 * fountain[half_frame:].mean()).all()


if __name__ == "__main__":
    main()
