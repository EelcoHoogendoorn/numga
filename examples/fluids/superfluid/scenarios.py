"""Superfluid helium's sounds and a heat pulse in a cavity divided by a wall of powder.

The liquid's properties at saturated vapour pressure are the recommended values of Donnelly and Barenghi (1998): the
density (their table 1.2), the normal fluid's density (2.3), the first, second and fourth sound speeds (3.3, 4.3 and
6.3), the heat capacity (7.4) and the entropy from it (8.5).
"""

from __future__ import annotations

import numpy as np

from numga import stack

from examples.fluids.superfluid import core

# The temperatures in kelvin; the liquid's density and its normal fluid's in kilograms per cubic metre; its first sound in
# metres per second; its heat capacity in joules per mole and kelvin, and helium's molar mass in kilograms per mole; its
# entropy in joules per kilogram and kelvin; the measured second and fourth sounds in metres per second.
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

# The cavity, in millimetres: its width and height, its vertices across and up, the middle and the half thickness of the
# wall of powder across it, where the heater sits, and the pulse's width; the temperature the cavity is held at, as an
# index into the temperatures; the pulse's rise of temperature at its middle, in kelvin.
width, height = 16.0, 8.0
n_columns, n_rows = 65, 33
wall_middle, wall_half_thickness = 9.0, 0.75
heater = np.array([4.0, 4.5])
pulse_width = 0.5
at = 6
pulse_rise = 1e-3
# The time step in milliseconds, a share of the time first sound takes across a triangle's shorter side; the steps and
# the steps between frames; the frames' duration in milliseconds; the shares of the pulse's rise of temperature, and of
# the fountain pressure of that rise, at which the pulse's colours saturate.
courant = 0.5
n_steps, stride = 1400, 14
frame_duration_ms = 40
pulse_temperature_share, pulse_pressure_share = 0.2, 0.1
# The steady heater's warming of its chamber on average, in kelvin per millisecond, and the multiple of the chamber's
# last warming, and of its fountain pressure, at which the heater's colours saturate.
warming_rate = 1e-3
heater_share = 1.5


def components(at: int | slice) -> tuple[core.Point, np.ndarray]:
    """Per unit volume, the parcels of the two components, the superfluid's mass alone and the normal fluid's mass with
    all the entropy, `[..., components] Amount-`, and their masses `[..., components]`."""
    masses = np.stack([densities[at] - normal_densities[at], normal_densities[at]], axis=-1)
    held_entropy = np.stack([0 * entropies[at], densities[at] * entropies[at]], axis=-1)
    return masses * core.mass + held_entropy * core.entropy, masses


def stiffness(at: int | slice):
    """The liquid's thermodynamic stiffness at these temperatures `[...] Potential+ <- Amount-`."""
    return core.stiffness_of(densities[at], first_sounds[at], temperatures[at], entropies[at], heat_capacities[at])


def sounds() -> tuple[core.Scalar, core.Scalar]:
    """The squared speeds of sound at every temperature, of the liquid and of the liquid with its normal fluid held still,
    `[temperatures, 2] Scalar+` each."""
    parcels, masses = components(slice(None))
    return core.sound_speeds(stiffness(slice(None)), parcels, masses), core.sound_speeds(
        stiffness(slice(None)), parcels[..., :1], masses[..., :1])


def scene() -> tuple[core.Topology, np.ndarray, np.ndarray, np.ndarray]:
    """The cavity: its topology, its vertices' places across and up `[vertices, 2]`, whether each triangle lies in the
    powder `[faces]`, and the heater's pulse of temperature at each vertex `[vertices]`."""
    topology, coordinates = core.cavity(n_columns, n_rows, width, height)
    centroids_across = coordinates[topology.elements[2], 0].mean(axis=-1)
    distances_squared = ((coordinates - heater) ** 2).sum(axis=-1)
    return (topology, coordinates, np.abs(centroids_across - wall_middle) < wall_half_thickness,
            pulse_rise * np.exp(-distances_squared / (2 * pulse_width ** 2)))


def pulse(topology: core.Topology, vertices: core.Point, in_powder: np.ndarray, rise: np.ndarray):
    """The cavity's mass and entropy per volume over time `Point-[vertices]`, from the heater's pulse of temperature,
    entropy added at constant density, with the normal fluid held still in the powder."""
    parcels, masses = components(at)
    # The entropy that raises the temperature by the pulse at constant density.
    added = rise * densities[at] * heat_capacities[at] / temperatures[at]   # [vertices]
    dt = courant * (width / (n_columns - 1)) / first_sounds[at]
    return core.stepped(topology, vertices, stiffness(at), parcels, masses, core.mv.scalar(in_powder[:, None] * 1.0).field(),
                        (0 * core.mass + core.entropy * added).field(), (0 * core.mass + 0 * core.entropy * added).field(),
                        dt, n_steps, stride)


def steady_heater(topology: core.Topology, vertices: core.Point, in_powder: np.ndarray, rise: np.ndarray):
    """The cavity's mass and entropy per volume over time `Point-[vertices]`, from rest, with the heater adding entropy
    at constant density at a steady rate shaped like the pulse, warming its chamber on average by `warming_rate`."""
    parcels, masses = components(at)
    # The entropy per time that warms the heater's chamber at the warming rate on average, spread like the pulse.
    chamber_area = (wall_middle - wall_half_thickness) * height
    rate = rise / (rise.sum() * (width / (n_columns - 1)) * (height / (n_rows - 1))) * warming_rate * chamber_area
    added = rate * densities[at] * heat_capacities[at] / temperatures[at]   # [vertices]
    dt = courant * (width / (n_columns - 1)) / first_sounds[at]
    return core.stepped(topology, vertices, stiffness(at), parcels, masses, core.mv.scalar(in_powder[:, None] * 1.0).field(),
                        (0 * core.mass + 0 * core.entropy * added).field(), (0 * core.mass + core.entropy * added).field(),
                        dt, n_steps, stride)


def chambers(coordinates: np.ndarray, cells: core.Scalar) -> core.Scalar:
    """The reading of the difference between the mean of a field over the chamber the heater is in and over the other,
    each vertex weighed by its share of the area, `Scalar+[vertices]`."""
    hot = coordinates[:, 0] < wall_middle - wall_half_thickness
    cold = coordinates[:, 0] > wall_middle + wall_half_thickness
    hot_cells, cold_cells = (cells * core.mv.scalar(chamber[:, None] * 1.0).field() for chamber in (hot, cold))
    return hot_cells / hot_cells.sites.sum() - cold_cells / cold_cells.sites.sum()


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.superfluid import render

    free, held = sounds()
    save_figure(render.draw_sounds(temperatures, free, held, first_sounds, second_sounds, fourth_sounds),
                "superfluid_sounds")
    topology, coordinates, in_powder, rise = scene()
    vertices = (core.mass + core.axis * coordinates[:, 0] + core.across * coordinates[:, 1]).field()
    save_figure(render.draw_cavity(topology, vertices, in_powder), "superfluid_cavity")
    liquid = stiffness(at)
    parcels, _ = components(at)
    _, _, cells = core.triangles(topology, vertices)
    # The fountain pressure of a rise of temperature: the density times the entropy per mass times the rise.
    fountain_per_kelvin = densities[at] * entropies[at]
    # Each frame's rise of temperature and of pressure: its potentials read by a unit of entropy, and by the whole
    # liquid's parcel, the Gibbs-Duhem relation.
    readings = ((liquid(amounts) & core.entropy, liquid(amounts) & parcels.sum(axis=-1))
                for amounts in pulse(topology, vertices, in_powder, rise))
    save_animation(render.animate(topology, vertices, in_powder, readings, pulse_temperature_share * pulse_rise,
                                  pulse_pressure_share * fountain_per_kelvin * pulse_rise),
                   "superfluid_pulse", frame_duration_ms)
    # The heater left on, and the chambers' difference of pressure against the fountain pressure of their difference of
    # temperature.
    readings = [(liquid(amounts) & core.entropy, liquid(amounts) & parcels.sum(axis=-1))
                for amounts in steady_heater(topology, vertices, in_powder, rise)]
    times = np.arange(len(readings)) * stride * courant * (width / (n_columns - 1)) / first_sounds[at]
    warmest = heater_share * warming_rate * times[-1]
    save_animation(render.animate(topology, vertices, in_powder, readings, warmest, fountain_per_kelvin * warmest),
                   "superfluid_heater", frame_duration_ms)
    difference = chambers(coordinates, cells)
    pressures = stack([(difference * pressure).sites.sum() for _, pressure in readings])
    fountain = fountain_per_kelvin * stack([(difference * warmth).sites.sum() for warmth, _ in readings])
    save_figure(render.draw_fountain(times, pressures, fountain), "superfluid_fountain")


if __name__ == "__main__":
    main()
