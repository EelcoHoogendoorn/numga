"""Renders the heated cylinder under a loaded piston alongside three property charts of the van der Waals fluid.

The pressure–volume chart shows red isotherms beneath the black coexistence dome, where the boiling isotherm crosses
its loop at the saturation pressure and encloses two equal areas. The temperature–enthalpy chart shows purple isobars
that rise through the liquid, cross the dome horizontally during boiling, and rise through the vapour, crossed by
grey-blue curves of constant vapour fraction. The phase diagram shows the saturation curve up to the critical point
alongside green lines of constant volume. Because heat absorbed at constant pressure equals the change in enthalpy,
the heated fluid advances along its isobar in equal enthalpy steps.

Callers supply the fluid's thermodynamic functions, which this module samples onto the chart diagrams.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from numga import concatenate, stack
from examples.thermodynamics import rendering as art
from examples.thermodynamics.phase_change import core
from numga.render import solids

mv = core.mv
PressureAt = Callable[[core.Scalar, np.ndarray | float], core.Scalar]
EnthalpyAt = Callable[[core.Scalar, np.ndarray | float], core.Scalar]
CoexistenceAt = Callable[[np.ndarray | float], core.State]
IsobarAt = Callable[[core.Scalar, core.Scalar], core.Scalar]

n_rows, n_columns = 300, 330
volume_range, pressure_range = (0.0, 4.4), (0.0, 0.44)
isotherm_temperatures = np.array([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1])
enthalpy_range, temperature_range = (-1.6, 2.4), (0.6, 1.12)
phase_temperature_range, phase_pressure_range = (0.6, 1.12), (0.0, 0.55)
isochore_volumes = np.array([0.5, 0.6, 0.8, 1.0, 1.4, 2.0, 3.0])
n_dome = 40
dome_sample_temperatures = np.linspace(0.6, 0.995, n_dome)
isobar_temperatures = np.array([0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
vapour_fractions = np.array([0.2, 0.4, 0.6, 0.8])
n_samples = 160
n_isotherms, n_isochores, n_fractions = len(isotherm_temperatures), len(isochore_volumes), len(vapour_fractions)
n_isochore_temperatures = 60
# No fluid state exists at or below the excluded volume.
exclusion = 1 / 3
critical_volume, critical_pressure, critical_temperature = 1.0, 3 / 8, 1.0
height_per_volume, cylinder_pixels = 0.55, 150
liquid_colour, vapour_colour, vapour_opacity = art.blue, art.rgb("#9ccfe0"), 0.35
dome_colour = art.boundary


def isotherms(pressure_at: PressureAt, temperatures: np.ndarray) -> art.Point:
    """Samples isotherms across the volume range on the pressure–volume chart."""
    volumes = np.broadcast_to(np.linspace(0.36, volume_range[1], 2 * n_samples), (len(temperatures), 2 * n_samples))
    return art.chart_points(volumes, art.context.include_from_algebra(
        pressure_at(mv.scalar(volumes[..., None]), temperatures[:, None])))


def isobar(isobar_at: IsobarAt, enthalpy_at: EnthalpyAt, state: core.State, saturation: core.Scalar
           ) -> art.Point:
    """Samples an isobar at a given saturation pressure across the liquid, boiling and vapour regimes."""
    liquid, vapour = mv.l & state, mv.g & state
    # Geometric spacing clusters liquid samples near the excluded volume and spreads vapour samples across expansion.
    liquid_volumes = exclusion + (liquid - exclusion) * np.geomspace(0.05, 1, n_samples)
    expansion = 4 * temperature_range[1] / (saturation * vapour)
    vapour_volumes = vapour * (expansion.log() * np.linspace(0, 1, n_samples)).exp()
    volumes = concatenate((liquid_volumes, vapour_volumes))
    temperatures = isobar_at(volumes, saturation)
    return art.chart_points(art.context.include_from_algebra(enthalpy_at(volumes, temperatures)),
                            art.context.include_from_algebra(temperatures))


def region_labels(liquid: tuple[float, float], mixture: tuple[float, float], vapour: tuple[float, float]
                  ) -> list[art.Label]:
    """Places text labels on the liquid, two-phase and vapour regions of a property chart."""
    return [art.Label(art.chart_points(*liquid), "liquid", art.ink, art.text_size, "mm", (0, 0)),
            art.Label(art.chart_points(*mixture), "liquid + vapour", art.ink, art.text_size, "mm", (0, 0)),
            art.Label(art.chart_points(*vapour), "vapour", art.ink, art.text_size, "mm", (0, 0))]


def trail(diagram: art.Diagram, path: art.Point) -> list:
    """Draws the orange trajectory traversed so far with a dot at its current state."""
    if path.shape[0] == 0:
        return []
    return (art.within(diagram, path[None], art.bold, art.orange[None], 1.0)
            + art.dots(diagram, path[-1:], art.orange))


class Charts:
    """Samples the fixed thermodynamic curves once and draws the three property charts at each frame."""

    def __init__(self, pressure_at: PressureAt, enthalpy_at: EnthalpyAt, coexistence_at: CoexistenceAt,
                 isobar_at: IsobarAt, temperature: float) -> None:
        self.pressure_volume = art.Diagram(volume_range, pressure_range, (0.0, 0.0), n_rows, n_columns, ("V", "p"),
                                           (art.mv.x, art.mv.y), (art.mv.w, 0.0))
        self.temperature_enthalpy = art.Diagram(enthalpy_range, temperature_range,
                                                (enthalpy_range[0], temperature_range[0]), n_rows, n_columns,
                                                ("H", "T"), (art.mv.x, art.mv.y), (art.mv.w, 0.0))
        dome_states = coexistence_at(dome_sample_temperatures)
        liquid_dome, vapour_dome = mv.l & dome_states, mv.g & dome_states
        temperatures = dome_sample_temperatures
        levels = art.context.include_from_algebra(pressure_at(liquid_dome, temperatures))            # [n_dome] Scalar+
        liquid_heat = art.chart_points(
            art.context.include_from_algebra(enthalpy_at(liquid_dome, temperatures)), temperatures)  # [n_dome] Point-
        vapour_heat = art.chart_points(
            art.context.include_from_algebra(enthalpy_at(vapour_dome, temperatures)), temperatures)  # [n_dome] Point-
        critical_enthalpy = art.context.include_from_algebra(
            enthalpy_at(mv.scalar([critical_volume]), critical_temperature))
        # Trace the coexistence dome up the liquid branch to the critical point and down the vapour branch.
        self.pressure_dome = concatenate([
            art.chart_points(art.context.include_from_algebra(liquid_dome), levels),
            art.chart_points(critical_volume, critical_pressure)[None],
            art.chart_points(art.context.include_from_algebra(vapour_dome), levels)[::-1],
        ])                                                                                           # [2 * n_dome + 1] Point-
        self.heat_dome = concatenate([liquid_heat, art.chart_points(critical_enthalpy, critical_temperature)[None],
                                      vapour_heat[::-1]])                                            # [2 * n_dome + 1] Point-
        # Constant-quality curves interpolate linearly between the liquid and vapour branches at each temperature.
        self.fractions = (liquid_heat * (1 - vapour_fractions[:, None])
                          + vapour_heat * vapour_fractions[:, None])                     # [n_fractions, n_dome] Point-
        self.isotherm_lines = isotherms(pressure_at, isotherm_temperatures)              # [n_isotherms, 2 * n_samples] Point-
        self.pressure_temperature = art.Diagram(phase_temperature_range, phase_pressure_range,
                                                (phase_temperature_range[0], 0.0), n_rows, n_columns, ("T", "p"),
                                                (art.mv.x, art.mv.y), (art.mv.w, 0.0))
        self.saturation_curve = concatenate([art.chart_points(temperatures, levels),
                                             art.chart_points(critical_temperature, critical_pressure)[None]])
        isochore_temperatures = np.linspace(*phase_temperature_range, n_isochore_temperatures)
        isochores = art.chart_points(isochore_temperatures, art.context.include_from_algebra(pressure_at(
            mv.scalar(isochore_volumes[:, None, None]), isochore_temperatures[None, :])))   # [n_isochores, n_isochore_temperatures] Point-
        self.isochores = [isochores[volume] for volume in range(isochores.shape[0])]
        self.boiling_mask = isotherm_temperatures == temperature
        self.temperature = temperature
        states = coexistence_at(isobar_temperatures)
        saturations = pressure_at(mv.l & states, isobar_temperatures)
        self.isobars = [isobar(isobar_at, enthalpy_at, states[index], saturations[index])
                        for index in range(len(isobar_temperatures))]
        self.boiling_isobar_mask = isobar_temperatures == temperature
        # Split the boiling isotherm loop where it crosses the saturation tie line into its two equal-area lobes.
        boiling_state = coexistence_at(temperature)
        liquid, vapour = mv.l & boiling_state, mv.g & boiling_state
        self.saturation = art.context.include_from_algebra(pressure_at(liquid, temperature))
        loop_volumes = liquid + (vapour - liquid) * np.linspace(0, 1, 2 * n_samples)
        loop = art.chart_points(art.context.include_from_algebra(loop_volumes),
                                art.context.include_from_algebra(pressure_at(loop_volumes, temperature)))
        saturation_line = art.mv.y - art.mv.w * self.saturation                          # [] Plane+
        middle = 1 + np.flatnonzero(np.diff((saturation_line & loop[1:-1]) > 0))[0]
        # Meet the segment across the sign change with the saturation line to locate the crossing point.
        crossing_point = (loop[middle] & loop[middle + 1]) ^ saturation_line             # [] Point-
        # Project the two loop endpoints vertically onto the saturation line.
        endpoints = (loop[np.array([0, -1])] & art.mv.y.dual()) ^ saturation_line        # [2] Point-
        lobes = [concatenate([endpoints[:1], loop[1:middle + 1], crossing_point[None]]),
                 concatenate([crossing_point[None], loop[middle + 1:-1], endpoints[1:]])]
        # Pair each lobe curve with its vertical projection onto the saturation line for shading.
        self.lobes = [(lobe, (lobe & art.mv.y.dual()) ^ saturation_line, colour)
                      for lobe, colour in zip(lobes, (liquid_colour, art.orange))]

    def pressure_volume_chart(self, path: art.Point) -> np.ndarray:
        diagram = self.pressure_volume
        isotherm = art.families["temperature"]
        other_isotherms = self.isotherm_lines[~self.boiling_mask]                        # [n_isotherms - 1, 2 * n_samples] Point-
        boiling_isotherm = self.isotherm_lines[self.boiling_mask]                        # [1, 2 * n_samples] Point-
        layers = ([art.between(diagram, lobe, base, colour, 0.28) for lobe, base, colour in self.lobes]
                  + art.within(diagram, other_isotherms, art.thin,
                               np.broadcast_to(isotherm, (other_isotherms.shape[0], 3)), 1.0)
                  + art.within(diagram, boiling_isotherm, art.medium, isotherm[None], 1.0)
                  + art.within(diagram, self.pressure_dome[None], art.medium, dome_colour[None], 1.0)
                  + trail(diagram, path))
        labels = region_labels((0.2, 0.06), (2.0, 0.12), (3.3, 0.33)) + [
            art.Label(art.chart_points(critical_volume, critical_pressure), "critical point", dome_colour,
                      art.text_size, "lb", (6, 3))]
        names = [art.Floating(art.along([boiling_isotherm[0]], 6), f"T = {self.temperature:g}", isotherm,
                              art.text_size, "lb"),
                 art.Floating(art.along([other_isotherms[line] for line in range(other_isotherms.shape[0])], 6),
                              "T = const", isotherm, art.text_size, "lb"),
                 art.Floating(art.along([self.pressure_dome], 3), "coexistence", dome_colour, art.text_size, "lt")]
        return art.floated(diagram, art.picture(diagram, layers, labels), names)

    def temperature_enthalpy_chart(self, path: art.Point) -> np.ndarray:
        diagram = self.temperature_enthalpy
        isobar = art.families["pressure"]
        lines = [art.within(diagram, isobar_line[None], art.medium if at_boiling else art.thin, isobar[None], 1.0)
                 for isobar_line, at_boiling in zip(self.isobars, self.boiling_isobar_mask)]
        fractions = [self.fractions[fraction] for fraction in range(len(vapour_fractions))]
        layers = (art.polylines(diagram, fractions, art.thin, art.families["level"], 1.0, art.SOLID)
                  + [stroke for strokes in lines for stroke in strokes]
                  + art.within(diagram, self.heat_dome[None], art.medium, dome_colour[None], 1.0)
                  + trail(diagram, path))
        bottom = np.argmin(np.abs(dome_sample_temperatures - temperature_range[0] - 0.03))
        labels = region_labels((-1.15, 1.02), (0.45, 0.665), (2.0, 0.8)) + [
            art.Label(self.heat_dome[len(dome_sample_temperatures)], "critical point", dome_colour, art.text_size, "lb",
                      (6, 3))] + [
            art.Label(self.fractions[index, bottom], f"{fraction:g}", art.families["level"], art.tick_size, "mt",
                      (0, -2))
            for index, fraction in enumerate(vapour_fractions)]
        boiling_isobars = [isobar_line for isobar_line, flag in zip(self.isobars, self.boiling_isobar_mask) if flag]
        other_isobars = [isobar_line for isobar_line, flag in zip(self.isobars, self.boiling_isobar_mask) if not flag]
        names = [art.Floating(art.along(boiling_isobars, 8), f"p = {float(self.saturation.to_array()):.2f}", isobar,
                              art.text_size, "lb"),
                 art.Floating(art.along(other_isobars, 8), "p = const", isobar, art.text_size, "lb"),
                 art.Floating(art.along(fractions, 4), "x = const", art.families["level"], art.text_size, "lb")]
        return art.floated(diagram, art.picture(diagram, layers, labels), names)

    def pressure_temperature_chart(self, path: art.Point) -> np.ndarray:
        """Draws the pressure–temperature phase diagram with isochores, the saturation curve and the current
        trajectory."""
        diagram = self.pressure_temperature
        layers = (art.polylines(diagram, self.isochores, art.thin, art.families["volume"], 1.0, art.SOLID)
                  + art.polylines(diagram, [self.saturation_curve], art.medium, dome_colour, 1.0, art.SOLID)
                  + art.dots(diagram, self.saturation_curve[-1:], dome_colour)
                  + trail(diagram, path))
        labels = [art.Label(art.chart_points(0.68, 0.33), "liquid", art.ink, art.text_size, "mm", (0, 0)),
                  art.Label(art.chart_points(0.98, 0.1), "vapour", art.ink, art.text_size, "mm", (0, 0)),
                  art.Label(self.saturation_curve[-1], "critical point", dome_colour, art.text_size, "rb", (-6, 3))]
        names = [art.Floating(art.along(self.isochores, 4), "V = const", art.families["volume"], art.text_size, "lb"),
                 art.Floating(art.along([self.saturation_curve], 3), "saturation", dome_colour, art.text_size, "lt")]
        return art.floated(diagram, art.picture(diagram, layers, labels), names)


def boiler(volume: solids.Scalar, liquid_volume: solids.Scalar, largest_volume: float, warmth: float) -> np.ndarray:
    """Renders the vertical cylinder under its loaded piston with liquid below and translucent vapour above."""
    height, liquid = volume * height_per_volume, liquid_volume * height_per_volume
    maximum = solids.mv.scalar([largest_volume * height_per_volume])
    view = art.cylinder_view(maximum, cylinder_pixels)
    parts = art.cylinder(height, maximum) + art.load(height) + [(art.fill(0.0, liquid), liquid_colour)]
    vapour = art.thermal_colour(warmth) * 0.5 + vapour_colour * 0.5
    return solids.draw(parts, view, art.white, inks=[],
                       translucent=[(art.fill(liquid, height), vapour, vapour_opacity)], meshes=[])


def draw_charts(pressure_at: PressureAt, enthalpy_at: EnthalpyAt, coexistence_at: CoexistenceAt,
                isobar_at: IsobarAt, temperature: float):
    """Draws the three static property charts with the boiling isotherm and isobar highlighted."""
    charts = Charts(pressure_at, enthalpy_at, coexistence_at, isobar_at, temperature)
    empty = art.chart_points(np.zeros(0), np.zeros(0))
    return art.figure(art.beside([charts.pressure_volume_chart(empty), charts.temperature_enthalpy_chart(empty),
                                  charts.pressure_temperature_chart(empty)]))


def heating(pressure_at: PressureAt, enthalpy_at: EnthalpyAt, coexistence_at: CoexistenceAt, isobar_at: IsobarAt,
            temperature: float, states: Iterator[tuple[core.Scalar, core.Scalar, core.Scalar, core.Scalar]]
            ) -> Iterator[np.ndarray]:
    """Streams composite frames showing the cylinder alongside the growing trajectory on all three charts."""
    charts = Charts(pressure_at, enthalpy_at, coexistence_at, isobar_at, temperature)
    path = []
    for volume, fluid_temperature, enthalpy, liquid_volume in states:
        path.append(art.context.include_from_algebra(stack((volume, fluid_temperature, enthalpy))))
        history = stack(path)
        volumes, temperatures, enthalpies = history[:, 0], history[:, 1], history[:, 2]
        warmth = float(((fluid_temperature - temperature_range[0])
                        / (temperature_range[1] - temperature_range[0])).to_array())
        yield art.beside([boiler(solids.context.include_from_algebra(volume),
                                 solids.context.include_from_algebra(liquid_volume), volume_range[1], warmth),
                          charts.pressure_volume_chart(art.chart_points(volumes, charts.saturation)),
                          charts.temperature_enthalpy_chart(art.chart_points(enthalpies, temperatures)),
                          charts.pressure_temperature_chart(art.chart_points(temperatures, charts.saturation))])
