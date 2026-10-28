"""Boiling under a loaded piston beside three property charts of the van der Waals fluid.

The pressure–volume chart holds the isotherms, red, under the dome of coexisting volumes, black; the isotherm at the
boiling temperature crosses its loop at the saturation pressure, cutting off two equal areas. The temperature–enthalpy
chart holds the dome again and the isobars, purple, each rising through the liquid, flat across the dome while the
fluid boils and rising through the vapour, with lines of equal vapour fraction, grey-blue, across the dome. The phase
diagram holds the saturation pressure against the temperature, ending at the critical point, and lines of equal
volume, green; the heated fluid crosses the saturation curve at its fixed pressure. Heat taken in at fixed pressure is
the enthalpy's change, so the heated fluid walks along its isobar in equal steps of enthalpy.

The charts' curves come from the fluid's own functions, which the notebook hands over; this module only samples them.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from examples.thermodynamics import rendering as art
from examples.thermodynamics.phase_change import core
from examples.util.wgsl import solids

PressureAt = Callable[[core.Scalar, np.ndarray], core.Scalar]
EnthalpyAt = Callable[[core.Scalar, np.ndarray], core.Scalar]
CoexistenceAt = Callable[[np.ndarray], core.State]
IsobarAt = Callable[[core.Scalar, core.Scalar], core.Scalar]

# The charts' size in pixels; the pressure–volume chart's ranges and isotherms; the temperature–enthalpy chart's
# ranges, the temperatures of the dome's samples, the saturation temperatures of its isobars, and the vapour fractions
# of its lines across the dome.
n_rows, n_columns = 300, 330
volume_range, pressure_range = (0.0, 4.4), (0.0, 0.44)
isotherm_temperatures = np.array([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1])
enthalpy_range, temperature_range = (-1.6, 2.4), (0.6, 1.12)
# The phase diagram's ranges, temperature and pressure, and the volumes of its lines of equal volume.
phase_temperature_range, phase_pressure_range = (0.6, 1.12), (0.0, 0.55)
isochore_volumes = np.array([0.5, 0.6, 0.8, 1.0, 1.4, 2.0, 3.0])
dome_sample_temperatures = np.linspace(0.6, 0.995, 40)
isobar_temperatures = np.array([0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
vapour_fractions = np.array([0.2, 0.4, 0.6, 0.8])
n_samples = 160
# The excluded volume, below which no isobar reaches.
exclusion = 1 / 3
# The critical point.
critical_volume, critical_pressure, critical_temperature = 1.0, 3 / 8, 1.0
# The cylinder: the height of a unit of volume, and its picture's pixels across.
height_per_volume, cylinder_pixels = 0.55, 150
# The colours of the liquid and of the vapour, and how much of what lies behind it the vapour hides; the dome's.
liquid_colour, vapour_colour, vapour_opacity = art.BLUE, art.rgb("#9ccfe0"), 0.35
dome_colour = art.BOUNDARY


def numbers(value: core.Scalar) -> np.ndarray:
    """The values of a scalar as a NumPy array, which the charts index cheaply frame after frame."""
    return np.asarray(value.to_array())


def isotherms(mv, pressure_at: PressureAt, temperatures: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Volumes and pressures along isotherms `[temperatures, samples]`."""
    volumes = np.broadcast_to(np.linspace(0.36, volume_range[1], 2 * n_samples), (len(temperatures), 2 * n_samples))
    return volumes, numbers(pressure_at(mv.scalar(volumes[..., None]), temperatures[:, None]))


def isobar(isobar_at: IsobarAt, enthalpy_at: EnthalpyAt, coexisting: core.State, saturation: core.Scalar
           ) -> tuple[np.ndarray, np.ndarray]:
    """Along the isobar at a saturation pressure: the liquid squeezed toward its excluded volume up to boiling, the
    fluid boiling, and the vapour expanding beyond; the enthalpies and temperatures of its samples."""
    mv = coexisting.context.multivector
    liquid, vapour = float(numbers(mv.l & coexisting)), float(numbers(mv.g & coexisting))
    # The liquid's volumes crowd toward the excluded volume, the vapour's spread out, each by equal ratios.
    squeezed = exclusion + (liquid - exclusion) * np.geomspace(0.05, 1, n_samples)
    expanded = vapour * np.geomspace(1, 4 * temperature_range[1] / (float(numbers(saturation)) * vapour), n_samples)
    volumes = mv.scalar(np.concatenate([squeezed, expanded])[:, None])
    temperatures = isobar_at(volumes, saturation)
    return numbers(enthalpy_at(volumes, numbers(temperatures))), numbers(temperatures)


def region_labels(liquid: tuple[float, float], mixed: tuple[float, float], vapour: tuple[float, float]
                  ) -> list[art.Label]:
    """The names of the three regions a chart of the fluid falls into."""
    return [art.Label(*liquid, "liquid", art.INK, art.TEXT_SIZE, "mm", (0, 0)),
            art.Label(*mixed, "liquid + vapour", art.INK, art.TEXT_SIZE, "mm", (0, 0)),
            art.Label(*vapour, "vapour", art.INK, art.TEXT_SIZE, "mm", (0, 0))]


def marked(diagram: art.Diagram, path: tuple[np.ndarray, np.ndarray]) -> list:
    """The path so far, bold, and a dot at its end; nothing for a path not yet begun."""
    if len(path[0]) == 0:
        return []
    return (art.within(diagram, path[0][None], path[1][None], art.BOLD, art.ORANGE[None], 1.0)
            + art.dots(diagram, path[0][-1], path[1][-1], art.ORANGE))


class Charts:
    """The three charts' fixed curves, sampled once, and the drawing of a state on them."""

    def __init__(self, pressure_at: PressureAt, enthalpy_at: EnthalpyAt, coexistence_at: CoexistenceAt,
                 isobar_at: IsobarAt, temperature: float) -> None:
        self.pressure_volume = art.Diagram(volume_range, pressure_range, (0.0, 0.0), n_rows, n_columns, ("V", "p"))
        self.temperature_enthalpy = art.Diagram(enthalpy_range, temperature_range,
                                                (enthalpy_range[0], temperature_range[0]), n_rows, n_columns,
                                                ("H", "T"))
        # The dome: the coexisting volumes up to the critical point, read in the algebra the states come in.
        coexisting_dome = coexistence_at(dome_sample_temperatures)
        mv = coexisting_dome.context.multivector
        liquid_dome, vapour_dome = mv.l & coexisting_dome, mv.g & coexisting_dome
        liquids, vapours, temperatures = numbers(liquid_dome), numbers(vapour_dome), dome_sample_temperatures
        levels = numbers(pressure_at(liquid_dome, temperatures))                         # [dome]
        liquid_enthalpies = numbers(enthalpy_at(liquid_dome, temperatures))
        vapour_enthalpies = numbers(enthalpy_at(vapour_dome, temperatures))
        critical_enthalpy = float(numbers(enthalpy_at(mv.scalar([critical_volume]), critical_temperature)))
        # Each chart's dome: up the liquid's side to the critical point and down the vapour's.
        self.dome_volumes = np.concatenate([liquids, [critical_volume], vapours[::-1]])
        self.dome_pressures = np.concatenate([levels, [critical_pressure], levels[::-1]])
        self.dome_enthalpies = np.concatenate([liquid_enthalpies, [critical_enthalpy], vapour_enthalpies[::-1]])
        self.dome_temperatures = np.concatenate([temperatures, [critical_temperature], temperatures[::-1]])
        self.fractions = (liquid_enthalpies * (1 - vapour_fractions[:, None])
                          + vapour_enthalpies * vapour_fractions[:, None],
                          np.broadcast_to(temperatures, (len(vapour_fractions), len(temperatures))))
        self.isotherm_volumes, self.isotherm_pressures = isotherms(mv, pressure_at, isotherm_temperatures)
        # The phase diagram: the saturation curve, and lines of equal volume, straight in temperature.
        self.pressure_temperature = art.Diagram(phase_temperature_range, phase_pressure_range,
                                                (phase_temperature_range[0], 0.0), n_rows, n_columns, ("T", "p"))
        self.saturation_temperatures = np.concatenate([temperatures, [critical_temperature]])
        self.saturation_pressures = np.concatenate([levels, [critical_pressure]])
        self.isochore_temperatures = np.linspace(*phase_temperature_range, 60)
        self.isochore_pressures = list(numbers(pressure_at(mv.scalar(isochore_volumes[:, None, None]),
                                                           self.isochore_temperatures[None, :])))
        self.own = isotherm_temperatures == temperature
        self.temperature = temperature
        # The isobars at the saturation pressures of their temperatures.
        states = coexistence_at(isobar_temperatures)
        saturations = pressure_at(mv.l & states, isobar_temperatures)
        self.isobars = [isobar(isobar_at, enthalpy_at, states[index], saturations[index])
                        for index in range(len(isobar_temperatures))]
        self.own_isobar = isobar_temperatures == temperature
        # The loop of the isotherm at the boiling temperature, cut where it crosses the saturation pressure into its
        # two lobes, each closed back along the flat stretch.
        coexisting = coexistence_at(np.array(temperature))
        self.liquid, self.vapour = float(numbers(mv.l & coexisting)), float(numbers(mv.g & coexisting))
        self.saturation = float(numbers(pressure_at(mv.l & coexisting, temperature)))
        loop = np.linspace(self.liquid, self.vapour, 2 * n_samples)
        heights = numbers(pressure_at(mv.scalar(loop[:, None]), temperature))
        middle = 1 + np.flatnonzero(np.diff(heights[1:-1] > self.saturation))[0]
        crossing = loop[middle] + (loop[middle + 1] - loop[middle]) * (self.saturation - heights[middle]) / (
            heights[middle + 1] - heights[middle])
        self.lobes = [(np.concatenate([[self.liquid], loop[1:middle + 1], [crossing]]),
                       np.concatenate([[self.saturation], heights[1:middle + 1], [self.saturation]]), liquid_colour),
                      (np.concatenate([[crossing], loop[middle + 1:-1], [self.vapour]]),
                       np.concatenate([[self.saturation], heights[middle + 1:-1], [self.saturation]]), art.ORANGE)]

    def pressure_volume_chart(self, path: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        diagram = self.pressure_volume
        family = self.isotherm_pressures
        isotherm = art.FAMILIES["temperature"]
        layers = ([art.areas(diagram, x, y, colour, 0.28) for x, y, colour in self.lobes]
                  + art.within(diagram, self.isotherm_volumes[~self.own], family[~self.own], art.THIN,
                               np.broadcast_to(isotherm, (int((~self.own).sum()), 3)), 1.0)
                  + art.within(diagram, self.isotherm_volumes[self.own], family[self.own], art.LINE, isotherm[None],
                               1.0)
                  + art.within(diagram, self.dome_volumes[None], self.dome_pressures[None], art.LINE,
                               dome_colour[None], 1.0)
                  + marked(diagram, path))
        labels = region_labels((0.2, 0.06), (2.0, 0.12), (3.3, 0.33)) + [
            art.Label(critical_volume, critical_pressure, "critical point", dome_colour, art.TEXT_SIZE, "lb", (6, 3))]
        others = list(zip(self.isotherm_volumes[~self.own], family[~self.own]))
        names = [art.Floating(*art.along(list(zip(self.isotherm_volumes[self.own], family[self.own])), 6),
                              f"T = {self.temperature:g}", isotherm, art.TEXT_SIZE, "lb"),
                 art.Floating(*art.along(others, 6), "T = const", isotherm, art.TEXT_SIZE, "lb"),
                 art.Floating(*art.along([(self.dome_volumes, self.dome_pressures)], 3), "coexistence", dome_colour,
                              art.TEXT_SIZE, "lt")]
        return art.floated(diagram, art.picture(diagram, layers, labels), names)

    def temperature_enthalpy_chart(self, path: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        diagram = self.temperature_enthalpy
        isobar = art.FAMILIES["pressure"]
        lines = [art.within(diagram, enthalpies[None], temperatures[None], art.LINE if own else art.THIN,
                            isobar[None], 1.0)
                 for (enthalpies, temperatures), own in zip(self.isobars, self.own_isobar)]
        layers = (art.polylines(diagram, list(zip(*self.fractions)), art.THIN, art.FAMILIES["level"], 1.0, art.SOLID)
                  + [stroke for strokes in lines for stroke in strokes]
                  + art.within(diagram, self.dome_enthalpies[None], self.dome_temperatures[None], art.LINE,
                               dome_colour[None], 1.0)
                  + marked(diagram, path))
        bottom = np.argmin(np.abs(dome_sample_temperatures - temperature_range[0] - 0.03))
        labels = region_labels((-1.15, 1.02), (0.45, 0.665), (2.0, 0.8)) + [
            art.Label(self.dome_enthalpies[len(dome_sample_temperatures)], critical_temperature, "critical point",
                      dome_colour, art.TEXT_SIZE, "lb", (6, 3))] + [
            art.Label(enthalpy, dome_sample_temperatures[bottom], f"{fraction:g}", art.FAMILIES["level"],
                      art.TICK_SIZE, "mt", (0, -2))
            for enthalpy, fraction in zip(self.fractions[0][:, bottom], vapour_fractions)]
        own = [isobar_line for isobar_line, flag in zip(self.isobars, self.own_isobar) if flag]
        others = [isobar_line for isobar_line, flag in zip(self.isobars, self.own_isobar) if not flag]
        names = [art.Floating(*art.along(own, 8), f"p = {self.saturation:.2f}", isobar, art.TEXT_SIZE, "lb"),
                 art.Floating(*art.along(others, 8), "p = const", isobar, art.TEXT_SIZE, "lb"),
                 art.Floating(*art.along(list(zip(*self.fractions)), 4), "x = const", art.FAMILIES["level"],
                              art.TEXT_SIZE, "lb")]
        return art.floated(diagram, art.picture(diagram, layers, labels), names)

    def pressure_temperature_chart(self, path: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """The phase diagram: the saturation pressure against the temperature up to the critical point, with the lines
        of equal volume, and the heated fluid's path across the saturation curve at its fixed pressure."""
        diagram = self.pressure_temperature
        layers = (art.polylines(diagram, [(self.isochore_temperatures, line) for line in self.isochore_pressures],
                                art.THIN, art.FAMILIES["volume"], 1.0, art.SOLID)
                  + art.polylines(diagram, [(self.saturation_temperatures, self.saturation_pressures)], art.LINE,
                                  dome_colour, 1.0, art.SOLID)
                  + art.dots(diagram, critical_temperature, critical_pressure, dome_colour)
                  + marked(diagram, path))
        labels = [art.Label(0.68, 0.33, "liquid", art.INK, art.TEXT_SIZE, "mm", (0, 0)),
                  art.Label(0.98, 0.1, "vapour", art.INK, art.TEXT_SIZE, "mm", (0, 0)),
                  art.Label(critical_temperature, critical_pressure, "critical point", dome_colour, art.TEXT_SIZE,
                            "rb", (-6, 3))]
        names = [art.Floating(*art.along([(self.isochore_temperatures, line) for line in self.isochore_pressures], 4),
                              "V = const", art.FAMILIES["volume"], art.TEXT_SIZE, "lb"),
                 art.Floating(*art.along([(self.saturation_temperatures, self.saturation_pressures)], 3), "saturation",
                              dome_colour, art.TEXT_SIZE, "lt")]
        return art.floated(diagram, art.picture(diagram, layers, labels), names)


def boiler(volume: float, liquid_volume: float, largest_volume: float, warmth: float) -> np.ndarray:
    """The cylinder under its loaded piston, the liquid at the bottom and the vapour above it."""
    height, liquid = volume * height_per_volume, liquid_volume * height_per_volume
    maximum = largest_volume * height_per_volume
    view = art.cylinder_view(maximum, cylinder_pixels)
    parts = art.cylinder(height, maximum) + art.load(height) + [(art.fill(0.0, liquid), liquid_colour)]
    vapour = art.thermal_colour(warmth) * 0.5 + vapour_colour * 0.5
    return solids.draw(parts, view, art.WHITE, inks=[],
                       translucent=[(art.fill(liquid, height), vapour, vapour_opacity)], meshes=[])


def draw_charts(pressure_at: PressureAt, enthalpy_at: EnthalpyAt, coexistence_at: CoexistenceAt,
                isobar_at: IsobarAt, temperature: float):
    """The three property charts, the isotherm and the isobar at the temperature drawn dark."""
    charts = Charts(pressure_at, enthalpy_at, coexistence_at, isobar_at, temperature)
    empty = (np.zeros(0), np.zeros(0))
    return art.figure(art.beside([charts.pressure_volume_chart(empty), charts.temperature_enthalpy_chart(empty),
                                  charts.pressure_temperature_chart(empty)]))


def heating(pressure_at: PressureAt, enthalpy_at: EnthalpyAt, coexistence_at: CoexistenceAt, isobar_at: IsobarAt,
            temperature: float, states: Iterator[tuple[core.Scalar, core.Scalar, core.Scalar, core.Scalar]]
            ) -> Iterator[np.ndarray]:
    """Each state of the heated fluid as it comes, its volume, temperature, enthalpy and the volume its liquid takes up:
    the cylinder, and the path so far on the three charts."""
    charts = Charts(pressure_at, enthalpy_at, coexistence_at, isobar_at, temperature)
    path = []
    for volume, fluid_temperature, enthalpy, liquid_volume in states:
        values = [float(numbers(value)) for value in (volume, fluid_temperature, enthalpy, liquid_volume)]
        path.append(values)
        traced = np.array(path)
        warmth = (values[1] - temperature_range[0]) / (temperature_range[1] - temperature_range[0])
        pressures = np.full(len(traced), charts.saturation)
        yield art.beside([boiler(values[0], values[3], volume_range[1], warmth),
                          charts.pressure_volume_chart((traced[:, 0], pressures)),
                          charts.temperature_enthalpy_chart((traced[:, 2], traced[:, 1])),
                          charts.pressure_temperature_chart((traced[:, 1], pressures))])
