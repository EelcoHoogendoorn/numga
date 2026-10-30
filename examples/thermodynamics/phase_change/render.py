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

from numga import concatenate, stack
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
liquid_colour, vapour_colour, vapour_opacity = art.blue, art.rgb("#9ccfe0"), 0.35
dome_colour = art.boundary


def isotherms(mv, pressure_at: PressureAt, temperatures: np.ndarray) -> art.Point:
    """Isotherms on the pressure–volume chart `[temperatures, samples]`."""
    volumes = np.broadcast_to(np.linspace(0.36, volume_range[1], 2 * n_samples), (len(temperatures), 2 * n_samples))
    return art.chart_points(volumes, art.context.include_from_algebra(pressure_at(mv.scalar(volumes[..., None]), temperatures[:, None])))


def isobar(isobar_at: IsobarAt, enthalpy_at: EnthalpyAt, coexisting: core.State, saturation: core.Scalar
           ) -> art.Point:
    """The isobar at a saturation pressure on the temperature–enthalpy chart `[samples]`: the liquid squeezed toward
    its excluded volume up to boiling, the fluid boiling, and the vapour expanding beyond."""
    mv = coexisting.context.multivector
    liquid, vapour = mv.l & coexisting, mv.g & coexisting
    # The liquid's volumes crowd toward the excluded volume, the vapour's spread out, each by equal ratios.
    squeezed = exclusion + (liquid - exclusion) * np.geomspace(0.05, 1, n_samples)
    expansion = 4 * temperature_range[1] / (saturation * vapour)
    expanded = vapour * (expansion.log() * np.linspace(0, 1, n_samples)).exp()
    volumes = concatenate((squeezed, expanded))
    temperatures = isobar_at(volumes, saturation)
    return art.chart_points(art.context.include_from_algebra(enthalpy_at(volumes, temperatures)), art.context.include_from_algebra(temperatures))


def region_labels(liquid: tuple[float, float], mixed: tuple[float, float], vapour: tuple[float, float]
                  ) -> list[art.Label]:
    """The names of the three regions a chart of the fluid falls into, at values of its two quantities."""
    return [art.Label(art.chart_points(*liquid), "liquid", art.ink, art.text_size, "mm", (0, 0)),
            art.Label(art.chart_points(*mixed), "liquid + vapour", art.ink, art.text_size, "mm", (0, 0)),
            art.Label(art.chart_points(*vapour), "vapour", art.ink, art.text_size, "mm", (0, 0))]


def marked(diagram: art.Diagram, path: art.Point) -> list:
    """The path so far `[steps]`, bold, and a dot at its end; nothing for a path not yet begun."""
    if path.shape[0] == 0:
        return []
    return (art.within(diagram, path[None], art.bold, art.orange[None], 1.0)
            + art.dots(diagram, path[-1:], art.orange))


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
        temperatures = dome_sample_temperatures
        levels = art.context.include_from_algebra(pressure_at(liquid_dome, temperatures))                         # [dome]
        liquid_heat = art.chart_points(art.context.include_from_algebra(enthalpy_at(liquid_dome, temperatures)), temperatures)   # [dome] Point-
        vapour_heat = art.chart_points(art.context.include_from_algebra(enthalpy_at(vapour_dome, temperatures)), temperatures)   # [dome] Point-
        critical_enthalpy = art.context.include_from_algebra(enthalpy_at(mv.scalar([critical_volume]), critical_temperature))
        # Each chart's dome: up the liquid's side to the critical point and down the vapour's.
        self.pressure_dome = concatenate([art.chart_points(art.context.include_from_algebra(liquid_dome), levels),
                                          art.chart_points(critical_volume, critical_pressure)[None],
                                          art.chart_points(art.context.include_from_algebra(vapour_dome), levels)[::-1]])   # [dome] Point-
        self.heat_dome = concatenate([liquid_heat, art.chart_points(critical_enthalpy, critical_temperature)[None],
                                      vapour_heat[::-1]])                                     # [dome] Point-
        # The mixtures of equal vapour fraction across the dome, each shared out between the phases at its temperature.
        self.fractions = (liquid_heat * (1 - vapour_fractions[:, None])
                          + vapour_heat * vapour_fractions[:, None])                 # [fractions, dome] Point-
        self.isotherm_lines = isotherms(mv, pressure_at, isotherm_temperatures)     # [isotherms, samples] Point-
        # The phase diagram: the saturation curve, and lines of equal volume, straight in temperature.
        self.pressure_temperature = art.Diagram(phase_temperature_range, phase_pressure_range,
                                                (phase_temperature_range[0], 0.0), n_rows, n_columns, ("T", "p"))
        self.saturation_curve = concatenate([art.chart_points(temperatures, levels),
                                             art.chart_points(critical_temperature, critical_pressure)[None]])
        isochore_temperatures = np.linspace(*phase_temperature_range, 60)
        isochores = art.chart_points(isochore_temperatures, art.context.include_from_algebra(pressure_at(
            mv.scalar(isochore_volumes[:, None, None]), isochore_temperatures[None, :])))   # [volumes, samples] Point-
        self.isochores = [isochores[volume] for volume in range(isochores.shape[0])]
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
        liquid, vapour = mv.l & coexisting, mv.g & coexisting
        self.saturation = art.context.include_from_algebra(pressure_at(mv.l & coexisting, temperature))
        loop_volumes = liquid + (vapour - liquid) * np.linspace(0, 1, 2 * n_samples)
        loop = art.chart_points(art.context.include_from_algebra(loop_volumes), art.context.include_from_algebra(pressure_at(loop_volumes, temperature)))
        # The plane the flat stretch lies in, where the pressure is the saturation pressure.
        flat = art.mv.y - art.mv.w * self.saturation                                 # [] Plane+
        middle = 1 + np.flatnonzero(np.diff((flat & loop[1:-1]) > 0))[0]
        # Where the loop crosses the flat stretch between its samples on either side: the line through them met with it.
        crossing = art.planar.unit((loop[middle] & loop[middle + 1]) ^ flat)[None]   # [1] Point-
        # The loop's two ends on the flat stretch: the lines through them along the pressure, met with it.
        ends = art.planar.unit((loop[np.array([0, -1])] & art.mv.y.dual()) ^ flat)    # [2] Point-
        self.lobes = [(concatenate([ends[:1], loop[1:middle + 1], crossing]), liquid_colour),
                      (concatenate([crossing, loop[middle + 1:-1], ends[1:]]), art.orange)]

    def pressure_volume_chart(self, path: art.Point) -> np.ndarray:
        diagram = self.pressure_volume
        isotherm = art.families["temperature"]
        others = self.isotherm_lines[~self.own]                                      # [isotherms, samples] Point-
        own = self.isotherm_lines[self.own]                                          # [1, samples] Point-
        layers = ([art.areas(diagram, corners, colour, 0.28) for corners, colour in self.lobes]
                  + art.within(diagram, others, art.thin, np.broadcast_to(isotherm, (others.shape[0], 3)), 1.0)
                  + art.within(diagram, own, art.medium, isotherm[None], 1.0)
                  + art.within(diagram, self.pressure_dome[None], art.medium, dome_colour[None], 1.0)
                  + marked(diagram, path))
        labels = region_labels((0.2, 0.06), (2.0, 0.12), (3.3, 0.33)) + [
            art.Label(art.chart_points(critical_volume, critical_pressure), "critical point", dome_colour,
                      art.text_size, "lb", (6, 3))]
        names = [art.Floating(art.along([own[0]], 6), f"T = {self.temperature:g}", isotherm, art.text_size, "lb"),
                 art.Floating(art.along([others[line] for line in range(others.shape[0])], 6), "T = const", isotherm,
                              art.text_size, "lb"),
                 art.Floating(art.along([self.pressure_dome], 3), "coexistence", dome_colour, art.text_size, "lt")]
        return art.floated(diagram, art.picture(diagram, layers, labels), names)

    def temperature_enthalpy_chart(self, path: art.Point) -> np.ndarray:
        diagram = self.temperature_enthalpy
        isobar = art.families["pressure"]
        lines = [art.within(diagram, isobar_line[None], art.medium if own else art.thin, isobar[None], 1.0)
                 for isobar_line, own in zip(self.isobars, self.own_isobar)]
        fractions = [self.fractions[fraction] for fraction in range(len(vapour_fractions))]
        layers = (art.polylines(diagram, fractions, art.thin, art.families["level"], 1.0, art.SOLID)
                  + [stroke for strokes in lines for stroke in strokes]
                  + art.within(diagram, self.heat_dome[None], art.medium, dome_colour[None], 1.0)
                  + marked(diagram, path))
        bottom = np.argmin(np.abs(dome_sample_temperatures - temperature_range[0] - 0.03))
        labels = region_labels((-1.15, 1.02), (0.45, 0.665), (2.0, 0.8)) + [
            art.Label(self.heat_dome[len(dome_sample_temperatures)], "critical point", dome_colour, art.text_size, "lb",
                      (6, 3))] + [
            art.Label(self.fractions[index, bottom], f"{fraction:g}", art.families["level"], art.tick_size, "mt",
                      (0, -2))
            for index, fraction in enumerate(vapour_fractions)]
        own = [isobar_line for isobar_line, flag in zip(self.isobars, self.own_isobar) if flag]
        others = [isobar_line for isobar_line, flag in zip(self.isobars, self.own_isobar) if not flag]
        names = [art.Floating(art.along(own, 8), f"p = {float(self.saturation.to_array()):.2f}", isobar, art.text_size, "lb"),
                 art.Floating(art.along(others, 8), "p = const", isobar, art.text_size, "lb"),
                 art.Floating(art.along(fractions, 4), "x = const", art.families["level"], art.text_size, "lb")]
        return art.floated(diagram, art.picture(diagram, layers, labels), names)

    def pressure_temperature_chart(self, path: art.Point) -> np.ndarray:
        """The phase diagram: the saturation pressure against the temperature up to the critical point, with the lines
        of equal volume, and the heated fluid's path across the saturation curve at its fixed pressure."""
        diagram = self.pressure_temperature
        layers = (art.polylines(diagram, self.isochores, art.thin, art.families["volume"], 1.0, art.SOLID)
                  + art.polylines(diagram, [self.saturation_curve], art.medium, dome_colour, 1.0, art.SOLID)
                  + art.dots(diagram, self.saturation_curve[-1:], dome_colour)
                  + marked(diagram, path))
        labels = [art.Label(art.chart_points(0.68, 0.33), "liquid", art.ink, art.text_size, "mm", (0, 0)),
                  art.Label(art.chart_points(0.98, 0.1), "vapour", art.ink, art.text_size, "mm", (0, 0)),
                  art.Label(self.saturation_curve[-1], "critical point", dome_colour, art.text_size, "rb", (-6, 3))]
        names = [art.Floating(art.along(self.isochores, 4), "V = const", art.families["volume"], art.text_size, "lb"),
                 art.Floating(art.along([self.saturation_curve], 3), "saturation", dome_colour, art.text_size, "lt")]
        return art.floated(diagram, art.picture(diagram, layers, labels), names)


def boiler(volume: solids.Scalar, liquid_volume: solids.Scalar, largest_volume: float, warmth: float) -> np.ndarray:
    """The cylinder under its loaded piston, the liquid at the bottom and the vapour above it."""
    height, liquid = volume * height_per_volume, liquid_volume * height_per_volume
    maximum = solids.mv.scalar([largest_volume * height_per_volume])
    view = art.cylinder_view(maximum, cylinder_pixels)
    parts = art.cylinder(height, maximum) + art.load(height) + [(art.fill(0.0, liquid), liquid_colour)]
    vapour = art.thermal_colour(warmth) * 0.5 + vapour_colour * 0.5
    return solids.draw(parts, view, art.white, inks=[],
                       translucent=[(art.fill(liquid, height), vapour, vapour_opacity)], meshes=[])


def draw_charts(pressure_at: PressureAt, enthalpy_at: EnthalpyAt, coexistence_at: CoexistenceAt,
                isobar_at: IsobarAt, temperature: float):
    """The three property charts, the isotherm and the isobar at the temperature drawn dark."""
    charts = Charts(pressure_at, enthalpy_at, coexistence_at, isobar_at, temperature)
    empty = art.chart_points(np.zeros(0), np.zeros(0))
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
        path.append(art.context.include_from_algebra(stack((volume, fluid_temperature, enthalpy))))
        history = stack(path)
        volumes, temperatures, enthalpies = history[:, 0], history[:, 1], history[:, 2]
        warmth = float(((fluid_temperature - temperature_range[0]) / (temperature_range[1] - temperature_range[0])).to_array())
        yield art.beside([boiler(solids.context.include_from_algebra(volume), solids.context.include_from_algebra(liquid_volume),
                                volume_range[1], warmth),
                          charts.pressure_volume_chart(art.chart_points(volumes, charts.saturation)),
                          charts.temperature_enthalpy_chart(art.chart_points(enthalpies, temperatures)),
                          charts.pressure_temperature_chart(art.chart_points(temperatures, charts.saturation))])
