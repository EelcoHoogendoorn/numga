"""Boiling under a loaded piston beside two property charts of the van der Waals fluid.

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
IsobarAt = Callable[[core.Scalar, float], core.Scalar]

# The charts' size in pixels; the pressure–volume chart's ranges and isotherms; the temperature–enthalpy chart's
# ranges, the temperatures of the dome's samples, the saturation temperatures of its isobars, and the vapour fractions
# of its lines across the dome.
ROWS, COLUMNS = 300, 330
VOLUMES, PRESSURES = (0.0, 4.4), (0.0, 0.44)
ISOTHERMS = np.array([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1])
ENTHALPIES, TEMPERATURES = (-1.6, 2.4), (0.6, 1.12)
# The phase diagram's ranges, temperature and pressure, and the volumes of its lines of equal volume.
PHASE_TEMPERATURES, PHASE_PRESSURES = (0.6, 1.12), (0.0, 0.55)
ISOCHORES = np.array([0.5, 0.6, 0.8, 1.0, 1.4, 2.0, 3.0])
DOME = np.linspace(0.6, 0.995, 40)
ISOBARS = np.array([0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
FRACTIONS = np.array([0.2, 0.4, 0.6, 0.8])
N_SAMPLES = 160
# The excluded volume, below which no isobar reaches.
EXCLUSION = 1 / 3
# The critical point.
CRITICAL_VOLUME, CRITICAL_PRESSURE, CRITICAL_TEMPERATURE = 1.0, 3 / 8, 1.0
# The cylinder: the height of a unit of volume, and its picture's pixels across.
HEIGHT_PER_VOLUME, CYLINDER_PIXELS = 0.55, 150
# The colours of the liquid and of the vapour, and how much of what lies behind it the vapour hides; the dome's.
LIQUID, VAPOUR, VAPOUR_OPACITY = art.BLUE, art.rgb("#9ccfe0"), 0.35
DOME_COLOUR = art.BOUNDARY
# Line widths, in pixels.
THIN, BOLD = 1.0, 2.2


def numbers(value: core.Scalar) -> np.ndarray:
    return value.to_array()


def isotherms(mv, pressure_at: PressureAt, temperatures: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Volumes and pressures along isotherms `[temperatures, samples]`."""
    volumes = np.broadcast_to(np.linspace(0.36, VOLUMES[1], 2 * N_SAMPLES), (len(temperatures), 2 * N_SAMPLES))
    return volumes, numbers(pressure_at(mv.scalar(volumes[..., None]), temperatures[:, None]))


def isobar(isobar_at: IsobarAt, enthalpy_at: EnthalpyAt, coexisting: core.State, level: float
           ) -> tuple[np.ndarray, np.ndarray]:
    """Along the isobar at a saturation pressure: the liquid squeezed toward its excluded volume up to boiling, the
    fluid boiling, and the vapour expanding beyond; the enthalpies and temperatures of its samples."""
    mv = coexisting.context.multivector
    liquid, vapour = float(numbers(mv.l & coexisting)), float(numbers(mv.g & coexisting))
    # The liquid's volumes crowd toward the excluded volume, the vapour's spread out, each by equal ratios.
    squeezed = EXCLUSION + (liquid - EXCLUSION) * np.geomspace(0.05, 1, N_SAMPLES)
    expanded = vapour * np.geomspace(1, 4 * TEMPERATURES[1] / (level * vapour), N_SAMPLES)
    volumes = mv.scalar(np.concatenate([squeezed, expanded])[:, None])
    temperatures = isobar_at(volumes, level)
    return numbers(enthalpy_at(volumes, temperatures.to_array())), numbers(temperatures)


def region_labels(liquid: tuple[float, float], mixed: tuple[float, float], vapour: tuple[float, float]
                  ) -> list[art.Label]:
    """The names of the three regions a chart of the fluid falls into."""
    return [art.Label(*liquid, "liquid", art.INK, art.NAME_SIZE, "mm", (0, 0)),
            art.Label(*mixed, "liquid + vapour", art.INK, art.NAME_SIZE, "mm", (0, 0)),
            art.Label(*vapour, "vapour", art.INK, art.NAME_SIZE, "mm", (0, 0))]


def marked(diagram: art.Diagram, path: tuple[np.ndarray, np.ndarray]) -> list:
    """The path so far, bold, and a dot at its end; nothing for a path not yet begun."""
    if len(path[0]) == 0:
        return []
    return (art.within(diagram, path[0][None], path[1][None], BOLD, art.ORANGE[None], 1.0)
            + art.dots(diagram, path[0][-1], path[1][-1], art.ORANGE))


class Charts:
    """The two charts' fixed curves, sampled once, and the drawing of a state on them."""

    def __init__(self, pressure_at: PressureAt, enthalpy_at: EnthalpyAt, coexistence_at: CoexistenceAt,
                 isobar_at: IsobarAt, temperature: float) -> None:
        self.pressure_volume = art.Diagram(VOLUMES, PRESSURES, (0.0, 0.0), ROWS, COLUMNS, ("V", "p"))
        self.temperature_enthalpy = art.Diagram(ENTHALPIES, TEMPERATURES, (ENTHALPIES[0], TEMPERATURES[0]), ROWS, COLUMNS, ("H", "T"))
        # The dome: the coexisting volumes up to the critical point, read in the algebra the states come in.
        coexisting_dome = coexistence_at(DOME)
        mv = coexisting_dome.context.multivector
        liquid_dome, vapour_dome = mv.l & coexisting_dome, mv.g & coexisting_dome
        liquids, vapours, temperatures = numbers(liquid_dome), numbers(vapour_dome), DOME
        levels = numbers(pressure_at(liquid_dome, temperatures))                         # [dome]
        liquid_enthalpies = numbers(enthalpy_at(liquid_dome, temperatures))
        vapour_enthalpies = numbers(enthalpy_at(vapour_dome, temperatures))
        critical_enthalpy = float(numbers(enthalpy_at(mv.scalar([CRITICAL_VOLUME]), CRITICAL_TEMPERATURE)))
        # Each chart's dome: up the liquid's side to the critical point and down the vapour's.
        self.dome_volumes = np.concatenate([liquids, [CRITICAL_VOLUME], vapours[::-1]])
        self.dome_pressures = np.concatenate([levels, [CRITICAL_PRESSURE], levels[::-1]])
        self.dome_enthalpies = np.concatenate([liquid_enthalpies, [critical_enthalpy], vapour_enthalpies[::-1]])
        self.dome_temperatures = np.concatenate([temperatures, [CRITICAL_TEMPERATURE], temperatures[::-1]])
        self.fractions = (liquid_enthalpies * (1 - FRACTIONS[:, None]) + vapour_enthalpies * FRACTIONS[:, None],
                          np.broadcast_to(temperatures, (len(FRACTIONS), len(temperatures))))
        self.isotherm_volumes, self.isotherm_pressures = isotherms(mv, pressure_at, ISOTHERMS)
        # The phase diagram: the saturation curve, and lines of equal volume, straight in temperature.
        self.pressure_temperature = art.Diagram(PHASE_TEMPERATURES, PHASE_PRESSURES, (PHASE_TEMPERATURES[0], 0.0), ROWS,
                                                COLUMNS, ("T", "p"))
        self.saturation_temperatures = np.concatenate([temperatures, [CRITICAL_TEMPERATURE]])
        self.saturation_pressures = np.concatenate([levels, [CRITICAL_PRESSURE]])
        self.isochore_temperatures = np.linspace(*PHASE_TEMPERATURES, 60)
        self.isochore_pressures = list(numbers(pressure_at(mv.scalar(ISOCHORES[:, None, None]),
                                                           self.isochore_temperatures[None, :])))
        self.own = ISOTHERMS == temperature
        self.temperature = temperature
        # The isobars at the saturation pressures of their temperatures.
        states = coexistence_at(ISOBARS)
        saturations = numbers(pressure_at(mv.l & states, ISOBARS))
        self.isobars = [isobar(isobar_at, enthalpy_at, states[index], saturations[index])
                        for index in range(len(ISOBARS))]
        self.own_isobar = ISOBARS == temperature
        # The loop of the isotherm at the boiling temperature, cut where it crosses the saturation pressure into its
        # two lobes, each closed back along the flat stretch.
        coexisting = coexistence_at(np.array(temperature))
        self.liquid, self.vapour = float(numbers(mv.l & coexisting)), float(numbers(mv.g & coexisting))
        self.saturation = float(numbers(pressure_at(mv.l & coexisting, temperature)))
        loop = np.linspace(self.liquid, self.vapour, 2 * N_SAMPLES)
        heights = numbers(pressure_at(mv.scalar(loop[:, None]), temperature))
        middle = 1 + np.flatnonzero(np.diff(heights[1:-1] > self.saturation))[0]
        crossing = loop[middle] + (loop[middle + 1] - loop[middle]) * (self.saturation - heights[middle]) / (
            heights[middle + 1] - heights[middle])
        self.lobes = [(np.concatenate([[self.liquid], loop[1:middle + 1], [crossing]]),
                       np.concatenate([[self.saturation], heights[1:middle + 1], [self.saturation]]), LIQUID),
                      (np.concatenate([[crossing], loop[middle + 1:-1], [self.vapour]]),
                       np.concatenate([[self.saturation], heights[middle + 1:-1], [self.saturation]]), art.ORANGE)]

    def pressure_volume_chart(self, path: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        diagram = self.pressure_volume
        family = self.isotherm_pressures
        isotherm = art.FAMILIES["temperature"]
        layers = ([art.areas(diagram, x, y, colour, 0.28) for x, y, colour in self.lobes]
                  + art.within(diagram, self.isotherm_volumes[~self.own], family[~self.own], THIN,
                               np.broadcast_to(isotherm, (int((~self.own).sum()), 3)), 1.0)
                  + art.within(diagram, self.isotherm_volumes[self.own], family[self.own], 1.8, isotherm[None], 1.0)
                  + art.within(diagram, self.dome_volumes[None], self.dome_pressures[None], 1.8, DOME_COLOUR[None], 1.0)
                  + marked(diagram, path))
        labels = region_labels((0.2, 0.06), (2.0, 0.12), (3.3, 0.33)) + [
            art.Label(CRITICAL_VOLUME, CRITICAL_PRESSURE, "critical point", DOME_COLOUR, art.LABEL_SIZE, "lb", (6, 3)),
            art.Label(VOLUMES[1], self.isotherm_pressures[self.own][0, -1], f"T = {self.temperature:g}", isotherm,
                      art.LABEL_SIZE, "rb", (0, 3))]
        return art.picture(diagram, layers, labels)

    def temperature_enthalpy_chart(self, path: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        diagram = self.temperature_enthalpy
        isobar = art.FAMILIES["pressure"]
        lines = [art.within(diagram, enthalpies[None], temperatures[None], 1.8 if own else THIN, isobar[None], 1.0)
                 for (enthalpies, temperatures), own in zip(self.isobars, self.own_isobar)]
        layers = (art.polylines(diagram, list(zip(*self.fractions)), THIN, art.FAMILIES["level"], 1.0, art.SOLID)
                  + [stroke for strokes in lines for stroke in strokes]
                  + art.within(diagram, self.dome_enthalpies[None], self.dome_temperatures[None], 1.8, DOME_COLOUR[None], 1.0)
                  + marked(diagram, path))
        bottom = np.argmin(np.abs(DOME - TEMPERATURES[0] - 0.03))
        labels = region_labels((-1.15, 1.02), (0.45, 0.665), (2.0, 0.8)) + [
            art.Label(self.dome_enthalpies[len(DOME)], CRITICAL_TEMPERATURE, "critical point", DOME_COLOUR,
                      art.LABEL_SIZE, "lb", (6, 3)),
            art.Label(ENTHALPIES[1], self.temperature, f"p = {self.saturation:.2f}", isobar, art.LABEL_SIZE, "rb",
                      (0, 3))] + [
            art.Label(enthalpy, DOME[bottom], f"{fraction:g}", art.FAMILIES["level"], art.LABEL_SIZE, "mt", (0, -2))
            for enthalpy, fraction in zip(self.fractions[0][:, bottom], FRACTIONS)]
        return art.picture(diagram, layers, labels)

    def pressure_temperature_chart(self, path: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """The phase diagram: the saturation pressure against the temperature up to the critical point, with the lines
        of equal volume, and the heated fluid's path across the saturation curve at its fixed pressure."""
        diagram = self.pressure_temperature
        layers = (art.polylines(diagram, [(self.isochore_temperatures, line) for line in self.isochore_pressures], THIN,
                                art.FAMILIES["volume"], 1.0, art.SOLID)
                  + art.polylines(diagram, [(self.saturation_temperatures, self.saturation_pressures)], 1.8, DOME_COLOUR,
                                  1.0, art.SOLID)
                  + art.dots(diagram, CRITICAL_TEMPERATURE, CRITICAL_PRESSURE, DOME_COLOUR)
                  + marked(diagram, path))
        labels = [art.Label(0.68, 0.33, "liquid", art.INK, art.NAME_SIZE, "mm", (0, 0)),
                  art.Label(0.98, 0.1, "vapour", art.INK, art.NAME_SIZE, "mm", (0, 0)),
                  art.Label(CRITICAL_TEMPERATURE, CRITICAL_PRESSURE, "critical point", DOME_COLOUR, art.LABEL_SIZE, "rb",
                            (-6, 3)),
                  art.Label(PHASE_TEMPERATURES[1], self.isochore_pressures[-1][-1], "V = const", art.FAMILIES["volume"],
                            art.LABEL_SIZE, "rt", (0, -3))]
        return art.picture(diagram, layers, labels)


def boiler(volume: float, liquid_volume: float, largest_volume: float, warmth: float) -> np.ndarray:
    """The cylinder under its loaded piston, the liquid at the bottom and the vapour above it."""
    height, liquid = volume * HEIGHT_PER_VOLUME, liquid_volume * HEIGHT_PER_VOLUME
    maximum = largest_volume * HEIGHT_PER_VOLUME
    view = art.cylinder_view(maximum, CYLINDER_PIXELS)
    parts = art.cylinder(height, maximum) + art.load(height) + [(art.fill(0.0, liquid), LIQUID)]
    vapour = art.thermal_colour(warmth) * 0.5 + VAPOUR * 0.5
    return solids.draw(parts, view, art.WHITE, inks=[], translucent=[(art.fill(liquid, height), vapour, VAPOUR_OPACITY)],
                       meshes=[])


def draw_charts(pressure_at: PressureAt, enthalpy_at: EnthalpyAt, coexistence_at: CoexistenceAt,
                isobar_at: IsobarAt, temperature: float):
    """The two property charts, the isotherm and the isobar at the temperature drawn dark."""
    charts = Charts(pressure_at, enthalpy_at, coexistence_at, isobar_at, temperature)
    empty = (np.zeros(0), np.zeros(0))
    return art.figure(art.beside([charts.pressure_volume_chart(empty), charts.temperature_enthalpy_chart(empty),
                                  charts.pressure_temperature_chart(empty)]))


def heating(pressure_at: PressureAt, enthalpy_at: EnthalpyAt, coexistence_at: CoexistenceAt, isobar_at: IsobarAt,
            temperature: float, states: Iterator[tuple[core.Scalar, core.Scalar, core.Scalar, core.Scalar]]
            ) -> Iterator[np.ndarray]:
    """Each state of the heated fluid as it comes, its volume, temperature, enthalpy and the volume its liquid takes up:
    the cylinder, and the path so far on the two charts."""
    charts = Charts(pressure_at, enthalpy_at, coexistence_at, isobar_at, temperature)
    path = []
    for volume, warmth, enthalpy, liquid_volume in states:
        values = [float(numbers(value)) for value in (volume, warmth, enthalpy, liquid_volume)]
        path.append(values)
        traced = np.array(path)
        share = (values[1] - TEMPERATURES[0]) / (TEMPERATURES[1] - TEMPERATURES[0])
        pressures = np.full(len(traced), charts.saturation)
        yield art.beside([boiler(values[0], values[3], VOLUMES[1], share),
                          charts.pressure_volume_chart((traced[:, 0], pressures)),
                          charts.temperature_enthalpy_chart((traced[:, 2], traced[:, 1])),
                          charts.pressure_temperature_chart((traced[:, 1], pressures))])
