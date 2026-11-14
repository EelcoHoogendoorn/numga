"""Render two bodies exchanging heat alongside their state-space, entropy and temperature diagrams.

Two blocks scaled by the bodies' heat capacities and joined by a conducting bar sit beside three diagrams. Each block is
coloured by its temperature gap from equilibrium, blending linear and logarithmic scales so the colour keeps changing as
the gap closes. The first diagram reads the two energies directly off the state with `a` and `b`, showing teal entropy
contours, the line of constant total energy along `transfer`, and the black contour tangent to that line at equilibrium.
The second chart plots total entropy along the constant-energy line against the energy of body `a`, with the tangent
showing the entropic slope that drives the state uphill to equilibrium. The third chart plots the two temperatures over
time as they converge to their equilibrium value.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import chain

import numpy as np

from numga import Extensor, stack
from examples.thermodynamics import rendering as art
from examples.thermodynamics.entropy_production import core
from numga.render import solids

mv = core.mv
# Pixel dimensions and axis ranges for the state-space and entropy diagrams.
n_rows, n_columns = 300, 310
energy_range = (0.0, 4.3)
entropy_range = (-1.6, 1.3)
# Contour levels, sampling grid for entropy contours, and sample count along the constant-energy line.
entropy_levels = np.linspace(-0.4, 1.6, 11)
n_grid = 160
grid_energies = np.linspace(0.04, 4.3, n_grid)
n_line_samples = 240
tangent_half_length = 0.8
# Pixel width of the 3D block view, logarithmic span of the colour ramp, and ranges of the temperature chart.
n_block_pixels = 240
colour_decades = 2.5
time_range, temperature_range = (0.0, 5.8), (0.0, 3.0)


def bodies(temperatures: np.ndarray, capacities: np.ndarray, max_gap: float) -> np.ndarray:
    """Render the two blocks scaled by their heat capacities, coloured by their temperature gaps and joined by a bar."""
    sizes = 0.5 * np.cbrt(capacities)
    # Each block rests on the floor with its centre half its side length above z = 0.
    centres = solids.mv.point([[-1, 0, 0], [1, 0, 0]]) + art.rise * sizes         # [2] Point-
    blocks = solids.boxes(solids.translator(centres - solids.origin), np.repeat(sizes[:, None], 3, axis=1))
    equilibrium_temperature = (capacities * temperatures).sum() / capacities.sum()
    colours = np.stack([
        art.thermal_colour(art.relaxing(temperature - equilibrium_temperature, max_gap, colour_decades))
        for temperature in temperatures
    ])
    bar = art.box(solids.mv.point([0, 0, 0.45]), np.array([0.5, 0.12, 0.1]))
    corners = solids.mv.point(np.array([[x, y, z] for x in (-1.75, 1.75) for y in (-0.75, 0.75) for z in (0, 1.6)]))
    view = solids.view([corners], 0.0, np.radians(18), np.radians(-16), 0.06, n_block_pixels)
    return solids.draw([(blocks, colours), (bar, art.grey)], view, art.white, inks=[], translucent=[], meshes=[])


class Charts:
    """Hold the fixed contour and curve layers of the three diagrams and draw each frame's state over them."""

    def __init__(
        self,
        entropy_of: Callable[[Extensor], Extensor],
        entropy_potential_of: Callable[[Extensor], Extensor],
        transfer: Extensor,
        state: Extensor,
    ) -> None:
        self.entropy_of = entropy_of
        self.entropy_potential_of = entropy_potential_of
        self.transfer = transfer
        self.energies = art.Diagram(
            energy_range, energy_range, (0.0, 0.0), n_rows, n_columns, ("U_a", "U_b"), (mv.a, mv.b), (0 * mv.a, 1.0)
        )
        self.cooling = art.Diagram(
            time_range, temperature_range, (0.0, 0.0), n_rows, n_columns, ("t", "T"),
            (art.mv.x, art.mv.y), (art.mv.w, 0.0),
        )
        self.hill = art.Diagram(
            energy_range, entropy_range, (0.0, entropy_range[0]), n_rows, n_columns, ("U_a", "S"),
            (art.mv.x, art.mv.y), (art.mv.w, 0.0),
        )
        grid = mv.a.dual() * grid_energies[None, :] + mv.b.dual() * grid_energies[:, None]    # [n_grid, n_grid] State-
        grid_entropy = np.asarray(entropy_of(grid).to_array())
        self.levels = art.Levels(grid, grid_entropy, entropy_levels)
        # Sample the line of constant total energy through the initial state from nearly empty to nearly full body a.
        energies_a = (core.energy_potential & state) * np.linspace(0.02, 0.98, n_line_samples)  # [n_line_samples] Scalar+
        energy_line = state + transfer * ((energies_a - (mv.a & state)) / (mv.a & transfer))    # [n_line_samples] State-
        entropies = entropy_of(energy_line)                                                     # [n_line_samples] Scalar+
        peak_index = entropies.argmax()
        entropy_curve = self.entropy_points(energy_line)                                        # [n_line_samples] Point-
        # The entropy contour through the maximum along the constant-energy line is tangent to that line there.
        equilibrium_contour = art.Levels(grid, grid_entropy, np.asarray(entropies[peak_index].to_array())[None])
        self.energy_layers = (
            art.level_family(self.energies, self.levels, "entropy", art.thin)
            + art.level_lines(self.energies, equilibrium_contour, art.boundary, art.medium)
            + art.polylines(self.energies, [energy_line], art.medium, art.ink, 1.0, art.SOLID)
            + art.dots(self.energies, energy_line[peak_index][None], art.green)
        )
        self.hill_layers = (
            art.polylines(self.hill, [entropy_curve], art.medium, art.ink, 1.0, art.SOLID)
            + art.dots(self.hill, entropy_curve[peak_index][None], art.green)
        )

    def entropy_points(self, states: Extensor) -> art.Point:
        """Map states onto the chart of total entropy against the energy of body a."""
        return art.chart_points(
            art.context.include_from_algebra(mv.a & states),
            art.context.include_from_algebra(self.entropy_of(states)),
        )

    def pictures(self, state: Extensor, state_path: list[Extensor], entropy_path: list[art.Point]) -> list[np.ndarray]:
        """Render the state-space and entropy diagrams with the trajectory up to the current frame."""
        current_state, current_point = state_path[-1], entropy_path[-1]                         # [] State-, Point-
        energy_layers = (
            self.energy_layers
            + art.polylines(self.energies, [stack(state_path)], art.bold, art.orange, 1.0, art.SOLID)
            + art.dots(self.energies, current_state[None], art.orange)
        )
        # Pairing the entropic potential with transfer gives the slope of entropy along transfer without autodiff.
        entropy_slope = self.entropy_potential_of(state) & self.transfer                        # [] Scalar+
        tangent = art.chart_directions(
            art.context.include_from_algebra(mv.a & self.transfer),
            art.context.include_from_algebra(entropy_slope),
        )                                                                                       # [] Point-
        energy_span = art.context.include_from_algebra(mv.a & state) + tangent_half_length * np.array([-1, 1])
        hill_layers = (
            self.hill_layers
            + art.polylines(self.hill, [stack(entropy_path)], art.bold, art.orange, 1.0, art.SOLID)
            + art.polylines(
                self.hill, [art.tangent_lines(self.hill, current_point, tangent, energy_span)],
                art.medium, art.orange, 1.0, art.SOLID,
            )
            + art.dots(self.hill, current_point[None], art.orange)
        )
        floating_labels = [
            art.Floating(art.crossings(self.levels, 8), "S = const", art.families["entropy"], art.text_size, "lb")
        ]
        return [
            art.floated(self.energies, art.picture(self.energies, energy_layers, []), floating_labels),
            art.picture(self.hill, hill_layers, []),
        ]

    def temperatures(
        self, times: np.ndarray, temperatures: art.Scalar, equilibrium_temperature: art.Scalar
    ) -> np.ndarray:
        """Render both body temperatures over time, red for the hot body and blue for the cold body, over the dashed
        equilibrium temperature."""
        diagram = self.cooling
        colours = [art.thermal_colour(1.0), art.thermal_colour(0.0)]
        n_steps = len(times)
        curves = art.chart_points(times[:, None], temperatures)                                 # [n_steps, 2] Point-
        layers = art.dashed(
            diagram, art.chart_points(np.array(time_range), equilibrium_temperature), art.medium, art.ink, (4.0, 3.0)
        )
        for body in range(curves.shape[1]):
            layers += art.polylines(diagram, [curves[:, body]], art.bold, colours[body], 1.0, art.SOLID)
            layers += art.dots(diagram, curves[-1, body][None], colours[body])
        labels = [
            art.Label(
                art.chart_points(time_range[1], equilibrium_temperature), "common", art.ink, art.text_size, "rb", (0, 3)
            )
        ]
        return art.picture(diagram, layers, labels)


def exchange(
    entropy_of: Callable[[Extensor], Extensor],
    entropy_potential_of: Callable[[Extensor], Extensor],
    temperatures_of: Callable[[Extensor], Extensor],
    transfer: Extensor,
    capacities: np.ndarray,
    states: Iterator[tuple[Extensor, float]],
) -> Iterator[np.ndarray]:
    """Yield side-by-side frames of the two blocks and the three diagrams across the exchange."""
    first_frame = next(states)
    charts = Charts(entropy_of, entropy_potential_of, transfer, first_frame[0])
    state_path, entropy_path, times, temperature_steps = [], [], [], []
    for state, time in chain([first_frame], states):
        state_path.append(art.on_numpy(state))
        entropy_path.append(charts.entropy_points(state))
        times.append(time)
        temperature_steps.append(art.context.include_from_algebra(temperatures_of(state)))
        n_steps = len(times)
        temperatures = stack(temperature_steps)                                                 # [n_steps, 2] Scalar+
        equilibrium_temperature = (capacities * temperatures[0]).sum() / capacities.sum()       # [] Scalar+
        initial_gaps = (temperatures[0] - equilibrium_temperature).abs()                        # [2] Scalar+
        max_gap = float(initial_gaps.max().to_array())
        yield art.beside(
            [bodies(temperatures[-1].to_array(), capacities, max_gap)]
            + charts.pictures(state, state_path, entropy_path)
            + [charts.temperatures(np.array(times), temperatures, equilibrium_temperature)]
        )


def draw(
    entropy_of: Callable[[Extensor], Extensor],
    entropy_potential_of: Callable[[Extensor], Extensor],
    temperatures_of: Callable[[Extensor], Extensor],
    transfer: Extensor,
    capacities: np.ndarray,
    state: Extensor,
):
    """Render the two blocks and the three diagrams at a single state."""
    frames = exchange(entropy_of, entropy_potential_of, temperatures_of, transfer, capacities, iter([(state, 0.0)]))
    return art.figure(next(frames))
