"""Render a damped spring-mass oscillator beside its phase-space, entropy-growth and energy-exchange diagrams.

A mass on a floor connected to a wall by a coiled spring and a viscous damper sits beside three diagrams. At constant
total energy, the reservoir holds whatever energy is not in the spring or the mass, so reservoir entropy is a function
of position and momentum alone whose teal contours on the phase portrait coincide with ellipses of constant mechanical
energy. Reversible motion follows those ellipses at constant entropy, while viscous damping crosses them inward toward
higher entropy. The middle chart plots entropy over time, rising steepest at peak speed and flattening at turning
points. The right chart plots entropy against remaining mechanical energy along the constant-energy curve up to its peak
at rest. The damper is coloured from icy blue to red heat by blending the remaining thermal rise with its logarithm so
that late warming remains visible.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import chain

import numpy as np

from numga import Extensor, stack
from examples.thermodynamics import rendering as art
from examples.thermodynamics.generic import core
from examples.util.wgsl import simplices, solids

mv = core.mv
# Pixel dimensions and axis ranges for the phase-portrait, entropy-time and entropy-energy diagrams.
n_rows, n_columns = 300, 310
position_range, momentum_range = (-2.6, 2.6), (-4.6, 4.6)
time_range, entropy_range = (0.0, 12.5), (0.0, 3.6)
mechanical_range = (0.0, 8.6)
# Contour levels of constant entropy and sampling grid across the position-momentum plane.
entropy_levels = np.array([0.0, 1.0, 1.8, 2.3, 2.7, 3.0, 3.15])
n_grid = 121
grid_positions, grid_momenta = np.linspace(-2.6, 2.6, n_grid), np.linspace(-4.6, 4.6, n_grid)
n_coil_points, n_curve_samples = 160, 200
# Pixel width of the 3D oscillator view and logarithmic span of the damper colour ramp.
n_apparatus_pixels = 230
colour_decades = 2.0


def oscillator(position: solids.Scalar, warmth: float) -> np.ndarray:
    """Render the sliding mass, spring coil and temperature-coloured damper cylinder on a floor anchored to a wall."""
    solid_mv = solids.mv
    mass_shift = solids.translator(solid_mv.direction([1, 0, 0]) * position)                # [] Motor+
    body = mass_shift >> art.box(solid_mv.point([0, 0, 0.65]), np.array([0.3, 0.4, 0.38]))
    floor = art.box(solid_mv.point([0, 0, 0.18]), np.array([2.9, 0.62, 0.05]))
    wall = art.box(solid_mv.point([-2.75, 0, 0.78]), np.array([0.08, 0.58, 0.55]))
    damper = art.box(solid_mv.point([-2.0, 0, 0.4]), np.array([0.5, 0.17, 0.15]))
    rod = solids.rods(solid_mv.point([-1.5, 0, 0.4]), mass_shift >> solid_mv.point([-0.3, 0, 0.4]), 0.045)[None]
    # Sample helical points along the spring axis between the wall anchor and the moving mass hook.
    coil_angles = np.linspace(0, 12 * np.pi, n_coil_points)
    wall_anchor, mass_hook = solid_mv.point([-2.67, 0, 0.98]), mass_shift >> solid_mv.point([-0.3, 0, 0.98])  # [] Point-, Point-
    centres = wall_anchor + (mass_hook - wall_anchor) * np.linspace(0, 1, n_coil_points)    # [n_coil_points] Point-
    coil = centres + ((solid_mv.yz * (coil_angles / 2)).exp() >> solid_mv.direction([0, 0, 0.13]))  # [n_coil_points] Point-
    corners = solid_mv.point([[x, y, z] for x in (-2.95, 2.5) for y in (-0.65, 0.65) for z in (0.1, 1.4)])
    view = solids.view([corners], 0.0, np.radians(18), np.radians(-12), 0.04, n_apparatus_pixels)
    spring = simplices.tubes(coil, simplices.Topology.chains([n_coil_points]), art.ink, 1.6, view, solids.gloss)
    return solids.draw(
        [(floor, art.grey), (wall, art.grey), (body, art.blue), (rod, art.grey), (damper, art.thermal_colour(warmth))],
        view,
        art.white,
        inks=[],
        translucent=[],
        meshes=[(spring, 1.0)],
    )


class Charts:
    """Hold the fixed entropy contours and constant-energy curve and draw the oscillator trajectory over them."""

    def __init__(self, energies_of: Callable, surface_entropy_of: Callable) -> None:
        self.phase = art.Diagram(
            position_range, momentum_range, (0.0, 0.0), n_rows, n_columns, ("q", "p"), (mv.q, mv.p), (0 * mv.q, 1.0)
        )
        self.rising = art.Diagram(
            time_range, entropy_range, (0.0, 0.0), n_rows, n_columns, ("t", "S"), (art.mv.x, art.mv.y), (art.mv.w, 0.0)
        )
        self.hill = art.Diagram(
            mechanical_range, entropy_range, (0.0, 0.0), n_rows, n_columns, ("E_mech", "S"),
            (art.mv.x, art.mv.y), (art.mv.w, 0.0),
        )
        grid = mv.q.dual() * grid_positions[None, :] + mv.p.dual() * grid_momenta[:, None]  # [n_grid, n_grid] State-
        with np.errstate(divide="ignore", invalid="ignore"):
            grid_entropy = np.asarray(surface_entropy_of(grid).to_array())
        self.entropy = art.Levels(grid, grid_entropy, entropy_levels)
        self.levels = art.level_family(self.phase, self.entropy, "entropy", art.thin)
        # Place each contour value label at the upper-right point maximizing the normalized diagonal reading.
        diagonal = mv.q / np.ptp(position_range) + mv.p / np.ptp(momentum_range)            # [] Potential+
        self.level_labels = []
        for level, corner in art.furthest(self.entropy, diagonal):
            if (mv.p & corner) < momentum_range[1] - 0.3:
                self.level_labels.append(
                    art.Label(corner, f"{level:g}", art.families["entropy"], art.tick_size, "lb", (2, 2))
                )
        # Sample the constant-energy curve of reservoir entropy against mechanical energy across displacements at rest.
        rest_states = mv.q.dual() * np.linspace(0, 2.0, n_curve_samples)                    # [n_curve_samples] State-
        kinetic, spring, _ = energies_of(rest_states)
        self.curve = art.chart_points(
            art.context.include_from_algebra(kinetic + spring),
            art.context.include_from_algebra(surface_entropy_of(rest_states)),
        )                                                                                   # [n_curve_samples] Point-

    def pictures(
        self, state_path: list[Extensor], times: np.ndarray, entropies: art.Scalar, mechanical: art.Scalar
    ) -> list[np.ndarray]:
        """Render the phase spiral, entropy-time step curve, and entropy-energy trajectory up to the current frame."""
        current_state = state_path[-1]                                                      # [] State-
        phase = (
            self.levels
            + art.polylines(self.phase, [stack(state_path)], art.bold, art.orange, 1.0, art.SOLID)
            + art.dots(self.phase, current_state[None], art.orange)
        )
        n_steps = len(times)
        entropy_time_curve = art.chart_points(times, entropies)                             # [n_steps] Point-
        entropy_energy_curve = art.chart_points(mechanical, entropies)                      # [n_steps] Point-
        rising = (
            art.polylines(self.rising, [entropy_time_curve], art.bold, art.orange, 1.0, art.SOLID)
            + art.dots(self.rising, entropy_time_curve[-1:], art.orange)
        )
        hill = (
            art.polylines(self.hill, [self.curve], art.medium, art.families["entropy"], 1.0, art.SOLID)
            + art.polylines(self.hill, [entropy_energy_curve], art.bold, art.orange, 1.0, art.SOLID)
            + art.dots(self.hill, entropy_energy_curve[-1:], art.orange)
        )
        # Anchor the constant-energy label at the right plot border level with the curve's right endpoint.
        right_edge = art.mv.x - art.mv.w * mechanical_range[1]                              # [] Plane+
        label_anchor = (self.curve[-1] & art.mv.x.dual()) ^ right_edge                      # [] Point-
        hill_labels = [art.Label(label_anchor, "E = const", art.families["entropy"], art.text_size, "rb", (0, 4))]
        floating_labels = [
            art.Floating(art.crossings(self.entropy, 6), "S = const", art.families["entropy"], art.text_size, "lt")
        ]
        return [
            art.floated(self.phase, art.picture(self.phase, phase, self.level_labels), floating_labels),
            art.picture(self.rising, rising, []),
            art.picture(self.hill, hill, hill_labels),
        ]


def damping(
    energies_of: Callable, surface_entropy_of: Callable, frame_time: float, states: Iterable[Extensor]
) -> Iterator[np.ndarray]:
    """Yield side-by-side frames of the damped oscillator and the three diagrams across `states`."""
    states = iter(states)
    first_state = next(states)
    charts = Charts(energies_of, surface_entropy_of)
    state_path, entropies, mechanical = [], [], []
    initial_energies = stack(energies_of(first_state))                                      # [3] Scalar+
    for state in chain([first_state], states):
        kinetic, spring, internal = energies_of(state)
        state_path.append(art.on_numpy(state))
        entropies.append(art.context.include_from_algebra(mv.s & state))
        mechanical.append(art.context.include_from_algebra(kinetic + spring))
        # Blend linear and logarithmic fractions of remaining mechanical energy so late warming stays visible.
        remaining_fraction = float(
            ((initial_energies.sum() - internal).clip(1e-300, np.inf) / (initial_energies.sum() - initial_energies[2])).to_array()
        )
        warmth = 1.0 - 0.5 * remaining_fraction - 0.5 * np.clip(np.log10(remaining_fraction) / colour_decades + 1, 0.0, 1.0)
        times = np.arange(len(state_path)) * frame_time
        yield art.beside(
            [oscillator(solids.context.include_from_algebra(mv.q & state), warmth)]
            + charts.pictures(state_path, times, stack(entropies), stack(mechanical))
        )


def draw(energies_of: Callable, surface_entropy_of: Callable, state: Extensor):
    """Render the oscillator and the three diagrams at a single state."""
    return art.figure(next(damping(energies_of, surface_entropy_of, 0.0, iter([state]))))
