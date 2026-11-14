"""A Hall bar and a Corbino ring carry electrons and holes in a perpendicular magnetic field as the carrier balance sweeps from electrons only through compensation to holes only.

Under an electric current, the Hall bar's transverse Hall voltage passes through zero at compensation while its longitudinal magnetoresistance peaks; under open-circuit heating, its transverse Nernst voltage peaks where the Hall voltage vanishes. Around the Corbino ring, carriers spiral at every balance and radial heating at compensation drives a circulating azimuthal current that reverses with the magnetic field. In a magnetized conductor at zero external magnetic field, sweeping magnetization from spin-up to spin-down electrons under skew scattering traces the anomalous Hall voltage through an unmagnetized spin Hall peak in longitudinal resistance.

Electrons and holes share unit mobility and unit entropy per carrier in dimensionless units where the bar width, the outer ring radius, the total carrier density, and the zero-field carrier conductivity equal one, so the magnetic field strength equals the tangent of the carrier Hall angle.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from numga.topology import Topology
from examples.thermodynamics.magnetotransport import core
from examples.util import shapes

mv = core.mv
States = Iterator[tuple[np.ndarray, core.Potential, core.Spatial, core.Potential, core.Spatial]]

# Electrons and holes carry opposite charges, opposite entropy per unit charge (equal positive entropy per carrier), and unit mobility, with zero spin skew.
valences = np.array([-1, 1])                                                  # [2]
entropies_per_charge = np.array([-1.0, 1.0])                                  # [2]
mobilities = np.array([1.0, 1.0])                                             # [2]
skews = np.zeros(2)                                                           # [2]
carriers = (valences, entropies_per_charge, mobilities, skews)
conductivity = 1.5
strength = 2.0
current, temperature_fall = 1.0, 1.0

# Cosine sweep from pure electrons (-1) through compensation (0) to pure holes (+1) at fixed magnetic field.
n_frames = 75
frame_duration_ms = 40
balances = -np.cos(np.linspace(0, np.pi, n_frames))                           # [n_frames]
densities = np.stack([(1 - balances) / 2, (1 + balances) / 2], axis=-1)      # [n_frames, 2]
magnetic = np.full(n_frames, strength) * mv.xy                                # [n_frames] Magnetic+

# Cosine sweep of the magnetic field from +strength through zero to -strength at carrier compensation.
reversal = strength * np.cos(np.linspace(0, np.pi, n_frames)) * mv.xy         # [n_frames] Magnetic+
compensation = np.full_like(densities, 0.5)                                   # [n_frames, 2]
n_gallery_colours = 64

# Spin-up and spin-down electrons at zero external field with opposite spin skew-scattering tangents.
spin_valences = np.array([-1, -1])                                            # [2]
spin_entropies_per_charge = np.array([-1.0, -1.0])                            # [2]
spin_mobilities = np.array([1.0, 1.0])                                        # [2]
spin_skews = np.array([1.0, -1.0])                                            # [2]
spin_carriers = (spin_valences, spin_entropies_per_charge, spin_mobilities, spin_skews)
magnetizations = np.cos(np.linspace(0, np.pi, n_frames))                      # [n_frames]
spin_densities = np.stack([(1 + magnetizations) / 2, (1 - magnetizations) / 2], axis=-1)   # [n_frames, 2]
no_field = np.zeros(n_frames) * mv.xy                                         # [n_frames] Magnetic+

# Conductor dimensions, triangulation counts, and matched pixel resolutions.
length, n_rows, n_columns = 3.0, 12, 30
inner_radius, n_rings, n_sectors = 0.4, 6, 48
ring_resolution = 110
bar_resolution = 2 * ring_resolution / length


# --- plumbing -------------------------------------------------------------------------
def bar_lattice(length: float, n_rows: int, n_columns: int) -> np.ndarray:
    """Staggering alternate horizontal rows of `n_columns` spacings across `n_rows` strips places equilateral bar coordinates (`[V, 2]`) with mirror symmetry."""
    even_columns = np.linspace(0, length, n_columns + 1)
    odd_columns = np.concatenate([[0], (even_columns[:-1] + even_columns[1:]) / 2, [length]])
    rows = [np.stack([row_columns, np.full(len(row_columns), -0.5 + row / n_rows)], axis=-1)
            for row, row_columns in enumerate([odd_columns if row % 2 else even_columns for row in range(n_rows + 1)])]
    return np.concatenate(rows)


def nearest_vertices(places: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Manhattan distance selects the vertex index (`[...]`) in `places` (`[V, 2]`) closest to each target coordinate (`[..., 2]`)."""
    return np.abs(places[None, :, :] - targets.reshape(-1, 1, 2)).sum(axis=-1).argmin(axis=-1).reshape(targets.shape[:-1])


# --- scenes ---------------------------------------------------------------------------
def bar(n_rows: int, n_columns: int
        ) -> tuple[Topology, core.Point, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Delaunay triangulation of the staggered lattice builds the Hall bar topology, unit-weight `vertices` (`Point-[V]`), end contacts, and probe pairs (`[2]`)."""
    places = bar_lattice(length, n_rows, n_columns)                           # [V, 2]
    left_contacts, right_contacts = (np.flatnonzero(places[:, 0] == end) for end in (0, length))
    middle = length / 2
    plus_probes = nearest_vertices(places, np.array([[middle - 0.5, -0.5], [middle, -0.5]]))    # [2]
    minus_probes = nearest_vertices(places, np.array([[middle + 0.5, -0.5], [middle, 0.5]]))    # [2]
    # Dualizing spatial coordinates plus a unit along w embeds vertices as unit-weight points.
    vertices = (mv(core.Upright, places) + mv.w).dual().field()               # Point-[V]
    topology, _ = shapes.delaunay(places)
    return topology, vertices, left_contacts, right_contacts, plus_probes, minus_probes


def ring(n_rings: int, n_sectors: int) -> tuple[Topology, core.Point, np.ndarray, np.ndarray]:
    """Concentric staggered circles around the origin form the Corbino ring topology, unit-weight `vertices` (`Point-[V]`), and inner and outer rim contacts (`[n_sectors]`)."""
    radii = np.linspace(inner_radius, 1.0, n_rings + 1)                       # [n_rings + 1]
    staggers = (np.arange(n_rings + 1) % 2)[:, None] / 2                      # [n_rings + 1, 1]
    angles = 2 * np.pi * (np.arange(n_sectors) + staggers) / n_sectors        # [n_rings + 1, n_sectors]
    turns = (mv.xy * (-angles / 2)).exp()                                     # [n_rings + 1, n_sectors] Rotor+
    vertices = (core.origin + radii[:, None] * (turns >> core.x_direction)).reshape(-1).field()   # Point-[V]
    inner_contacts, outer_contacts = np.arange(n_sectors), np.arange(n_sectors) + n_rings * n_sectors
    coordinates = shapes.read(vertices.batch(), stack((mv.x, mv.y)))          # [V, 2]
    disc_topology, _ = shapes.delaunay(coordinates)
    hole_faces = np.isin(disc_topology.elements[2], inner_contacts).all(axis=-1)
    return disc_topology.subset(~hole_faces, 2), vertices, inner_contacts, outer_contacts


def conductor_states(topology: Topology, vertices: core.Point, high_contacts: np.ndarray, low_contacts: np.ndarray,
                     span: float, magnetic: core.Magnetic,
                     carriers: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                     densities: np.ndarray) -> States:
    """Yields steady transport states across `n_frames` parameter values for a conductor whose contacts span `span`."""
    areas, gradients = core.triangles(topology, vertices)
    return core.steady_states(topology, areas, gradients, high_contacts, low_contacts, current,
                              temperature_fall * span, magnetic, *carriers, densities, conductivity)


def bar_readings(topology: Topology, plus_probes: np.ndarray, minus_probes: np.ndarray, states: list
                 ) -> tuple[core.Scalar, core.Scalar, core.Scalar, core.Scalar, core.Scalar]:
    """Longitudinal and transverse probe pairs read longitudinal resistance, Hall resistance, Seebeck thermopower, Nernst signal, and Righi-Leduc temperature ratio (`[n_frames] Scalar+`)."""
    electric_potentials = stack([potentials for _, potentials, _, _, _ in states])   # [n_frames] Potential+[V]
    thermal_potentials = stack([potentials for _, _, _, potentials, _ in states])    # [n_frames] Potential+[V]
    voltmeters = topology.reading(plus_probes, 0) - topology.reading(minus_probes, 0)   # Scalar+[2 <- V]
    electric_voltages = (voltmeters * (electric_potentials & core.time)).batch() / current   # [n_frames, 2] Scalar+
    thermal_voltages = (voltmeters * (thermal_potentials & core.time)).batch()               # [n_frames, 2] Scalar+
    probe_temperatures = (voltmeters * (thermal_potentials & core.entropy)).batch()          # [n_frames, 2] Scalar+
    longitudinal_drop = probe_temperatures[:, 0]                              # [n_frames] Scalar+
    return (electric_voltages[:, 0], electric_voltages[:, 1], -thermal_voltages[:, 0] / longitudinal_drop,
            -thermal_voltages[:, 1] / longitudinal_drop, probe_temperatures[:, 1] / longitudinal_drop)


def ring_resistances(topology: Topology, high_contacts: np.ndarray, low_contacts: np.ndarray,
                     states: list) -> core.Scalar:
    """Rim voltage divided by drive current gives the Corbino ring's two-terminal resistance (`[n_frames] Scalar+`)."""
    electric_potentials = stack([potentials for _, potentials, _, _, _ in states])   # [n_frames] Potential+[V]
    voltmeter = topology.reading(high_contacts[:1], 0) - topology.reading(low_contacts[:1], 0)   # Scalar+[1 <- V]
    return (voltmeter * (electric_potentials & core.time)).batch()[:, 0] / current


def end_fluxes() -> core.Flux:
    """Evaluating the single-species responses on a unit longitudinal electric drive gives the pure-electron and pure-hole fluxes (`[2] Flux-`)."""
    pure_species = np.eye(2)                                                  # [2, 2]
    return core.response(strength * mv.xy, *carriers, pure_species, conductivity)(mv.xt)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.magnetotransport import render

    save_figure(render.turns_figure(mv.xt, end_fluxes()), "magnetotransport_turns")
    bar_topology, bar_vertices, left_contacts, right_contacts, plus_probes, minus_probes = bar(n_rows, n_columns)
    ring_topology, ring_vertices, inner_contacts, outer_contacts = ring(n_rings, n_sectors)
    save_figure(render.conductors_figure(bar_topology, bar_vertices, (left_contacts, right_contacts), ring_topology,
                                         ring_vertices, (inner_contacts, outer_contacts), bar_resolution,
                                         ring_resolution), "magnetotransport_conductors")

    conductors = (bar_topology, bar_vertices, (left_contacts, right_contacts), ring_topology, ring_vertices,
                  (inner_contacts, outer_contacts), bar_resolution, ring_resolution)
    frames = list(zip(
        magnetic,
        conductor_states(bar_topology, bar_vertices, left_contacts, right_contacts, length, magnetic, carriers,
                         densities),
        conductor_states(ring_topology, ring_vertices, inner_contacts, outer_contacts, 1 - inner_radius, magnetic,
                         carriers, densities)))
    save_animation(render.current_driven_animation(*conductors, frames[0], frames), "magnetotransport_charge",
                   frame_duration_ms)
    save_animation(render.heat_driven_animation(*conductors, frames[0], frames), "magnetotransport_heat",
                   frame_duration_ms)

    reversal_frames = zip(
        reversal,
        conductor_states(bar_topology, bar_vertices, left_contacts, right_contacts, length, reversal, carriers,
                         compensation),
        conductor_states(ring_topology, ring_vertices, inner_contacts, outer_contacts, 1 - inner_radius, reversal,
                         carriers, compensation))
    save_animation(render.heat_driven_animation(*conductors, frames[0], reversal_frames), "magnetotransport_reversed",
                   frame_duration_ms)

    bar_states = [bar_state for _, bar_state, _ in frames]
    ring_states = [ring_state for _, _, ring_state in frames]
    resistance, hall, seebeck, nernst, righi_leduc = bar_readings(bar_topology, plus_probes, minus_probes, bar_states)
    save_figure(render.readings_figure(balances, resistance, hall, seebeck, nernst, righi_leduc, render.balance_names),
                "magnetotransport_readings")

    spin_frames = list(zip(
        no_field,
        conductor_states(bar_topology, bar_vertices, left_contacts, right_contacts, length, no_field, spin_carriers,
                         spin_densities),
        conductor_states(ring_topology, ring_vertices, inner_contacts, outer_contacts, 1 - inner_radius, no_field,
                         spin_carriers, spin_densities)))
    save_animation(render.current_driven_animation(*conductors, frames[0], spin_frames), "magnetotransport_spin",
                   frame_duration_ms)
    spin_resistance, spin_hall, _, spin_nernst, _ = bar_readings(bar_topology, plus_probes, minus_probes,
                                                                 [bar_state for _, bar_state, _ in spin_frames])
    save_figure(render.spin_readings_figure(magnetizations, spin_resistance, spin_hall),
                "magnetotransport_spin_readings")

    # checks
    middle = n_frames // 2
    assert ((resistance[[0, -1]] - 1.0).abs() < 1e-2).all()
    assert ((hall[[0, -1]] - strength / valences).abs() < 2e-2 * strength).all()
    assert hall[middle].abs() < 1e-12 * strength
    assert nernst.abs().argmax() == middle
    assert (righi_leduc[[0, middle, -1]].abs() < 1e-10).all()
    assert righi_leduc[middle // 2] * righi_leduc[-1 - middle // 2] < 0
    corbino_resistance = (1 + strength ** 2) * np.log(1 / inner_radius) / (2 * np.pi)
    assert ((ring_resistances(ring_topology, inner_contacts, outer_contacts, ring_states)
             - corbino_resistance).abs() < 1e-2 * corbino_resistance).all()

    assert ((spin_hall[[0, -1]] - spin_skews).abs() < 2e-2).all()
    assert spin_hall[middle].abs() < 1e-12
    assert spin_resistance.argmax() == middle
    assert spin_nernst.abs().max() < 1e-10

    spin_responses = core.response(no_field, *spin_carriers, spin_densities, conductivity)   # [n_frames] Flux- <- Drive+
    np.testing.assert_allclose(spin_responses[::-1].kernel, spin_responses.adjugate().kernel,
                               atol=1e-12 * np.abs(spin_responses.kernel).max())


def pinned() -> None:
    """Writes the gallery animation of the heat-driven Hall bar and Corbino ring across the electron-to-hole balance sweep."""
    from examples.util.animation import pin_animation
    from examples.thermodynamics.magnetotransport import render

    bar_topology, bar_vertices, left_contacts, right_contacts, _, _ = bar(n_rows, n_columns)
    ring_topology, ring_vertices, inner_contacts, outer_contacts = ring(n_rings, n_sectors)
    frames = list(zip(
        magnetic,
        conductor_states(bar_topology, bar_vertices, left_contacts, right_contacts, length, magnetic, carriers,
                         densities),
        conductor_states(ring_topology, ring_vertices, inner_contacts, outer_contacts, 1 - inner_radius, magnetic,
                         carriers, densities)))
    pin_animation(render.heat_driven_animation(bar_topology, bar_vertices, (left_contacts, right_contacts),
                                               ring_topology, ring_vertices, (inner_contacts, outer_contacts),
                                               bar_resolution, ring_resolution, frames[0], frames),
                  "magnetotransport", frame_duration_ms, colors=n_gallery_colours)


if __name__ == "__main__":
    main()
