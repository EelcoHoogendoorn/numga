"""A Hall bar and a Corbino ring of one conductor in a fixed magnetic field, as the balance of its carriers runs from
electrons only, a doped semiconductor, through compensation, a semimetal such as bismuth, to holes only; each conductor
driven by a current and, apart, by heat. Driven by a current, the bar's Hall voltage falls through nothing at
compensation and comes back reversed, while its resistance along it peaks; driven by heat, its Nernst voltage grows
from nothing at either end to its largest at compensation. The ring's carriers spiral, and heat drives a current around
it where the Hall voltage would vanish.

Electrons and holes share one mobility and one entropy per carrier, so that compensation, the Hall voltage's zero and
the Nernst voltage's peak fall at one balance. The carriers' and the lattice's constants are of order one, in units
where the bar is one wide, the ring's outer radius one, the carriers' total density one and their conductivity with no
field one; the field's strength is the tangent of the angle it turns a carrier's drift by.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack

from examples.thermodynamics.magnetotransport import core
from examples.util.topology import Topology

mv = core.mv
States = Iterator[tuple[np.ndarray, core.Potential, core.Spatial, core.Potential, core.Spatial]]

# The carriers, electrons and holes: the charge each holds; the entropy it carries per charge it holds, its
# thermopower, an electron's negative, so that both kinds carry entropy alike; and its mobility, the velocity a unit
# push gives it. The lattice's conductivity, the entropy it conducts per unit fall of temperature.
valences = np.array([-1, 1])
entropies_per_charge = np.array([-1.0, 1.0])
mobilities = np.array([1.0, 1.0])
conductivity = 1.5
# The magnetic field's strength, the tangent of the Hall angle.
strength = 2.0
# The two drives: the current, and the fall of temperature between the contacts per unit of the span between them,
# the bar's length or the ring's width, so that a fall of temperature pushes a carrier's entropy as hard as the
# current's fall of the electrochemical potential pushes its charge.
current, temperature_fall = 1.0, 1.0
# The carriers' balance, the holes' share of them less the electrons', swept from electrons only to holes only in a
# cosine, so that it dwells at the ends; the frames and their duration in milliseconds.
n_frames = 75
frame_duration_ms = 40
balances = -np.cos(np.linspace(0, np.pi, n_frames))                          # [frames]
densities = np.stack([(1 - balances) / 2, (1 + balances) / 2], axis=-1)     # [frames, kinds]
magnetic = np.full(n_frames, strength) * mv.xy                               # [frames] Bivector+
# The bar: its length in widths, its rows of triangles across, an even count so that the bar mirrors in its middle
# line, and its columns of places along, so that the triangles are nearly equilateral and a column falls at every
# tenth of a width; and its pictures' pixels to a width.
length, n_rows, n_columns = 3.0, 12, 30
bar_resolution = 80
# The ring: its inner radius in its outer, its rings of triangles from rim to rim and its places around each circle of
# places; and its pictures' pixels to an outer radius.
inner_radius, n_rings, n_sectors = 0.4, 6, 48
ring_resolution = 110


# --- plumbing -------------------------------------------------------------------------
def lattice(length: float, n_rows: int, n_columns: int) -> np.ndarray:
    """Places of a bar `length` long and one wide `[vertices, 2]`, along and across: `n_rows` rows of triangles, a row
    of places every `1 / n_rows` from half a width below the bar's middle line to half a width above, `n_columns`
    spacings along, every other row shifted by half a spacing with the ends kept, so that the triangulation prefers no
    diagonal."""
    straight = np.linspace(0, length, n_columns + 1)
    shifted = np.concatenate([[0], (straight[:-1] + straight[1:]) / 2, [length]])
    rows = [np.stack([places, np.full(len(places), -0.5 + row / n_rows)], axis=-1)
            for row, places in enumerate([shifted if row % 2 else straight for row in range(n_rows + 1)])]
    return np.concatenate(rows)


def at(places: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """The vertices at the wanted places `[..., 2]` among the places `[vertices, 2]`, `[...]`."""
    return np.abs(places[None, :, :] - wanted.reshape(-1, 1, 2)).sum(axis=-1).argmin(axis=-1).reshape(wanted.shape[:-1])


# --- scenes ---------------------------------------------------------------------------
def bar() -> tuple[Topology, core.Point, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The Hall bar: its triangles, its vertices' places `Point-[vertices]`, the vertices of its contact at its left
    end `[across]` and at its right end `[across]`, and its probes in pairs, a voltmeter's plus and minus terminals:
    along its lower side a width apart about its middle, plus upstream, and across its middle, plus below; the plus
    probes `[pairs]` and the minus probes `[pairs]`."""
    places = lattice(length, n_rows, n_columns)                              # [vertices, 2]
    left, right = (np.flatnonzero(places[:, 0] == end) for end in (0, length))
    middle = length / 2
    plus = at(places, np.array([[middle - 0.5, -0.5], [middle, -0.5]]))
    minus = at(places, np.array([[middle + 0.5, -0.5], [middle, 0.5]]))
    # The places as the vertices of a mesh, each of unit weight along the entropy.
    vertices = (core.entropy + mv(core.Upright, places).dual()).field()     # Point-[vertices]
    return Topology.triangulate(vertices), vertices, left, right, plus, minus


def ring() -> tuple[Topology, core.Point, np.ndarray, np.ndarray]:
    """The Corbino ring, of unit outer radius about the origin: its triangles, its vertices' places `Point-[vertices]`,
    and the vertices of its contact along its inner rim `[around]` and along its outer rim `[around]`. Its places lie
    on `n_rings + 1` circles from the inner rim to the outer, `n_sectors` on each, alternate circles turned by half a
    sector, so that the triangulation is uniform; it fills the hole too, with triangles whose every corner is on the
    inner rim, and those are left out."""
    radii = np.linspace(inner_radius, 1.0, n_rings + 1)                       # [circles]
    staggers = (np.arange(n_rings + 1) % 2)[:, None] / 2                     # [circles, 1]
    angles = 2 * np.pi * (np.arange(n_sectors) + staggers) / n_sectors        # [circles, sectors]
    turns = (mv.xy * (-angles / 2)).exp()                                     # [circles, sectors] Rotor+
    # Each place the direction along x turned to its angle and stretched to its circle's radius, from the centre.
    vertices = (core.entropy + radii[:, None] * (turns >> core.axis)).reshape(-1).field()   # Point-[vertices]
    inner, outer = np.arange(n_sectors), np.arange(n_sectors) + n_rings * n_sectors
    filled = Topology.triangulate(vertices)
    in_hole = np.isin(filled.elements[2], inner).all(axis=-1)                # [faces]
    return filled.subset(~in_hole), vertices, inner, outer


def states(topology: Topology, vertices: core.Point, high: np.ndarray, low: np.ndarray, span: float,
           magnetic: core.Magnetic, densities: np.ndarray) -> States:
    """The conductor's steady states, its vertices at these places `Point-[vertices]` and its contacts `high` `[high]`
    and `low` `[low]` this span apart, at each frame's magnetic field `[frames] Bivector+` and densities of the carriers
    `[frames, kinds]`, as `core.sweep` gives them, driven by the current and by the fall of temperature over the
    span."""
    areas, gradients = core.triangles(topology, vertices)
    return core.sweep(topology, areas, gradients, high, low, current, temperature_fall * span, magnetic, valences,
                      entropies_per_charge, mobilities, densities, conductivity)


def bar_readings(topology: Topology, plus: np.ndarray, minus: np.ndarray, states: list
                 ) -> tuple[core.Scalar, core.Scalar, core.Scalar, core.Scalar, core.Scalar]:
    """At each of the bar's states `[frames] Scalar+` each: driven by the current, its resistance along it, the voltage
    per current between its probes on its lower side a width apart, and its Hall resistance, the voltage per current
    from its lower side to its upper at its middle; driven by heat, its Seebeck and its Nernst signal, the electric
    field along it and across it per fall of temperature along it, each the voltage between the same probes over the
    difference of temperature between the probes along it, with the sign of a field down a fall, and its Righi-Leduc
    signal, the fall of temperature across it per fall along it."""
    electric = stack([potentials for _, potentials, _, _, _ in states])      # [frames] Potential+[vertices]
    thermal = stack([potentials for _, _, _, potentials, _ in states])       # [frames] Potential+[vertices]
    # A voltmeter across each pair of the bar's probes: the potential at its plus probe less that at its minus probe.
    voltmeters = topology.reading(plus) - topology.reading(minus)            # Scalar+[pairs <- vertices]
    electric_voltages = (voltmeters * (electric & core.time)).batch() / current   # [frames, pairs] Scalar+
    thermal_voltages = (voltmeters * (thermal & core.time)).batch()         # [frames, pairs] Scalar+
    # Driven by heat, the differences of temperature between the probes, along the bar the upstream one the warmer.
    temperatures = (voltmeters * (thermal & core.entropy)).batch()          # [frames, pairs] Scalar+
    warmer = temperatures[:, 0]                                               # [frames] Scalar+
    return (electric_voltages[:, 0], electric_voltages[:, 1], -thermal_voltages[:, 0] / warmer,
            -thermal_voltages[:, 1] / warmer, temperatures[:, 1] / warmer)


def ring_resistances(topology: Topology, high: np.ndarray, low: np.ndarray, states: list) -> core.Scalar:
    """The ring's resistance at each of its states `[frames] Scalar+`: driven by the current, the voltage between its
    contacts per current."""
    electric = stack([potentials for _, potentials, _, _, _ in states])      # [frames] Potential+[vertices]
    voltmeter = topology.reading(high[:1]) - topology.reading(low[:1])       # Scalar+[1 <- vertices]
    return (voltmeter * (electric & core.time)).batch()[:, 0] / current


def end_fluxes() -> core.Flux:
    """The flux a unit electric field along the bar drives `[ends] Flux-`, with electrons only and with holes only."""
    ends = np.eye(2)                                                          # [ends, kinds]
    return core.response(strength * mv.xy, valences, entropies_per_charge, mobilities, ends, conductivity)(mv.xt)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.magnetotransport import render

    save_figure(render.draw_turns(mv.xt, end_fluxes()), "magnetotransport_turns")
    bar_topology, bar_vertices, left, right, plus, minus = bar()
    ring_topology, ring_vertices, inner, outer = ring()
    save_figure(render.draw_conductors(bar_topology, bar_vertices, (left, right), ring_topology, ring_vertices,
                                       (inner, outer), bar_resolution, ring_resolution), "magnetotransport_conductors")
    bar_states = list(states(bar_topology, bar_vertices, left, right, length, magnetic, densities))
    save_animation(render.animate_bars(bar_topology, bar_vertices, (left, right), magnetic[0], bar_resolution,
                                       bar_states), "magnetotransport", frame_duration_ms)
    ring_states = list(states(ring_topology, ring_vertices, inner, outer, 1 - inner_radius, magnetic, densities))
    save_animation(render.animate_rings(ring_topology, ring_vertices, (inner, outer), magnetic[0], ring_resolution,
                                        ring_states), "magnetotransport_corbino", frame_duration_ms)
    resistance, hall, seebeck, nernst, righi_leduc = bar_readings(bar_topology, plus, minus, bar_states)
    save_figure(render.draw_readings(balances, resistance, hall, seebeck, nernst, righi_leduc),
                "magnetotransport_readings")

    # checks
    # With one kind of carrier the bar's resistance along it is a square's with no field and its Hall resistance the
    # strength over the carriers' charge, less the contacts' share; at compensation the Hall resistance vanishes where
    # the Nernst signal is largest. The Righi-Leduc signal needs both kinds, unbalanced: it is nothing at either end and
    # at compensation, and opposite on either side. The ring's resistance is one plus the strength squared times a
    # ring's with no field, the logarithm of its radii's ratio over two pi, at every balance.
    middle = n_frames // 2
    np.testing.assert_allclose(resistance.to_array()[[0, -1]], 1.0, rtol=1e-2)
    np.testing.assert_allclose(hall.to_array()[[0, -1]], strength / valences, rtol=2e-2)
    assert abs(hall.to_array()[middle]) < 1e-12 * strength
    assert np.abs(nernst.to_array()).argmax() == middle
    np.testing.assert_allclose(righi_leduc.to_array()[[0, middle, -1]], 0.0, atol=1e-10)
    assert righi_leduc.to_array()[middle // 2] * righi_leduc.to_array()[-1 - middle // 2] < 0
    np.testing.assert_allclose(ring_resistances(ring_topology, inner, outer, ring_states).to_array(),
                               (1 + strength ** 2) * np.log(1 / inner_radius) / (2 * np.pi), rtol=1e-2)


if __name__ == "__main__":
    main()
