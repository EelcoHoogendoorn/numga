"""Magnetic skyrmions: the exchange carried across a twisted bond is the Heisenberg exchange, the Dzyaloshinskii-Moriya
interaction and an anisotropy along the bond; as a lattice of skyrmions crystallizes from noise, damped, the energy
never rises and the charge comes out whole, one for each skyrmion; a high field erases every skyrmion; a current carries
the lattice without unwinding it; the written effective field is the energy's derivative; the spins stay unit throughout; and the lattice and the frames draw, alone and side
by side."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numga.backend.jax import derivative

from examples.quantum.skyrmions import core, render, scenarios


@pytest.fixture(scope="module")
def film():
    return scenarios.film()


@pytest.fixture(scope="module")
def crystal_states(film):
    topology, vertices, bonds, periods, carriers = film
    return list(scenarios.crystallizing(topology, bonds, carriers))


def total_charge(topology, spins) -> float:
    return float(core.charges(topology, spins).sites.sum().to_array())


def test_twisted_exchange_is_dzyaloshinskii_moriya(film):
    """The exchange carried across a bond by a rotor turning about it is, to round-off, the Heisenberg exchange times
    the twist's cosine, the Dzyaloshinskii-Moriya interaction times its sine, the plane through the two spins paired
    with the bond's twist, and an anisotropy along the bond times one less its cosine."""
    topology, vertices, bonds, periods, carriers = film
    spins = scenarios.noise(topology, 0, scenarios.seed)                      # Spin-[vertices]
    applied = scenarios.crystal_field * core.film                             # [] Field+
    tails, heads = topology.incidence[1, 0] * spins                           # Spin-[edges], Spin-[edges]
    twist_planes = (core.mv.antipseudoscalar([scenarios.twist_angle]) * bonds).normalized()   # Field+[edges]
    angle = scenarios.twist_angle
    heisenberg = -np.cos(angle) * (core.exchange(tails) & heads).sites.sum()
    dzyaloshinskii_moriya = np.sin(angle) * ((heads ^ tails) & twist_planes.dual()).sites.sum()
    anisotropy = -(1 - np.cos(angle)) * ((twist_planes & tails) * (twist_planes & heads)).sites.sum()
    zeeman = -(applied & spins).sites.sum()
    np.testing.assert_allclose(core.energy(topology, carriers, applied, spins).to_array(),
                               (heisenberg + dzyaloshinskii_moriya + anisotropy + zeeman).to_array(), rtol=1e-12)


def test_effective_field_is_the_energys_derivative(film):
    """The effective field written out, each neighbour's spin carried across its bond to the site plus the applied field,
    paired with the spin at its own site and summed over the sites, is JAX's derivative of the energy, negated."""
    topology, vertices, bonds, periods, carriers = film
    spins = scenarios.noise(topology, scenarios.n_smoothing, scenarios.seed)  # Spin-[vertices]
    applied = scenarios.crystal_field * core.film                             # [] Field+
    slope = derivative(lambda spins: core.energy(topology, carriers, applied, spins))(spins)   # [] Scalar+ <- Spin-[vertices]
    field = core.effective_field(topology, carriers, applied, spins)          # Field+[vertices]
    written = core.pairing(field).on_diagonal().sites.sum()                   # [] Scalar+ <- Spin-[vertices]
    np.testing.assert_allclose(np.asarray((slope + written).kernel), 0.0, atol=1e-12)


def test_lattice_of_skyrmions_crystallizes(film, crystal_states):
    """Damped and cold, the energy never rises from frame to frame; the spins stay unit; and the crystal's charge is a
    whole number, one for each skyrmion, a patch of spins pointing against the field."""
    topology, vertices, bonds, periods, carriers = film
    energies = np.array([core.energy(topology, carriers, scenarios.crystal_field * core.film, spins).to_array()
                         for spins in crystal_states])
    assert np.all(np.diff(energies) <= 1e-12 * np.abs(energies[0]))
    crystal = crystal_states[-1]
    np.testing.assert_allclose(crystal.norm().batch().to_array(), 1.0, atol=1e-10)
    pointing_down = (core.film & (topology.incidence[2, 0].mean(axis=0) * crystal)) < 0
    n_skyrmions = topology.subset(np.asarray(pointing_down), 2).n_components
    assert n_skyrmions > 1
    np.testing.assert_allclose(total_charge(topology, crystal), -n_skyrmions, atol=1e-9)


def test_high_field_erases_the_skyrmions(film, crystal_states):
    """Over the sweep, the skyrmions keep their charge while the field is low, and from the field's peak until it is
    low again none is left."""
    topology, vertices, bonds, periods, carriers = film
    crystal = crystal_states[-1]
    sweep = list(scenarios.sweeping(topology, bonds, carriers, crystal))
    np.testing.assert_allclose(total_charge(topology, sweep[scenarios.sweep_frames[1]]), total_charge(topology, crystal),
                               atol=1e-9)
    ferromagnet = sweep[scenarios.sweep_frames[2]:scenarios.sweep_frames[3]]
    np.testing.assert_allclose([total_charge(topology, spins) for spins in ferromagnet], 0.0, atol=1e-9)
    np.testing.assert_allclose(sweep[-1].norm().batch().to_array(), 1.0, atol=1e-10)


def test_current_carries_the_lattice_whole(film, crystal_states):
    """A current moves the skyrmions without unwinding any, and the spins stay unit; the lattice and its frames draw."""
    topology, vertices, bonds, periods, carriers = film
    crystal = crystal_states[-1]
    driven = list(scenarios.driving(topology, bonds, carriers, crystal))
    np.testing.assert_allclose(total_charge(topology, driven[-1]), total_charge(topology, crystal), atol=1e-9)
    np.testing.assert_allclose(driven[-1].norm().batch().to_array(), 1.0, atol=1e-10)
    frame = next(render.animate(topology, vertices, bonds, periods, [driven[-1]]))
    assert frame.shape[-1] == 3 and frame.shape[1] == render.n_pixels
    cases = (crystal, driven[-1], crystal)
    together = next(render.animate_together(topology, vertices, bonds, periods, [cases]))
    assert together.shape[1] == len(cases) * render.n_panel_pixels + (len(cases) - 1) * render.gap
    plt.close(render.draw_lattice(topology, vertices, bonds, periods))
