"""Magnetic skyrmions in a thin chiral magnet evolve on a periodic triangular lattice in three-dimensional Euclidean space.

Chiral crystals lack mirror symmetry, so neighbouring magnetic moments favour a slight relative twist of fixed handedness alongside ferromagnetic exchange. Without an external field that competition forms a helical spiral, whereas a moderate perpendicular field stabilizes a hexagonal lattice of skyrmions whose spins turn smoothly from aligned with the field at the rim to opposed at the core. Wrapping the unit sphere once around each core gives every skyrmion an integer topological charge, and a spin-polarized current drives the lattice sideways at the skyrmion Hall angle.

Each spin `Spin` is modelled as an intrinsic turning-plane patch and each magnetic field `Field` as an extrinsic turning plane, so the Zeeman energy pairs them directly as `-(applied_field & spins)`. Chiral exchange transports a neighbour's spin across each bond by a rotor `carriers` generated from `twist * bonds` for an intrinsic scalar `twist`, maps the transported spin to its exchange field via `exchange = Spin.dual()`, and pairs that field with the spin at the opposite end. Parallel transport of bond rotors around each triangle yields a curvature holonomy in the film plane, while parallel transport of shortest-arc spin rotors yields each triangle's solid-angle topological charge. Each time step turns every spin by a rotor built from its effective field, thermal kicks, current advection via simplex gradients averaged onto dual areas, and the Gilbert damping quarter turn `undamped_field.commutator(spins.dual())`.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax
import numpy as np

from numga import stack, where
from numga.algebras import VGA3D as ga
from numga.backend.jax import JaxContext
from numga.topology import Topology

jax.config.update("jax_enable_x64", True)

mv = JaxContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Spin = ga.gatype.antibivector()                                               # Antibivector-
Field = ga.gatype.bivector()                                                  # Bivector+
Rotor = ga.gatype.rotor()                                                     # Rotor+
# Perpendicular applied fields turn spins in the film plane.
film = mv.xy                                                                  # [] Field+
# Isotropic exchange maps a neighbour's spin patch to the field plane it exerts.
exchange = Spin.dual()                                                        # [] Field+ <- Spin-


def energy(topology: Topology, carriers: Rotor, applied_field: Field, spins: Spin) -> Scalar:
    """Chiral bond exchange across `carriers` (`Rotor+[E]`) and Zeeman pairing with `applied_field` (`[] Field+`) sum to the total energy (`[] Scalar+`) of `spins` (`Spin-[V]`)."""
    tail_spins, head_spins = topology.incidence[1, 0] * spins                 # Spin-[E], Spin-[E]
    exchange_fields = exchange(carriers >> tail_spins)                        # Field+[E]
    return -(exchange_fields & head_spins).sites.sum() - (applied_field & spins).sites.sum()


def holonomies(topology: Topology, rotors: Rotor) -> Rotor:
    """Parallel transport around each triangle composes the oriented edge `rotors` (`Rotor+[E]`) into a face holonomy (`Rotor+[F]`)."""
    side_rotors = topology.incidence[2, 1] * rotors                           # [3] Rotor+[F]
    edge_signs = topology.boundary[2, 1] * rotors.norm()                      # [3] Scalar+[F]
    oriented_sides = where(edge_signs > 0, side_rotors, ~side_rotors)         # [3] Rotor+[F]
    # Left-multiplying around the boundary composes each edge transport onto the arriving spin.
    return oriented_sides.cumprod(axis=0)[-1]


def effective_field(topology: Topology, carriers: Rotor, applied_field: Field, spins: Spin) -> Field:
    """Parallel transport of neighbour `spins` (`Spin-[V]`) across `carriers` (`Rotor+[E]`) combines with `applied_field` (`[] Field+`) into the site effective field (`Field+[V]`)."""
    tail_spins, head_spins = topology.incidence[1, 0] * spins                 # Spin-[E], Spin-[E]
    # Reversing the rotor sandwich transports a head spin back across its bond to the tail site.
    tail_fields = exchange(carriers << head_spins)                            # Field+[E]
    head_fields = exchange(carriers >> tail_spins)                            # Field+[E]
    return (topology.incidence[1, 0].adjugate() * stack((tail_fields, head_fields))).sum(axis=0) + applied_field


def dynamics(
    topology: Topology,
    bonds: Point,
    carriers: Rotor,
    spins: Spin,
    applied_fields: Field,
    drifts: Point,
    time_steps: np.ndarray,
    damping: float,
    temperature: float,
    n_substeps: int,
    seed: int,
) -> Iterator[Spin]:
    """Yields the lattice `spins` (`Spin-[V]`) across frames, advancing `n_substeps` damped rotor steps per frame under `applied_fields`, current `drifts`, and thermal kicks."""
    V, E, F = topology.n_elements
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    sides = topology.boundary[2, 1] * bonds                                   # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                       # Spin-[F]
    gradients = (sides | patches.inverse()) / 2                               # [3] Plane+[F]
    areas = patches.norm()                                                    # Scalar+[F]
    dual_areas = (corners.adjugate() * (areas / 3)).sum(axis=0)               # Scalar+[V]

    def substep(spins: Spin, thermal_noise: jax.Array, applied_field: Field, drift: Point, time_step: float) -> Spin:
        """One damped rotor step of duration `time_step` updates `spins` (`Spin-[V]`) under its effective field, thermal kick, and drift advection."""
        field = effective_field(topology, carriers, applied_field, spins)     # Field+[V]
        # Fluctuation-dissipation sets the thermal field spread that holds damped spins at the bath temperature.
        thermal_scale = (2 * damping * temperature / time_step) ** 0.5
        thermal_field = thermal_scale * mv(Field, thermal_noise).field()      # Field+[V]
        # Simplex gradients evaluate the directional derivative along the drift on each triangle before dual-area averaging.
        face_derivatives = ((gradients & drift) * (corners * spins)).sum(axis=0)   # Spin-[F]
        drift_derivatives = (corners.adjugate() * (areas / 3 * face_derivatives)).sum(axis=0) / dual_areas   # Spin-[V]
        undamped_field = field + thermal_field - (spins ^ drift_derivatives)  # Field+[V]
        # Gilbert damping adds a quarter turn about each spin, turning the spin toward its field.
        turning_rate = (undamped_field + damping * undamped_field.commutator(spins.dual())) / (1 + damping ** 2)   # Field+[V]
        step_rotors = (turning_rate * (-time_step / 2)).exp()                 # Rotor+[V]
        return step_rotors >> spins

    @jax.jit
    def frame_step(spins: Spin, key: jax.Array, applied_field: Field, drift: Point, time_step: float) -> Spin:
        """A compiled scan of `n_substeps` rotor steps advances `spins` (`Spin-[V]`) across one frame."""
        noise_samples = jax.random.normal(key, (n_substeps, V, 3))
        return jax.lax.scan(
            lambda spins, sample: (substep(spins, sample, applied_field, drift, time_step), None),
            spins,
            noise_samples,
        )[0]

    n_frames = len(time_steps)
    frame_keys = jax.random.split(jax.random.PRNGKey(seed), n_frames)
    for frame in range(n_frames):
        yield spins
        spins = frame_step(spins, frame_keys[frame], applied_fields[frame], drifts[frame], time_steps[frame])


def charges(topology: Topology, spins: Spin) -> Scalar:
    """Spherical solid angles enclosed by each triangle's three `spins` (`Spin-[V]`) give the topological charge per face (`Scalar+[F]`)."""
    first, second, third = topology.incidence[2, 0] * spins                   # Spin-[F], Spin-[F], Spin-[F]
    # Half the turn of a spin product transports one corner's spin along the shortest arc to the next corner's,
    # and the three transports around the triangle compose into its holonomy.
    holonomy = ((1 + first * third) * (1 + third * second) * (1 + second * first)).normalized()   # Rotor+[F]
    # The holonomy logarithm lies in the first corner spin's plane at half the enclosed solid angle.
    return -2 * (holonomy.log() & first) / (4 * np.pi)
