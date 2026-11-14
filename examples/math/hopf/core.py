"""Hopf fibration of unit three-dimensional spinors grouped by their spin on the unit sphere.

Each unit spinor in `Even` turns the north `pole` of the unit sphere into its spin in `Antivector`, so sandwiching `pole` with `Even` forms the quadratic Hopf map `hopf = Even >> pole`. Taking the inner product `spin.dot(hopf)` produces a symmetric scalar form whose top eigenspace intersects the unit three-sphere in the circular fibre over `spin`, swept out by right-multiplying a fibre spinor by equatorial rotors. Stereographic projection maps each spinor's bivector grade to an intrinsic spatial point through `handedness`, sending latitude circles of spins to nested tori of once-linked circles, while minimal-angle rotor transport around a closed spin loop shifts a spinor along its fibre by half the enclosed solid angle.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Antivector = ga.gatype.antivector()                # Antivector-
HandedScalar = ga.gatype.antipseudoscalar()        # Scalar-
Even = ga.gatype.even()                            # Even+
Rotor = ga.gatype.rotor()                          # Rotor+

# North pole of the unit spin sphere and quadratic Hopf map sending (psi, psi) to psi >> pole.
pole = mv.z.dual()                                 # [] Antivector-
hopf = Even >> pole                                # [] Antivector- <- (Even+, Even+)
# Intrinsic unit scalar fixing the right-hand rule from bivector planes to spatial points.
handedness = mv(HandedScalar, [1])                 # [] Scalar-


def fibre_start(spin: Antivector) -> Even:
    """The top eigenvector of `spin.dot(hopf)` selects a unit spinor (`[...] Even+`) on the fibre over each `spin`."""
    _, spinors = spin.dot(hopf).eigh()                                         # [..., 4] Even+
    return spinors[..., -1]                                                    # [...] Even+


def sphere_spins(polar: np.ndarray, azimuth: np.ndarray) -> Antivector:
    """Polar tilts in `zx` followed by azimuthal turns in `xy` carry `pole` to unit spins (`[...] Antivector-`) on the sphere."""
    tilts = (mv.zx * (-polar / 2)).exp()                                       # [...] Rotor+
    turns = (mv.xy * (-azimuth / 2)).exp()                                     # [...] Rotor+
    return (turns * tilts) >> pole                                             # [...] Antivector-


def stereographic_projection(spinor: Even) -> Antivector:
    """Stereographic projection from `-1` maps unit spinors into three-dimensional spatial points (`[...] Antivector-`)."""
    # Scale the bivector grade by 1 / (1 + scalar) and convert to an intrinsic point via handedness.
    return handedness * spinor.select[2] / (1 + spinor.select[0])              # [...] Antivector-


def linking_number(first: Antivector, second: Antivector) -> HandedScalar:
    """The discrete Gauss sum over segment pairs gives the signed linking number (`[...] Scalar-`) between closed polygonal curves."""
    step_first = first[..., 1:] - first[..., :-1]                              # [...] Antivector-
    step_second = second[..., 1:] - second[..., :-1]                           # [...] Antivector-
    middle_first = 0.5 * (first[..., 1:] + first[..., :-1])                    # [...] Antivector-
    middle_second = 0.5 * (second[..., 1:] + second[..., :-1])                 # [...] Antivector-
    separation = middle_first[..., :, None] - middle_second[..., None, :]      # [...] Antivector-
    # Regressive product of separation and both segment displacements spans their signed volume.
    volume = separation & step_first[..., :, None] & step_second[..., None, :] # [...] Scalar-
    distance = separation.antinorm()                                           # [...] Scalar+
    return (volume / distance ** 3).sum(axis=(-2, -1)) / (4 * np.pi)           # [...] Scalar-


def fibre_sweep(polars: np.ndarray, n_turns: int, n_frames: int,
                n_samples: int) -> Iterator[tuple[Antivector, Antivector]]:
    """Yields `(spin, projected_fibre)` pairs as `spin` spirals between `polars[0]` and `polars[1]`."""
    fibre_turns = (mv.xy * np.linspace(0.0, 2 * np.pi, n_samples + 1)).exp()   # [n_samples + 1] Rotor+
    for fraction in np.linspace(0.0, 1.0, n_frames):
        polar = polars[0] + (polars[1] - polars[0]) * fraction
        spiral_spin = sphere_spins(polar, 2 * np.pi * n_turns * fraction)      # [] Antivector-
        fibre = fibre_start(spiral_spin) * fibre_turns                         # [n_samples + 1] Even+
        yield spiral_spin, stereographic_projection(fibre)


def transport_rotors(spins: Antivector) -> Rotor:
    """Minimal-angle step rotors multiplied along `spins` carry a frame (`[...] Rotor+`) from `spins[0]` to each subsequent spin."""
    # Adding 1 halves the double-angle rotor spins[1:] * ~spins[:-1] between consecutive unit spins.
    step_rotors = (1 + spins[1:] * ~spins[:-1]).normalized()                   # [...] Rotor+
    return step_rotors.cumprod(axis=0)                                         # [...] Rotor+
