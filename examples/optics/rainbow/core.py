"""Sunlight refracted into a spherical raindrop, reflected internally once or twice, and refracted out again forms the primary and secondary rainbows in planar spacetime.

Sunlight entering a spherical raindrop bends at its front surface, reflects off the back of the drop, and bends again on the way out, returning toward the sun at a viewing angle determined by its impact offset from the drop's centre. That deflection reaches a stationary extreme near 42 degrees from the antisolar point for one internal reflection, concentrating rays into the primary bow, and a second extreme near 51 degrees for two internal reflections, leaving Alexander's dark band between the two bows while dispersion separates violet from red.

Planar spacetime models wavevectors and interface normals as extrinsic vectors, while spatial displacements, ray directions, and the drop's rest-frame time direction are intrinsic antivectors. Reflecting an incident wavevector across a unit interface normal plane via `face >> wavevector` isolates its continuous tangential-and-temporal part `(wavevector + (face >> wavevector)) / 2`, and enforcing the medium dispersion relation fixes the transmitted normal wavenumber directly without angles. Because every chord of a circle meets the rim at equal angles at both ends, reflecting an entry point across the chord plane gives the next bounce point and one fixed spatial rotor `turn` steps every vertex and wavevector around the drop, while the normal wavenumbers `(wavevector | face)` give both Fresnel reflectances directly.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack

# Planar spacetime `x+y+t-` through the drop's cross-section, space positive and time negative.
ga = Algebra("x+y+t-")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Wavevector = ga.gatype.vector()                                               # Vector+
Plane = Wavevector.derive.within(Algebra("x+y+"))                             # Vector+
Event = ga.gatype.antivector()                                                # Antivector-
# Spatial displacements from the drop centre and ray directions in the drop rest frame.
Direction = Plane.derive.dual                                                 # Antivector-
Rotor = ga.gatype.rotor()                                                     # Rotor+

# Drop rest-frame time direction pairing to +1 with unit-frequency sunlight, and spatial axes.
time = -mv.t.dual()                                                           # [] Event-
forward, up = mv.x.dual(), mv.y.dual()                                        # [] Direction-, Direction-
sunlight_wave = mv.x - mv.t                                                   # [] Wavevector+


# --- math -----------------------------------------------------------------------------
def refracted_wavevector(wavevector: Wavevector, face: Plane, index: np.ndarray) -> Wavevector:
    """Crossing a unit interface `face` (`[...] Plane+`) into a medium of refractive `index` preserves the
    tangential-and-temporal wavevector part and sets the normal wavenumber from dispersion (`[...] Wavevector+`)."""
    # `face >> wavevector` reverses the normal component while keeping the tangential spatial and temporal parts.
    tangential = (wavevector + (face >> wavevector)) / 2                      # [...] Wavevector+
    frequency = wavevector & time                                             # [...] Scalar+
    normal_wavenumber = ((index**2 - 1) * frequency.squared() - tangential.squared()).square_root()   # [...] Scalar+
    return tangential + face * normal_wavenumber                              # [...] Wavevector+


def entry_point(sunlight: Direction, offsets: Direction) -> Direction:
    """Rays travelling along unit `sunlight` (`[] Direction-`) at transverse `offsets` (`[...] Direction-`)
    meet the sunlit hemisphere of the unit drop at `entry_point` (`[...] Direction-`)."""
    return offsets - sunlight * (1 - offsets.antinorm().squared()).square_root()   # [...] Direction-


def bounce_rotor(vertex: Direction, wavevector: Wavevector) -> Rotor:
    """Reflecting `vertex` (`[...] Direction-`) across the spatial chord along `wavevector` (`[...] Wavevector+`)
    defines the drop-centre rotor (`[...] Rotor+`) stepping each internal bounce to the next."""
    spatial = wavevector.cast(Plane)                                          # [...] Plane+
    chord_plane = spatial / spatial.norm()                                    # [...] Plane+
    # `chord_plane >> vertex` reverses the chordwise component and keeps the perpendicular offset from the centre.
    next_vertex = chord_plane >> vertex                                       # [...] Direction-
    return (next_vertex * vertex).square_root()                               # [...] Rotor+


def fresnel_reflectances(
    incident: Wavevector, refracted: Wavevector, face: Plane, index: np.ndarray,
) -> tuple[Scalar, Scalar]:
    """Normal wavenumbers of `incident` and `refracted` (`[...] Wavevector+`) across `face` (`[...] Plane+`)
    into refractive `index` determine the perpendicular and in-plane Fresnel reflectances (`[...] Scalar+` each)."""
    incident_normal = incident | face                                         # [...] Scalar+
    refracted_normal = refracted | face                                       # [...] Scalar+
    reflectance_across = ((incident_normal - refracted_normal) / (incident_normal + refracted_normal)).squared()   # [...] Scalar+
    reflectance_along = (
        (index**2 * incident_normal - refracted_normal) / (index**2 * incident_normal + refracted_normal)
    ).squared()                                                               # [...] Scalar+
    return reflectance_across, reflectance_along


def drop_paths(
    offsets: Direction, index: np.ndarray, n_reflections: int,
) -> Iterator[tuple[Direction, Direction, Scalar]]:
    """Yields `(vertex, exit_direction, transmittance)` (`[...] Direction-`, `[...] Direction-`, `[...] Scalar+`)
    for zero up to `n_reflections` internal reflections of unit-frequency sunlight at impact `offsets`."""
    vertex = entry_point(forward, offsets)                                    # [...] Direction-
    inward_face = (-vertex).dual_inverse()                                    # [...] Plane+
    inside_wave = refracted_wavevector(sunlight_wave, inward_face, index)     # [...] Wavevector+
    # Equal chord angles around the circle let a single rotor step every internal bounce and wavevector.
    turn = bounce_rotor(vertex, inside_wave)                                  # [...] Rotor+
    reflectance_across, reflectance_along = fresnel_reflectances(sunlight_wave, inside_wave, inward_face, index)   # [...] Scalar+, Scalar+
    passed_across = (1 - reflectance_across).squared()                        # [...] Scalar+
    passed_along = (1 - reflectance_along).squared()                          # [...] Scalar+
    for _ in range(n_reflections + 1):
        vertex = turn >> vertex                                               # [...] Direction-
        exit_wave = refracted_wavevector(inside_wave, vertex.dual_inverse(), np.array(1.0))   # [...] Wavevector+
        exit_direction = exit_wave.cast(Plane).dual()                         # [...] Direction-
        yield vertex, exit_direction, (passed_across + passed_along) / 2
        inside_wave = turn >> inside_wave                                     # [...] Wavevector+
        passed_across = passed_across * reflectance_across                    # [...] Scalar+
        passed_along = passed_along * reflectance_along                       # [...] Scalar+


def sky_bows(
    offsets: Direction, indices: np.ndarray, n_reflections: int,
) -> tuple[Scalar, Scalar]:
    """Tracing `offsets` (`[n_rays] Direction-`) across `indices` (`[n_colours, 1]`) for `1 .. n_reflections`
    internal reflections gives the antisolar viewing cosines and ring-weighted fluxes (`[n_reflections, n_colours, n_rays] Scalar+` each)."""
    _, *bows = drop_paths(offsets, indices, n_reflections)
    # Looking back against `exit_direction` measures the angle from the antisolar direction `forward`.
    sight_cosines = stack([-(mv.x & exit_direction) for _, exit_direction, _ in bows])   # [..., ...] Scalar+
    ring_sizes = offsets.antinorm()                                           # [...] Scalar+
    return sight_cosines, stack([passed * ring_sizes for _, _, passed in bows])   # [..., ...] Scalar+


def ray_bundle(
    offsets: Direction, index: np.ndarray, n_reflections: int,
) -> tuple[Direction, Direction, Direction]:
    """Entry points, bounce vertices, and exit directions (`[...] Direction-`, `[n_reflections + 1, ...] Direction-`,
    `[n_reflections + 1, ...] Direction-`) form the ray fan across `offsets`."""
    vertices, exits, _ = zip(*drop_paths(offsets, index, n_reflections))
    return entry_point(forward, offsets), stack(list(vertices)), stack(list(exits))


def swept_ray_states(
    offsets: Direction, index: np.ndarray, n_reflections: int,
) -> Iterator[tuple[Direction, Direction, Direction, Scalar]]:
    """Yields `(entry, vertices, exits, sight_cosines)` (`[] Direction-`, `[n_reflections + 1] Direction-`,
    `[n_reflections + 1] Direction-`, `[n_reflections + 1] Scalar+`) as a single ray steps across `offsets`."""
    for offset in offsets:
        entered, vertices, exits = ray_bundle(offset, index, n_reflections)
        yield entered, vertices, exits, -(mv.x & exits)

