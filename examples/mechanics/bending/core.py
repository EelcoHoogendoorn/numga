"""Euler–Bernoulli beam bending with Saint-Venant torsion and Timoshenko shear models thin-walled
cantilevers in the projective geometric algebra of three-dimensional space: loaded along any line,
beams bend sideways when the load lies off their principal bending directions and twist when it
misses their shear centre.

The beam runs along `z`, and its cross-section at the clamp lies in `section_plane` at `z = 0`.
Thin walls of given thickness follow a midline of straight wall segments, each segment joining a
pair of nodes. Bending keeps each cross-section planar: the motion from one cross-section to the
next is a twist, which on the unstrained beam is the ideal line `shift_along_beam` carrying the
section along the beam. Bending adds a line in the section's plane, `Strain`, turning the section
about a neutral axis or stretching it along the beam. Each fibre is a line along the beam through a
point of the section; a strain stretches it by `strain & fibre`, and the fibre pulls along itself.
Summed over the walls, the fibre pulls form the bending stiffness `Resultant- <- Strain+`.

Under a twist about a line along the beam, each midline point moves axially in proportion to its
warping, the signed area swept along the walls by the line from a pole. At the shear centre that
warping produces zero bending couple across the beam. Open walls resist a rate of twist by a third
of their length times the cube of their thickness, and each wall resists transverse sliding along
its own midline in proportion to its area. Because those slide forces pass through the shear centre,
a twist about the axial line through the shear centre costs no slide, and the linear solve places
the twisting axis there automatically.

The full section stiffness `Forque- <- Twist+` sums dyads of lines for axial stretch and bending,
Saint-Venant torsion, and wall shear, taking any strain twist to the resultant forque exerted by
the section. Solving against the load seen at each station gives the piece strains, whose motor
exponentials compose into the cross-section frames along the beam. Each constant-strain piece is
integrated exactly, while station loads are evaluated on the straight reference beam with free
warping at the clamp.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, concatenate
from numga.algebras import PGA3D as ga
from examples.mechanics.simplex import barycentric_samples, simplex_inertia_weights
from numga.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
Twist = ga.gatype.bivector()
# Lines of force: loads, and the resultant of everything a section exerts.
Forque = ga.gatype.antibivector()
Motor = ga.gatype.rotor()
Plane = ga.gatype.vector()
# Lines in the section's plane.
Strain = Twist.derive.touching(Algebra("z+"))
# Lines in the section's plane as joins of its points, such as the line through a wall's ends.
WallLine = Strain.derive.flipped_orientation
# Lines along the beam, and couples across it.
Resultant = Strain.derive.dual
# Fibre pulls weighted by signed swept-area warping, and their couples across the beam.
Warped = Resultant.derive.flipped_orientation
Couples = Warped.derive.touching(Algebra("w0"))
# Shifts of a pole across the beam: points at infinity in the section's plane.
Shift = Plane.derive.within(Algebra("x+y+")).derive.dual
# Transverse planes across and up the section, for placing points in `z = 0`.
SectionPlane = Plane.derive.within(Algebra("x+y+"))
# A bending strain to the resultant of the fibres' pulls.
Bending = ga.gatype((Resultant, Strain))                    # Resultant- <- Strain+
# Any strain to the forque of everything the section exerts.
Stiffness = ga.gatype((Forque, Twist))                      # Forque- <- Twist+

# Cross-section plane at the clamp, axial direction, and unit-length unstrained shift twist.
section_plane = mv.z                                        # [] Plane+
beam_direction = section_plane.dual()                       # [] Point-
shift_along_beam = section_plane ^ mv.w                     # [] Twist+
# Pure axial torque couple, the join of the transverse directions across and up.
couple_about_beam = mv.xy.dual()                            # [] Forque-


# --- math -----------------------------------------------------------------------------
def wall_samples(walls: Point, thickness: Scalar) -> Point:
    """Two quadrature points along each wall (`[2] Point-[E]`), weighted by the square root of each
    point's share of the wall area, integrate any quadratic over the wall exactly."""
    weights = simplex_inertia_weights(2)
    tail, head = walls                                                      # Point-[E], Point-[E]
    wall_lengths = (tail & head).norm()                                     # Scalar+[E]
    return barycentric_samples(weights, walls) * (wall_lengths * thickness).square_root()


def centroid(walls: Point, thickness: Scalar) -> Point:
    """Area-weighted wall midpoints sum to the normalized centroid (`[] Point-`) of the section."""
    tail, head = walls                                                      # Point-[E], Point-[E]
    wall_lengths = (tail & head).norm()                                     # Scalar+[E]
    return ((tail + head) * (wall_lengths * thickness / 2)).sites.sum().normalized()


def bending_stiffness(walls: Point, thickness: Scalar, modulus: float) -> Bending:
    """Summing fibre tension dyads over the exact wall quadrature points gives the section's bending
    stiffness (`[] Resultant- <- Strain+`)."""
    fibres = beam_direction & wall_samples(walls, thickness)                # [2] Resultant-[E]
    stretches = Strain & fibres                                             # [2] (Scalar+ <- Strain+)[E]
    # Each sample point appears twice in the dyad, so its square-root area weight contributes once.
    tensions = modulus * fibres * stretches                                 # [2] (Resultant- <- Strain+)[E]
    return tensions.sites.sum().sum(axis=0)                                 # [] Resultant- <- Strain+


def warping(topology: Topology, nodes: Point, thickness: Scalar, pole: Point) -> Warped:
    """Integrating fibres weighted by the signed area swept from `pole` along the walls (`[E]`) gives
    the warping resultant (`[] Warped+`)."""
    walls = topology.incidence[1, 0] * nodes                                # [2] Point-[E]
    tail, head = walls                                                      # Point-[E], Point-[E]
    # Least-squares integration of wall lines along `d[0]` accumulates path lines at the nodes up to a common constant.
    paths = (topology.d[0] * WallLine).lstsq(tail & head)                   # WallLine-[V]
    # Joining the accumulated wall lines with `pole` and `beam_direction` measures the signed swept area.
    node_warpings = beam_direction & pole & paths                           # Scalar-[V]
    wall_lengths = (tail & head).norm()                                     # Scalar+[E]
    area_roots = (wall_lengths * thickness).square_root()                   # Scalar+[E]
    end_warpings = topology.incidence[1, 0] * node_warpings                 # [2] Scalar-[E]
    sample_warpings = barycentric_samples(simplex_inertia_weights(2), end_warpings) * area_roots  # [2] Scalar-[E]
    return (sample_warpings * (beam_direction & wall_samples(walls, thickness))).sites.sum().sum(axis=0)  # [] Warped+


def shear_centre(topology: Topology, nodes: Point, thickness: Scalar) -> Point:
    """Shifting the pole from the centroid cancels the centroid's warping couples across the beam to
    locate the shear centre (`[] Point-`)."""
    centre = centroid(topology.incidence[1, 0] * nodes, thickness)          # [] Point-
    couples_per_shift = warping(topology, nodes, thickness, Shift).cast(Couples)  # [] Couples+ <- Shift-
    centroid_couples = warping(topology, nodes, thickness, centre).cast(Couples)  # [] Couples+
    return centre - couples_per_shift.solve(centroid_couples)               # [] Point-


def torsional_stiffness(walls: Point, thickness: Scalar, shear_modulus: float) -> Scalar:
    """Open walls resist Saint-Venant torsion by `shear_modulus` times one third of each wall's length
    times the cube of its thickness (`[] Scalar+`)."""
    tail, head = walls                                                      # Point-[E], Point-[E]
    wall_lengths = (tail & head).norm()                                     # Scalar+[E]
    return shear_modulus * (wall_lengths * thickness**3 / 3).sites.sum()    # [] Scalar+


def section_stiffness(topology: Topology, nodes: Point, thickness: Scalar, modulus: float,
                      shear_modulus: float) -> Stiffness:
    """Fibre tension, Saint-Venant torsion, and wall shear through the shear centre sum as line dyads
    into the full section stiffness (`[] Forque- <- Twist+`)."""
    walls = topology.incidence[1, 0] * nodes                                # [2] Point-[E]
    tail, head = walls                                                      # Point-[E], Point-[E]
    fibres = beam_direction & wall_samples(walls, thickness)                # [2] Resultant-[E]
    axial_and_bending = (modulus * fibres * (Twist & fibres)).sites.sum().sum(axis=0)  # [] Resultant- <- Twist+
    torsion_constant = torsional_stiffness(walls, thickness, shear_modulus)  # [] Scalar+
    torsion = torsion_constant * couple_about_beam * (couple_about_beam & Twist)  # [] Forque- <- Twist+
    wall_lengths = (tail & head).norm()                                     # Scalar+[E]
    wall_directions = (topology.d[0] * nodes) / wall_lengths                # Point-[E]
    slide_lines = shear_centre(topology, nodes, thickness) & wall_directions  # Forque-[E]
    shear = (shear_modulus * wall_lengths * thickness * slide_lines * (slide_lines & Twist)).sites.sum()  # [] Forque- <- Twist+
    return axial_and_bending + torsion + shear                              # [] Forque- <- Twist+


def station_loads(loads: Forque, stations: np.ndarray) -> Forque:
    """Translating tip `loads` back by `stations` (`[n_pieces]`) expresses the internal load in each
    station's cross-section plane (`[..., n_pieces] Forque-`)."""
    shifts = (shift_along_beam * (stations / 2)).exp()                      # [n_pieces] Motor+
    return shifts << loads[..., None]                                       # [..., n_pieces] Forque-


def section_frames(strains: Twist, step: float) -> Motor:
    """Exponentiating each piece twist (`[..., n_pieces] Twist+`) over `step` and multiplying rightwards
    from the clamp builds the cross-section frames (`[..., n_pieces + 1] Motor+`)."""
    piece_motors = ((shift_along_beam + strains) * (step / 2)).exp()        # [..., n_pieces] Motor+
    clamp = mv.rotor().broadcast_to(piece_motors.shape[:-1] + (1,))         # [..., 1] Motor+
    # Each piece twist acts in the local frame of the preceding section, so cumulative right-multiplication
    # equals the reverse of the cumulative left-product of reversed motors.
    return concatenate([clamp, piece_motors.reverse().cumprod(axis=-1).reverse()], axis=-1)  # [..., n_pieces + 1] Motor+


def axial_stress(strains: Twist, modulus: float, points: Point) -> Scalar:
    """Pairing `strains` with the axial fibres through `points` and scaling by `modulus` gives the axial
    normal stress (`[...] Scalar+`)."""
    return modulus * (strains & (beam_direction & points))                  # [...] Scalar+


def polar_strains(rigidity: Bending, points: Point) -> Strain:
    """Solving `rigidity` against unit axial pulls through `points` maps each point to its polar neutral
    axis (`[...] Strain+`)."""
    return rigidity.solve(beam_direction & points)                          # [...] Strain+


def tip_points(points: Point, length: float) -> Point:
    """Translating clamp-plane `points` by `length` along `shift_along_beam` places them on the unbent
    tip section (`[...] Point-`)."""
    shift = (shift_along_beam * (length / 2)).exp()                         # [] Motor+
    return shift >> points                                                  # [...] Point-


def cantilever(topology: Topology, nodes: Point, thickness: Scalar, loads: Forque, length: float,
               modulus: float, shear_modulus: float, n_pieces: int) -> tuple[Twist, Motor, Strain, Scalar]:
    """Solving the section stiffness along `n_pieces` stations of a cantilever of `length` under tip
    `loads` gives the piece strains (`[..., n_pieces] Twist+`), section frames (`[..., n_pieces + 1] Motor+`),
    two principal bending axes (`[2] Strain+`), and clamp wall-end stresses (`[..., 2] Scalar+[E]`)."""
    step = length / n_pieces
    stations = (np.arange(n_pieces) + 0.5) * step                           # [n_pieces]
    walls = topology.incidence[1, 0] * nodes                                # [2] Point-[E]
    rigidity = bending_stiffness(walls, thickness, modulus)                 # [] Resultant- <- Strain+
    stiffness = section_stiffness(topology, nodes, thickness, modulus, shear_modulus)  # [] Forque- <- Twist+
    # Station lever arms are evaluated on the straight reference beam; `exp(strain * step / 2)` moves frames
    # opposite to the bivector's own sign, so the stiffness solves against `-station_loads(...)`.
    strains = stiffness.solve(-station_loads(loads, stations))              # [..., n_pieces] Twist+
    frames = section_frames(strains, step)                                  # [..., n_pieces + 1] Motor+
    # The Euclidean metric on `Strain` vanishes on pure axial shift, ordering the two bending axes before it.
    _, axes = (Strain | Strain).eigh(Strain & rigidity)                     # [3] Strain+
    clamp_stresses = axial_stress(strains[..., 0, None], modulus, walls)    # [..., 2] Scalar+[E]
    return strains, frames, axes[:2], clamp_stresses


def axial_pushes(topology: Topology, nodes: Point, thickness: Scalar, points: Point,
                 modulus: float, push: float) -> tuple[Strain, Scalar, Strain]:
    """Compressive axial `push` loads through `points` produce bending strains (`[...] Strain+`), wall-end
    stresses (`[..., 2] Scalar+[E]`), and the node polar lines (`[V] Strain+`) whose envelope bounds the kern."""
    walls = topology.incidence[1, 0] * nodes                                # [2] Point-[E]
    rigidity = bending_stiffness(walls, thickness, modulus)                 # [] Resultant- <- Strain+
    strains = -push * polar_strains(rigidity, points)                       # [...] Strain+
    stresses = axial_stress(strains[..., None], modulus, walls)             # [..., 2] Scalar+[E]
    return strains, stresses, polar_strains(rigidity, nodes.batch())


def turning_states(topology: Topology, nodes: Point, thickness: Scalar, centre: Point,
                   down_load: Point, length: float, modulus: float, shear_modulus: float,
                   n_pieces: int, n_frames: int
                   ) -> Iterator[tuple[Point, Point, Twist, Motor, Strain, Scalar]]:
    """Yields `(load_point, load_direction, strains, frames, principal_axes, clamp_stresses)` across
    `n_frames` as the tip load rotates one full turn in the cross-section plane."""
    angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)            # [n_frames]
    tip_centre = tip_points(centre, length)                                 # [] Point-
    for angle in angles:
        turn = (mv.xy * (-angle / 2)).exp()                                 # [] Motor+
        load_direction = turn >> down_load                                  # [] Point-
        strains, frames, axes, stresses = cantilever(
            topology, nodes, thickness, tip_centre & load_direction,
            length, modulus, shear_modulus, n_pieces)
        yield centre, load_direction, strains, frames, axes, stresses


def crossing_states(topology: Topology, nodes: Point, thickness: Scalar, start: Point, end: Point,
                    down_load: Point, length: float, modulus: float, shear_modulus: float,
                    n_pieces: int, n_frames: int
                    ) -> Iterator[tuple[Point, Point, Twist, Motor, Strain, Scalar]]:
    """Yields `(load_point, load_direction, strains, frames, principal_axes, clamp_stresses)` across
    `n_frames` as a downward tip load translates linearly from `start` to `end` across the section."""
    fractions = np.linspace(0, 1, n_frames)                                 # [n_frames]
    for fraction in fractions:
        load_point = start * (1 - fraction) + end * fraction                # [] Point-
        strains, frames, axes, stresses = cantilever(
            topology, nodes, thickness, tip_points(load_point, length) & down_load,
            length, modulus, shear_modulus, n_pieces)
        yield load_point, down_load, strains, frames, axes, stresses


# --- plumbing -------------------------------------------------------------------------
def section_point(coordinates: np.ndarray) -> Point:
    """Two-dimensional `(x, y)` `coordinates` (`[..., 2]`) embed into unit-weight points (`[...] Point-`)
    in the clamp plane `z = 0`."""
    return (mv(SectionPlane, coordinates) + mv.w).dual()                    # [...] Point-
