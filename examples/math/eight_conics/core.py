"""Penrose's eight-conic theorem computes an eighth quadric in double contact with three neighbours on a cube.

Each quadric is modelled as a polarity map from points to polar planes, so adding the dyad of a plane changes the map everywhere except on that plane and produces double contact along it. Three compatible cube faces determine the eighth quadric from each of its three neighbours by one further contact dyad, and all three sums evaluate to the same map.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA2D as ga

mv = NumpyContext(ga).multivector
Plane = ga.gatype.vector()                                              # Vector+
Point = ga.gatype.antivector()                                          # Antivector-
Quadric = ga.gatype((Plane, Point))                                     # Plane+ <- Point-

first_axes = np.array([1, 2, 0])
second_axes = np.array([2, 0, 1])
edges = np.array([
    [0, 1], [0, 2], [0, 4], [1, 3], [2, 3], [1, 5],
    [4, 5], [2, 6], [4, 6], [3, 7], [5, 7], [6, 7],
])


def ellipsoid(center: Point, semi_axes: Point) -> Quadric:
    """The dual ellipsoid on `center` (`Point-`) and `semi_axes` (`[2] Point-`) inverts to its point-to-plane polarity map (`Plane+ <- Point-`)."""
    dual_quadric = (semi_axes * (Plane & semi_axes)).sum(axis=0) - center * (Plane & center)  # [] Point- <- Plane+
    return dual_quadric.inverse()                                                             # [] Plane+ <- Point-


def intersections(quadrics: Quadric, planes: Plane) -> Point:
    """Each quadric meets its plane in two real points (`[..., 2] Point-`) along the plane's tangent direction."""
    origin = mv.w.dual()                                                    # [] Point-
    unit_planes = planes.normalized()                                       # [...] Plane+
    # Averaging the chart origin with its reflection lands on the perpendicular foot on each plane.
    anchors = (origin + (unit_planes >> origin)) / 2                        # [...] Point-
    directions = (unit_planes | mv.xy).dual()                               # [...] Point-
    quadratic_terms = quadrics(directions) & directions                     # [...] Scalar+
    linear_terms = quadrics(anchors) & directions                           # [...] Scalar+
    constant_terms = quadrics(anchors) & anchors                            # [...] Scalar+
    root_signs = np.array([-1, 1])
    discriminants = linear_terms.squared() - quadratic_terms * constant_terms  # [...] Scalar+
    offsets = (-linear_terms[..., None] + discriminants.square_root()[..., None] * root_signs) / quadratic_terms[..., None]  # [..., 2] Scalar+
    return anchors[..., None] + directions[..., None] * offsets             # [..., 2] Point-


def configuration(base: Quadric, contact_planes: Plane, couplings: np.ndarray) -> tuple[Quadric, Plane, Quadric]:
    """A base quadric, three contact planes (`[3] Plane+`), and three face couplings (`[..., 3]`) determine the cube's eight quadrics (`[..., 8]`), twelve edge contact planes (`[..., 12]`), and three eighth-vertex completions (`[..., 3]`)."""
    double_planes = contact_planes * (contact_planes & Point)               # [3] Plane+ <- Point-
    neighbours = base + double_planes                                       # [3] Plane+ <- Point-
    first_planes = contact_planes[first_axes]                               # [3] Plane+
    second_planes = contact_planes[second_axes]                             # [3] Plane+
    face_scales = 1 - couplings ** 2

    # Each face quadric shares one contact plane with each of two neighbours.
    cross_terms = first_planes * (second_planes & Point) + second_planes * (first_planes & Point)  # [3] Plane+ <- Point-
    faces = (
        base * face_scales
        + double_planes[first_axes]
        + double_planes[second_axes]
        + cross_terms * couplings
    )                                                                       # [..., 3] Plane+ <- Point-

    # The three closing contact planes determine the same eighth quadric independently.
    closing_scale = 1 - np.sum(couplings ** 2, axis=-1) - 2 * np.prod(couplings, axis=-1)
    closing_planes = (
        contact_planes * face_scales
        + first_planes * (couplings[..., second_axes] + couplings * couplings[..., first_axes])
        + second_planes * (couplings[..., first_axes] + couplings * couplings[..., second_axes])
    )                                                                       # [..., 3] Plane+
    completions = (
        faces * closing_scale[..., None]
        + closing_planes * (closing_planes & Point)
    ) / face_scales                                                         # [..., 3] Plane+ <- Point-

    # Binary vertex labels flip one of the three choices along each cube edge.
    batch_shape = completions.shape[:-1]
    quadrics = stack((
        base.broadcast_to(batch_shape),
        neighbours[0].broadcast_to(batch_shape),
        neighbours[1].broadcast_to(batch_shape),
        faces[..., 2],
        neighbours[2].broadcast_to(batch_shape),
        faces[..., 1],
        faces[..., 0],
        completions[..., 0],
    ), axis=-1)                                                             # [..., 8] Plane+ <- Point-
    first_edge_planes = first_planes * couplings + second_planes            # [..., 3] Plane+
    second_edge_planes = first_planes + second_planes * couplings           # [..., 3] Plane+
    edge_planes = stack((
        contact_planes[0].broadcast_to(batch_shape),
        contact_planes[1].broadcast_to(batch_shape),
        contact_planes[2].broadcast_to(batch_shape),
        first_edge_planes[..., 2],
        second_edge_planes[..., 2],
        second_edge_planes[..., 1],
        first_edge_planes[..., 1],
        first_edge_planes[..., 0],
        second_edge_planes[..., 0],
        closing_planes[..., 2],
        closing_planes[..., 1],
        closing_planes[..., 0],
    ), axis=-1)                                                             # [..., 12] Plane+
    return quadrics, edge_planes, completions


def deformation(
    base: Quadric,
    contact_planes: Plane,
    couplings: np.ndarray,
) -> Iterator[tuple[Quadric, Plane, Point, Quadric]]:
    """Yields `(quadrics, edge_planes, contacts, completions)` (`[8]`, `[12]`, `[12, 2]`, `[3]`) as the face couplings (`[n_frames, 3]`) deform the cube."""
    for coupling in couplings:
        quadrics, edge_planes, completions = configuration(base, contact_planes, coupling)
        contacts = intersections(quadrics[edges[:, 0]], edge_planes)        # [12, 2] Point-
        yield quadrics, edge_planes, contacts, completions
