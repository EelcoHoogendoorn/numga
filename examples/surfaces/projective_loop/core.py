"""Loop subdivision smooths a triangulated surface by splitting every triangle into four, moving each existing vertex to a weighted average of itself and its neighbours, and placing a new vertex on each edge between its endpoints and the two opposite corners. When the vertices are points of projective space carrying homogeneous weights and the subdivision averages those weighted points directly, each vertex weight acts as a handle on the limit surface so that a heavy vertex pulls the surface toward itself and a light vertex releases it. Subdivision also commutes with perspective, because refining and then viewing in perspective gives the same surface as viewing in perspective and then refining, whereas averaging Euclidean positions does not.

The vertices are points `Point` of projective geometric algebra, and one Loop step `step * controls` is a linear map from the coarser topology's vertex sites to the finer topology's sites that composes across repeated steps into one map. A central collineation `Point + centre * (axis & Point) * strength` maps points to points at each site while the subdivision step acts across sites, so the two commute.
"""

from __future__ import annotations

from numga import NumpyContext
from numga.algebras import PGA3D as ga
from numga.topology import Topology

ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()                      # Antivector-


