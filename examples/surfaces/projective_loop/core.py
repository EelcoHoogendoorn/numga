r"""Loop subdivision smooths a triangle mesh by splitting every triangle into four, moving each existing vertex to a weighted average of itself and its neighbours, and placing a new vertex on each edge between its endpoints and the two opposite corners. When the vertices are points of projective space carrying homogeneous weights and the subdivision averages those weighted points directly, each vertex weight acts as a handle on the limit surface: a heavy vertex pulls the surface toward itself, and a light vertex releases it. And subdivision commutes with perspective: refining and then viewing in perspective gives what viewing in perspective and then refining gives, where averaging Euclidean positions does not.

The vertices are points of projective geometric algebra, and one Loop step is a linear map from the coarser mesh's vertex sites to the finer one's, `step * controls`, and the steps compose into one map. A perspective is a map on points, the identity plus one term, `Point + centre * (axis & Point) * strength`. The step acts across the sites and the perspective on the point at each site, so the two commute.

In the homogeneous-coordinate notation of computer-aided geometric design, each weighted vertex reads as a four-vector $(w\,\mathbf p,\ w)$, a Loop step reads as a subdivision matrix $S$ acting across the vertices, and a projective transformation reads as a $4 \times 4$ homography matrix $H$ acting on each vertex, so $(S V) H^{\mathsf T} = S (V H^{\mathsf T})$.
"""

from __future__ import annotations

from numga import NumpyContext
from numga.algebras import PGA3D as ga
from examples import instantiate

ctx = NumpyContext(ga)
mv = ctx.multivector
Mesh = instantiate("examples.util.mesh", ga).Mesh
Point = ga.gatype.antivector()                      # Antivector-

