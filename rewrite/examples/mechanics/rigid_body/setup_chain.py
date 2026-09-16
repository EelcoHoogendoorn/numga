"""Initialize an N-link rigid body chain in arbitrary dimensions for physics benchmarks."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

from numga import Context, Extensor
from examples.mechanics.rigid_body.core import Body, Constraint


def setup_bodies(
    context: Context,
    fixing: bool = True,
    n_bodies: int = 12,
    compliance: float = 1e-9,
    distance: float = 9e-2,
    size: float = 5e-2,
    damping: float = 1e-3,
) -> Tuple[Body, List[Constraint]]:
    """Initialize an N-link chain of rigid bodies in arbitrary PGA dimensions.

    Parameters
    ----------
    context : Context
        A NumpyContext or JaxContext bound to a PGA algebra (e.g. `x+y+w0` or `x+y+z+w0`).
    fixing : bool
        If True, anchor body 0 at translation directions so the chain hangs.
    n_bodies : int
        Number of links in the chain.
    compliance : float
        Constraint compliance (inverse stiffness).
    distance : float
        Link separation distance.
    size : float
        Size of each rigid body link.
    damping : float
        Velocity damping factor.
    """
    n_dim = context.algebra.dimension
    ndim = n_dim - 1
    origin_mask = 1 << ndim

    # Helper to construct translator motor for displacement d: [..., ndim]
    def translator(dist: np.ndarray) -> Extensor:
        pad = np.pad(dist, [(0, 0)] * (dist.ndim - 1) + [(0, 1)])
        vec = context.multivector.vector(pad)
        origin = context.multivector.vector(
            np.array([0] * ndim + [1], dtype=float)
        )
        generator = (origin * -0.5).wedge(vec)
        return generator.exp()

    def make_cube(n: int) -> np.ndarray:
        return np.array(np.kron(np.diag(np.arange(n) + 1.0) / n, [+1.0, -1.0]).T)

    # Origin point in PGA: (0, ..., 0, 1) in antivector space
    origin_pt_coords = np.zeros((len(context.algebra.subspace.antivector()),), dtype=float)
    origin_pt_coords[-1] = 1.0
    origin_pt = context.multivector.antivector(origin_pt_coords)

    def point_embed(coords: np.ndarray) -> Extensor:
        t = translator(coords)
        return t.sandwich(origin_pt)

    # Construct single body template from point cloud
    cube = make_cube(ndim) * size
    points = point_embed(cube)
    body_single = Body.from_point_cloud(points)

    # Gravitational acceleration along coordinate axis 0
    grav_coords = np.zeros((len(context.algebra.subspace.antivector()),), dtype=float)
    grav_coords[-1] = 5e-2  # direction
    body_single = body_single.copy(
        gravity=context.multivector.antivector(grav_coords),
        damping=context.multivector.scalar([damping]),
    )

    # Displacements of each body along the last model axis: [n_bodies, ndim]
    d = np.zeros((n_bodies, ndim), dtype=float)
    d[:, -1] = np.arange(n_bodies) * distance
    qs = translator(d)

    # Replicate template body across all n_bodies
    bodies = body_single[None][np.zeros(n_bodies, dtype=int)]
    bodies = bodies.copy(motor=bodies.motor * qs)

    # Initialize constraints between adjacent bodies
    i = np.arange(n_bodies - 1)
    body_idx = np.array([i, i + 1])

    # Anchor offsets in body-local frame
    a = np.zeros((2, ndim), dtype=float)
    a[0, -1] = +distance / 2.0
    a[1, -1] = -distance / 2.0
    a = a[:, None, :] * np.ones((1, n_bodies - 1, 1), dtype=float)
    anchors = point_embed(a)

    compliance_ext = context.multivector.scalar(
        np.full((n_bodies - 1, 1), compliance, dtype=float)
    )
    constraints = Constraint(
        body_idx=body_idx,
        anchors=anchors,
        compliance=compliance_ext,
    )

    # Anchor body 0 if requested
    if fixing:
        bivector = context.algebra.subspace.bivector()
        trans_indices = [idx for idx, m in enumerate(bivector.masks) if (m & origin_mask)]

        def fix_anchor(kernel: np.ndarray) -> np.ndarray:
            if hasattr(kernel, "at"):
                return kernel.at[0, :, trans_indices].set(0)
            k = np.array(kernel, copy=True)
            k[0, :, trans_indices] = 0
            return k

        bodies.inertia_inv = bodies.inertia_inv.map_kernel(fix_anchor)

    # Red/black partition of constraint set for independent relaxation
    return bodies, [constraints[0::2], constraints[1::2]]
