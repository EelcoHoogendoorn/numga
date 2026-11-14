"""Closed-form specializations accelerate common `PGA3D` motor and rigid-body inertia operations on
the NumPy backend. Calling `register()` places the closed-form motor exponential and normalization
kernels at the front of `Extensor.exp` and `Extensor.normalized` dispatch, while
`principal_inertia_pga3` constructs compact principal-frame inertia maps directly.

Every specialization requires `PGA3D`'s default cyclic blade layout (`yz zx xy` paired position by
position with `wx wy wz`), checking exact blade order and signs in its dispatch predicate. Inputs in
other layouts fall through to the generic algebraic implementations because converting blade layouts
at runtime would spend the work the closed forms save. In the cyclic order, each rotation blade sits
opposite its conjugate translation blade without alternating signs, and near-zero screw angles switch
from `(cos - sinc) / angle^2` to a truncated Taylor series to preserve precision.

Each NumPy kernel writes directly into a single preallocated output array, whereas a JAX backend
counterpart would target XLA fusion separately. `PrincipalInertiaPGA3` stores a rigid body's
principal-frame map from extrinsic bivector rates to intrinsic bivector momenta as four scalars per
body (mass on `wx wy wz` and three principal moments on `yz zx xy`), reducing evaluation and
inversion to four multiplications or reciprocals and materializing the dense `6 x 6` kernel only when
general `Extensor` operations request it. Predicate dispatch and `PrincipalInertiaPGA3` are described
in `docs/internals.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from numga.algebras import PGA3D
from numga.extensor import Extensor
from numga.gatype import GAType, Orientation, ReverseProductOne, Versor

if TYPE_CHECKING:
    from numga.backend.context import Context


def layout(names: str) -> Callable:
    """Return a dispatch predicate matching algebras with `PGA3D`'s description and the exact signed
    blade order in `names`."""
    return lambda t: t.algebra.description == PGA3D.description and t.subspaces == (t.algebra.subspace(names),)


def exp_pga3(bivector: Extensor, *, n: int = 15) -> Extensor:
    """Chasles's screw formula exponentiates a `PGA3D` bivector in layout `yz zx xy wx wy wz` into a
    unit motor."""
    yz, zx, xy, wx, wy, wz = np.moveaxis(bivector._kernel, -1, 0)
    length_squared = yz*yz + zx*zx + xy*xy
    angle = np.sqrt(length_squared)
    cosine = np.cos(angle)
    sinc = np.sinc(angle / np.pi)
    pitch = yz*wx + zx*wy + xy*wz
    # Replace (cosine - sinc) / length_squared with its Taylor series near zero to avoid cancellation.
    small = length_squared < 1e-4
    screw = pitch * np.where(small, -1/3 + length_squared/30 - length_squared**2/840 + length_squared**3/45360,
                             (cosine - sinc) / np.where(small, 1, length_squared))
    m = np.empty(bivector.shape + (8,))
    m[..., 0] = cosine
    m[..., 1] = sinc*yz
    m[..., 2] = sinc*zx
    m[..., 3] = sinc*xy
    m[..., 4] = sinc*wx + screw*yz
    m[..., 5] = sinc*wy + screw*zx
    m[..., 6] = sinc*wz + screw*xy
    m[..., 7] = pitch*sinc
    return Extensor._from_prepared_kernel(bivector.context, bivector.algebra.gatype.rotor(), m)


def exp_rotation_pga3(bivector: Extensor, *, n: int = 15) -> Extensor:
    """Rodrigues's formula exponentiates a pure rotation bivector in layout `yz zx xy` into a unit
    rotor."""
    angle = np.sqrt((bivector._kernel**2).sum(axis=-1))
    m = np.empty(bivector.shape + (4,))
    m[..., 0] = np.cos(angle)
    m[..., 1:] = bivector._kernel * np.sinc(angle / np.pi)[..., None]
    gatype = bivector.algebra.gatype.even().derive.nondegenerate.derive.with_traits(ReverseProductOne, Versor)
    return Extensor._from_prepared_kernel(bivector.context, gatype, m)


def exp_translation_pga3(bivector: Extensor, *, n: int = 15) -> Extensor:
    """Exponentiating an ideal translation bivector in layout `wx wy wz` adds `1` because ideal
    squares vanish."""
    m = np.empty(bivector.shape + (4,))
    m[..., 0] = 1
    m[..., 1:] = bivector._kernel
    gatype = bivector.algebra.gatype(bivector.algebra.subspace("1 wx wy wz"), orientation=Orientation.extrinsic).derive.with_traits(ReverseProductOne, Versor)
    return Extensor._from_prepared_kernel(bivector.context, gatype, m)


def normalize_pga3(motor: Extensor) -> Extensor:
    """Projecting an even `PGA3D` element in layout `1 yz zx xy wx wy wz wxyz` onto the Study quadric
    produces a unit motor."""
    m = motor._kernel.copy()
    # Views onto the copied components let the Study-quadric updates below modify m in place.
    e, yz, zx, xy, wx, wy, wz, volume = (m[..., i] for i in range(8))
    scale = (e*e + yz*yz + zx*zx + xy*xy)**(-0.5)
    correction = (e*volume - yz*wx - zx*wy - xy*wz) * scale**2
    m *= scale[..., None]
    wx += yz*correction; wy += zx*correction; wz += xy*correction; volume -= e*correction
    return Extensor._from_prepared_kernel(motor.context, motor.algebra.gatype.rotor(), m)


class PrincipalInertiaPGA3(Extensor):
    """A principal-frame rigid-body inertia map (`Bivector- <- Bivector+`) stores mass in `upper`
    (`[..., 1]`) and principal moments on `yz zx xy` in `lower` (`[..., 3]`), swapping and scaling
    the translation and rotation blocks directly while rebuilding a dense `6 x 6` kernel on demand."""
    __slots__ = ("upper", "lower")

    def __init__(self, context: Context, upper: np.ndarray, lower: np.ndarray, gatype: GAType) -> None:
        self._context = context
        self._gatype = gatype
        self.upper = upper
        self.lower = lower

    @classmethod
    def _from_prepared_kernel(cls, context: Context, gatype: GAType, kernel: Any) -> Extensor:
        # Derived results return ordinary dense extensors.
        return Extensor._from_prepared_kernel(context, gatype, kernel)

    @property
    def shape(self) -> tuple[int, ...]:
        return np.broadcast_shapes(self.upper.shape[:-1], self.lower.shape[:-1])

    @property
    def _kernel(self) -> np.ndarray:
        dense = np.zeros(self.shape + (6, 6))
        dense[..., (0, 1, 2), (3, 4, 5)] = self.upper
        dense[..., (3, 4, 5), (0, 1, 2)] = self.lower
        return dense

    def __call__(self, *operands: Any) -> Extensor:
        rate = operands[0]
        if (len(operands) != 1 or not isinstance(rate, Extensor) or rate.gatype.output_subspace is not self.axes[1]
                or rate.gatype.orientation is not self.gatype.input_orientations[0]):
            return Extensor.__call__(self, *operands)
        r = rate._kernel
        momentum = np.empty(np.broadcast_shapes(r.shape[:-1], self.shape) + (6,))
        momentum[..., 0:3] = r[..., 3:6] * self.upper
        momentum[..., 3:6] = r[..., 0:3] * self.lower
        return Extensor._from_prepared_kernel(rate.context, self.gatype.derive.output, momentum)

    def inverse(self) -> PrincipalInertiaPGA3:
        """Inverting a principal-frame inertia map swaps `upper` and `lower` and takes four
        elementwise reciprocals."""
        return PrincipalInertiaPGA3(self._context, 1 / self.lower, 1 / self.upper, self.gatype.derive.transposed)


def principal_inertia_pga3(mass: Extensor, moments: Extensor) -> PrincipalInertiaPGA3:
    """Constructing a principal-frame inertia map pairs a `Scalar` mass with principal moments selected
    on the rotation blades `yz zx xy`."""
    bivector = mass.algebra.subspace.bivector()
    gatype = mass.algebra.gatype.oriented((bivector, bivector), (Orientation.intrinsic, Orientation.extrinsic))
    lower = moments.select_subspace(mass.algebra.subspace("yz zx xy"))._kernel
    return PrincipalInertiaPGA3(mass.context, mass._kernel, lower, gatype)


def register() -> None:
    """Registering the closed-form `PGA3D` exponential and motor normalization kernels at position `0`
    places them ahead of the generic `Extensor` implementations."""
    Extensor.exp.register(layout("yz zx xy wx wy wz"), position=0)(exp_pga3)
    Extensor.exp.register(layout("yz zx xy"), position=0)(exp_rotation_pga3)
    Extensor.exp.register(layout("wx wy wz"), position=0)(exp_translation_pga3)
    Extensor.normalized.register(layout("1 yz zx xy wx wy wz wxyz"), position=0)(normalize_pga3)
