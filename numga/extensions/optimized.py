"""Opt-in closed forms for Cl(3,0,1) on the NumPy backend; call register() to prioritize them.

Each formula is written against one exact coefficient layout, the default one, and is
dispatched only on that layout: the predicates compare subspaces, blade order and signs
included. Other layouts fall through to the generic implementations; converting them here
would cost what the formulas save. In the cyclic order yz zx xy, each rotation blade pairs
with the translation blade xw yw zw at the same position, and no signs appear.

The bodies are NumPy-specific: they fill one preallocated output in place. A JAX version
would be written separately, against what XLA fuses well.
"""

import numpy as np

from numga.algebras import PGA3D
from numga.extensor import Extensor
from numga.gatype import ReverseProductOne, Versor

def layout(names: str):
    """A predicate matching exactly this blade layout of PGA3D, or of an algebra built with its
    description."""
    return lambda t: t.algebra.description == PGA3D.description and t.subspaces == (t.algebra.subspace(names),)


def exp_pga3(b: Extensor, *, n: int = 15) -> Extensor:
    """Motor exp of a bivector in the layout yz zx xy xw yw zw."""
    yz, zx, xy, xw, yw, zw = np.moveaxis(b._kernel, -1, 0)
    length_squared = yz*yz + zx*zx + xy*xy
    angle = np.sqrt(length_squared)
    cosine = np.cos(angle)
    sinc = np.sinc(angle / np.pi)
    pitch = yz*xw + zx*yw + xy*zw
    # (cos(a) - sinc(a))/a² loses precision near zero; its analytic series does not.
    small = length_squared < 1e-4
    screw = pitch * np.where(small, -1/3 + length_squared/30 - length_squared**2/840 + length_squared**3/45360,
                             (cosine - sinc) / np.where(small, 1, length_squared))
    m = np.empty(b.shape + (8,))
    m[..., 0] = cosine
    m[..., 1] = sinc*yz
    m[..., 2] = sinc*zx
    m[..., 3] = sinc*xy
    m[..., 4] = sinc*xw + screw*yz
    m[..., 5] = sinc*yw + screw*zx
    m[..., 6] = sinc*zw + screw*xy
    m[..., 7] = pitch*sinc
    return Extensor._from_prepared_kernel(b.context, b.algebra.gatype.rotor(), m)


def exp_rotation_pga3(b: Extensor, *, n: int = 15) -> Extensor:
    """Rotor exp of a bivector in the layout yz zx xy."""
    angle = np.sqrt((b._kernel**2).sum(axis=-1))
    m = np.empty(b.shape + (4,))
    m[..., 0] = np.cos(angle)
    m[..., 1:] = b._kernel * np.sinc(angle / np.pi)[..., None]
    gatype = b.algebra.gatype(b.algebra.subspace("1 yz zx xy")).with_traits(ReverseProductOne, Versor)
    return Extensor._from_prepared_kernel(b.context, gatype, m)


def exp_translation_pga3(b: Extensor, *, n: int = 15) -> Extensor:
    """Translator exp of a bivector in the layout xw yw zw: one plus the bivector."""
    m = np.empty(b.shape + (4,))
    m[..., 0] = 1
    m[..., 1:] = b._kernel
    gatype = b.algebra.gatype(b.algebra.subspace("1 xw yw zw")).with_traits(ReverseProductOne, Versor)
    return Extensor._from_prepared_kernel(b.context, gatype, m)


def normalize_pga3(motor: Extensor) -> Extensor:
    """Unit motor from an even element in the layout 1 yz zx xy xw yw zw xyzw."""
    m = motor._kernel.copy()
    # views onto the components of the copy; the in-place updates below write through them
    e, yz, zx, xy, xw, yw, zw, volume = (m[..., i] for i in range(8))
    scale = (e*e + yz*yz + zx*zx + xy*xy)**(-0.5)
    correction = (e*volume - yz*xw - zx*yw - xy*zw) * scale**2
    m *= scale[..., None]
    xw += yz*correction; yw += zx*correction; zw += xy*correction; volume -= e*correction
    return Extensor._from_prepared_kernel(motor.context, motor.algebra.gatype.rotor(), m)


class PrincipalInertiaPGA3(Extensor):
    """Inertia of a rigid body in its principal frame, Bivector <- Bivector, stored as four numbers.

    Rate and momentum share the bivectors yz zx xy xw yw zw. The map sends the translation block
    to the momentum's first block and the rotation block to its second, each scaled blade-wise:
    upper = m [..., 1] and lower = (I_yz, I_zx, I_xy) [..., 3] for the inertia. The inverse has
    the same form with upper = 1 / lower and lower = 1 / upper, so it is four reciprocals.

    Applying and inverting use these numbers directly, and the batch shape is theirs. Every other
    operation works on the dense 6x6 map, rebuilt from the four numbers whenever one needs it
    rather than stored, and returns an ordinary Extensor.
    """
    __slots__ = ("upper", "lower")

    def __init__(self, context, upper: np.ndarray, lower: np.ndarray) -> None:
        Bivector = context.algebra.gatype.bivector()
        self._context = context
        self._gatype = context.algebra.gatype((Bivector, Bivector))
        self.upper = upper
        self.lower = lower

    @classmethod
    def _from_prepared_kernel(cls, context, gatype, kernel) -> Extensor:
        # derived results are dense, ordinary extensors
        return Extensor._from_prepared_kernel(context, gatype, kernel)

    @property
    def shape(self) -> tuple[int, ...]:
        return np.broadcast_shapes(self.upper.shape[:-1], self.lower.shape[:-1])

    @property
    def _kernel(self) -> np.ndarray:
        """The dense 6x6 map, rebuilt from the four numbers on every access; it is not stored."""
        dense = np.zeros(self.shape + (6, 6))
        dense[..., (0, 1, 2), (3, 4, 5)] = self.upper
        dense[..., (3, 4, 5), (0, 1, 2)] = self.lower
        return dense

    def __call__(self, *operands):
        rate = operands[0]
        if len(operands) != 1 or not isinstance(rate, Extensor) or rate.gatype.output_subspace is not self.axes[1]:
            return Extensor.__call__(self, *operands)
        r = rate._kernel
        momentum = np.empty(np.broadcast_shapes(r.shape[:-1], self.shape) + (6,))
        momentum[..., 0:3] = r[..., 3:6] * self.upper
        momentum[..., 3:6] = r[..., 0:3] * self.lower
        return Extensor._from_prepared_kernel(rate.context, rate.gatype, momentum)

    def inverse(self) -> "PrincipalInertiaPGA3":
        return PrincipalInertiaPGA3(self._context, 1 / self.lower, 1 / self.upper)


def principal_inertia_pga3(mass: Extensor, moments: Extensor) -> PrincipalInertiaPGA3:
    """From a Scalar mass [...] and moments on the rotation blades yz zx xy [...]."""
    return PrincipalInertiaPGA3(mass.context, mass._kernel, moments.select_subspace(mass.algebra.subspace("yz zx xy"))._kernel)


def register() -> None:
    Extensor.exp.register(layout("yz zx xy xw yw zw"), position=0)(exp_pga3)
    Extensor.exp.register(layout("yz zx xy"), position=0)(exp_rotation_pga3)
    Extensor.exp.register(layout("xw yw zw"), position=0)(exp_translation_pga3)
    Extensor.normalized.register(layout("1 yz zx xy xw yw zw xyzw"), position=0)(normalize_pga3)
