"""Opt-in closed forms for Cl(3,0,1); call register() to prioritize them.

Layouts and signs are converted at the numerical boundary. NumPy and JAX use
the same formulas, including the analytic small-angle limit of the screw term.
"""

from numga.extensor import Extensor
from numga.gatype import ReverseProductOne, Versor


def exp_pga3(b: Extensor, *, n: int = 15) -> Extensor:
    layout = b.algebra.subspace.from_layout((3, 5, 6, 9, 10, 12), (1,) * 6)
    xp = b.context.xp
    xy, xz, yz, xw, yw, zw = xp.moveaxis(b.select_subspace(layout)._kernel, -1, 0)
    length_squared = xy*xy + xz*xz + yz*yz
    angle = xp.sqrt(length_squared)
    cosine = xp.cos(angle)
    sinc = xp.sinc(angle / xp.pi)
    pitch = xy*zw - xz*yw + yz*xw
    # (cos(a) - sinc(a))/a² loses precision near zero; its analytic series does not.
    series = -1/3 + length_squared/30 - length_squared**2/840 + length_squared**3/45360
    quotient = (cosine - sinc) / xp.where(length_squared == 0, 1, length_squared)
    screw = pitch * xp.where(length_squared < 1e-4, series, quotient)
    coefficients = xp.stack((cosine, sinc*xy, sinc*xz, sinc*yz,
                             sinc*xw + screw*yz, sinc*yw - screw*xz,
                             sinc*zw + screw*xy, pitch*sinc), axis=-1)
    output = b.algebra.subspace.from_layout((0, 3, 5, 6, 9, 10, 12, 15), (1,) * 8)
    gatype = b.algebra.gatype(output).with_traits(ReverseProductOne, Versor)
    return Extensor._from_prepared_kernel(b.context, gatype, coefficients)


def normalize_pga3(m: Extensor) -> Extensor:
    output = m.algebra.subspace.from_layout((0, 3, 5, 6, 9, 10, 12, 15), (1,) * 8)
    xp = m.context.xp
    e, xy, xz, yz, xw, yw, zw, volume = xp.moveaxis(m.select_subspace(output)._kernel, -1, 0)
    scale = (e*e + xy*xy + xz*xz + yz*yz)**(-0.5)
    correction = (e*volume - xy*zw + xz*yw - yz*xw) * scale**2
    coefficients = xp.stack((e, xy, xz, yz, xw + yz*correction,
                             yw - xz*correction, zw + xy*correction,
                             volume - e*correction), axis=-1) * scale[..., None]
    gatype = m.algebra.gatype(output).with_traits(ReverseProductOne, Versor)
    return Extensor._from_prepared_kernel(m.context, gatype, coefficients)


def register() -> None:
    Extensor.exp.register(
        lambda t: t.algebra.signature == (1, 1, 1, 0) and t <= t.algebra.gatype.bivector(),
        position=0,
    )(exp_pga3)
    Extensor.normalized.register(
        lambda t: t.algebra.signature == (1, 1, 1, 0) and t <= t.algebra.gatype.even(),
        position=0,
    )(normalize_pga3)
