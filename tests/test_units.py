"""Static per-blade physical dimension tracking across UnitGrid, Context opt-in, GATypes,
products, duality, linear algebra, extensions, fields, sparse operators, and JAX."""

from __future__ import annotations

from fractions import Fraction
import pickle

import numpy as np
import pytest

from numga import (
    Algebra,
    GATypeError,
    NumpyContext,
    Orientation,
    Unit,
    UnitGrid,
    dim,
    stack,
    where,
)
from numga.algebras import PGA2D, PGA3D, STA, VGA3D
from numga.extensions import invariant_decomposition, optimized
from numga.extensions.optimized import principal_inertia_pga3
from numga.sparse import spdiag
from numga.topology import Topology


def _pga3_types():
    """Standard Model B PGA3D rigid-body and projective geometry types."""
    Scalar = PGA3D.gatype.scalar()
    Plane = PGA3D.gatype.vector()
    Point = PGA3D.gatype.antivector()
    Bivector = PGA3D.gatype.bivector()
    Antibivector = PGA3D.gatype.antibivector()
    Origin = Point.derive.nondegenerate
    Direction = Point.derive.degenerate
    Rotation = Bivector.derive.nondegenerate
    Translation = Bivector.derive.degenerate
    Torque = Antibivector.derive.degenerate
    Force = Antibivector.derive.nondegenerate

    WorldPoint = Point.derive.projective(dim.length)
    Screw = Bivector.derive.projective(dim.length)
    Rate = Bivector.derive.unit(1 / dim.time).derive.projective(dim.length)
    force_dim = dim.mass * dim.length / dim.time**2
    Forque = Antibivector.derive.unit(force_dim).derive.projective(dim.length)
    Momentum = Forque.derive.scaled_unit(dim.time)
    Inertia = PGA3D.gatype((Momentum, Rate))
    return dict(
        Scalar=Scalar,
        Plane=Plane,
        Point=Point,
        Bivector=Bivector,
        Antibivector=Antibivector,
        Origin=Origin,
        Direction=Direction,
        Rotation=Rotation,
        Translation=Translation,
        Torque=Torque,
        Force=Force,
        WorldPoint=WorldPoint,
        Screw=Screw,
        Rate=Rate,
        Forque=Forque,
        Momentum=Momentum,
        Inertia=Inertia,
    )


def _blade_units(gatype) -> dict[str, Unit]:
    """Map nonzero blade names of a nullary `GAType` to their `Unit`."""
    space = gatype.output_subspace
    return {
        space.algebra.blade_name(space.masks[idx]): unit
        for idx, unit in gatype.units.items
    }


# --- Unit value type, UnitGrid, and dim namespace ---------------------------------------

def test_unit_dim_arithmetic_roots_and_grid():
    force = dim.mass * dim.length / dim.time**2
    assert force is (dim.length * dim.mass * (dim.time ** -2))
    assert str(force) == "length*mass/time^2"
    assert not force.is_dimensionless

    assert dim.none.is_dimensionless
    assert str(dim.none) == "1"
    assert pickle.loads(pickle.dumps(dim.none)) is dim.none
    assert pickle.loads(pickle.dumps(force)) is force

    # Rational exponents and roots.
    assert (dim.length / dim.length) is dim.none
    assert ((1 / dim.length) * dim.length) is dim.none
    assert (dim.length / dim.time) ** 2 == dim.length**2 / dim.time**2
    assert (dim.length**2 / dim.time**2).root(2) == dim.length / dim.time
    assert (dim.length**3).root(3) == dim.length
    assert dim.length.root(2) == dim.length ** Fraction(1, 2)
    assert str(dim.length.root(2)) == "length^(1/2)"
    assert (dim.mass * dim.length**2) ** Fraction(-1, 2) == (dim.mass * dim.length**2).root(2).inverse()
    assert str((dim.mass * dim.length**2).root(2).inverse()) == "1/(length*mass^(1/2))"

    # UnitGrid interning, pickling, uniform detection, unchecked, and structural zero states.
    g1 = UnitGrid.uniform_grid((3,), dim.length)
    g2 = UnitGrid.uniform_grid((3,), dim.length)
    assert g1 is g2
    assert g1.uniform is dim.length
    assert not g1.is_unchecked and not g1.is_zero
    assert pickle.loads(pickle.dumps(g1)) == g1

    g_unchecked = UnitGrid.unchecked((3,))
    assert g_unchecked.is_unchecked and not g_unchecked.is_zero and g_unchecked.uniform is None
    assert pickle.loads(pickle.dumps(g_unchecked)) == g_unchecked

    g_zero = UnitGrid.zero((3,))
    assert g_zero.is_zero and not g_zero.is_unchecked and g_zero.uniform is None
    assert pickle.loads(pickle.dumps(g_zero)) == g_zero

    g_mix = UnitGrid.from_entries((2,), {0: dim.none, 1: dim.length})
    assert g_mix.uniform is None and not g_mix.is_unchecked and not g_mix.is_zero

    # Strings are rejected: only Unit objects from dim are accepted.
    with pytest.raises(TypeError):
        Unit.require("m")
    with pytest.raises(ValueError):
        (dim.length**2).root(0)
    with pytest.raises(AttributeError):
        _ = dim._private


# --- Algebra neutrality, value-level opt-in, and pixels vs metres in one PGA ------------

def test_algebra_has_no_units_value_level_opt_in_and_pixels_vs_metres():
    # Algebras carry zero physical units: no `units=` argument, no hidden `w = 1/length`.
    assert not hasattr(PGA3D, "has_units")
    assert not hasattr(PGA3D, "antiscalar_unit")
    assert not hasattr(PGA2D, "has_units")
    with pytest.raises(TypeError):
        Algebra("x+y+w0", units={"w": 1 / dim.length})

    Point = PGA3D.gatype.antivector()
    Bivector = PGA3D.gatype.bivector()
    assert not Point.has_units and Point.units.is_unchecked
    assert not Bivector.has_units and Bivector.units.is_unchecked
    with pytest.raises(GATypeError):
        _ = Point.unit
    with pytest.raises(GATypeError):
        _ = Bivector.unit

    # Plain context (units=False) builds unchecked values that match algebra types and never fail checks.
    plain_ctx = NumpyContext(PGA3D)
    plain_mv = plain_ctx.multivector
    assert plain_ctx.gatype is PGA3D.gatype
    plain_p = plain_mv.point([1.0, 2.0, 3.0])
    assert not plain_p.has_units and plain_p.units.is_unchecked
    assert plain_p.gatype == Point
    assert plain_mv.w.units.is_unchecked
    assert (plain_mv.x + 2.0 * plain_mv.w).units.is_unchecked
    assert plain_p.dual().units.is_unchecked
    plain_b = plain_mv.bivector([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert Bivector.dual()(plain_b).units.is_unchecked

    # Opted-in context (units=True) shares PGA3D and its single gatype factory, tracking units per blade.
    ctx = NumpyContext(PGA3D, units=True)
    mv = ctx.multivector
    assert ctx.algebra is plain_ctx.algebra
    assert ctx.gatype is PGA3D.gatype
    assert ctx.is_compatible_with(plain_ctx)
    assert mv.w.has_units and mv.w.unit is dim.none and mv.x.unit is dim.none

    # Dimensionless point, metre point, and pixel point coexist in the SAME PGA3D algebra.
    p_bare = mv.point([1.0, 2.0, 3.0])
    p_m = mv.point([1.0, 2.0, 3.0], unit=dim.length)
    p_px = mv.point([100.0, 200.0, 300.0], unit=dim.pixel)
    assert p_bare.unit is dim.none
    assert _blade_units(p_m.gatype) == {
        "xyz": dim.none, "wyz": dim.length, "wxz": dim.length, "wxy": dim.length,
    }
    assert _blade_units(p_px.gatype) == {
        "xyz": dim.none, "wyz": dim.pixel, "wxz": dim.pixel, "wxy": dim.pixel,
    }
    assert (mv.w & p_m).unit is dim.none
    assert (mv.x & p_m).unit == dim.length
    assert (mv.x & p_px).unit == dim.pixel

    # Adding or subtracting metres and pixels is refused; same-unit differences succeed.
    with pytest.raises(GATypeError):
        _ = p_m + p_px
    with pytest.raises(GATypeError):
        _ = p_m - p_px
    diff_m = p_m - mv.point([0.0, 1.0, 1.0], unit=dim.length)
    assert (mv.x & diff_m).unit == dim.length

    # In PGA2D, a camera projection map sends world metres to screen pixels in the same algebra.
    mv2 = NumpyContext(PGA2D, units=True).multivector
    Point2 = PGA2D.gatype.antivector()
    Origin2 = Point2.derive.nondegenerate
    Dir2 = Point2.derive.degenerate
    World2 = PGA2D.gatype(Point2.output_subspace, orientation=Point2.orientation,
                          unit={Origin2: dim.none, Dir2: dim.length})
    Screen2 = PGA2D.gatype(Point2.output_subspace, orientation=Point2.orientation,
                           unit={Origin2: dim.none, Dir2: dim.pixel})
    CameraMap = PGA2D.gatype((Screen2, World2))
    focal = mv2.scalar(500.0, unit=dim.pixel / dim.length)
    cam = (
        mv2.w.dual() * (mv2.w & Point2)
        + focal * (mv2.x.dual() * (mv2.x & Point2) + mv2.y.dual() * (mv2.y & Point2))
    )
    assert all(CameraMap.units.get(idx) == u for idx, u in cam.gatype.units.items)
    w_pt = mv2.point([0.2, 0.4], unit=dim.length)
    s_pt = cam(w_pt)
    assert s_pt.gatype == Screen2
    np.testing.assert_allclose((mv2.x & s_pt).to_array(), 100.0)
    with pytest.raises(GATypeError):
        _ = s_pt + w_pt

    # Checked and unchecked values on the single PGA3D interoperate.
    rotor = (0.3 * mv.xy).exp()
    assert rotor.unit is dim.none
    assert (rotor >> plain_p).units.is_unchecked
    assert (mv.scalar(2.0, unit=dim.length) + plain_mv.scalar([3.0])).units.is_unchecked


# --- GAType identity semantics, aliases, typed zeros, and unit scaling ------------------

def test_gatype_identity_semantics_aliases_and_typed_zeros():
    mv = NumpyContext(PGA3D, units=True).multivector
    t = _pga3_types()
    Scalar, Point, Plane, Bivector, Screw, Rate, Momentum, Inertia = (
        t["Scalar"], t["Point"], t["Plane"], t["Bivector"],
        t["Screw"], t["Rate"], t["Momentum"], t["Inertia"],
    )
    Origin, Direction = t["Origin"], t["Direction"]
    Antibivector, WorldPoint, Forque = t["Antibivector"], t["WorldPoint"], t["Forque"]
    Duration = Scalar.derive.unit(dim.time)
    Length = Scalar.derive.unit(dim.length)

    # A GAType is its identity map wherever it appears: `+` builds an identity-map Extensor,
    # never a GAType, and `* Unit` / `/ Unit` on GAType are refused.
    from numga.extensor import Extensor

    assert isinstance(Origin + Direction, Extensor)
    assert isinstance(Plane + Bivector, Extensor)
    assert isinstance(Scalar + Scalar, Extensor)
    with pytest.raises(GATypeError):
        _ = Point + Plane
    with pytest.raises(TypeError):
        _ = Scalar * dim.time
    with pytest.raises(TypeError):
        _ = Scalar / dim.time
    with pytest.raises(TypeError):
        _ = dim.time * Scalar
    with pytest.raises(TypeError):
        _ = dim.time / Scalar

    # ga.gatype(expr) preserves UnitGrid from a value expression.
    assert PGA3D.gatype(mv.point([1.0, 2.0, 3.0], unit=dim.length)) == WorldPoint
    assert PGA3D.gatype(mv.scalar(2.0, unit=dim.time)) == Duration

    assert _blade_units(Rate) == {
        "yz": 1 / dim.time, "xz": 1 / dim.time, "xy": 1 / dim.time,
        "wx": dim.length / dim.time, "wy": dim.length / dim.time, "wz": dim.length / dim.time,
    }
    # `.units` is always a UnitGrid; `.unit` returns a Unit for uniform types and raises otherwise.
    assert isinstance(Inertia.units, UnitGrid)
    assert Inertia.has_units and Inertia.units.uniform is None
    with pytest.raises(GATypeError):
        _ = Inertia.unit
    with pytest.raises(GATypeError):
        _ = Rate.unit
    assert "1/time | length/time" in Rate.signature
    assert Rate != Bivector
    reconstructor, reduce_args = Inertia.__reduce__()
    assert reconstructor(*reduce_args) == Inertia

    # 0D Python scalars and typed zeros.
    dt = mv(Duration, 0.025)
    assert dt.shape == ()
    assert dt.gatype == Duration
    assert dt.unit == dim.time
    assert dt.to_array() == pytest.approx(0.025)
    assert mv.scalar(0.025, unit=dim.time).gatype == Duration

    zero_rates = mv.zeros(Rate, (4,))
    assert zero_rates.shape == (4,)
    assert zero_rates.gatype == Rate
    np.testing.assert_array_equal(zero_rates.kernel, np.zeros((4, 6)))

    # Neutral mv.zeros stamps structural zero so accumulators need no unit alias at runtime.
    poly_zero = mv.zeros(Bivector, (4,))
    assert poly_zero.has_units and poly_zero.units.is_zero
    with pytest.raises(GATypeError):
        _ = poly_zero.unit
    assert (poly_zero + zero_rates).gatype == Rate
    assert (-poly_zero / 2).exp().gatype.is_rotor

    # `.derive.projective(unit)` and `.derive.scaled_unit(unit)` build projective types cleanly.
    force_dim = dim.mass * dim.length / dim.time**2
    assert Antibivector.derive.projective(dim.length).derive.scaled_unit(force_dim) == Forque
    assert Bivector.derive.projective(dim.length).derive.scaled_unit(1 / dim.time) == Rate
    proj_motor = PGA3D.gatype.rotor().derive.projective(dim.length)
    assert proj_motor.is_rotor
    assert _blade_units(proj_motor) == {
        "1": dim.none, "yz": dim.none, "xz": dim.none, "xy": dim.none,
        "wx": dim.length, "wy": dim.length, "wz": dim.length, "wxyz": dim.length,
    }
    with pytest.raises(GATypeError):
        VGA3D.gatype.vector().derive.projective(dim.length)
    with pytest.raises(GATypeError):
        Algebra("x+y+w0s0").gatype.vector().derive.projective(dim.length)

    # Extensor * Unit and Extensor / Unit scale every nonzero blade unit.
    assert (dt / dim.time).unit is dim.none
    assert ((dt / dim.time) * dim.length).gatype == Length
    assert (mv.x / dim.length).unit == 1 / dim.length
    assert (zero_rates * dim.time).gatype == Screw

    # ReverseProductOne refuses non-dimensionless scalar/rotation blades while permitting nilpotent translations.
    with pytest.raises(GATypeError):
        PGA3D.gatype.rotor().derive.unit(dim.length)
    with pytest.raises(GATypeError):
        mv.rotor(unit=dim.length)
    trans_motor = (0.5 * mv.wx * dim.length).exp()
    assert trans_motor.gatype.is_rotor
    assert _blade_units(trans_motor.gatype) == {"1": dim.none, "wx": dim.length}

    # cast and retyped preserve UnitGrid on unchecked aliases, fill zero-padded target blade units
    # on checked targets, and refuse conflicting explicit units.
    r = mv(Rate, [1, 0, 0, 0, 0, 0])
    assert _blade_units(r.cast(PGA3D.gatype.even()).gatype) == _blade_units(Rate)
    assert r.cast(Rate).gatype == Rate
    assert r.retyped(Bivector).gatype == Rate
    assert r.retyped(Rate).gatype == Rate
    full_motor_type = PGA3D.gatype.rotor().derive.projective(dim.length)
    embedded_motor = trans_motor.cast(full_motor_type)
    assert embedded_motor.gatype.units == full_motor_type.units
    batch_motors = embedded_motor[None].broadcast_to((2,))
    # Functional .at.set merges sparser compatible blade units and refuses conflicting blade units.
    sparse_motor = trans_motor.cast(PGA3D.gatype.even())
    assert batch_motors.at[1].set(sparse_motor).gatype.units == full_motor_type.units
    wrong_motor = (0.5 * mv.wx * dim.time).exp().cast(PGA3D.gatype.even())
    with pytest.raises(GATypeError):
        batch_motors.at[1].set(wrong_motor)
    with pytest.raises(GATypeError):
        r.cast(Bivector.derive.unit(dim.length))
    with pytest.raises(GATypeError):
        r.cast(PGA3D.gatype.bivector(unit=dim.none))
    with pytest.raises(GATypeError):
        r.retyped(Bivector.derive.unit(dim.length))
    # clip accepts scale-invariant open bounds (0, +/-inf) while rejecting finite dimensionless bounds.
    d_len = mv.scalar(2.5, unit=dim.length)
    assert d_len.clip(0.0, np.inf).unit == dim.length
    assert (-d_len).clip(-np.inf, 0.0).unit == dim.length
    with pytest.raises(GATypeError):
        d_len.clip(0.0, 1.0)


# --- Arithmetic, literal zero, nilpotent exp, and centralized ExtensionMethod -----------

def test_arithmetic_literal_zero_nilpotent_exp_and_transcendentals():
    mv = NumpyContext(PGA3D, units=True).multivector
    t = _pga3_types()
    Scalar, Rate, Screw = t["Scalar"], t["Rate"], t["Screw"]
    Length = Scalar.derive.unit(dim.length)
    Duration = Scalar.derive.unit(dim.time)
    Gravity = Scalar.derive.unit(dim.length / dim.time**2)

    d1, d2 = mv(Length, 3.0), mv(Length, 4.0)
    dt = mv(Duration, 2.0)

    # Literal Python 0 / 0.0 acts as additive zero and sign threshold; booleans and non-zeros do not.
    assert (d1 + d2).unit == dim.length
    assert (d1 + 0).unit == dim.length
    assert (0.0 + d1).unit == dim.length
    assert sum([d1, d2]).to_array() == pytest.approx(7.0)
    assert sum([d1, d2]).unit == dim.length
    assert (d1 / dt).unit == dim.length / dim.time
    assert d1.abs().unit == dim.length
    assert bool(d1 > 0) and bool(d1 < d2)
    assert d1.clip(0, mv(Length, 2.0)).unit == dim.length
    assert where(d1 > 0, d1, 0).unit == dim.length

    for bad in (
        lambda: d1 + dt,
        lambda: d1 + 1.0,
        lambda: d1 + False,
        lambda: d1 + np.zeros(()),
        lambda: d1 > 1.0,
        lambda: d1 < dt,
        lambda: d1.clip(0, 2.0),
        lambda: d1.clip(0, dt),
        lambda: stack([d1, dt]),
    ):
        with pytest.raises(GATypeError):
            bad()

    # Square roots halve rational exponents.
    assert (d1 * d1 + d2 * d2).square_root().to_array() == pytest.approx(5.0)
    assert (d1 * d1 + d2 * d2).square_root().unit == dim.length
    assert (d1 * d1).inverse_square_root().unit == 1 / dim.length
    assert d1.square_root().unit == dim.length ** Fraction(1, 2)
    g = mv(Gravity, 9.8)
    assert (g.square_root() * d1.square_root()).unit == dim.length / dim.time

    # Scalar transcendentals require dimensionless input.
    ratio = d1 / d2
    assert ratio.sin().unit is dim.none
    assert ratio.exp().unit is dim.none
    assert ratio.log().unit is dim.none
    for method in ("sin", "cos", "tan", "sinh", "cosh", "tanh", "exp", "log"):
        with pytest.raises(GATypeError):
            getattr(d1, method)()

    # Nilpotent exp succeeds for both position shifts and Galilean velocity boosts in the SAME PGA3D;
    # applying a velocity boost to a position point is refused, and rotating metres into pixels is refused.
    pos_shift = (0.5 * d1 * mv.wx).exp()
    vel_boost = (0.5 * (d1 / dt) * mv.wx).exp()
    assert _blade_units(pos_shift.gatype) == {"1": dim.none, "wx": dim.length}
    assert _blade_units(vel_boost.gatype) == {"1": dim.none, "wx": dim.length / dim.time}
    p_pos = mv.point([1.0, 2.0, 3.0], unit=dim.length)
    p_shifted = pos_shift >> p_pos
    assert p_shifted.gatype == p_pos.gatype
    np.testing.assert_allclose((mv.x & p_shifted).to_array(), 1.0 - 3.0, atol=1e-12)
    with pytest.raises(GATypeError):
        _ = vel_boost >> p_pos

    # Rotating metres into pixels fails power-series closure (`xy * wx = wy` conflicts with pixel).
    mixed_xy_m_px = 0.3 * mv.xy + 2.0 * mv.wx * dim.length + 5.0 * mv.wy * dim.pixel
    with pytest.raises(GATypeError):
        mixed_xy_m_px.exp()

    # Dimensioned rotation rates are refused by every power-series exp method.
    rate = mv(Rate, [0.3, -0.4, 1.2, 0.5, -0.2, 0.1])
    for method in ("exp", "exp_quadratic", "exp_bisect", "exp_derivative"):
        with pytest.raises(GATypeError):
            getattr(rate, method)()

    # Centralized ExtensionMethod enforcement holds even with invariant_decomposition and optimized registered.
    from numga.extensor import Extensor

    saved = [
        (
            m._dispatch,
            list(m._dispatch._registrations),
            list(m._dispatch._predicates),
            list(m._dispatch._field_predicates),
        )
        for m in (Extensor.exp, Extensor.log, Extensor.normalized)
    ]
    try:
        invariant_decomposition.register()
        optimized.register()
        with pytest.raises(GATypeError):
            rate.exp()
        with pytest.raises(GATypeError):
            (mv.xy * dim.length).exp()
        with pytest.raises(GATypeError):
            (mv.rotor() * dim.length).log()
        with pytest.raises(GATypeError):
            mixed_xy_m_px.exp()
        # Proper screw displacement `rate * dt` succeeds through exp and log with exact blade units.
        motor = (rate * dt).exp()
        assert _blade_units(motor.gatype) == {
            "1": dim.none, "yz": dim.none, "xz": dim.none, "xy": dim.none,
            "wx": dim.length, "wy": dim.length, "wz": dim.length, "wxyz": dim.length,
        }
        logged = motor.log()
        assert logged.gatype == Screw
        np.testing.assert_allclose(logged.kernel, (rate * dt).kernel, atol=1e-10)
    finally:
        for dispatch, regs, preds, field_preds in saved:
            dispatch._registrations[:] = regs
            dispatch._predicates[:] = preds
            dispatch._field_predicates[:] = field_preds
            dispatch._cache.clear()


# --- Per-blade duality and interior products --------------------------------------------

def test_dual_and_interior_products_per_blade():
    mv = NumpyContext(PGA3D, units=True).multivector
    Length = PGA3D.gatype.scalar(unit=dim.length)

    p = mv.point([3.0, 4.0, 0.0], unit=dim.length)
    d = mv.direction([3.0, 4.0, 0.0], unit=dim.length)
    plane = 0.6 * mv.x + 0.8 * mv.y - mv(Length, 5.0) * mv.w

    # Duality is a pure blade permutation on UnitGrid; round-tripping restores exact per-blade units.
    p_dual = p.dual()
    assert _blade_units(p_dual.gatype) == {
        "w": dim.none, "x": dim.length, "y": dim.length, "z": dim.length,
    }
    assert p_dual.dual_inverse().gatype == p.gatype
    np.testing.assert_allclose(p_dual.dual_inverse().kernel, p.kernel)

    d_dual = d.dual()
    assert d_dual.unit == dim.length
    assert d_dual.dual_inverse().unit == dim.length
    np.testing.assert_allclose(d_dual.dual_inverse().kernel, d.kernel)

    # Regressive pairing of affine plane `{x/y: 1, w: L}` with point `{w.dual: 1, space: L}` is homogeneous L.
    assert (plane & p).unit == dim.length
    assert (plane & p).to_array() == pytest.approx(0.0)

    # Interior product preserves blade units and rejects conflicting contributions to the same blade.
    assert mv.w.left_interior(mv.wx * dim.length).unit == dim.length
    assert mv.x.left_interior(mv.wx * dim.length).unit == dim.length
    bad_vector = mv.x * dim.time + mv.y * dim.length
    with pytest.raises(GATypeError):
        plane.left_interior(bad_vector)


# --- PGA3D projective geometry ----------------------------------------------------------

def test_pga3d_geometry_and_simplex_gradient():
    mv = NumpyContext(PGA3D, units=True).multivector
    Scalar = PGA3D.gatype.scalar()
    Length = Scalar.derive.unit(dim.length)
    Temperature = Scalar.derive.unit(dim.temperature)

    origin = mv.w.dual()
    p = mv.point([3.0, 4.0, 0.0], unit=dim.length)
    p_explicit = origin + mv(Length, 3.0) * mv.x.dual() + mv(Length, 4.0) * mv.y.dual()
    np.testing.assert_allclose(p.kernel, p_explicit.cast(p.subspace).kernel)
    assert _blade_units(p_explicit.gatype) == {"xyz": dim.none, "wyz": dim.length, "wxz": dim.length}
    assert p_explicit.retyped(p.gatype).gatype == p.gatype
    assert p.norm().unit is dim.none
    assert p.antinorm().unit == dim.length
    assert p.antinorm().to_array() == pytest.approx(5.0)

    plane = 0.6 * mv.x + 0.8 * mv.y - mv(Length, 10.0) * mv.w
    signed_dist = plane & p
    assert signed_dist.unit == dim.length
    assert signed_dist.to_array() == pytest.approx(-5.0)

    target = mv.point([1.0, 2.0, 2.0], unit=dim.length)
    disp = (target & origin) ^ mv.w
    assert disp.unit == dim.length
    assert disp.antinorm().unit == dim.length
    assert disp.antinorm().to_array() == pytest.approx(3.0)

    translator = (1 + target * origin.inverse()) / 2
    assert _blade_units(translator.gatype) == {
        "1": dim.none, "wx": dim.length, "wy": dim.length, "wz": dim.length,
    }
    moved = translator >> origin
    assert moved.gatype == target.gatype
    np.testing.assert_allclose(moved.kernel, target.kernel)

    line = origin & p
    assert set(_blade_units(line.gatype).values()) == {dim.length}
    assert line.norm().unit == dim.length
    assert line.normalized().unit is dim.none

    p0, p1 = mv.point([0.0, 0.0, 0.0], unit=dim.length), mv.point([1.0, 0.0, 0.0], unit=dim.length)
    p2, p3 = mv.point([0.0, 2.0, 0.0], unit=dim.length), mv.point([0.0, 0.0, 3.0], unit=dim.length)
    volume = p0 & p1 & p2 & p3
    assert volume.unit == dim.length**3
    f0, f1, f2, f3 = (mv(Temperature, v) for v in (300.0, 310.0, 320.0, 330.0))
    grad = -(
        f0 * (p1 & p2 & p3)
        - f1 * (p0 & p2 & p3)
        + f2 * (p0 & p1 & p3)
        - f3 * (p0 & p1 & p2)
    ) / volume
    assert _blade_units(grad.gatype) == {
        "x": dim.temperature / dim.length,
        "y": dim.temperature / dim.length,
        "z": dim.temperature / dim.length,
        "w": dim.temperature,
    }
    probe = mv.point([0.5, 1.0, 1.5], unit=dim.length)
    interpolated = grad & probe
    assert interpolated.unit == dim.temperature
    assert interpolated.to_array() == pytest.approx(330.0)


# --- PGA3D rigid-body mechanics, spectra, forms, rank-1 checks, outermorphism, trace ----

def test_pga3d_mechanics_rattleback_step_and_spectral_rules():
    mv = NumpyContext(PGA3D, units=True).multivector
    t = _pga3_types()
    Scalar, Plane, Point, Bivector = t["Scalar"], t["Plane"], t["Point"], t["Bivector"]
    Screw, Rate, Forque, Momentum, Inertia = t["Screw"], t["Rate"], t["Forque"], t["Momentum"], t["Inertia"]
    Spin = Bivector.derive.nondegenerate
    Length = Scalar.derive.unit(dim.length)
    Mass = Scalar.derive.unit(dim.mass)
    Duration = Scalar.derive.unit(dim.time)
    Stiffness = Scalar.derive.unit(dim.mass / dim.time**2)

    rng = np.random.default_rng(11)
    pts = mv.point(rng.uniform(-0.1, 0.1, (16, 3)), unit=dim.length)
    masses = mv(Mass, rng.uniform(0.2, 0.8, (16, 1)))
    inertia = (pts * masses).regressive(pts.commutator(Bivector)).sum(axis=0)
    assert all(Inertia.units.get(idx) == u for idx, u in inertia.gatype.units.items)

    rate = mv(Rate, [0.3, -0.4, 1.2, 0.5, -0.2, 0.1])
    momentum = inertia(rate)
    assert momentum.gatype == Momentum

    energy = (momentum & rate) / 2
    assert energy.unit == dim.mass * dim.length**2 / dim.time**2

    dt = mv(Duration, 0.01)
    motor = (rate * dt).exp()
    assert motor.gatype.is_rotor
    assert _blade_units(motor.gatype)["1"] is dim.none
    assert _blade_units(motor.gatype)["wx"] == dim.length

    screw_norm = rate.norm()
    assert _blade_units(screw_norm.gatype) == {"1": 1 / dim.time, "wxyz": dim.length / dim.time}
    direction, scale = rate.decompose_polar()
    assert direction.gatype == Screw
    assert _blade_units(scale.gatype) == {"1": 1 / dim.time, "wxyz": dim.length / dim.time}
    assert (direction * scale).gatype == Rate
    np.testing.assert_allclose((direction * scale).kernel, rate.kernel)

    recovered = inertia.inverse()(momentum)
    assert recovered.gatype == Rate
    np.testing.assert_allclose(recovered.kernel, rate.kernel)
    assert inertia.solve(momentum).gatype == Rate
    assert (inertia(rate).commutator(rate)).gatype == Forque

    Moments = Spin.derive.unit(dim.mass * dim.length**2)
    p_inertia = principal_inertia_pga3(mv(Mass, 2.0), mv(Moments, [0.1, 0.2, 0.3]))
    assert p_inertia(rate).gatype == Momentum
    assert p_inertia.inverse()(p_inertia(rate)).gatype == Rate
    # A principal inertia built for dimensionless lengths conflicts with a metre-typed rate on energy and commutator.
    bad_inertia = principal_inertia_pga3(mv(Mass, 2.0), mv(Spin.derive.unit(dim.mass), [0.1, 0.2, 0.3]))
    with pytest.raises(GATypeError):
        _ = bad_inertia(rate) & rate
    with pytest.raises(GATypeError):
        _ = bad_inertia(rate).commutator(rate)

    stiffness = (pts * mv(Stiffness, rng.uniform(10.0, 20.0, (16, 1)))).regressive(pts.commutator(Bivector)).sum(axis=0)
    K_form = Bivector & stiffness
    M_form = Bivector & inertia
    omega_sq, modes = K_form.eigh(M_form)
    assert omega_sq.unit == 1 / dim.time**2
    assert modes.gatype == Screw.derive.scaled_unit(1 / (dim.mass.root(2) * dim.length))
    assert M_form(modes, modes).unit is dim.none
    np.testing.assert_allclose(M_form(modes, modes).to_array(), np.ones(6), atol=1e-10)
    assert K_form(modes, modes).unit == 1 / dim.time**2
    np.testing.assert_allclose(K_form(modes, modes).to_array(), omega_sq.to_array(), atol=1e-10)
    assert omega_sq.square_root().unit == 1 / dim.time

    solved = K_form.lstsq(K_form(rate), M_form)
    assert solved.gatype == Rate
    np.testing.assert_allclose(solved.kernel, rate.kernel, atol=1e-10)

    # Schur-complement PGA eigh carries exact per-blade units `{rot: 1, trans: length}` on modes.
    schur_vals, schur_modes = K_form.eigh()
    assert schur_vals.unit == dim.mass * dim.length**2 / dim.time**2
    assert schur_modes.gatype == Screw

    endo = inertia.inverse()(stiffness)
    assert endo.det().unit == 1 / dim.time**12
    assert endo.trace().unit == 1 / dim.time**2
    vals, vecs = endo.eig()
    assert vals.unit == 1 / dim.time**2
    assert endo(vecs).gatype == (vals * vecs).gatype

    f_point = ((motor >> Point) * 3.0) / dim.time
    assert (Plane & f_point).trace(1, 2).unit == 1 / dim.time
    np.testing.assert_allclose((Plane & f_point).trace(1, 2).kernel, f_point.trace().kernel)
    probe_plane = 0.6 * mv.x + 0.8 * mv.y - mv(Length, 2.0) * mv.w
    assert (f_point.adjugate()(probe_plane) & pts[0]).gatype == (probe_plane & f_point(pts[0])).gatype
    np.testing.assert_allclose(
        (f_point.adjugate()(probe_plane) & pts[0]).to_array(),
        (probe_plane & f_point(pts[0])).to_array(),
        atol=1e-12,
    )

    # Euclidean-metric methods refuse non-uniform endomorphisms and mixed bilinear forms.
    for method in ("eigh", "eigvalsh", "cholesky", "svd", "svdvals", "pinv"):
        with pytest.raises(GATypeError):
            getattr(endo, method)()
    with pytest.raises(GATypeError):
        K_form.cholesky()

    spin_form = Spin & stiffness(Spin)
    assert spin_form.cholesky().unit == dim.mass.root(2) * dim.length / dim.time

    # Rank-1 matrix refusal: inhomogeneous `[[1, L], [1, 1]]` has incompatible paths and is refused
    # by det, inverse, and solve; block-diagonal maps with distinct sector units succeed.
    vmv = NumpyContext(VGA3D, units=True).multivector
    V = VGA3D.gatype.vector()
    inhom = (
        vmv.x * (vmv.x | V)
        + vmv.x * (vmv.y | V) * dim.length
        + vmv.y * (vmv.x | V)
        + vmv.y * (vmv.y | V)
        + vmv.z * (vmv.z | V)
    )
    assert inhom.gatype.subspaces == (V.output_subspace, V.output_subspace)
    for bad_op in (
        lambda: inhom.det(),
        lambda: inhom.inverse(),
        lambda: inhom.solve(vmv.x + vmv.y + vmv.z),
    ):
        with pytest.raises(GATypeError):
            bad_op()

    block_diag = (
        vmv.x * (vmv.x | V) * dim.mass
        + (vmv.y * (vmv.y | V) + vmv.z * (vmv.z | V)) * dim.time
    )
    assert block_diag.det().unit == dim.mass * dim.time**2
    inv_block = block_diag.inverse()
    rhs_block = vmv.x * dim.mass + (vmv.y + vmv.z) * dim.time
    assert inv_block(rhs_block).unit is dim.none
    assert block_diag.solve(rhs_block).unit is dim.none

    spd_unary = vmv(VGA3D.gatype.scalar().derive.unit(dim.length**2), 4.0) * V
    chol = spd_unary.cholesky()
    assert chol.unit == dim.length
    np.testing.assert_allclose(chol(chol).kernel, spd_unary.kernel)
    assert spd_unary.eigh()[0].unit == dim.length**2
    assert spd_unary.svd()[1].unit == dim.length**2
    assert spd_unary.pinv().unit == 1 / dim.length**2

    # Outermorphism on a correlation in PGA3D.
    mirrors = stack((mv.x, mv.y, mv.z))
    radius = mv(Length, 2.0)
    ball = (mirrors * (mirrors & Point)).sum(axis=-1) / radius**2 - mv.w * (mv.w & Point)
    Line = PGA3D.gatype.antibivector()
    ball_line = ball.outermorphism(Line)
    p1, p2 = mv.point([1.0, 0.5, -0.3], unit=dim.length), mv.point([-0.2, 1.1, 0.7], unit=dim.length)
    lhs = ball_line(p1 & p2)
    rhs = ball(p1) ^ ball(p2)
    assert lhs.gatype.units == rhs.gatype.units
    assert _blade_units(lhs.gatype) == {
        "yz": 1 / dim.length**2, "xz": 1 / dim.length**2, "xy": 1 / dim.length**2,
        "wx": 1 / dim.length, "wy": 1 / dim.length, "wz": 1 / dim.length,
    }
    np.testing.assert_allclose(lhs.kernel, rhs.kernel, atol=1e-12)


# --- Metric-free SI Maxwell without c ---------------------------------------------------

def test_metric_free_si_maxwell_without_c():
    """In Model B, the metric-free SI Maxwell equations, rest-frame constitutive map (and its
    inverse), and stress-energy tensor all typecheck with exact SI units and zero factors of `c`."""
    mv = NumpyContext(STA, units=True).multivector
    space = Algebra("x+y+z+")
    time_alg = Algebra("t-")

    Vector = STA.gatype.vector()
    Bivector = STA.gatype.bivector()
    Antibivector = STA.gatype.antibivector()
    Direction = STA.gatype.antivector()

    TimeCovector = Vector.derive.within(time_alg)
    SpaceCovector = Vector.derive.within(space)
    ElectricPlane = Bivector.derive.touching(time_alg)
    MagneticPlane = Bivector.derive.within(space)
    ElectricPatch = ElectricPlane.derive.dual
    MagneticPatch = MagneticPlane.derive.dual

    # Pure SI units: no c anywhere.
    E_dim = dim.voltage / dim.length                          # V/m
    B_dim = dim.voltage * dim.time / dim.length**2            # T = V*s/m^2
    D_dim = dim.charge / dim.length**2                        # C/m^2
    H_dim = dim.charge / (dim.length * dim.time)              # A/m
    eps_dim = D_dim / E_dim                                   # F/m = C/(V*m)
    inv_mu_dim = H_dim / B_dim                                # m/H = C/(V*s^2)
    energy_density = dim.charge * dim.voltage / dim.length**3 # J/m^3

    Wavevector = STA.gatype(
        Vector.output_subspace,
        orientation=Vector.orientation,
        unit={TimeCovector: 1 / dim.time, SpaceCovector: 1 / dim.length},
    )
    EMField = STA.gatype(
        Bivector.output_subspace,
        orientation=Bivector.orientation,
        unit={ElectricPlane: E_dim, MagneticPlane: B_dim},
    )
    EMExcitation = STA.gatype(
        Antibivector.output_subspace,
        orientation=Antibivector.orientation,
        unit={ElectricPatch: D_dim, MagneticPatch: H_dim},
    )

    k = mv.scalar(2.0, unit=1 / dim.time) * mv.t - mv.scalar(3.0, unit=1 / dim.length) * mv.z
    assert k.retyped(Wavevector).gatype == Wavevector
    F = mv.scalar(5.0, unit=E_dim) * mv.xt + mv.scalar(0.2, unit=B_dim) * mv.yz
    assert F.retyped(EMField).gatype == EMField

    # Rest-frame constitutive map: pure eps on E-dyads and 1/mu on B-dyads, no c anywhere.
    time_dir = -mv.t.dual()
    axes = stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))
    swept = axes & time_dir
    patches = stack((mv.yz, mv.zx, mv.xy)).dual()
    electric_dyads = -swept * (Bivector & swept)
    magnetic_dyads = patches * (Bivector & patches)

    eps = mv.scalar(8.854e-12, unit=eps_dim)
    inv_mu = mv.scalar(1.0 / (4e-7 * np.pi), unit=inv_mu_dim)
    medium = (eps * electric_dyads + inv_mu * magnetic_dyads).sum(axis=0)
    G = medium(F)
    assert G.retyped(EMExcitation).gatype == EMExcitation

    # Block-diagonal constitutive map inverts cleanly back to EMField!
    F_back = medium.inverse()(G)
    assert F_back.retyped(EMField).gatype == EMField
    np.testing.assert_allclose(F_back.cast(F.subspace).kernel, F.kernel, rtol=1e-12)

    # Metric-free Maxwell exterior products and Lagrangian density.
    dF = k ^ F
    assert _blade_units(dF.gatype)["xzt"] == dim.voltage / dim.length**2
    dG = k ^ G
    assert _blade_units(dG.gatype)["yzt"] == dim.charge / (dim.length**2 * dim.time)

    lagrangian = -(F & G) / 2
    assert lagrangian.gatype.unit == energy_density

    # Stress-energy tensor: exact SI energy density, Poynting flux, momentum density, and Maxwell stress!
    T = ((F & G) / 2) * Direction - ((Direction & F) ^ G)
    assert (mv.t & T(time_dir)).unit == energy_density
    assert (mv.z & T(time_dir)).unit == energy_density * dim.length / dim.time   # Poynting flux W/m^2
    assert (mv.t & T(mv.z.dual())).unit == energy_density * dim.time / dim.length  # momentum density
    assert (mv.z & T(mv.z.dual())).unit == energy_density                        # Maxwell stress N/m^2


# --- Multi-null thermodynamics, topology, fields, sparse, and JAX -----------------------

def test_multi_null_thermodynamics_sparse_and_fields():
    # Zero units on Algebra: spatial coordinates carry dim.length with zero entropy/volume pollution.
    multi = Algebra("x+y+s0v0")
    mmv = NumpyContext(multi, units=True).multivector
    space = Algebra("x+y+")
    UprightMulti = multi.gatype.vector().derive.within(space)
    SpatialMulti = UprightMulti.derive.dual
    place_m = mmv(UprightMulti.derive.unit(dim.length), [3.0, 4.0]).dual()
    assert place_m.gatype.unit == dim.length
    assert place_m.antinorm().gatype.unit == dim.length
    assert place_m.antinorm().to_array() == pytest.approx(5.0)
    assert place_m.dual().gatype.unit == dim.length

    # Rubber-band algebra x+y+ * s0 without algebra-level units.
    ga = space * Algebra("s0")
    mv = NumpyContext(ga, units=True).multivector
    Scalar = ga.gatype.scalar()
    Plane = ga.gatype.vector()
    Upright = Plane.derive.within(space)
    Spatial = Upright.derive.dual
    Displacement = Upright.derive.unit(dim.length)
    entropy = mv.s.dual()

    dilation = Upright * (Upright & Spatial)
    shear = 2 * ((Upright | Upright) * Spatial.dual_inverse()) - (Upright | (Upright ^ Spatial.dual_inverse()))
    assert dilation.unit is dim.none and shear.unit is dim.none

    topo = Topology(4, np.array([[0, 1, 2], [1, 3, 2]]))
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    places = mv(Displacement, xy).dual().field()
    assert places.unit == dim.length
    sides = topo.boundary[2, 1] * (topo.d[0] * places)
    double_areas = (sides[0] & sides[1]) & entropy
    gradients = (entropy & sides) / double_areas
    areas = double_areas.norm() / 2
    assert areas.unit == dim.length**2
    assert gradients.unit == 1 / dim.length

    # Sparse maps scaled by a unit keep exact units; site-free scalars scale Sparse/SparseExtensor.
    Rate = Scalar.derive.unit(1 / dim.time)
    scaled_id = spdiag(mv(Rate, np.ones((4, 1))).field())
    assert (scaled_id * places).unit == dim.length / dim.time

    Conductance = Scalar.derive.unit(dim.energy / (dim.time * dim.temperature))
    kappa = mv(Conductance, 1.0)
    assert (kappa * scaled_id).cells.unit == dim.energy / (dim.temperature * dim.time**2)
    assert (scaled_id * kappa).cells.unit == dim.energy / (dim.temperature * dim.time**2)
    assert (scaled_id / dim.length).cells.unit == 1 / (dim.length * dim.time)
    assert (topo.d[0] * kappa).cells.unit == dim.energy / (dim.temperature * dim.time)
    assert (kappa * topo.d[0]).cells.unit == dim.energy / (dim.temperature * dim.time)

    # SparseExtensor solve and generalized eigh propagate exact rational units.
    Stiffness = Scalar.derive.unit(dim.mass / dim.time**2)
    Mass = Scalar.derive.unit(dim.mass)
    Force = Upright.derive.unit(dim.mass * dim.length / dim.time**2)
    stiff_cells = spdiag(mv(Stiffness, np.array([[2.0], [3.0], [4.0], [5.0]])).field()) * Upright
    mass_cells = spdiag(mv(Mass, np.array([[1.0], [1.0], [2.0], [2.0]])).field()) * Upright
    rhs_field = mv(Force, np.ones((4, 2))).field()
    sol = stiff_cells.solve(rhs_field)
    assert sol.unit == dim.length
    eigvals, eigfields = stiff_cells.eigh(mass_cells, n_modes=2)
    assert eigvals.unit == 1 / dim.time**2
    assert eigfields.unit == 1 / dim.mass.root(2)
    assert eigfields.scalar_product(mass_cells(eigfields)).sites.sum().unit is dim.none

    dense_k, dense_m = stiff_cells.to_dense(), mass_cells.to_dense()
    assert dense_k.solve(rhs_field).unit == dim.length
    assert dense_k.inverse().unit == dim.time**2 / dim.mass
    assert dense_k.trace().unit == dim.mass / dim.time**2
    assert dense_k.det().unit == dim.mass**8 / dim.time**16
    f_vals, f_modes = dense_k.eigh(dense_m)
    assert f_vals.unit == 1 / dim.time**2
    assert f_modes.unit == 1 / dim.mass.root(2)


def test_magnetotransport_two_null_units():
    ga = Algebra("x+y+t-w0") * Algebra("s0")
    quantities = Algebra("t-s0")
    mv = NumpyContext(ga, units=True).multivector
    Scalar = ga.gatype.scalar()
    Plane = ga.gatype.vector()
    Parcel = Plane.derive.within(quantities).derive.dual

    time = -mv.t.dual()
    entropy = mv.s.dual()
    Thermopower = Scalar.derive.unit(dim.voltage / dim.temperature)
    Voltage = Scalar.derive.unit(dim.voltage)
    Temperature = Scalar.derive.unit(dim.temperature)
    Mobility = Scalar.derive.unit(dim.length**2 / (dim.voltage * dim.time))
    MagneticStrength = Scalar.derive.unit(dim.voltage * dim.time / dim.length**2)
    SheetDensity = Scalar.derive.unit(dim.charge / dim.length**2)

    parcel = time + mv(Thermopower, 0.2) * entropy
    assert parcel.gatype.output_subspace == Parcel.output_subspace
    assert _blade_units(parcel.gatype) == {
        "xyws": dim.none, "xytw": dim.voltage / dim.temperature,
    }

    potential = mv(Voltage, 1.5) * mv.t + mv(Temperature, 300.0) * mv.s
    assert _blade_units(potential.gatype) == {"t": dim.voltage, "s": dim.temperature}
    assert (potential & parcel).unit == dim.voltage

    drive = -((0.1 * mv.x / dim.length) ^ potential)
    assert _blade_units(drive.gatype) == {
        "xt": dim.voltage / dim.length,
        "xs": dim.temperature / dim.length,
    }
    mobility = mv(Mobility, 0.05)
    magnetic = mv(MagneticStrength, 0.3) * mv.xy
    hall = 1 - mobility * magnetic
    assert hall.unit is dim.none
    drift = hall.inverse() * (mobility * drive.commutator(parcel))
    assert drift.unit == dim.length / dim.time
    flux = mv(SheetDensity, 1.0) * (drift & parcel)
    dissipation = drive & flux
    assert dissipation.unit == dim.charge * dim.voltage / (dim.length**2 * dim.time)


def test_jax_jit_scan_and_derivative_preserve_units():
    jax = pytest.importorskip("jax")
    from numga.backend.jax import JaxContext, derivative

    mv = JaxContext(PGA3D, units=True).multivector
    t = _pga3_types()
    Scalar, Bivector, Rate, Momentum, Inertia = (
        t["Scalar"], t["Bivector"], t["Rate"], t["Momentum"], t["Inertia"],
    )
    Mass = Scalar.derive.unit(dim.mass)
    Duration = Scalar.derive.unit(dim.time)

    pts = mv.point(np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]), unit=dim.length)
    masses = mv(Mass, np.array([[1.0], [2.0], [3.0]]))
    inertia = (pts * masses).regressive(pts.commutator(Bivector)).sum(axis=0)
    rate = mv(Rate, [0.2, -0.3, 0.5, 0.1, 0.4, -0.2])

    @jax.jit
    def kinetic_energy(r):
        return (r & inertia(r)) / 2

    E = kinetic_energy(rate)
    dE = derivative(kinetic_energy)(rate)
    ddE = derivative(derivative(kinetic_energy))(rate)
    assert E.unit == dim.mass * dim.length**2 / dim.time**2
    # dE is a covector `Scalar <- Rate`: its blade units match `Momentum` (energy / Rate).
    assert dE(rate).unit == E.unit
    assert ddE(rate, rate).unit == E.unit
    np.testing.assert_allclose(dE(rate).to_array(), 2 * E.to_array(), rtol=1e-12)
    np.testing.assert_allclose(ddE(rate, rate).to_array(), 2 * E.to_array(), rtol=1e-12)
    # Pairing `dE` back to a bivector/antibivector yields Momentum units `{rot: M L^2/T, trans: M L/T}`.
    dE_grid = {
        Bivector.output_subspace.algebra.blade_name(Bivector.output_subspace.masks[idx]): u
        for idx, u in dE.gatype.units.items
    }
    assert dE_grid == {
        "yz": dim.mass * dim.length**2 / dim.time,
        "xz": dim.mass * dim.length**2 / dim.time,
        "xy": dim.mass * dim.length**2 / dim.time,
        "wx": dim.mass * dim.length / dim.time,
        "wy": dim.mass * dim.length / dim.time,
        "wz": dim.mass * dim.length / dim.time,
    }
    # Differentiating with respect to a polymorphic zero (`mv.zeros(Bivector)`) cannot determine
    # per-blade input units and must raise instead of tagging a nonzero Jacobian as structural zero.
    with pytest.raises(GATypeError):
        derivative(lambda tw: (tw * 0.5).exp() >> pts[0])(mv.zeros(Bivector))
    typed_zero_twist = mv.zeros(Bivector.derive.projective(dim.length))
    d_moved = derivative(lambda tw: (tw * 0.5).exp() >> pts[0])(typed_zero_twist)
    assert d_moved(typed_zero_twist).gatype.units == pts[0].gatype.units

    dt = mv(Duration, 0.01)
    Motor = (rate * dt).exp().gatype

    def step(carry, _):
        motor, r = carry
        return ((motor * (r * dt).exp()).retyped(Motor), r), kinetic_energy(r)

    (final_motor, final_rate), energies = jax.lax.scan(step, (mv.rotor().retyped(Motor), rate), None, length=5)
    assert final_motor.gatype == Motor
    assert final_rate.gatype == Rate
    assert energies.unit == dim.mass * dim.length**2 / dim.time**2
    assert energies.shape == (5,)


def test_static_units_match_passive_rescaling_and_spacetime():
    """Passive rescaling in unchecked PGA3D matches static UnitGrid exponents (including
    M^(-1/2) on generalized eigh modes), and spacetime works with scaled time basis mv.t / c."""
    w0 = Algebra("w0")

    def rescale(value, factor: float):
        touching = value.gatype.derive.touching(w0)
        if len(touching.subspaces[0]) == 0:
            return value
        return value + (factor - 1.0) * value.cast(touching)

    bare_mv = NumpyContext(PGA3D).multivector
    BareBivector = PGA3D.gatype.bivector()
    lam, mu, tau = 2.3, 4.1, 1.7

    coords = np.array([[1.0, 0.0, 0.0], [-1.0, 0.5, 0.0], [0.0, -0.8, 0.9], [0.3, 0.2, -0.7]])
    anchors_xy = np.array([[2.0, 0.3, -0.1], [-1.8, 1.2, 0.5], [0.4, -1.7, 1.5], [0.8, 0.5, -1.9]])
    masses_num = np.array([1.2, 0.8, 1.5, 0.9])
    spring_k_num = np.array([10.0, 14.0, 8.0, 12.0])

    pts, anchors = bare_mv.point(coords), bare_mv.point(anchors_xy)
    inertia = (pts.regressive(pts.commutator(BareBivector)) * masses_num).sum(axis=0)
    lines = (pts & anchors).normalized()
    stiffness = (lines * (BareBivector & lines) * spring_k_num).sum(axis=0)
    omega_sq, modes = (BareBivector & stiffness).eigh(BareBivector & inertia)

    pts_s, anchors_s = rescale(pts, lam), rescale(anchors, lam)
    inertia_s = (pts_s.regressive(pts_s.commutator(BareBivector)) * (mu * masses_num)).sum(axis=0)
    lines_s = (pts_s & anchors_s).normalized()
    stiffness_s = (lines_s * (BareBivector & lines_s) * ((mu / tau**2) * spring_k_num)).sum(axis=0)
    omega_sq_s, modes_s = (BareBivector & stiffness_s).eigh(BareBivector & inertia_s)

    np.testing.assert_allclose(omega_sq_s.to_array(), omega_sq.to_array() / tau**2, rtol=1e-11, atol=1e-12)
    expected_modes_s = rescale(modes[2:], lam) / (np.sqrt(mu) * lam)
    signs = np.sign((modes_s[2:].kernel * expected_modes_s.kernel).sum(axis=-1, keepdims=True))
    np.testing.assert_allclose(modes_s[2:].kernel * signs, expected_modes_s.kernel, rtol=1e-11, atol=1e-11)

    mv = NumpyContext(PGA3D, units=True).multivector
    t = _pga3_types()
    Scalar, Bivector, Screw = t["Scalar"], t["Bivector"], t["Screw"]
    u_pts, u_anchors = mv.point(coords, unit=dim.length), mv.point(anchors_xy, unit=dim.length)
    u_inertia = (u_pts.regressive(u_pts.commutator(Bivector)) * mv(Scalar.derive.unit(dim.mass), masses_num[:, None])).sum(axis=0)
    u_lines = (u_pts & u_anchors).normalized()
    u_stiffness = (u_lines * (Bivector & u_lines) * mv(Scalar.derive.unit(dim.mass / dim.time**2), spring_k_num[:, None])).sum(axis=0)
    u_omega_sq, u_modes = (Bivector & u_stiffness).eigh(Bivector & u_inertia)
    assert u_omega_sq.unit == 1 / dim.time**2
    assert u_modes.gatype == Screw.derive.scaled_unit(1 / (dim.mass.root(2) * dim.length))
    np.testing.assert_allclose(u_omega_sq.to_array(), omega_sq.to_array(), rtol=1e-12)

    # Spacetime electromagnetism with Lorentz-compatible `t_s = mv.t / c` (works for any c).
    smv = NumpyContext(STA, units=True).multivector
    SScalar = STA.gatype.scalar()
    Speed = SScalar.derive.unit(dim.length / dim.time)
    Permeability = SScalar.derive.unit(dim.mass * dim.length / (dim.current**2 * dim.time**2))
    Magnetic = SScalar.derive.unit(dim.mass / (dim.current * dim.time**2))
    Electric = SScalar.derive.unit(dim.mass * dim.length / (dim.current * dim.time**3))

    c = smv(Speed, 2.5)
    mu_0 = smv(Permeability, 1.3)
    eps_0 = 1 / (mu_0 * c**2)
    t_s = smv.t / c
    assert t_s.unit == dim.time / dim.length

    wavevector = 2.0 * smv.x / dim.length + 3.0 * t_s / dim.time
    displacement = 0.5 * smv.x.dual() * dim.length + (c * smv(SScalar.derive.unit(dim.time), 0.4)) * (-smv.t.dual())
    assert wavevector.unit == 1 / dim.length
    assert displacement.unit == dim.length
    assert (wavevector & displacement).unit is dim.none

    B_val = smv(Magnetic, 0.4)
    E_val = smv(Electric, 1.2)
    field = B_val * smv.xy + E_val * (smv.x ^ t_s)
    excitation = field.dual() / mu_0
    assert field.unit == Magnetic.unit
    assert excitation.unit == dim.current / dim.length
    lagrangian = -(field & excitation) / 2
    expected_lagrangian = (eps_0 * E_val**2 - B_val**2 / mu_0) / 2
    assert lagrangian.unit == dim.mass / (dim.length * dim.time**2)
    np.testing.assert_allclose(lagrangian.to_array(), expected_lagrangian.to_array(), rtol=1e-12)


def test_extent_dimension_for_intensive_extensive_and_material_maps():
    """Extensivity sits inside `Unit` as `dim.extent`: degree 0 for intensive quantities, +1 for
    extensive amounts, -1 for material stiffness maps, +2 for products of extensives, and +-1/2 for
    spectral factors, with zero units on `Algebra`."""
    energy_per_extent = dim.energy
    pressure_dim = energy_per_extent / dim.length**3
    chemical_dim = energy_per_extent / dim.amount
    energy_dim = energy_per_extent * dim.extent
    entropy_dim = (energy_per_extent / dim.temperature) * dim.extent
    volume_dim = dim.length**3 * dim.extent
    amount_dim = dim.amount * dim.extent

    thermo = Algebra("s0v0n0")
    mv = NumpyContext(thermo, units=True).multivector
    Scalar = thermo.gatype.scalar()
    Plane = thermo.gatype.vector()
    Point = thermo.gatype.antivector()
    Thermal = Plane.derive.within(Algebra("s0"))
    Volumetric = Plane.derive.within(Algebra("v0"))
    Chemical = Plane.derive.within(Algebra("n0"))

    Temperature = Scalar.derive.unit(dim.temperature)
    Pressure = Scalar.derive.unit(pressure_dim)
    ChemicalPotential = Scalar.derive.unit(chemical_dim)
    Entropy = Scalar.derive.unit(entropy_dim)
    Volume = Scalar.derive.unit(volume_dim)
    Amount = Scalar.derive.unit(amount_dim)
    Duration = Scalar.derive.unit(dim.time)
    BarLength = Scalar.derive.unit(dim.length * dim.extent)
    Wavelength = Scalar.derive.unit(dim.length)

    Potential = thermo.gatype(
        Plane.output_subspace,
        orientation=Plane.orientation,
        unit={Thermal: dim.temperature, Volumetric: pressure_dim, Chemical: chemical_dim},
    )
    Parcel = thermo.gatype(
        Point.output_subspace,
        orientation=Point.orientation,
        unit={Thermal.derive.dual: entropy_dim, Volumetric.derive.dual: volume_dim, Chemical.derive.dual: amount_dim},
    )
    Stiffness = thermo.gatype((Potential, Parcel))

    T = mv(Temperature, 300.0)
    p = mv(Pressure, 1.0)
    mu = mv(ChemicalPotential, -0.5)
    S = mv(Entropy, 4.0)
    V = mv(Volume, 2.0)
    N = mv(Amount, 1.0)
    dt = mv(Duration, 0.1)

    potentials = T * mv.s - p * mv.v + mu * mv.n
    parcel = S * mv.s.dual() + V * mv.v.dual() + N * mv.n.dual()
    assert potentials.gatype == Potential
    assert all(u.exponents.get("extent", 0) == 0 for _, u in potentials.gatype.units.items)
    assert parcel.gatype == Parcel
    assert all(u.exponents.get("extent", 0) == 1 for _, u in parcel.gatype.units.items)

    # Euler pairing is extensive energy; densities are intensive; products of extensives have degree 2.
    euler_energy = potentials & parcel
    assert euler_energy.unit == energy_dim
    assert (S / V).unit.exponents.get("extent", 0) == 0
    assert (N / V).unit.exponents.get("extent", 0) == 0
    assert (S * V).unit.exponents["extent"] == 2

    # Extensive bar length and intensive wavelength share length but cannot be added; exp refuses extensive values.
    with pytest.raises(GATypeError):
        _ = mv(BarLength, 3.0) + mv(Wavelength, 0.5)
    with pytest.raises(GATypeError):
        _ = (S / (mv(Entropy, 1.0) / dim.extent)).exp()
    assert (S / mv(Entropy, 2.0)).exp().unit is dim.none

    # Material stiffness Potential <- Parcel has extent -1, compliance has extent +1.
    heat_capacity = mv(Entropy, 2.0)
    bulk_stiffness = mv(Scalar.derive.unit(pressure_dim / volume_dim), 3.0)
    chem_stiffness = mv(Scalar.derive.unit(chemical_dim / amount_dim), 5.0)
    stiffness = (
        (T / heat_capacity) * mv.s * (mv.s & Point)
        + bulk_stiffness * mv.v * (mv.v & Point)
        + chem_stiffness * mv.n * (mv.n & Point)
    )
    assert all(Stiffness.units.get(idx) == u for idx, u in stiffness.gatype.units.items)
    assert all(u.exponents["extent"] == -1 for _, u in stiffness.gatype.units.items)
    compliance = stiffness.inverse()
    assert all(u.exponents["extent"] == 1 for _, u in compliance.gatype.units.items)
    assert stiffness(parcel).gatype == Potential
    assert stiffness(parcel / dt).gatype == Potential.derive.scaled_unit(1 / dim.time)

    # Quadratic energy form Scalar <- (Parcel, Parcel); bare Cholesky across heterogeneous blades
    # is refused, while a uniform sector gives extent -1/2.
    energy_form = Point & stiffness(Point)
    assert energy_form(parcel, parcel).unit == energy_dim
    with pytest.raises(GATypeError):
        energy_form.cholesky()
    EntropyAxis = Thermal.derive.dual
    entropy_form = EntropyAxis & stiffness(EntropyAxis)
    assert entropy_form.cholesky().unit.exponents["extent"] == Fraction(-1, 2)

    # Mixed mechano-thermal algebra (x+y+ * s0) with zero algebra units.
    space = Algebra("x+y+")
    force_dim = (dim.mass * dim.extent) * dim.length / dim.time**2
    band_entropy = force_dim * dim.length / dim.temperature
    band = space * Algebra("s0")
    bmv = NumpyContext(band, units=True).multivector
    Upright = band.gatype.vector().derive.within(space)
    BPoint = band.gatype.antivector()

    place = bmv(Upright.derive.unit(dim.length), [1.0, 2.0]).dual()
    held_S = bmv(band.gatype.scalar().derive.unit(band_entropy), 0.5)
    points = place + held_S * bmv.s.dual()
    forces = bmv(Upright.derive.unit(force_dim), [3.0, 4.0])
    temp = bmv(band.gatype.scalar().derive.unit(dim.temperature), 300.0)
    band_potentials = temp * bmv.s - forces

    assert _blade_units(points.gatype) == {"ys": dim.length, "xs": dim.length, "xy": band_entropy}
    assert _blade_units(band_potentials.gatype) == {"x": force_dim, "y": force_dim, "s": dim.temperature}
    assert (band_potentials & points).unit == force_dim * dim.length
    assert (band_potentials & points).unit.exponents["extent"] == 1

    k_spring = bmv(band.gatype.scalar().derive.unit(force_dim / dim.length), 10.0)
    k_thermal = bmv(band.gatype.scalar().derive.unit(dim.temperature / band_entropy), 0.2)
    band_stiffness = (
        k_spring * (bmv.x * (bmv.x & BPoint) + bmv.y * (bmv.y & BPoint))
        + k_thermal * bmv.s * (bmv.s & BPoint)
    )
    response = band_stiffness(points)
    assert response.gatype == band_potentials.gatype
    assert (response & bmv.x.dual()).unit == force_dim
    assert (response & bmv.s.dual()).unit == dim.temperature


def test_pga3d_point_direction_and_sparse_field_operations():
    pytest.importorskip("scipy")

    ctx = NumpyContext(PGA3D, units=True)
    mv = ctx.multivector
    t = _pga3_types()
    Point, Direction, WorldPoint = t["Point"], t["Direction"], t["WorldPoint"]
    r = mv.scalar(2.0, unit=dim.length)
    mass = mv.scalar(3.0, unit=dim.mass)
    time = mv.scalar(4.0, unit=dim.time)

    # Projective points and directions carry explicit physical units cleanly.
    pts = mv.point([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], unit=dim.length)
    weighted = mv.point([1.0, 2.0, 3.0], unit=dim.length) * dim.mass
    velocity = mv.direction([1.0, 0.0, 0.0], unit=dim.length / dim.time)
    assert (mv.w & pts).unit is dim.none
    assert (mv.x & pts).unit == dim.length
    assert (mv.w & weighted).unit == dim.mass
    assert (mv.x & weighted).unit == dim.length * dim.mass
    assert (mv.x & velocity).unit == dim.length / dim.time
    assert velocity.antinorm().unit == dim.length / dim.time

    diff = pts[1] - pts[0]
    translation = ((mv.w ^ diff.cast(Direction).dual()) / 2).exp()
    assert _blade_units(translation.gatype) == {
        "1": dim.none, "wx": dim.length, "wy": dim.length, "wz": dim.length,
    }
    moved = translation >> pts[0]
    assert moved.gatype == WorldPoint
    np.testing.assert_allclose((mv.x & moved).to_array(), 1.0, atol=1e-12)

    # Sparse maps keep units through to_dense(), narrow-rhs solve, and correlation eigh.
    # A bare GAType is its dimensionless identity map, so `Sparse * GAType` stays checked (matching
    # `np.array(...) * GAType` and `Sparse * GAType.dual_inverse()`) while yielding unchecked output
    # whenever combined with an unchecked field.
    topo = Topology(3, np.array([[0, 1, 2]]))
    plain_sparse = topo.d[0] * Point
    assert plain_sparse.has_units
    assert plain_sparse.unit is dim.none
    assert plain_sparse.units == plain_sparse.cells.units
    dense_d0 = plain_sparse.to_dense()
    assert dense_d0.unit is dim.none
    diff_field = dense_d0(pts.field())
    assert diff_field.gatype.units == WorldPoint.units
    assert diff_field.gatype.fields == ((0, 3),)
    pts_unchecked = NumpyContext(PGA3D).multivector.point(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    assert plain_sparse(pts_unchecked.field()).units.is_unchecked

    q_cell = (
        (mv.x * (mv.x & Point) + mv.y * (mv.y & Point) + mv.z * (mv.z & Point)) / r**2
        + mv.w * (mv.w & Point)
    )
    K_diag = spdiag(q_cell[None].broadcast_to((3,)).field())
    M_diag = spdiag((q_cell * mass)[None].broadcast_to((3,)).field())
    rhs_narrow = (mv.x / r + mv.y / r + mv.w)[None].broadcast_to((3,)).field()
    sol = K_diag.solve(rhs_narrow)
    assert sol.gatype.fields == ((0, 3),)
    assert all(WorldPoint.units.get(idx) == u for idx, u in sol.gatype.units.items)
    np.testing.assert_allclose((K_diag(sol) - rhs_narrow).kernel, 0.0, atol=1e-12)

    with pytest.raises(GATypeError):
        K_diag.solve(pts.field())
    with pytest.raises(GATypeError):
        K_diag.lstsq(rhs_narrow)

    values, modes = K_diag.eigh(M_diag, 2)
    assert values.unit == 1 / dim.mass
    assert modes.gatype.units == WorldPoint.derive.scaled_unit(1 / dim.mass.root(2)).units
    assert modes.gatype.fields == ((0, 3),)
    np.testing.assert_allclose((K_diag(modes) - values * M_diag(modes)).kernel, 0.0, atol=1e-12)
    norm_check = (M_diag(modes) & modes).sites.sum()
    assert norm_check.unit is dim.none
    np.testing.assert_allclose(norm_check.to_array(), [1.0, 1.0], atol=1e-12)

    # Endomorphism eigh with inconsistent diagonal units or mismatched operator/metric types is refused.
    mixed_endo_cell = (
        (mv.x.dual() * (mv.x & Point) + mv.y.dual() * (mv.y & Point) + mv.z.dual() * (mv.z & Point)) / r
        + mv.w.dual() * (mv.w & Point)
    )
    A_endo = spdiag((mixed_endo_cell / time)[None].broadcast_to((3,)).field())
    M_endo = spdiag((Point * mass)[None].broadcast_to((3,)).field())
    with pytest.raises(GATypeError):
        A_endo.eigh(M_endo, 2)
    with pytest.raises(GATypeError):
        K_diag.eigh(M_endo, 2)
    with pytest.raises(GATypeError):
        A_endo.to_dense().eigh(M_endo.to_dense())

    # Uniform-unit VGA3D sparse endomorphism eigh propagates eigenvalue and eigenvector units.
    ctx_v = NumpyContext(VGA3D, units=True)
    mv_v = ctx_v.multivector
    Vec = VGA3D.gatype.vector()
    K_v = spdiag((Vec * mv_v(VGA3D.gatype.scalar(unit=dim.mass / dim.time**2), 8.0))[None].broadcast_to((3,)).field())
    M_v = spdiag((Vec * mv_v(VGA3D.gatype.scalar(unit=dim.mass), 2.0))[None].broadcast_to((3,)).field())
    vals_v, vecs_v = K_v.eigh(M_v, 2)
    assert vals_v.unit == 1 / dim.time**2
    assert vecs_v.unit == dim.mass ** Fraction(-1, 2)
    np.testing.assert_allclose(vals_v.to_array(), [4.0, 4.0], atol=1e-12)


# --- Passive-rescaling oracle across every registered escape law ------------------------

def _unit_scale_factor(unit: Unit, base_scales: dict[str, float]) -> float:
    """Numeric scale factor induced on `unit` by passive base-dimension rescaling."""
    factor = 1.0
    for name, exponent in unit.exponents.items():
        factor *= base_scales[name] ** float(exponent)
    return factor


def _scale_kernel(value, base_scales: dict[str, float]) -> np.ndarray:
    """Multiply each blade/entry of `value.kernel` by the passive scale factor of its static unit."""
    units = value.units
    assert not units.is_unchecked
    kernel = np.array(value.kernel, dtype=float, copy=True)
    factors = np.zeros(units.shape, dtype=float)
    for flat, unit in units.items:
        factors.flat[flat] = _unit_scale_factor(unit, base_scales)
    return kernel * factors


def _unchecked_scaled(value, base_scales: dict[str, float]):
    """Unchecked counterpart of `value` with entries rescaled passively by `base_scales`."""
    bare_ctx = NumpyContext(value.context.algebra)
    return bare_ctx.extensor(value.gatype.derive.unchecked, _scale_kernel(value, base_scales))


def _assert_passive_law(fn, args, base_scales: dict[str, float], *, up_to_sign: bool = False):
    """Verify `fn(*args)` against passive base-dimension rescaling of unchecked inputs."""
    checked_out = fn(*args)
    rescaled_out = fn(*(_unchecked_scaled(arg, base_scales) for arg in args))
    checked_tuple = checked_out if isinstance(checked_out, tuple) else (checked_out,)
    rescaled_tuple = rescaled_out if isinstance(rescaled_out, tuple) else (rescaled_out,)
    for checked, rescaled in zip(checked_tuple, rescaled_tuple):
        expected = _scale_kernel(checked, base_scales)
        actual = np.asarray(rescaled.kernel, dtype=float)
        if up_to_sign:
            # Resolve arbitrary sign per eigenmode / singular vector row.
            axes = tuple(range(1, actual.ndim)) if actual.ndim > 1 else ()
            if axes:
                dots = (actual * expected).sum(axis=axes, keepdims=True)
                signs = np.where(dots < 0, -1.0, 1.0)
                actual = actual * signs
            elif float(np.dot(actual.ravel(), expected.ravel())) < 0:
                actual = -actual
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    return checked_out


def test_all_escape_laws_commute_with_passive_rescaling():
    """Every registered unit escape law in `numga.units` commutes with independent passive
    rescaling of the base dimensions `(length, mass, time, current)`."""
    scales = {"length": 2.3, "mass": 4.1, "time": 1.7, "current": 3.1}
    mv = NumpyContext(PGA3D, units=True).multivector
    vmv = NumpyContext(VGA3D, units=True).multivector
    smv = NumpyContext(STA, units=True).multivector

    t = _pga3_types()
    Point, Plane, Bivector, Rate = t["Point"], t["Plane"], t["Bivector"], t["Rate"]
    V = VGA3D.gatype.vector()
    Field = STA.gatype.bivector()

    rng = np.random.default_rng(77)
    pts = mv.point(rng.uniform(-0.5, 0.5, (12, 3)), unit=dim.length)
    masses = mv.scalar(rng.uniform(0.3, 1.2, (12, 1)), unit=dim.mass)
    stiff_k = mv.scalar(rng.uniform(4.0, 15.0, (12, 1)), unit=dim.mass / dim.time**2)
    inertia = (pts * masses).regressive(pts.commutator(Bivector)).sum(axis=0)
    stiffness = (pts * stiff_k).regressive(pts.commutator(Bivector)).sum(axis=0)
    K_form = Bivector & stiffness
    M_form = Bivector & inertia
    endo = inertia.inverse()(stiffness)
    rate = mv(Rate, [0.3, -0.4, 0.7, 0.5, -0.2, 0.1])
    dt = mv.scalar(0.25, unit=dim.time)
    screw = rate * dt
    motor = screw.exp()
    momentum = inertia(rate)

    # 1. exp, log, square_root, inverse_square_root, normalized, decompose_polar
    _assert_passive_law(lambda x: x.exp(), (screw,), scales)
    _assert_passive_law(lambda x: x.log(), (motor,), scales)
    _assert_passive_law(lambda x: x.square_root(), (motor,), scales)
    _assert_passive_law(lambda x: x.inverse_square_root(), (motor,), scales)
    _assert_passive_law(lambda x: x.square_root(), (mv.scalar(9.0, unit=dim.length**2),), scales)
    _assert_passive_law(lambda x: x.inverse_square_root(), (mv.scalar(9.0, unit=dim.length**2),), scales)
    _assert_passive_law(lambda x: x.normalized(), ((pts[0] & pts[1]),), scales)
    _assert_passive_law(lambda x: x.decompose_polar(), (rate,), scales)

    # 2. inverse, solve, adjugate, det, trace (1-arg and 3-arg), outermorphism
    _assert_passive_law(lambda a: a.inverse(), (inertia,), scales)
    _assert_passive_law(lambda a, b: a.solve(b), (inertia, momentum), scales)
    _assert_passive_law(lambda a: a.adjugate(), (inertia,), scales)
    _assert_passive_law(lambda k, m: k.det(m), (K_form, M_form), scales)
    _assert_passive_law(lambda a: a.det(), (endo,), scales)
    _assert_passive_law(lambda a: a.trace(), (endo,), scales)
    f_point = ((motor >> Point) * 3.0) / dim.time
    _assert_passive_law(lambda a: a.trace(1, 2), (Plane & f_point,), scales)
    mirrors = stack((mv.x, mv.y, mv.z))
    radius = mv.scalar(2.0, unit=dim.length)
    ball = (mirrors * (mirrors & Point)).sum(axis=-1) / radius**2 - mv.w * (mv.w & Point)
    Line = PGA3D.gatype.antibivector()
    _assert_passive_law(lambda a: a.outermorphism(Line), (ball,), scales)

    # 3. eig, eigvals, eigh (1-arg uniform, 1-arg PGA Schur, 2-arg PGA generalized), eigvalsh, cholesky
    eig_only = endo.eigvals()
    rescaled_only = _unchecked_scaled(endo, scales).eigvals()
    np.testing.assert_allclose(
        np.sort(rescaled_only.kernel.ravel()),
        np.sort(_scale_kernel(eig_only, scales).ravel()),
        rtol=1e-10,
    )
    eig_vals, eig_vecs = endo.eig()
    rescaled_vals, rescaled_vecs = _unchecked_scaled(endo, scales).eig()
    order_checked = np.argsort(eig_vals.kernel.ravel())
    order_rescaled = np.argsort(rescaled_vals.kernel.ravel())
    np.testing.assert_allclose(
        rescaled_vals.kernel.ravel()[order_rescaled],
        _scale_kernel(eig_vals, scales).ravel()[order_checked],
        rtol=1e-10,
    )
    for k_c, k_r in zip(order_checked, order_rescaled):
        v_pred = _scale_kernel(eig_vecs[int(k_c)], scales)
        v_pred = v_pred / np.linalg.norm(v_pred)
        v_got = rescaled_vecs[int(k_r)].kernel / np.linalg.norm(rescaled_vecs[int(k_r)].kernel)
        sign = -1.0 if float(np.dot(v_pred, v_got)) < 0 else 1.0
        np.testing.assert_allclose(v_got * sign, v_pred, rtol=1e-9, atol=1e-9)

    _assert_passive_law(lambda k, m: k.eigh(m), (K_form, M_form), scales, up_to_sign=True)
    _assert_passive_law(lambda k, m: k.eigvalsh(m), (K_form, M_form), scales)
    _assert_passive_law(lambda k: k.eigh(), (K_form,), scales, up_to_sign=True)
    _assert_passive_law(lambda k: k.eigvalsh(), (K_form,), scales)

    A_v = (
        (vmv.x * (vmv.x | V) * 5.0 + vmv.y * (vmv.y | V) * 3.0 + vmv.z * (vmv.z | V) * 2.0
         + (vmv.x * (vmv.y | V) + vmv.y * (vmv.x | V)) * 0.4)
        * (dim.mass / dim.time**2)
    )
    _assert_passive_law(lambda a: a.eigh(), (A_v,), scales, up_to_sign=True)
    _assert_passive_law(lambda a: a.eigvalsh(), (A_v,), scales)
    _assert_passive_law(lambda a: a.cholesky(), (A_v,), scales)

    # 4. svd, svdvals, pinv, lstsq (2-arg uniform and 3-arg weighted PGA)
    wavevector = (smv.t * 0.7 - smv.z) / dim.length
    mu_0 = smv.scalar(1.3, unit=dim.mass * dim.length / (dim.current**2 * dim.time**2))
    wave_op = (wavevector ^ Field) + mu_0 * (wavevector ^ (Field.dual() / mu_0)).dual()
    _assert_passive_law(lambda a: a.svdvals(), (wave_op,), scales)
    _assert_passive_law(lambda a: a.svd(), (A_v,), scales, up_to_sign=True)
    _assert_passive_law(lambda a: a.pinv(), (A_v,), scales)
    b_v = (1.0 * vmv.x + 2.0 * vmv.y - 0.5 * vmv.z) * (dim.mass * dim.length / dim.time**2)
    _assert_passive_law(lambda a, b: a.lstsq(b), (A_v, b_v), scales)
    _assert_passive_law(lambda k, b, m: k.lstsq(b, m), (K_form, K_form(rate), M_form), scales)

    # 5. clip, abs, scalar transcendentals, and contraction
    d1, d2 = mv.scalar(3.0, unit=dim.length), mv.scalar(2.0, unit=dim.length)
    _assert_passive_law(lambda a, b: a.clip(0, b), (d1, d2), scales)
    _assert_passive_law(lambda a: (-a).abs(), (d1,), scales)
    _assert_passive_law(lambda a, b: (a / b).sin(), (d1, d2), scales)
    _assert_passive_law(lambda a, b: (a / b).exp(), (d1, d2), scales)
    _assert_passive_law(lambda a, b: (a / b).log(), (d1, d2), scales)
    _assert_passive_law(lambda a: a.contract(), (A_v,), scales)
    _assert_passive_law(lambda a: (V | a(V)).contract(), (A_v,), scales)
    _assert_passive_law(lambda a: (V | a(V)).contract(1, 2), (A_v,), scales)
    _assert_passive_law(lambda a: a.symmetric_reverse_product().contract(), (A_v,), scales)
    mixed_form = (vmv.x | V) * (vmv.x | V) * dim.length + (vmv.y | V) * (vmv.y | V) * dim.mass
    with pytest.raises(GATypeError, match="contract"):
        mixed_form.contract()

    # 6. Symmetric dtype promotion when combining real and complex checked contexts, and cyclic off-diagonal eig
    cmv = NumpyContext(STA, dtype=np.complex128, units=True).multivector
    real_s = smv.scalar(2.0, unit=dim.time / dim.length)
    complex_s = cmv.scalar(1.0 + 0.5j, unit=dim.time / dim.length)
    diff = real_s - complex_s
    assert diff.context.dtype == np.complex128 and diff.unit == dim.time / dim.length
    np.testing.assert_allclose(diff.kernel, [1.0 - 0.5j])
    prod_rc = real_s * (1j * cmv.x)
    assert prod_rc.context.dtype == np.complex128 and prod_rc.unit == dim.time / dim.length

    # Off-diagonal bipartite endomorphism (harmonic oscillator [[0, 1/m], [-k, 0]]) resolves eigenvalue unit 1/time.
    V2 = VGA3D.gatype(VGA3D.subspace("x y"), orientation=Orientation.extrinsic)
    osc = (
        vmv.x * (vmv.y | V2) * (1.0 / vmv.scalar(2.0, unit=dim.mass))
        - vmv.y * (vmv.x | V2) * vmv.scalar(8.0, unit=dim.mass / dim.time**2)
    )
    osc_vals, osc_vecs = osc.eig()
    assert osc_vals.unit == 1 / dim.time
    assert _blade_units(osc_vecs.gatype) == {"x": dim.none, "y": dim.mass / dim.time}
    np.testing.assert_allclose(np.sort(np.abs((osc_vals * dim.time).to_array())), [2.0, 2.0])
