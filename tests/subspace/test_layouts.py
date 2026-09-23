"""The same geometry, with deliberately different coefficient layouts."""

import numpy as np
import pytest

from numga import Algebra, GATypeDispatch, NumpyContext, ReverseProductOne
from numga.algebras import PGA2D, PGA3D


@pytest.fixture(params=("numpy", "jax"))
def backend(request):
    if request.param == "numpy":
        return NumpyContext, lambda function: function
    jax = pytest.importorskip("jax")
    from numga.backend.jax import JaxContext
    return JaxContext, jax.jit


def test_layout_identity_is_not_mathematical_inclusion():
    algebra = Algebra("x+y+z+")
    spaces = algebra.subspace
    lexical = spaces("xy xz yz")
    cyclic = spaces("yz zx xy")

    assert cyclic is spaces.from_blades(("yz", ("z", "x"), "xy"))
    assert cyclic is spaces("yz -xz xy")
    assert cyclic.masks == (6, 5, 3)
    assert cyclic.signs == (1, -1, 1)
    assert cyclic != lexical
    assert cyclic.same_support(lexical)
    assert cyclic.support_key == lexical.support_key
    assert cyclic.gatype != lexical.gatype
    assert cyclic.gatype <= lexical.gatype <= cyclic.gatype
    assert not cyclic.gatype < lexical.gatype

    dispatch = GATypeDispatch("layout", None, 1)

    @dispatch.register(lambda g: g.subspaces == (cyclic,))
    def specialized(value):
        return "coefficient-specific"

    @dispatch.register(lexical.gatype)
    def generic(value):
        return "mathematical support"

    assert dispatch.resolve(cyclic.gatype) is specialized
    assert dispatch.resolve(lexical.gatype) is generic
    assert dispatch.resolve(spaces("yx yz zx").gatype) is generic
    assert spaces("xy").gatype < cyclic.gatype
    assert algebra.operator.geometric_product(cyclic, cyclic) is algebra.operator.geometric_product(cyclic, cyclic)
    assert algebra.operator.geometric_product(cyclic, cyclic) is not algebra.operator.geometric_product(lexical, lexical)


def test_signed_relayout_and_products_agree(backend):
    context_type, compile = backend
    algebra = Algebra("x+y+z+")
    spaces = algebra.subspace
    mv = context_type(algebra).multivector
    cyclic, lexical = spaces("yz zx xy"), spaces("xy xz yz")
    a = mv(cyclic, [[1, 2, 3], [2, -1, 4]])
    b = mv(cyclic, [2, 1, -1])

    converted = compile(lambda x: x.select_subspace(lexical))(a)
    np.testing.assert_allclose(converted.kernel, [[3, -2, 1], [4, 1, 2]])
    np.testing.assert_allclose(converted.select_subspace(cyclic).kernel, a.kernel)
    np.testing.assert_allclose(a.restrict_subspace(spaces("xz yz")).kernel, [[1, 2], [2, -1]])
    assert a.restrict_subspace(spaces("xz yz")).subspace is spaces("yz zx")

    # An operator expecting lexical inputs must convert incoming cyclic arrays.
    product = lexical * lexical
    direct = compile(lambda x, y: x * y)(a, b)
    implicit = compile(lambda x, y: product(x, y))(a, b)
    explicit = product(converted, b.select_subspace(lexical))
    np.testing.assert_allclose(implicit.kernel, direct.kernel)
    np.testing.assert_allclose(explicit.kernel, direct.kernel)
    np.testing.assert_allclose(a.squared().kernel, (a * a).select_subspace(a.squared().subspace).kernel)
    np.testing.assert_allclose(a.reverse().select_subspace(lexical).kernel, converted.reverse().kernel)


def test_pga_defaults_make_the_dual_maps_coefficient_identities(backend):
    context_type, compile = backend
    spaces = PGA3D.subspace
    mv = context_type(PGA3D).multivector
    assert spaces.bivector() is spaces("yz zx xy xw yw zw")
    assert spaces.trivector() is spaces("yzw zxw xyw zyx")
    assert spaces.even() is spaces("1 yz zx xy xw yw zw xyzw")

    planes = mv.vector([[1, 2, 3, 4], [5, -1, 0, 2]])
    points = compile(lambda p: p.dual())(planes)
    assert points.subspace is spaces.trivector()
    np.testing.assert_allclose(points.kernel, planes.kernel)
    np.testing.assert_allclose(points.dual_inverse().kernel, planes.kernel)
    np.testing.assert_allclose(points.dual().kernel, -planes.kernel)

    rotation = mv(spaces("yz zx xy"), [1, 2, 3])
    translation = rotation.dual()
    assert translation.subspace is spaces("xw yw zw")
    np.testing.assert_allclose(translation.kernel, rotation.kernel)

    # Changing the storage convention doesn't change the geometric dual.
    lexical = spaces("xyz xyw xzw yzw")
    np.testing.assert_allclose(points.select_subspace(lexical).kernel, [[-4, 3, -2, 1], [-2, 0, 1, 5]])
    np.testing.assert_allclose(points.select_subspace(lexical).dual_inverse().kernel, planes.kernel)
    # The PGA regressive definition still uses dual-inverse(dual(a) ^ dual(b)).
    np.testing.assert_allclose(
        points.regressive(points).kernel,
        points.dual().wedge(points.dual()).dual_inverse().kernel,
    )


def test_pga2_defaults_make_line_point_duality_a_coefficient_identity(backend):
    context_type, compile = backend
    spaces = PGA2D.subspace
    mv = context_type(PGA2D).multivector

    assert spaces.bivector() is spaces("yw wx xy")
    assert spaces.even() is spaces("1 yw wx xy")
    lines = mv.vector([[2, 3, 1], [-1, 4, 1]])
    points = compile(lambda line: line.dual())(lines)

    assert points.subspace is spaces.antivector()
    np.testing.assert_allclose(points.kernel, lines.kernel)
    np.testing.assert_allclose(points.dual().kernel, lines.kernel)
    np.testing.assert_allclose(points.dual_inverse().kernel, lines.kernel)
    np.testing.assert_allclose(
        points.select_subspace(spaces("xy xw yw")).kernel,
        [[1, -3, 2], [1, -4, -1]],
    )


def test_unit_rotor_facts_survive_relayout_and_sandwich(backend):
    context_type, compile = backend
    algebra = Algebra("x+y+z+")
    spaces = algebra.subspace
    mv = context_type(algebra).multivector
    r = mv.even([3, 4, 0, 0]).normalized()
    layout = spaces("yz 1 zx xy")
    stored = compile(lambda x: x.select_subspace(layout))(r)

    assert stored.gatype <= algebra.gatype.rotor()
    assert stored.gatype.entails(ReverseProductOne)
    np.testing.assert_allclose(stored.kernel, [0, 0.6, 0, 0.8], atol=1e-6)
    np.testing.assert_allclose(stored.inverse().select_subspace(r.subspace).kernel, r.inverse().kernel)
    normal = mv.vector([1, 2, 3]).normalized()
    np.testing.assert_allclose(stored.sandwich(normal).kernel, r.sandwich(normal).kernel, atol=1e-6)


def test_reorienting_the_scalar_changes_coordinates_not_scalar_functions(backend):
    context_type, compile = backend
    algebra = Algebra("x+y+")
    mv = context_type(algebra).multivector
    reversed_scalar = algebra.subspace("-1")
    four = mv(reversed_scalar, [-4])

    unit = mv(reversed_scalar)
    np.testing.assert_allclose(unit.kernel, [-1])
    root, reciprocal_root, norm, logarithm = compile(
        lambda s: (s.square_root(), s.inverse_square_root(), s.norm(), s.log())
    )(four)
    np.testing.assert_allclose(root.kernel, [2])
    np.testing.assert_allclose(reciprocal_root.kernel, [0.5])
    np.testing.assert_allclose(norm.kernel, [4])
    np.testing.assert_allclose(logarithm.exp().kernel, [4], rtol=1e-6)
    np.testing.assert_allclose(four.exp().kernel, [np.exp(4)], rtol=1e-6)
    np.testing.assert_allclose((four + 1).kernel, [5])


@pytest.mark.parametrize("spelling", ["xy yx", "xx"])
def test_explicit_basis_requires_independent_blades(spelling):
    with pytest.raises(ValueError):
        Algebra("x+y+").subspace(spelling)
