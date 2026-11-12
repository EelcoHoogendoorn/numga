# Orientation in numga

Every `GAType` carries an orientation, `Orientation.extrinsic` (`+` in type comments) or `Orientation.intrinsic` (`-`), on its output and on each input. Constructors assign it, operations propagate and check it statically, and `>>` uses it to reflect extrinsic and intrinsic elements with the appropriate sign. [Why orientation](../articles/orientation.md) motivates the two orientations from geometric and physical requirements, and [Conventions](conventions.md) states the defining identities.

```python
from numga import Algebra, NumpyContext, Orientation
from numga.algebras import PGA3D

ga = PGA3D
mv = NumpyContext(ga).multivector

Plane = ga.gatype.vector()                                     # Vector+
Point = ga.gatype.antivector()                                 # Antivector-

mirror = mv.x                                                  # [] Vector+
red, blue = mv.x + mv.y, mv.y - mv.x                           # [] Vector+, [] Vector+
ends = mv.point([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])           # [points] Antivector-

# Swapping the two planes negates their meet; leaving the two points in the mirror preserves their join.
mirror >> (red ^ blue)                                         # [] Bivector+
mirror >> (ends[0] & ends[1])                                  # [] Antibivector-
# Complementary orientations pair to an extrinsic scalar.
red & ends[0]                                                  # [] Scalar+
```

## Propagation and static checks

Writing `+` for `extrinsic` and `-` for `intrinsic`, operations propagate and restrict orientation as follows:

| Operation | Result orientation |
| --- | --- |
| `a * b`, `a ^ b`, `a \| b`, `a.scalar_product(b)`, `a.commutator(b)`, `a.anticommutator(b)` | product of the signs |
| `a & b` | minus the product of the signs |
| `a.dual()`, `a.dual_inverse()` | flipped |
| `a + b`, `a - b`, `stack`, `concatenate` | equal orientations required; that orientation |
| `~a`, `a.involute()`, `-a`, `a.inverse()`, `a.normalized()`, `s * a` with `s` extrinsic | unchanged |
| `versor >> x`, `versor << x` | `x`'s orientation |
| `a.dot(b)`, `a.antinorm()` | equal orientations required; extrinsic output |
| `a.norm()`, Study norms, `cos`, `cosh` | extrinsic output |
| `sin`, `tan`, `sinh`, `tanh`, `arcsin`, `arctan`, `arcsinh`, `arctanh` | unchanged |
| `exp`, `log` and variants, square roots, `arccos`, `arccosh` | extrinsic input required |

Sums, differences, stacks and concatenations require matching orientations on every axis; binding checks each input's orientation, and `solve` checks the right-hand side against the output's. Power-series functions such as `exp` require extrinsic input because an intrinsic element's even powers are extrinsic while its odd powers are intrinsic, whereas odd scalar functions such as `sin` keep the mirror sign on `Scalar-` and even ones such as `cos` remove it. Mismatches raise `GATypeError`, a subclass of `TypeError`, before any kernel runs:

```python
Bivector = ga.gatype.bivector()                                # Bivector+
Antibivector = ga.gatype.antibivector()                        # Antibivector-
inertia = ends.regressive(ends.commutator(Bivector)).sum(axis=0)  # [] Antibivector- <- Bivector+
rate = mv.xy                                                   # [] Bivector+
momentum = inertia(rate)                                       # [] Antibivector-

rate + momentum                  # GATypeError: orientations differ
inertia(momentum)                # GATypeError: input requires extrinsic orientation
inertia.solve(rate)              # GATypeError: right-hand side requires intrinsic orientation
momentum.exp()                   # GATypeError: exp requires extrinsic orientation
rate.dot(momentum)               # GATypeError: dot takes operands of one orientation
inertia.eigh()                   # GATypeError: requires an endomorphism or a square form
```

## Reflections, pairings and sizes

`versor >> x` (or `versor.transform(x)`) uses the versor's reverse and assumes unit normalization (`versor * ~versor == +1`). For a unit mirror `m`, an extrinsic `x` maps to `m * x.involute() * m.inverse()` and an intrinsic `x` to `-m * x.involute() * m.inverse()`. Across `k` vector factors the grade involution and the intrinsic sign each enter `k` times, so an even versor acts as `v * x * v.inverse()` on both orientations; `versor << x` (or `versor.reverse_transform(x)`) applies the reverse action. Normalized versor transformations commute with `*`, `^`, `|`, `&`, `commutator` and `anticommutator`; extrinsic scalars stay invariant under reflection while intrinsic scalars negate.

Complementary elements pair by the regressive product, extrinsic first (`rate & momentum`), producing `Scalar+`. `.dot()` measures two elements of a single orientation, measuring intrinsic values through their duals (`a.dot(b) == a.dual().dot(b.dual())`) so that a direction's squared length is positive across spatial, spacetime and projective algebras alike. An intrinsic value therefore has two sizes: `.norm()` follows the reverse product (giving a projective point's weight, so `.normalized()` sets weight one while preserving `-`), whereas `.antinorm()` follows `.dot()` (giving a direction's length and a projective point's bulk):

```python
far, near = mv.point([[3, 4, 0], [0, 0, 0]])                   # [] Antivector-, [] Antivector-
step = far - near                                              # [] Antivector-
# Squared length 25, Euclidean length 5, unit weight 1, and bulk 5.
step.dot(step)                                                 # [] Scalar+
step.antinorm(), far.norm(), far.antinorm()                    # [] Scalar+, [] Scalar+, [] Scalar+
```

`.dual()` complements the blade layout and flips orientation, and `.dual_inverse()` reverses it. In `PGA3D`, `(Point.dual() ^ Point.dual()).dual_inverse()` equals `Point & Point`, its three orientation flips leaving one net flip; in nondegenerate signatures `.dual()` carries metric signs (`a.dual() == a.reverse() * antiscalar`, [Conventions](conventions.md#products)) and commutes with normalized versors, whereas in degenerate algebras such as PGA it does not commute with translations. Multiplying by a fixed pseudoscalar (`a * mv.pseudoscalar([1])` or `a * mv.antiscalar([1])`) follows `*` instead and transforms as a scene element; a blade complement is written with `.dual()`.

## Maps and forms

Each axis of a map carries its own orientation, printed output first (`intrinsic bivector <- extrinsic bivector`). In `PGA3D`, `inertia` above maps `Bivector+` to `Antibivector-`, a duality map between the two orientations of the grade-two space, so its unary trace is `Scalar-`, while swapping, complementing and flipping both axes leaves `inertia.adjugate()` with the same type `Antibivector- <- Bivector+`.

| Property | Meaning |
| --- | --- |
| `gatype.orientation` | the output's orientation |
| `gatype.orientations` | the orientations of all axes, output first |
| `gatype.maps_equal_dimensions` | output and input have the same dimension |
| `gatype.is_endomorphism` | output and input have the same blades and the same orientation |
| `gatype.is_correlation` | output and input have complementary blades and opposite orientations |

| Operation | Orientations |
| --- | --- |
| `f.inverse()`, `f.adjoint()` | axes swapped, each keeping its orientation |
| `f.adjugate()` | axes swapped and complemented, each orientation flipped |
| `f.trace()`, `f.contract()` on a unary map | matching input and output blade support required; scalar orientation is the product of their signs |
| `f.trace(i, j)` pairing two inputs | complementary spaces required; output orientation multiplied by the regressive pairing's orientation |
| `f.contract(i, j)` pairing two inputs | one space with an invertible metric required; output orientation multiplied by the metric pairing's orientation |
| `f.outermorphism(Type)` | `Type` matches products of the map's inputs; products of its outputs determine the result |

Both `adjugate` and two-input `trace` read their pairings extrinsic first ([Conventions](conventions.md#maps)), so `f.adjugate().adjugate() == f` and positive forms pull back to positive forms in every dimension.

`det`, `eig`, `eigh` and `cholesky` accept endomorphisms or square scalar-valued forms. Pairing a duality map's output with a second input builds its symmetric form:

```python
kinetic = Bivector & inertia(Bivector)                         # [] Scalar+ <- (Bivector+, Bivector+)
energy = kinetic(rate, rate) / 2                               # [] Scalar+
```

`form.eigh(partner)` measures eigenvalues against `partner` (defaulting to the input type's own metric when omitted). Two duality maps such as stiffness and inertia are paired into forms before calling `elastic.eigh(kinetic)`, with `kinetic` positive definite.

## Constructing and deriving types

Named constructors set both grade and orientation. Plain names and basis blades are extrinsic, while `anti` constructors select the complementary grade (`n - k`) and intrinsic orientation (`antiscalar` is grade `n` with `-`, and `antipseudoscalar` is grade `0` with `-`):

| Construction | Orientation |
| --- | --- |
| `scalar`, `vector`, `bivector`, `trivector`, `pseudoscalar`, `even`, `odd`, `rotor`, `motor`, `multivector` | extrinsic (`+`) |
| basis blades (`mv.x`, `mv.xy`, `mv.blade(mask)`) and plain numbers | extrinsic (`+`) |
| `antiscalar`, `antivector`, `antibivector`, `antitrivector`, `antiquadvector`, `antipseudoscalar` | intrinsic (`-`) |
| `mv.point(...)`, `mv.direction(...)` in `PGA2D` / `PGA3D`, and `point` in `Spherical3D` | intrinsic (`-`) |
| `mv(gatype, values)` and `ga.gatype(expression)` | inherited from `gatype` or `expression` |

`ga.gatype((Output, Input))` builds a map type from types that already carry orientations and refuses `orientation=`. Narrowing a type with `.derive.within(part)`, `.derive.touching(*parts)`, `.derive.without(part)`, `.derive.degenerate` or `.derive.nondegenerate` keeps its orientation (`part` being an algebra sharing generator names and squares); `.derive.flipped_orientation` flips only orientation (the output's on a map); `.derive.dual` complements blades and flips orientation; and constructing directly from a raw `SubSpace` requires an explicit `orientation=`:

```python
Surface = Plane.derive.flipped_orientation                     # Vector-
Line = ga.gatype(Point & Point)                                # Antibivector-
Strain = Bivector.derive.touching(Algebra("z+"))               # Bivector+
Resultant = Strain.derive.dual                                 # Antibivector-
Inertia = ga.gatype((Antibivector, Bivector))                  # Antibivector- <- Bivector+
Spatial = ga.gatype(ga.subspace("x y z"), orientation=Orientation.extrinsic)  # Vector+
ga.gatype(ga.subspace.vector())                                # TypeError: a subspace has no orientation
```

## Examples

* [Surface normals](../examples/orientation/normals/normals.ipynb): a plane's side and a triangle's winding under a mirror.
* [Mechanical work](../examples/orientation/mechanical_work/mechanical_work.ipynb): a mirrored crank receives the same positive power.
* [Energy in a mirror](../examples/orientation/mirror_energy/mirror_energy.ipynb): a spinning body and its mirror image have the same kinetic energy.
* [A circle constructed twice](../examples/orientation/conformal_circle/conformal_circle.ipynb): the same circle through ordered points and as a cut of two surfaces, with opposite orientations.
