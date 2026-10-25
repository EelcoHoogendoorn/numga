# Orientation in numga

Every `GAType` carries an orientation, `Orientation.extrinsic` or `Orientation.intrinsic`, on its output and on each input. Constructors set it, operations propagate it, sums and bindings check it, and the transform `>>` uses it to pick the sign of a reflection. Type comments use `+` for extrinsic and `-` for intrinsic. This page lists the rules; [Why orientation](why_orientation.md) explains why they are needed.

```python
from numga import NumpyContext, Orientation
from numga.algebras import PGA3D
```

## Constructing types

Every element can carry either orientation. Named constructors choose a convention: `bivector()` is extrinsic and has grade two; `antibivector()` is intrinsic and has the complementary grade, algebra dimension minus two. In `PGA3D`, whose algebra has four dimensions, both are grade two:

```python
ga = PGA3D
mv = NumpyContext(ga).multivector

Bivector = ga.gatype.bivector()                 # Bivector+
Antibivector = ga.gatype.antibivector()          # Antibivector-
```

To change only orientation, use `.derive.flipped_orientation`. It keeps the same blades:

```python
Plane = ga.gatype.vector()                                 # Vector+
Surface = Plane.derive.flipped_orientation                  # Vector-
Point = ga.gatype.antivector()                              # Antivector-
Scalar = ga.gatype.scalar()                                # Scalar+
IntrinsicScalar = Scalar.derive.flipped_orientation        # Scalar-
```

The `anti` constructors select both complementary grade and intrinsic orientation. In particular, `antiscalar()` is an intrinsic pseudoscalar; `IntrinsicScalar` above is an intrinsic scalar.

| Type constructor | Orientation |
| --- | --- |
| `scalar`, `vector`, `bivector`, `trivector`, `pseudoscalar`, `even`, `odd`, `rotor`, `motor`, `multivector` | extrinsic |
| `antiscalar`, `antivector`, `antibivector`, `antitrivector`, `antiquadvector` | intrinsic |
| `point`, in `Spherical3D` | intrinsic |

| Value construction | Orientation |
| --- | --- |
| `mv.vector([...])` and the other named constructors | as the type constructor of that name |
| basis blades `mv.x`, `mv.xy`, ..., and `mv.blade(mask)` | extrinsic |
| plain numbers in an expression | extrinsic scalars |
| `mv.point(coordinates)`, `mv.direction(coordinates)`, in `PGA2D` and `PGA3D` | intrinsic |
| `mv(gatype, values)` | the orientation of `gatype` |

A type taken from an expression carries the orientation its construction produces. Narrowing a type with `.within`, `.containing`, `.degenerate` or `.nondegenerate` keeps its orientation. Constructing one directly from a subspace requires an explicit orientation:

```python
Line = (Point & Point).output_gatype                         # Antibivector-
Resultant = Antibivector.derive.within("x y w")               # Antibivector-
Spatial = ga.gatype(ga.subspace("x y z"), orientation=Orientation.extrinsic)  # Vector+
```

`ga.gatype((Output, Input))` builds a map type from types that carry their own orientations, and refuses an `orientation=` argument. A printed type names the orientation of every axis: `intrinsic bivector <- extrinsic bivector` maps extrinsic bivectors to intrinsic ones. On a map type, `.derive.flipped_orientation` flips only the output.

## Propagation

Reading `extrinsic` as + and `intrinsic` as −:

| Operation | Result orientation |
| --- | --- |
| `a * b`, `a ^ b`, `a \| b`, `a.commutator(b)`, `a.anticommutator(b)` | the product of the signs |
| `a & b` | minus the product of the signs |
| `a.dual()`, `a.dual_inverse()` | flipped |
| `a + b`, `a - b`, `stack`, `concatenate` | equal orientations required; that orientation |
| `~a`, `a.involute()`, `-a`, `s * a` with `s` an extrinsic scalar | unchanged |
| `a.inverse()`, `a.normalized()` of an element | unchanged |
| `versor >> x`, `versor << x` | `x`'s orientation |
| `exp`, `log` and their variants, square roots, `arccos`, `arccosh` | extrinsic input required |
| norms, Study norms, `cos`, `cosh` | extrinsic output |
| `sin`, `tan`, `sinh`, `tanh`, `arcsin`, `arctan`, `arcsinh`, `arctanh` | unchanged |

For an intrinsic scalar, an odd function keeps the sign change under a mirror; an even function removes it. A norm is extrinsic, so dividing by it preserves the element's orientation:

```python
value: IntrinsicScalar = mv(IntrinsicScalar, [0.5])  # Scalar-
value.sin()                                        # Scalar-
value.cos()                                        # Scalar+
value.norm()                                       # Scalar+
value.normalized()                                 # Scalar-
```

## Reflection

`versor >> x`, or `versor.transform(x)`, uses the versor's reverse. For it to act as the composition of reflections in its vector factors, that reverse must equal its inverse: the reverse product must be +1. The formulas below assume this normalization. For a mirror `m` squaring to +1:

| | `m >> x` |
| --- | --- |
| extrinsic `x` | `m * x.involute() * m.inverse()` |
| intrinsic `x` | `-m * x.involute() * m.inverse()` |

A product of k vectors applies k reflections, so the grade involution and the intrinsic sign each enter k times: for an even versor they cancel, and its action is the sandwich `v * x * v.inverse()` whatever the orientation. `versor << x`, or `versor.reverse_transform(x)`, applies the reverse action.

These transformations commute with every product, the geometric, outer, inner and regressive products and the commutator and anticommutator alike: `m >> (a | b) == (m >> a) | (m >> b)`, and `m >> (a & b) == (m >> a) & (m >> b)`. Extrinsic scalars are invariant under reflections, and intrinsic scalars change sign.

## The dual

The dual complements the blade layout and flips orientation. `.dual_inverse()` reverses this operation, also flipping orientation. The regressive product uses volume complements, which agree with `.dual()` in the `PGA3D` algebra used here:

```python
Join = ga.gatype((Line, Point, Point))                         # Antibivector- <- (Antivector-, Antivector-)

join: Join = Point & Point                                    # Antibivector- <- (Antivector-, Antivector-)
join_from_duals: Join = (Point.dual() ^ Point.dual()).dual_inverse()  # Antibivector- <- (Antivector-, Antivector-)
```

These are the same map. The three orientation flips leave one net flip, which gives `&` its propagation rule. In a nondegenerate signature, `.dual()` carries metric signs that the volume complement does not; the formula with `.dual()` is therefore not a signature-independent definition of `&`.

In a non-degenerate algebra, normalized versor transformations commute with the dual: `m >> a.dual() == (m >> a).dual()`. In a degenerate algebra the complement remains defined, but this identity need not hold; in PGA, for example, the dual does not commute with translations.

Multiplication by a fixed pseudoscalar follows the geometric product's orientation rule. With the extrinsic pseudoscalar, `a * mv.pseudoscalar([1])` keeps `a`'s orientation; with the intrinsic one, `a * mv.antiscalar([1])` flips it. Either is an element that transforms along with the rest of the scene. Where a complement is meant, use `.dual()`.

The three operations line up. Multiplying by the intrinsic scalar one keeps every coefficient and flips only the orientation: a cast between extrinsic and intrinsic. Multiplying by the extrinsic pseudoscalar complements the grade and keeps the orientation. The dual does both: in a non-degenerate algebra it is the reverse times the intrinsic pseudoscalar, `a.dual() == a.reverse() * antiscalar`, at every grade. The reverse is what makes an element's outer product with a dual its dot product with the dualized element, `b ^ a.dual() == b.dot(a) * antiscalar`; without it the sign flips on bivectors and trivectors, which square to minus their size. In a degenerate algebra the pseudoscalar squares to zero and no product with it gives the dual, which remains the complement of the blade layout.

## Maps

Each axis of a map carries its own orientation. The inertia of three unit masses takes an extrinsic rate of motion to an intrinsic momentum:

```python
Inertia = ga.gatype((Antibivector, Bivector))                   # Antibivector- <- Bivector+

points: Point = mv.point([[1, 0, 0], [0, 1, 0], [0, 0, 1]])    # [points] Antivector-
inertia: Inertia = (points & points.commutator(Bivector)).sum(axis=0)  # Antibivector- <- Bivector+
rate: Bivector = mv.xy                                       # Bivector+
momentum: Antibivector = inertia(rate)                        # Antibivector-
```

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
| `f.trace()`, `f.contract()` on a unary map | same input and output blade support required; scalar orientation is the product of their signs |
| `f.trace(i, j)` pairing two inputs | complementary spaces required; output orientation multiplied by the regressive pairing's orientation |
| `f.contract(i, j)` pairing two inputs | the same space with an invertible metric required; output orientation multiplied by the metric pairing's orientation |
| `f.outermorphism(Type)` | `Type` must match products of the map's inputs; products of its outputs determine the result |

Pairing two inputs leaves the other axes unchanged. The inertia above is a correlation between two orientations of the same blade space, so its unary trace is intrinsic. Its adjugate has the same type as the inertia: swapping and then complementing the axes and flipping their orientations gives `intrinsic bivector <- extrinsic bivector` again.

`det`, `eig`, `eigh` and `cholesky` accept endomorphisms or square scalar-valued forms. Inertia is neither, but pairing its output with a second rate gives its energy form:

```python
EnergyForm = ga.gatype((Scalar, Bivector, Bivector))           # Scalar+ <- (Bivector+, Bivector+)

kinetic: EnergyForm = Bivector & inertia(Bivector)            # Scalar+ <- (Bivector+, Bivector+)
energy: Scalar = kinetic(rate, rate) / 2                      # Scalar+
```

A form's eigenvalues are measured against a metric form. Omitted, that metric is the input type's own metric; supplied explicitly, it must be a square form on the same input type. For `elastic.eigh(kinetic)`, `elastic` and `kinetic` are the stiffness and inertia forms, and `kinetic` must be positive definite. Two unary correlations must first be paired into forms this way.

## What is refused

Sums, differences, stacks and concatenations need matching orientations on every axis; binding needs the input's orientation, and `solve` needs the output's. Mismatches raise `GATypeError`, a subclass of `TypeError`, before any kernel runs:

```python
rate + momentum                  # GATypeError: orientations differ
inertia(momentum)                # GATypeError: input requires extrinsic orientation
inertia.solve(rate)              # GATypeError: right-hand side requires intrinsic orientation
momentum.exp()                   # GATypeError: exp requires extrinsic orientation
inertia.eigh()                   # GATypeError: requires an endomorphism or a square form
ga.gatype(ga.subspace.vector())  # TypeError: a subspace has no orientation
```

The restriction on `exp` follows from addition: an intrinsic element's even powers are extrinsic and its odd powers intrinsic, so the series would mix orientations in a sum. `log`, the exponential and logarithm variants, square roots, `arccos` and `arccosh` also require extrinsic input.

## Extension methods

An extension method declares the orientations it accepts and produces:

```python
from numga.extension import ExtensionMethod

ExtensionMethod("exp", takes="a value", input_orientation=Orientation.extrinsic)
ExtensionMethod("norm", takes="a value", output_orientation=Orientation.extrinsic)
```

An input declaration is checked before dispatch, and a mismatch raises `GATypeError`. An output declaration sets the result's orientation. A method with neither keeps the orientation its implementation produces.

## Examples

* [Surface normals](../examples/orientation/normals/normals.ipynb): a plane's side and a triangle's winding under a mirror.
* [Mechanical work](../examples/orientation/mechanical_work/mechanical_work.ipynb): a mirrored crank receives the same positive power.
* [Energy in a mirror](../examples/orientation/mirror_energy/mirror_energy.ipynb): a spinning body and its mirror image have the same kinetic energy.
* [A circle constructed twice](../examples/orientation/conformal_circle/conformal_circle.ipynb): the same circle through ordered points and as a cut of two surfaces, with opposite orientations.
