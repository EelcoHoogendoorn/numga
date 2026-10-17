# Orientation in numga

Every `GAType` carries an orientation, `Orientation.extrinsic` or `Orientation.intrinsic`, on its output and on each input slot. Constructors set it, operations propagate it, sums and bindings check it, and the reflection `>>` uses it to pick its sign. This page lists the rules; [Why orientation](why_orientation.md) explains why they are needed. A printed type marks only the intrinsic kind: `bivector` is an extrinsic bivector, `intrinsic bivector` its counterpart, and `intrinsic bivector <- bivector` a map from the first to the second.

```python
from numga import Orientation
```

## Reflection

`versor >> x` applies a versor to `x`. For an even versor it is the sandwich `m * x * m.inverse()`, whatever the orientation. For an odd versor `m`, the sign depends on the orientation:

| | `versor >> x` for odd `m` |
| --- | --- |
| extrinsic `x` | `m * x.involute() * m.inverse()` |
| intrinsic `x` | `-m * x.involute() * m.inverse()` |

With this rule every versor, odd ones included, commutes with every product, the geometric, outer, inner and regressive products and the commutator and anticommutator alike: `m >> (a | b) == (m >> a) | (m >> b)`, and `m >> (a & b) == (m >> a) & (m >> b)`. In a non-degenerate algebra it commutes with the dual as well, `m >> a.dual() == (m >> a).dual()`. Extrinsic scalars are invariant under reflections, and intrinsic scalars change sign.

## Propagation

Reading `extrinsic` as + and `intrinsic` as −:

| Operation | Result orientation |
| --- | --- |
| `a * b`, `a ^ b`, `a \| b`, `a.commutator(b)`, `a.anticommutator(b)` | the product of the signs |
| `a & b` | minus the product of the signs |
| `a.dual()` | flipped |
| `a + b`, `a - b`, `stack`, `concatenate` | equal orientations required; that orientation |
| `~a`, `a.involute()`, `-a`, `s * a` with `s` an extrinsic scalar | unchanged |
| `a.inverse()` of an element | unchanged |
| `versor >> x`, `versor << x` | `x`'s orientation |
| `exp`, `log` and their variants, square roots, `arccos`, `arccosh` | extrinsic input required |
| norms, Study norms, `cos`, `cosh` | extrinsic output |

## The dual

The dual exchanges the two kinds of orientation. It complements an element against the orientation of space, held fixed: a turn in a plane becomes the arrow along the plane's normal that the right-hand rule assigns it. A mirror reverses the handedness the right-hand rule refers to, so the arrow made from a turn reverses with the turn, where a plane with the same coefficients keeps its side:

```python
turn = mv.xy                     # a turn in the xy plane
normal = turn.dual()             # intrinsic vector z: the turn's arrow by the right-hand rule

mv.x >> turn                     # -xy: the mirror x = 0 reverses the turn
mv.x >> normal                   # -z: and the arrow made from it
mv.x >> mv.z                     # +z: the plane z = 0 keeps its positive side
```

This is why `&` carries one net flip: the regressive product is the outer product of the duals, dualized, `a & b == (a.dual() ^ b.dual()).dual()`, three flips in all.

A product with a constant pseudoscalar, `a * I`, has the coefficients of a dual but keeps `a`'s orientation: there `I` is an element of the scene, a product of basis planes that a mirror reflects along with everything else. Where a complement is meant, use `.dual()`.

## What is refused

Combining kinds where they must match raises before any kernel runs:

```python
spin + momentum                  # ValueError: cannot add in bivector and intrinsic bivector: their orientations differ
inertia(momentum)                # ValueError: cannot bind an intrinsic bivector into input slot 0, which takes an extrinsic orientation
inertia.solve(spin)              # ValueError: cannot solve for an extrinsic right-hand side against a map whose output takes an intrinsic orientation
momentum.exp()                   # LookupError: 'exp' takes extrinsic elements; got intrinsic bivector
inertia.eigh()                   # LookupError: no 'eigh' implementation matches GATypes (GAType(intrinsic bivector <- bivector),)
ga.gatype(ga.subspace.vector())  # TypeError: a subspace has no orientation
```

Sums, differences, stacks and concatenations need equal orientations; bindings need the slot's orientation; `solve` needs the output's. `exp`, `log`, their variants, square roots, `arccos` and `arccosh` take extrinsic elements only, so a force cannot generate a motion. `det`, `eig`, `eigh` and `cholesky` take endomorphisms only. A bare subspace is refused wherever a type is expected.

## Constructing types

Orientation is set at construction and never defaults. Plane equations, basis blades and meets of planes are extrinsic; points and joins of points are intrinsic; plain numbers are extrinsic scalars. Grade does not fix orientation. The join of three points is an intrinsic vector, and the meet of three planes an extrinsic point:

```python
winding = a & b & c                                     # intrinsic vector
corner = mv.x ^ mv.y ^ mv.z                             # trivector
```

Named type constructors:

| Constructor | Orientation |
| --- | --- |
| `scalar`, `vector`, `bivector`, `trivector`, `pseudoscalar`, `even`, `odd`, `rotor`, `motor`, `multivector` | extrinsic |
| `antiscalar`, `antivector`, `antibivector`, `antitrivector`, `antiquadvector` | intrinsic |
| `point`, in the spherical algebras | intrinsic |

Values:

| Construction | Orientation |
| --- | --- |
| `mv.vector([...])` and the other named constructors | as the type constructor of that name |
| basis blades `mv.x`, `mv.xy`, ..., and `mv.blade(mask)` | extrinsic |
| plain numbers in an expression | extrinsic scalars |
| `mv.point(coordinates)`, `mv.direction(coordinates)`, in projective algebras | intrinsic |
| `mv(gatype, values)` | the orientation of `gatype` |

Coefficients alone do not say which kind they hold, so a type for a partial layout is best taken from a construction, then narrowed from a named type, and only then built from a subspace with its orientation given:

```python
Line = (Point & Point).output_gatype                    # the type a construction produces
Resultant = ga.gatype.antibivector().derive.within("x y w")   # narrowed from a named type
Shift = ga.gatype.antivector().derive.containing("w z")
Direction = Point.derive.degenerate                     # the part along the degenerate direction
Spatial = Plane.derive.nondegenerate
ga.gatype(subspace, orientation=Orientation.intrinsic)  # from a subspace, orientation required
```

`ga.gatype((Output, Input))` builds a map type from types that already carry their orientations, and refuses an `orientation=` argument. `Type.derive.flipped_orientation` flips the orientation of a type's output.

## Maps

Each slot of a map carries its own orientation:

```python
inertia = (points & points.commutator(Line) * masses).sum()     # intrinsic bivector <- bivector
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
| `f.inverse()`, `f.adjoint()` | slots swapped, each keeping its orientation |
| `f.adjugate()` | acts on the complements, each slot flipped |
| `f.trace()`, `f.contract(...)` | the pairing's orientation times the output's: an extrinsic scalar for an endomorphism, intrinsic for a correlation |
| `f.outermorphism(Type)` | `Type` must have the orientation that products of the map's outputs have |
| `a.eigh(b)` | `a` and `b` of the same type |

An inertia map takes extrinsic rates of turning to intrinsic momenta. It is a correlation, so it has no eigenvalues of its own, while `stiffness.eigh(inertia)` against a second map of its type is well defined. Its adjugate flips both slots back to `intrinsic bivector <- bivector`.

## Extension methods

An extension method declares the orientations it accepts and produces:

```python
ExtensionMethod("exp", input_orientation=Orientation.extrinsic)
ExtensionMethod("norm", output_orientation=Orientation.extrinsic)
```

An input declaration is checked before dispatch, and a mismatch raises `LookupError`. An output declaration sets the result's orientation. A method with neither keeps the orientation its implementation produces.

## Examples

* [Surface normals](../examples/orientation/normals/normals.ipynb): a plane's side and a triangle's winding under a mirror.
* [Mechanical work](../examples/orientation/mechanical_work/mechanical_work.ipynb): a mirrored crank receives the same positive power.
* [Energy in a mirror](../examples/orientation/mirror_energy/mirror_energy.ipynb): a spinning body and its mirror image have the same kinetic energy.
* [A circle constructed twice](../examples/orientation/conformal_circle/conformal_circle.ipynb): the same circle through ordered points and as a cut of two surfaces, with opposite orientations.
