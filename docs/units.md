# Physical units in numga

This document describes how numga checks physical units: how a context opts in, how values get their units, how each operation carries units from its operands to its result, and how checked and unchecked code work together. [Conventions](conventions.md) states the products whose units are checked here, and [Orientation in numga](orientation.md) describes orientation, which types check in the same way.

numga keeps units on the coefficients of values. An algebra and its basis vectors carry none. A value's type holds one unit for each of its blades, and a map's type one unit for each pair of an output blade and an input blade. Units are worked out when the type of a result is formed, and types are cached, so checking adds little to a computation, and an inconsistent expression raises a `GATypeError` before its numbers are computed.

## Opting in

A context checks units when it is built with `units=True`:

```python
from numga import NumpyContext, dim
from numga.algebras import PGA3D

mv = NumpyContext(PGA3D, units=True).multivector
```

A context with and one without `units=True` share their algebra and its types. A value built in a context without it is unchecked, and so is every result that has an unchecked operand. Existing code therefore runs unchanged, and a checked value can be passed to code that does not check, such as a renderer.

## Dimensions

`dim` makes a dimension from any attribute name: `dim.length`, `dim.mass`, `dim.time`, `dim.temperature`, `dim.pixel`. Dimensions multiply, divide and take rational powers, and `dim.none` is the dimension of a number without a unit:

```python
force = dim.mass * dim.length / dim.time**2
dim.length.root(2)                                            # length^(1/2)
```

A type records dimensions and no particular unit. A value in metres and a value in millimetres both have the dimension length, so any coherent system of units passes, and a change of units stays a multiplication by a number. The examples give extensivity a dimension of its own, `dim.extent`: an extensive quantity carries it to the first power and an intensive one does not.

## Values with units

A number gets its unit where it is constructed, by a keyword or by multiplying a value by a dimension:

```python
dt = mv.scalar(0.025, unit=dim.time)                          # [] Scalar+, time
point = mv.point([1.0, 2.0, 3.0], unit=dim.length)            # [] Antivector-, 1 | length
shift = mv.wx * dim.length                                    # [] Bivector+, length
```

`mv.point` and `mv.direction` take the unit of their coordinates. A point then has no unit on its weight blade and the unit of its coordinates on its three blades with w. A bare number has no unit, and it multiplies a value without changing the value's units. Adding a nonzero number to a value with a unit raises, and the literal `0` is the one number that takes any unit, so `sum(values)`, `x + 0` and `x > 0` work.

A type alias carries units where an expression needs them, for a loop's carry, a cast or a signature:

```python
Scalar = PGA3D.gatype.scalar()
Bivector = PGA3D.gatype.bivector()
Duration = Scalar.derive.unit(dim.time)                       # Scalar+, time
Rate = Bivector.derive.unit(1 / dim.time).derive.projective(dim.length)   # Bivector+, 1/time | length/time

dt = mv(Duration, 0.025)                                      # [] Scalar+, time
```

`derive.projective(dim.length)` multiplies the units of the blades that contain the algebra's null basis vector by a length, which gives a projective type the pattern of its coefficients. A type whose units differ in some other way is taken from a value, `PGA3D.gatype(point)`, or given its units block by block:

```python
Point = PGA3D.gatype.antivector()
WorldPoint = PGA3D.gatype(Point.output_subspace, orientation=Point.orientation,
                          unit={Point.derive.nondegenerate: dim.none, Point.derive.degenerate: dim.length})
```

A type is its identity map, as everywhere in numga. `Duration + Duration` therefore builds a map, and `Scalar * dim.time` raises: a type gets units through `derive`, and a value through its constructor or a dimension.

## Reading units

`value.units` holds a unit for each nonzero blade or map entry, and `value.unit` returns the one unit that all of them share, raising where they differ. `value.has_units` says whether a value is checked. The signature of a type ends with its distinct units, so `Rate.signature` ends with `1/time | length/time`, and the trailing comments in the examples give units in the same form.

## How operations carry units

Every exact operation of numga is a table of numbers that combines input blades into output blades. Each term of the table multiplies the units of its input blades, and every term that lands on one output blade has to give the same unit. The rules below follow from that one condition.

| Operation | Rule |
| --- | --- |
| Sums, differences, stacks, comparisons, `where` | equal units on each blade; the literal `0` takes any unit |
| Geometric, outer, inner and regressive products, commutators, sandwiches | each term multiplies the units of its two blades, and the terms on one output blade agree |
| `.dual()` | each blade's unit moves to its complement |
| `exp`, `log` of a multivector | one unit on each blade at every order of the series |
| Functions of a scalar, such as `exp`, `log`, `cos`, `tanh` | an argument without a unit |
| `square_root`, Cholesky factors, normalized modes | exponents halved |
| `inverse`, `solve`, `det`, `eig`, `lstsq`, `eigh(metric)` | each connected block of a map's entries factors as an output unit over an input unit |
| `eigh`, `svd`, `cholesky` and `pinv` of a single map | one unit on all the blades they mix |
| `adjugate`, `trace`, transposes | units carried with their entries |

A plane of unit normal paired with a point in metres gives metres, since the normal pairs with the coordinates and the offset with the weight. The join of two such points has metres on its direction blades and square metres on its moment blades. A sandwich of a point by a translator in metres is consistent, while the same sandwich by a Galilean boost, whose blade with w is in metres per second, adds metres per second to metres and raises:

```python
translator = (0.5 * mv.wx * dim.length).exp()                 # [] Even+, 1 | length
boost = (0.5 * mv.wx * (dim.length / dim.time)).exp()         # [] Even+, 1 | length/time
translator >> point                                           # [] Antivector-, 1 | length
```

The exponential of a rotation's generator has a term on the scalar blade at every even order, so the generator has no unit. A rate in 1/time has no exponential, while `(rate * dt).exp()` is a motor with no unit on its rotation blades and length on its blades with w. A bivector that turns in `xy` and has metres on `wx` and pixels on `wy` has no exponential either, because its turning adds the coefficient on `wx` to the one on `wy`.

The complement is a permutation of blades, so it never raises. The complement of a point in metres has metres on its normal and no unit on its offset, which is not the unit pattern of a plane. A duality map that the model means, such as a polarity, an inertia or an electromagnetic medium, is written as a map, and its entries carry the constants that the complement leaves out.

## Maps

A map holds a unit on each nonzero entry, the unit of its output blade over the unit of its input blade. An inertia built from points in metres and masses in kilograms has kilogram square metres from angular rate to angular momentum, kilograms from velocity to linear momentum, and kilogram metres on the entries that couple them:

```python
masses = mv.scalar([[1.0], [2.0], [3.0]], unit=dim.mass)      # [3] Scalar+, mass
places = mv.point(coordinates, unit=dim.length)               # [3] Antivector-, 1 | length
inertia = (places * masses).regressive(places.commutator(Bivector)).sum(axis=0)   # [] Antibivector- <- Bivector+
momentum = inertia(rate)                                      # [] Antibivector-
energy = (rate & momentum) / 2                                # [] Scalar+, length^2*mass/time^2
```

Because the units of a map's entries are ratios, the same inertia applies to a rate and to its rate of change, and gives a momentum and a forque. `inertia.inverse()` and `inertia.solve(momentum)` have consistent units, because the inertia's entries factor as the units of a momentum's blades over the units of a rate's. A generalized eigenproblem gives squared frequencies in 1/time² and modes normalized against the inertia:

```python
omega_squared, modes = (Bivector & stiffness).eigh(Bivector & inertia)   # [6] Scalar+, 1/time^2
```

A decomposition of a single map, such as `eigh`, `svd` or `cholesky`, uses the Euclidean product on coefficients, and it raises where the blades it mixes have different units.

## Checked and unchecked code together

An operation with an unchecked operand gives an unchecked result, so a checked rotor applied to an unchecked point gives an unchecked point. `value.cast(Type)` and `value.retyped(Type)` compare units when both the value and the type have them, and they keep the value's units when the type has none.

A loop's carry gets its full type on entry, which for units means that a carry that starts with fewer blades than it fills is retyped to the type it reaches. The JAX backend keeps units through `jit`, `scan` and `derivative`:

```python
import jax

Motor = (rate * dt).exp().gatype                              # Even+, 1 | length

def step(carry, _):
    motor, turning = carry
    return ((motor * (turning * dt).exp()).retyped(Motor), turning), None

(motor, rate), _ = jax.lax.scan(step, (mv.rotor().retyped(Motor), rate), None, length=n_steps)
```

`mv.zeros(Type, shape)` with a type that has no units gives a zero that takes the units of the first value added to it, which serves as the start of an accumulation.
