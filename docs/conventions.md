# Conventions

The definitions numga's operations rest on, each by the identity that defines it. The [tutorials](tutorial/) motivate them; [orientation](orientation.md) and [extensor syntax](extensor_syntax.md) give the full rules and interface.

## Orientation

Every type is `intrinsic` or `extrinsic`. Writing `sign(x)` for `1` on an extrinsic `x` and `-1` on an intrinsic one, the geometric product and the reflection of `x` by a vector `n` are defined to satisfy

```python
sign(a * b) == sign(a) * sign(b)
reflection(x) == sign(x) * n * x.involute() * n
```

## Scalars

The unit elements of grades `0` and `n`, one of each orientation, are named

| Grade | `extrinsic` | `intrinsic` |
| --- | --- | --- |
| 0 | `scalar` | `antipseudoscalar` |
| n | `pseudoscalar` | `antiscalar` |

```python
antipseudoscalar * antipseudoscalar == scalar
antiscalar == antipseudoscalar * pseudoscalar
```

## Products

**Regressive product.** The regressive product is the associative product that satisfies, for every `a` and `b` of complementary grades,

```python
a ^ b == (a & b) * antiscalar
```

**Dot.** `a.dot(b)` is the scalar part of `a * ~b`.

**Dual.** The dual flips the orientation. In a nondegenerate algebra it is the product with the unit volume,

```python
b.dual() == b.reverse() * antiscalar
```

and so reads the metric: `a & b.dual() == a.dot(b)` for every `a` of `b`'s grade. In a degenerate algebra the product gives the dual of every blade of non-null basis directions, but sends a blade with a null direction to zero; such a blade has a rule of its own, pairing with its dual to the square of its other directions, as `w & w.dual() == 1` and `(y ^ w) & (y ^ w).dual() == y.dot(y)`. `a.dual_inverse()` is the dual's inverse.

**Inner product.** For `a` of grade `r` and `b` of grade `s`,

```python
a | b == (a * b).select[abs(r - s)]
```

**Commutator.** `a.commutator(b) == (a * b - b * a) / 2` and `a.anticommutator(b) == (a * b + b * a) / 2`.

**Sandwich.** For a vector `n`, extended to every versor by multiplication, with `V << x` being `~V >> x`,

```python
n >> x == sign(x) * n * x.involute() * n
```

## Maps

**Adjugate.** `T.adjugate()` is defined by

```python
T.adjugate()(c) & x == c & T(x)
```

**Adjoint.** `T.adjoint()` is defined by

```python
T.adjoint()(b).scalar_product(a) == b.scalar_product(T(a))
```

**Trace.** For inputs `x` and `y` from complementary spaces, `T.trace(1, 2)` is the trace of the map `M` with `T(x, y) == M(x) & y`; for the output against an input, it is the trace of the map.

**Contraction.** For inputs `x` and `y` from one space, `T.contract(1, 2)` is the trace of the map `M` with `T(x, y) == M(x).dot(y)`; for the output against an input, it is the trace of the map.

**Outermorphism.** `T.outermorphism` is defined by

```python
T(a ^ b) == T(a) ^ T(b)    # vectors to vectors
T(p & q) == T(p) & T(q)    # antivectors to antivectors
T(a ^ b) == T(a) & T(b)    # vectors to antivectors
```

**Spectra of forms.** For symmetric forms `F` and `G`, `F.eigh(G)` solves `F(v) == eigenvalue * G(v)`; without a partner, `G` is the dot form of `F`'s space.

## Between algebras

**Values of a subalgebra.** Where every generator of an algebra is among those of `mv`'s algebra, by name and with the same square, as the algebra of space `x+y+z+` lies in its projective algebra `w0x+y+z+`, `mv.from_subalgebra` takes its values by kind. With `small` the smaller algebra's value constructors and `x` any blade of it, it is the linear map defined by

```python
mv.from_subalgebra(small.x) == mv.x                  # extrinsic: the same blade
mv.from_subalgebra(small.x.dual()) == mv.x.dual()    # intrinsic: the same complement
mv.from_subalgebra(f)(mv.from_subalgebra(a)) == mv.from_subalgebra(f(a))    # a map, each axis by its kind
```

Products of extrinsic elements carry over, as do joins of intrinsic ones and an extrinsic element's products and pairings with an element on its right, `plane & point` and `rotor >> point`. A point of space, an offset from its origin, is the projective direction of that offset, and `origin + mv.from_subalgebra(offset)` the point at it, with `origin = mv.w.dual()`.

## Norms

**Norm.** `x.norm()` and `x.normalized()` are defined by

```python
x.norm() * x.norm() == x * ~x
x.normalized() == x.norm().inverse() * x
```
