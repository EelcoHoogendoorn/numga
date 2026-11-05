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

**Adjugate.** `T.adjugate()` is defined by the two pairings it carries, each read with its extrinsic factor first, as every reading is:

```python
T.adjugate()(c) & x == c & T(x)    # intrinsic <- intrinsic
x & T.adjugate()(c) == T(x) & c    # extrinsic <- extrinsic
T.adjugate()(c) & x == T(x) & c    # extrinsic <- intrinsic
x & T.adjugate()(c) == c & T(x)    # intrinsic <- extrinsic
```

The order matters in an even dimension. Swapping the factors of a pairing of complementary grades costs the sign of moving one of them across the pseudoscalar: nothing in an odd dimension, and the grade involute in an even one, where the complement of an odd grade is odd.

```python
b & a == a & b                  # n odd
b & a == a.involute() & b       # n even
```

A blade's complement can be taken on its left or on its right, and in an even dimension the two differ on odd grades. The adjugate with the complement on the left of both pairings, `L(c) & x == c & T(x)`, reads the extrinsic factor first only where both spaces of `T` are intrinsic; `T.adjugate()` is `L` with the involute on each extrinsic side, on its output where the input of `T` is extrinsic and on its input where the output of `T` is extrinsic. The two differ by a sign where exactly one extrinsic side has an odd grade, in an even dimension only: for a gradient `Vector <- Scalar` in the plane, `L == -T.adjugate()`. Read extrinsic first, in every dimension the adjugate is its own inverse, `T.adjugate().adjugate() == T`, and composes backwards, `T(S).adjugate() == S.adjugate()(T.adjugate())`; a form symmetric under the reading is its own adjugate, as the Euclidean metric on vectors, `Vector.dual()`, is; and a positive form pulls back to a positive one, as in `G.adjugate()(Vector.dual()(G))` for a gradient `G` from scalars to vectors. `L` makes that metric minus its own adjugate, and that pullback negative, in every even dimension.

The involute in the sandwich `n >> x` is the sign of moving a vector across `x`, `n ^ x == x.involute() ^ n`, in every dimension; the pseudoscalar moves across `x` the same way in an even dimension only, and so the pairing pays the involute there alone. A sandwich keeps the pairing in either order, `(V >> a) & (V >> b) == a & b`, and keeps grades, so its adjugate is the reverse sandwich, `(V >> Space).adjugate()(c) == V << c`. The adjoint has no sides: the scalar product commutes in every grade.

**Adjoint.** `T.adjoint()` is defined by

```python
T.adjoint()(b).scalar_product(a) == b.scalar_product(T(a))
```

**Trace.** For inputs `x` and `y` from complementary spaces, `T.trace(1, 2)` is the trace of the map `M` with `T(x, y)` the pairing of `M(x)` and `y` read with its extrinsic factor first, `M(x) & y`, or `y & M(x)` where only `y` is extrinsic; so a form traces alike with its inputs in either order. For the output against an input, it is the trace of the map.

**Contraction.** For inputs `x` and `y` from one space, `T.contract(1, 2)` is the trace of the map `M` with `T(x, y) == M(x).dot(y)`; for the output against an input, it is the trace of the map.

**Outermorphism.** `T.outermorphism` is defined by

```python
T(a ^ b) == T(a) ^ T(b)    # vectors to vectors
T(p & q) == T(p) & T(q)    # antivectors to antivectors
T(a ^ b) == T(a) & T(b)    # vectors to antivectors
```

**Spectra of forms.** For symmetric forms `F` and `G`, `F.eigh(G)` solves `F(v) == eigenvalue * G(v)`; without a partner, `G` is the dot form of `F`'s space.

## Norms

**Norm.** `x.norm()` and `x.normalized()` are defined by

```python
x.norm() * x.norm() == x * ~x
x.normalized() == x.norm().inverse() * x
```
