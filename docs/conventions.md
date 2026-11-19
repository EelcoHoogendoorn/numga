# Conventions

This document collects the identities that define numga's operations. The [geometric algebra](tutorial/geometric-algebra/00-introduction.md) and [extensor](tutorial/extensors/00-introduction.md) tutorials motivate them, while [Orientation in numga](orientation.md) and [Extensor syntax and methods](extensor_syntax.md) give the full rules and interface.

## Orientation

Every type is `intrinsic` or `extrinsic`. Writing `sign(x)` for `1` on an extrinsic `x` and `-1` on an intrinsic one, the geometric product and the reflection of `x` by a vector `n` satisfy

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

**Dual.** The dual flips orientation and complements blade support. In a nondegenerate algebra it multiplies by the unit volume,

```python
b.dual() == b.reverse() * antiscalar
```

so that `a & b.dual() == a.dot(b)` for extrinsic `a` and `b` of one grade.

In a degenerate algebra, the product with `antiscalar` gives the dual of every non-null blade and sends any blade containing a null direction to zero. A null-containing blade instead pairs with its dual to the square of its non-null directions, as `w & w.dual() == 1` and `(y ^ w) & (y ^ w).dual() == y.dot(y)`. `a.dual_inverse()` inverts `.dual()`.

**Dot.** `a.dot(b)` takes two values of one orientation. For two extrinsic values it is the scalar part of `a * ~b`; two intrinsic values are measured through their duals,

```python
a.dot(b) == a.dual().dot(b.dual())    # a, b intrinsic
```

so that a direction's dot with itself is its squared length in space, spacetime and projective algebras alike. An extrinsic value and an intrinsic one have no dot; they pair by `&`.

**Inner product.** For `a` of grade `r` and `b` of grade `s`,

```python
a | b == (a * b).select[abs(r - s)]
```

**Commutator.** `a.commutator(b) == (a * b - b * a) / 2` and `a.anticommutator(b) == (a * b + b * a) / 2`.

**Sandwich.** For a vector `n`, extended to every versor by multiplication, with `V << x` equal to `~V >> x`,

```python
n >> x == sign(x) * n * x.involute() * n
```

## Norms

**Norm and antinorm.** `x.norm()` and `x.normalized()` follow the reverse product, while `x.antinorm()` follows `.dot()`:

```python
x.norm() * x.norm() == x * ~x
x.normalized() == x.norm().inverse() * x
x.antinorm() * x.antinorm() == x.dot(x)
```

On an intrinsic value, these measure two distinct sizes. For a projective point, `.norm()` is its weight, so `.normalized()` sets weight one, while `.antinorm()` is its bulk, the weight times its distance from the origin. For a direction, `.antinorm()` is its length in every algebra, so `(p - q).antinorm()` is the distance between two points of weight one.

Each returns NaN where the squared measure under its root is negative, such as the norm of a spatial direction in spacetime or the antinorm of a timelike one.

## Maps

**Adjugate.** `T.adjugate()` is defined by the two regressive pairings it carries, each read with its extrinsic factor first:

```python
T.adjugate()(c) & x == c & T(x)    # intrinsic <- intrinsic
x & T.adjugate()(c) == T(x) & c    # extrinsic <- extrinsic
T.adjugate()(c) & x == T(x) & c    # extrinsic <- intrinsic
x & T.adjugate()(c) == c & T(x)    # intrinsic <- extrinsic
```

Factor order matters in even dimensions. Swapping the factors of a complementary-grade pairing costs the sign of moving one across the pseudoscalar: nothing in an odd dimension, and the grade involute in an even one, where the complement of an odd grade is odd:

```python
b & a == a & b                  # n odd
b & a == a.involute() & b       # n even
```

Read extrinsic first, `.adjugate()` satisfies four invariants in every dimension: it is its own inverse (`T.adjugate().adjugate() == T`), it reverses composition (`T(S).adjugate() == S.adjugate()(T.adjugate())`), a form symmetric under the reading is its own adjugate (such as `Vector.dual()`), and a positive form pulls back to a positive one (such as `G.adjugate()(Vector.dual()(G))` for a gradient `G` of type `Vector+ <- Scalar+`).

A single-formula left-complement operator `L(c) & x == c & T(x)` reads the extrinsic factor first only when both spaces of `T` are intrinsic. In general, `T.adjugate()` equals `L` composed with the grade involute on every extrinsic side of `T` (on the output of `L` when `T`'s input is extrinsic, and on the input of `L` when `T`'s output is extrinsic). In even dimensions the two therefore differ by a sign whenever `T` has an odd number of odd-grade extrinsic sides: for a planar gradient `Vector+ <- Scalar+`, `L == -T.adjugate()`, making `Vector.dual()` minus its own `L` and pulling back a positive form to a negative one.

The involute in the sandwich `n >> x` reflects the sign of moving a vector across `x` (`n ^ x == x.involute() ^ n`) in every dimension, whereas the pseudoscalar pays that involute only in even dimensions. A versor sandwich preserves `&` in either order (`(V >> a) & (V >> b) == a & b`) and preserves grades, so its adjugate is the reverse sandwich, `(V >> Space).adjugate()(c) == V << c`.

**Adjoint.** `T.adjoint()` is defined by the scalar product, which commutes in every grade and needs no side convention:

```python
T.adjoint()(b).scalar_product(a) == b.scalar_product(T(a))
```

**Trace.** For inputs `x` and `y` from complementary spaces, `T.trace(1, 2)` is the trace of the map `M` defined by `T(x, y) == M(x) & y`, or `T(x, y) == y & M(x)` when only `y` is extrinsic, so a form traces to the same value with its inputs in either order. Paired between the output and an input of the same space, it is the trace of the map.

**Contraction.** For inputs `x` and `y` from one space, `T.contract(1, 2)` is the trace of the map `M` defined by `T(x, y) == M(x).scalar_product(y.reverse())`, the metric that `.norm()` follows, which equals `M(x).dot(y)` on an extrinsic space. Paired between the output and an input of the same space, it is the trace of the map.

**Outermorphism.** `T.outermorphism(Grade)` extends a map between vectors and antivectors to `Grade` by

```python
T.outermorphism(Bivector)(a ^ b) == T(a) ^ T(b)        # vectors to vectors
T.outermorphism(Antibivector)(p & q) == T(p) & T(q)    # antivectors to antivectors
T.outermorphism(Antibivector)(a ^ b) == T(a) & T(b)    # vectors to antivectors
```

**Spectra of forms.** For symmetric forms `F` and `G`, `F.eigh(G)` solves `F(v) == eigenvalue * G(v)`. Without a partner, `G` defaults to the metric `.norm()` follows on `F`'s space, `Space.scalar_product(Space.reverse())`, which equals its dot form where the space is extrinsic.
