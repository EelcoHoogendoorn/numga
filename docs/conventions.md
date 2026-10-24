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

**Dual.** The dual is defined implicitly by the relationship, for every `a` and `b` of one grade,

```python
a.dot((b.reverse() * antiscalar).dual_inverse()) == a.dot(b)
```

In a nondegenerate algebra the dots can be dropped, and the relationship becomes the familiar

```python
b.dual() == b.reverse() * antiscalar
```

In a degenerate algebra the relationship leaves the dual of a null blade undefined, and the convention chosen for a null basis blade `e` is its complement, `e & e.dual() == 1`. `a.dual_inverse()` is the dual's inverse.

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

## Norms

**Norm.** `x.norm()` and `x.normalized()` are defined by

```python
x.norm() * x.norm() == x * ~x
x.normalized() == x.norm().inverse() * x
```
