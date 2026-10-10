# 14. Outermorphisms and the determinant

A map on vectors extends to every grade by mapping each factor of an outer product. The extension is the outermorphism: on bivectors it takes $a \wedge c$ to $A(a) \wedge A(c)$, and on the pseudoscalar, which has one dimension, it is a multiple of the identity, the determinant. Of all maps on bivectors, the outermorphism is the one the map on vectors determines.

For example, for the map `A` below, the outermorphism on bivectors maps the outer product of two vectors to the outer product of their images:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
c = 7 * x + 8 * y + 10 * z
A = Vector + a * (b | Vector)
A.outermorphism(Bivector)(a ^ c) - (A(a) ^ A(c))
```

```text
  yz  zx  xy
   0   0   0
```

On the pseudoscalar it multiplies by the determinant:

```python
A.outermorphism(Pseudoscalar)(x ^ y ^ z)
```

```text
33 xyz
```

```python
A.det()
```

```text
33
```

For the rotation of vectors by a rotor, the outermorphism on bivectors is the rotation of bivectors that the rotor gives directly:

```python
rotor = ((x ^ y) * (-pi / 4)).exp()
rotation = rotor >> Vector
rotation.outermorphism(Bivector) - (rotor >> Bivector)
```

```text
    yz  zx  xy
yz   0   0   0
zx   0   0   0
xy   0   0   0
```

The adjugate is the determinant times the inverse of the outermorphism on bivectors:

```python
A.adjugate() - A.det() * A.outermorphism(Bivector).inverse()
```

```text
    yz  zx  xy
yz   0   0   0
zx   0   0   0
xy   0   0   0
```

In matrix notation, with each bivector named by the basis vector it lacks, the outermorphism on bivectors reads as the cofactor matrix $C$ of $A$, the outermorphism on the pseudoscalar as $\det A$, and the relation with the adjugate as Cramer's rule, $\det(A)\, C^{-1} = A^T$.
