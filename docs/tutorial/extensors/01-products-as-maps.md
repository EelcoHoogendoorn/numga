# 1. Open axes

The inner product of two given vectors is a scalar. For example:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
a | b
```

```text
32
```

With $a$ fixed and $b$ allowed to vary, the expression describes a linear map from vectors to scalars. Writing `Vector` in place of $b$ leaves that factor open:

```python
a | Vector
```

```text
   x  y  z
1  1  2  3
```

Leaving a factor open turns the product into a map. Each column is the result for one basis vector in the open axis.

Leaving both factors open gives:

```python
Vector | Vector
```

```text
first input x:
   x  y  z
1  1  0  0

first input y:
   x  y  z
1  0  1  0

first input z:
   x  y  z
1  0  0  1
```

Leaving both factors open turns the inner product itself into a map with two axes, linear in each axis separately. Such maps in geometric algebra are extensors.

Every product of the algebra is linear in each factor, so the same construction applies to the outer product:

```python
a ^ Vector
```

```text
     x   y   z
yz   0  -3   2
zx   3   0  -1
xy  -2   1   0
```

The product determines what the map takes in and what it returns: here a vector goes in and a bivector comes out. Leaving both factors open, `Vector ^ Vector`, is the outer product itself as a map.

In index notation, an open axis reads as a free index. With each bivector named by the basis vector it lacks, the inner product reads as $a_i b_i$, with $b$ open as $a_i$, and with both open as $\delta_{ij}$; the outer product reads as $\epsilon_{ijk} a_j b_k$, with $b$ open as $\epsilon_{ijk} a_j$, and with both open as $\epsilon_{ijk}$.

Example: [as-rigid-as-possible surface deformation](../../../examples/surfaces/arap/arap.ipynb), where leaving the vector open in `rotors >> Vector` turns each vertex's rotor into the map that rotates its edges.
