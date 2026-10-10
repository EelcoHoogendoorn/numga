# 1. Open slots

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

The type marks an input slot. Each column is the result for one basis vector in that slot.

Leaving both factors open gives:

```python
Vector | Vector
```

```text
first slot x:
   x  y  z
1  1  0  0

first slot y:
   x  y  z
1  0  1  0

first slot z:
   x  y  z
1  0  0  1
```

This map takes two vectors and returns their inner product. It is linear in each slot separately. Such maps from multivectors to a multivector are extensors.

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

The output is a bivector: the product determines both the input and output types. Each column holds the bivector formed with one input basis vector. Leaving both factors open, `Vector ^ Vector`, gives the outer product as a map with two vector slots.

In index notation, an open slot reads as a free index. With each bivector named by the basis vector it lacks, the inner product reads as $a_i b_i$, with $b$ open as $a_i$, and with both open as $\delta_{ij}$; the outer product reads as $\epsilon_{ijk} a_j b_k$, with $b$ open as $\epsilon_{ijk} a_j$, and with both open as $\epsilon_{ijk}$.
