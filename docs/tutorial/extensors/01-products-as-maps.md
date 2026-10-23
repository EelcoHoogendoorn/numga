# 1. Products as maps

The inner product of two given vectors is a scalar. For example:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
a | b
```

```text
32
```

The inner product with $a$ is a linear map from vectors to scalars:

```python
a | Vector
```

```text
   x  y  z
1  1  2  3
```

Each column is the map's result for one basis vector.

The inner product itself is a bilinear form on pairs of vectors:

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

The form is linear in each argument separately. Such multilinear maps in geometric algebra are extensors.

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

The product determines what the map takes in and what it returns: here a vector goes in and a bivector comes out. The map `Vector ^ Vector` takes pairs of vectors to their outer product.

In index notation, with each bivector named by the basis vector it lacks, the inner product reads as $a_i b_i$, the linear form `a | Vector` as $a_i$, and the bilinear form `Vector | Vector` as $\delta_{ij}$. The outer product reads as $\epsilon_{ijk} a_j b_k$, the map `a ^ Vector` as $\epsilon_{ijk} a_j$, and the bilinear map `Vector ^ Vector` as $\epsilon_{ijk}$.

Example: [as-rigid-as-possible surface deformation](../../../examples/surfaces/arap/arap.ipynb), where `rotors >> Vector` maps each vertex's edges to their rotated directions.
