# 3. The scalar product within the geometric product

The geometric product of two vectors has a scalar part and a bivector part. For two vectors in numga:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
a * b
```

```text
32 - 3 xy - 6 xz - 3 yz
```

The scalar part is the scalar product of the two vectors, $a \cdot b$. Selecting it and discarding the bivector part gives the scalar product of vector algebra, as taking the real part of a complex number gives a real number.

The number of basis vectors in a product is its grade: scalars have grade 0, vectors grade 1 and bivectors grade 2. The scalar part is the part of grade 0, the bivector part the part of grade 2.

The scalar part of the geometric product and the scalar product, one under the other:

```python
(a * b).select_grade(0)
```

```text
32
```

```python
a.scalar_product(b)
```

```text
32
```

The bivector part is what is discarded:

```python
(a * b).select_grade(2)
```

```text
-3 xy - 6 xz - 3 yz
```

The algebraic structure of the scalar product is therefore all still there. Everything that can be done with the scalar product can be done with the geometric product, by discarding its bivector part: vector algebra is a subset of geometric algebra.
