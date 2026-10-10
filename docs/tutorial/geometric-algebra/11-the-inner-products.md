# 11. The inner products

The inner product of a part of grade $r$ and a part of grade $s$ is the part of grade $|r - s|$ of their geometric product, with $\langle M \rangle_k$ the part of grade $k$ of a multivector $M$:

$$A \cdot B = \langle A B \rangle_{|r - s|}$$

For basis elements:

```python
x | x
```

```text
1
```

```python
x | (x * y)
```

```text
y
```

```python
(x * y) | (x * y * z)
```

```text
-z
```

```python
x | y
```

```text
0
```

```python
x | (y * z)
```

```text
0
```

The inner product of two basis elements is their geometric product when the basis vectors of one are all among those of the other, and zero otherwise. The shared basis vectors square to one, and only the rest remains. The inner product contracts, and lowers the grade.

For two vectors, it is the scalar part of their geometric product, the scalar product:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
a | b
```

```text
32
```

The scalar part of $a b$ is the same in both orders, so for two vectors the inner product is also half the sum of the two orders, and together with the outer product it gives back the geometric product:

$$a \cdot b = \tfrac{1}{2} (a b + b a), \qquad a b = a \cdot b + a \wedge b$$

The product of a vector and a bivector, here $b \wedge d$ with $d = 7 x + 8 y + 10 z$:

```python
d = 7 * x + 8 * y + 10 * z
a * (b ^ d)
```

```text
12 x - 9 y + 2 z - 3 xyz
```

```python
a | (b ^ d)
```

```text
12 x - 9 y + 2 z
```

```python
a ^ (b ^ d)
```

```text
-3 xyz
```

The vector part, of grade $2 - 1$, is the inner product, and the trivector part, of grade $2 + 1$, the outer product. The product of a vector and a bivector splits the same way as the product of two vectors:

$$a B = a \cdot B + a \wedge B$$

Here the halves trade places: the outer product of a vector and a bivector is half the sum $\tfrac{1}{2} (a B + B a)$, and the inner product half the difference. The grades, not the order of the factors, define the two products.

The literature defines several inner products: the left contraction, the right contraction, Hestenes' inner product and the scalar product among them. They differ in which grade they select, mostly when the left factor has the higher grade. All of them are grade selections of the geometric product.
