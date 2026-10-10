# 10. The outer product

The outer product of a part of grade $r$ and a part of grade $s$ is the part of grade $r + s$ of their geometric product. Writing $\langle M \rangle_k$ for the part of grade $k$ of a multivector $M$:

$$A \wedge B = \langle A B \rangle_{r + s}$$

For basis elements:

```python
x ^ y
```

```text
xy
```

```python
x ^ (y * z)
```

```text
xyz
```

```python
x ^ x
```

```text
0
```

```python
x ^ (x * y)
```

```text
0
```

The outer product of two basis elements is their geometric product when they share no basis vector, and zero when they share one: a shared basis vector squares to one and leaves the product short of grade $r + s$. The outer product only joins basis vectors that are new to each other, and raises the grade.

For two vectors, it is the bivector part of their geometric product:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
a ^ b
```

```text
-3 xy - 6 xz - 3 yz
```

The products $a b$ and $b a$:

```python
a * b
```

```text
32 - 3 xy - 6 xz - 3 yz
```

```python
b * a
```

```text
32 + 3 xy + 6 xz + 3 yz
```

The scalar part is the same in both orders, and the bivector part changes sign. For two vectors, the outer product is therefore also half the difference of the two orders:

$$a \wedge b = \tfrac{1}{2} (a b - b a)$$

The outer product of a vector with itself:

```python
a ^ a
```

```text
0
```

It is zero. The outer product of three vectors, with $d = 7 x + 8 y + 10 z$:

```python
d = 7 * x + 8 * y + 10 * z
a ^ b ^ d
```

```text
-3 xyz
```

The vector $c = 2 b - a$ is a sum of multiples of $a$ and $b$. Its outer product with them:

```python
c = 2 * b - a
a ^ b ^ c
```

```text
0
```

It is zero. The outer product of vectors is zero exactly when the vectors are linearly dependent; otherwise its grade is the number of vectors.

In matrix notation, the coefficient of $xyz$ in $a \wedge b \wedge d$ reads as the determinant of the matrix with rows $a$, $b$ and $d$:

$$\det \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 10 \end{pmatrix} = -3$$
