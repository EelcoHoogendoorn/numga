# 12. The reverse

The reverse of a product of vectors is the same vectors in the opposite order. For the basis elements:

```python
(x * y).reverse()
```

```text
-xy
```

```python
(x * y * z).reverse()
```

```text
-xyz
```

Reversing a product of $k$ different basis vectors takes a fixed number of exchanges of neighbours, each flipping the sign, so the sign depends only on the grade:

| grade              | 0   | 1   | 2   | 3   | 4   |
| ------------------ | --- | --- | --- | --- | --- |
| sign of the reverse | $+$ | $+$ | $-$ | $-$ | $+$ |

For two vectors $a$ and $b$, the reverse of the product $a b$ is $b a$. The product of the two:

$$a b\, b a = a\, (b b)\, a = (b \cdot b)(a \cdot a)$$

For two vectors with squares $14$ and $77$:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
(a * b) * (a * b).reverse()
```

```text
1078
```

The product of a product of vectors with its reverse is a scalar, the product of the squares of the vectors.
