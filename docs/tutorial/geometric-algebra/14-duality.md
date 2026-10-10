# 14. Duality

The trivector $I = xyz$ is the pseudoscalar of three-dimensional space. The product of each basis element with it:

|           | $1$     | $x$    | $y$     | $z$    | $xy$   | $xz$   | $yz$   | $xyz$  |
| --------- | ------- | ------ | ------- | ------ | ------ | ------ | ------ | ------ |
| times $I$ | $xyz$   | $yz$   | $-xz$   | $xy$   | $-z$   | $y$    | $-x$   | $-1$   |

```python
I = x * y * z
x * I
```

```text
yz
```

Each product consists of exactly the basis vectors the element lacks. Multiplying by the pseudoscalar pairs every basis element with its complement, up to sign, and takes grade $k$ to grade $3 - k$. This pairing is duality.

The square of the pseudoscalar, and the difference of its products with $x$ in the two orders:

```python
I * I
```

```text
-1
```

```python
x * I - I * x
```

```text
0
```

The pseudoscalar squares to $-1$, and it commutes with $x$; the same holds for every element of three-dimensional space.

The outer product of two vectors, taken to its dual:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
(a ^ b) * I.inverse()
```

```text
-3 x + 6 y - 3 z
```

In vector algebra notation, $(a \wedge b)\, I^{-1}$ reads as the cross product $a \times b$.
