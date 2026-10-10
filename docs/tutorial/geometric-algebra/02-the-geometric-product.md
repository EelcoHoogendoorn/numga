# 2. The geometric product of vectors

The geometric product of vectors differs from the scalar product in one of the multiplication rules of basis vectors: two different basis vectors no longer multiply to zero, but their products in the two orders sum to zero. For $i \neq j$:

$$\begin{aligned} \text{scalar product:} \quad & e_i e_i = 1, \qquad & e_i e_j &= e_j e_i = 0 \\ \text{geometric product:} \quad & e_i e_i = 1, \qquad & e_i e_j &+ e_j e_i = 0 \end{aligned}$$

Two vectors $a$ and $b$, written in the basis, multiply term by term into a grid of nine terms:

$$a = a_1 x + a_2 y + a_3 z, \qquad b = b_1 x + b_2 y + b_3 z$$

Under the geometric product, the diagonal terms become numbers, as in the scalar product, but the terms off the diagonal no longer drop out:

|           | $b_1 x$           | $b_2 y$           | $b_3 z$           |
| --------- | ----------------- | ----------------- | ----------------- |
| $a_1 x$   | $a_1 b_1$         | $a_1 b_2\, x y$   | $a_1 b_3\, x z$   |
| $a_2 y$   | $a_2 b_1\, y x$   | $a_2 b_2$         | $a_2 b_3\, y z$   |
| $a_3 z$   | $a_3 b_1\, z x$   | $a_3 b_2\, z y$   | $a_3 b_3$         |

Unlike the scalar product, the geometric product of two vectors does not reduce to a scalar. Beside the scalar product on the diagonal, the terms off it remain, each with a product of two different basis vectors. Written out as a multiplication table of the basis vectors:

|       | $x$    | $y$    | $z$    |
| ----- | ------ | ------ | ------ |
| $x$   | 1      | $xy$   | $xz$   |
| $y$   | $yx$   | 1      | $yz$   |
| $z$   | $zx$   | $zy$   | 1      |

By the multiplication rule $e_i e_j + e_j e_i = 0$, each product is the negative of its mirror across the diagonal, $y x = -x y$, so the six terms off the diagonal pair up into three. The product of two vectors has four terms, a scalar and three products of two different basis vectors:

$$a b = (a_1 b_1 + a_2 b_2 + a_3 b_3) + (a_1 b_2 - a_2 b_1)\, xy + (a_1 b_3 - a_3 b_1)\, xz + (a_2 b_3 - a_3 b_2)\, yz$$

The products of two different basis vectors are neither numbers nor vectors. In geometric algebra they are called bivectors.

The same in numga, for the vectors

$$a = x + 2 y + 3 z, \qquad b = 4 x + 5 y + 6 z$$

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
a * b
```

```text
32 - 3 xy - 6 xz - 3 yz
```
