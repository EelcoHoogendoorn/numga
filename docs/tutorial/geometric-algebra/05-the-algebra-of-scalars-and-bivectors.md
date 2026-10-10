# 5. The algebra of scalars and bivectors

The product of two vectors is a scalar plus a bivector. Two such products, each a scalar $s$ plus a bivector $B$, multiply term by term:

$$(s_1 + B_1)(s_2 + B_2) = s_1 s_2 + s_1 B_2 + s_2 B_1 + B_1 B_2$$

The first three terms are a scalar and two bivectors. The last is a product of two bivectors, which is a scalar plus a bivector. The result is again a scalar plus a bivector, the same structure as each factor. The products of the scalar and the three bivectors, viewed as a multiplication table:

|        | $1$    | $xy$   | $yz$   | $xz$   |
| ------ | ------ | ------ | ------ | ------ |
| $1$    | $1$    | $xy$   | $yz$   | $xz$   |
| $xy$   | $xy$   | $-1$   | $xz$   | $-yz$  |
| $yz$   | $yz$   | $-xz$  | $-1$   | $xy$   |
| $xz$   | $xz$   | $yz$   | $-xy$  | $-1$   |

Every entry is a scalar or a bivector. The scalars and the three bivectors are closed under the geometric product: an algebra of their own.

In numga, for two products of vectors $ab$ and $bc$:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
c = 7 * x + 8 * y + 9 * z
(a * b) * (b * c)
```

```text
3850 - 462 xy - 924 xz - 462 yz
```

This algebraic structure is identical to that of the quaternions. Hamilton's quaternions have three units $i$, $j$ and $k$, related by

$$i^2 = j^2 = k^2 = ijk = -1$$

which gives their multiplication table:

|       | $1$   | $i$   | $j$   | $k$   |
| ----- | ----- | ----- | ----- | ----- |
| $1$   | $1$   | $i$   | $j$   | $k$   |
| $i$   | $i$   | $-1$  | $k$   | $-j$  |
| $j$   | $j$   | $-k$  | $-1$  | $i$   |
| $k$   | $k$   | $j$   | $-i$  | $-1$  |

With $i = xy$, $j = yz$ and $k = xz$, it matches the table of the scalars and bivectors above cell for cell. The geometric product of two vectors, a scalar plus three bivectors, therefore has the algebraic structure of a quaternion: two such products multiply as two quaternions do.
