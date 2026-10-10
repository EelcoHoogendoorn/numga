# 4. Products of bivectors

The square of $xy$ follows from the multiplication rules of the geometric product. Exchanging the two middle basis vectors flips the sign, and each basis vector then squares to one:

$$(xy)(xy) = x\, y\, x\, y = -x\, x\, y\, y = -1$$

The square of a bivector is a scalar, $-1$, and the squares of $xz$ and $yz$ come out the same. The product of two different bivectors, such as $xy$ and $yz$, follows from the same multiplication rules:

$$(xy)(yz) = x\, y\, y\, z = x z$$

The product of two different bivectors is a bivector.

The same in numga:

```python
(x * y) * (x * y)
```

```text
-1
```

```python
(x * y) * (y * z)
```

```text
xz
```
