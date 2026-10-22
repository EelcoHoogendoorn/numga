# 12. Trace and contraction

The trace pairs the output of a map with one of its axes and drops both. It needs no metric: an output and an input of the same type pair through the complement.

A contraction pairs two inputs with each other. Two inputs of the same type pair only through the inner product of their axis, the metric, so a contraction is a separate operation from the trace.

For example, the trace of the second moment of three vectors is the sum of its eigenvalues:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
c = 7 * x + 8 * y + 10 * z
moment = a * (a | Vector) + b * (b | Vector) + c * (c | Vector)
moment.trace()
```

```text
304
```

For a map with several axes, the trace can pair the output with any matching input. Pairing the output of `Vector * (Vector | Vector)` with its first input sums over the three basis directions and leaves three times the inner product of the other two; pairing it with its second input leaves the inner product of the first and the third:

```python
(Vector * (Vector | Vector)).trace(0, 1)(a, b)
```

```text
96
```

```python
(Vector * (Vector | Vector)).trace(0, 2)(a, b)
```

```text
32
```

The form `Vector | moment` has two inputs of the same type, and its contraction pairs them through the inner product:

```python
(Vector | moment).contract()
```

```text
304
```

In index notation, the trace reads as an index repeated between an upper and a lower position, $M^i{}_i$, and the contraction of two lower indices as one through the metric, $g^{ij} F_{ij}$.

Example: [vortices in a plane of fluid](../../../examples/fluids/vortices/vortices.ipynb), where the trace and the contraction of the flow's derivative read off how it spreads and how it turns.
