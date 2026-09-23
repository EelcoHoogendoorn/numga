# Dimension Polymorphism and Squeeze Rules

- No dimension polymorphism: functions must accept clean array inputs (`coords: np.ndarray` of shape `(..., d)`), never varargs (`*coords`) or dimension-branching logic (`if len(coords) ...`).
- Certainly no `squeeze`: never call `np.squeeze` or drop axes unpredictably; preserve batch and coordinate axes uniformly.
