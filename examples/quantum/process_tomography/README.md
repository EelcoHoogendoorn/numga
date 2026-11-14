# Quantum Process Tomography

Four prepared qubit states and a four-outcome detector determine a gate's action on every state. `process_tomography.ipynb` models the states of a qubit as unit-weight points inside a ball in three-dimensional projective space and the detector's effects as planes, so that an outcome probability is the pairing `effect & state`. Rotation, dephasing, and relaxation channels act linearly as `Point <- Point` maps, and each map is reconstructed from reciprocal frames built out of meets of effect planes and joins of prepared points.

The reconstructed maps carry the pure-state sphere to a rotated sphere or a contracted ellipsoid, with relaxation shifting its centre toward the north pole. Validation across pure and mixed states confirms exact probability predictions, a four-point join contrasts spanning tetrahedral probes against two axial probes, and an animation composes the reconstructed maps to predict repeated gate passes.

`process_tomography_bloch.ipynb` is the conventional twin notebook in the Pauli algebra, representing each state as half the unit scalar plus half its Bloch vector, probabilities as scalar products, and reconstruction via dual-frame overlap solves.

Open `process_tomography.ipynb` in Jupyter, or generate the figures and animation headlessly from the repository root:

```sh
MPLBACKEND=Agg python -m examples.quantum.process_tomography.scenarios
```

The construction follows [Chuang and Nielsen](https://arxiv.org/abs/quant-ph/9610001) and the symmetric tetrahedral qubit measurement of [Renes et al.](https://arxiv.org/abs/quant-ph/0310075).
