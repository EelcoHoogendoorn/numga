# Learn a noisy quantum gate

Four prepared qubit states and a four-outcome detector determine a gate's action on
every state. The notebook constructs rotation, dephasing and relaxation channels as
sums of sandwiches, simulates their measurement probabilities, and reconstructs each
`State <- State` extensor with dual frames and typed solves.

The learned maps turn the Bloch sphere into a sphere or ellipsoid, with relaxation
shifting its centre. A further figure tests predictions on other pure and mixed
states. Singular values show why two opposite probes cannot determine the whole
map. An animation composes the learned channels to predict repeated use.

Preparations and detector effects are calibrated, and probabilities are exact. The
example isolates noise in the gate; it does not sample finite measurement counts.

Open `process_tomography.ipynb` in Jupyter. To generate the figures and animation
headlessly from the repository root:

```sh
MPLBACKEND=Agg python -m examples.quantum.process_tomography.scenarios
```

Outputs are saved under incrementing names in `plots/`.

The construction follows [Chuang and Nielsen's quantum process tomography](https://arxiv.org/abs/quant-ph/9610001)
and the tetrahedral qubit measurement in [Renes et al.](https://arxiv.org/abs/quant-ph/0310075).
