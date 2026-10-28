# Learn a noisy quantum gate

Four prepared qubit states and a four-outcome detector determine a gate's action on every state. The notebook models the states of a qubit as the points of a ball in projective space and the detector's effects as planes, so that a probability is the pairing `effect & state`. It constructs rotation, dephasing and relaxation channels as maps from points to points, simulates their measurement probabilities, and reconstructs each `Point <- Point` map from reciprocal frames that are the meets of the effects' planes and the joins of the preparations.

The learned maps turn the Bloch sphere into a sphere or ellipsoid, with relaxation shifting its centre. A further figure tests predictions on other pure and mixed states. Four probes in general position join to a volume, while two opposite probes join only to a line, which is why they cannot determine the whole map. An animation composes the learned channels to predict repeated use.

Preparations and detector effects are calibrated, and probabilities are exact. The example isolates noise in the gate; it does not sample finite measurement counts.

`process_tomography_bloch.ipynb` is the conventional twin: it computes the same experiment with a state as a scalar plus its Bloch vector, a probability as a scalar product, and the reconstruction by dual frames and typed solves.

Open `process_tomography.ipynb` in Jupyter. To generate the figures and animation headlessly from the repository root:

```sh
MPLBACKEND=Agg python -m examples.quantum.process_tomography.scenarios
```

Outputs are saved under incrementing names in `plots/`.

The construction follows [Chuang and Nielsen's quantum process tomography](https://arxiv.org/abs/quant-ph/9610001) and the tetrahedral qubit measurement in [Renes et al.](https://arxiv.org/abs/quant-ph/0310075).
