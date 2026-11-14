# Second-Harmonic Generation

A tetrahedral crystal's nonlinear electric response is modelled as a bilinear map `Polarization- <- (Field+, Field+)` from two extrinsic spacetime field bivectors to an intrinsic polarization antibivector. The notebook builds that map from bond planes, rotates the crystal about the beam, binds a fixed pump to probe linear mixing, and integrates the cosine and sine quadratures of the doubled light across matched, mismatched, and periodically poled crystals. The electric-dipole model uses normalized amplitudes and omits pump depletion and interface reflections.

Open `second_harmonic.ipynb` in Jupyter, or write the figures and animation to `plots/` from the repository root:

```sh
MPLBACKEND=Agg python -m examples.electromagnetism.second_harmonic.scenarios
```

The material model follows [Hardhienata et al., *Bond Model and Group Theory of Second Harmonic Generation in GaAs(001)*](https://arxiv.org/abs/1408.1185), and the phase-matching propagation follows [MIT's *Optical Properties of Solids*, chapter 11](https://web.mit.edu/6.732/www/opt.pdf).
