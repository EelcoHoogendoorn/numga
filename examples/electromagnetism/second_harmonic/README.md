# A crystal that doubles the frequency

A tetrahedral crystal's nonlinear electric response as `Bivector <- (Bivector, Bivector)`.
The notebook builds it from bond planes, turns the crystal, binds a pump, and accumulates two
temporal field quadratures through matched, mismatched and periodically flipped crystals.
The electric-dipole model uses normalized amplitudes and omits pump depletion and interface transmission.

Open `second_harmonic.ipynb` in Jupyter, or write the figures and the animation to `plots/` from
the repository root:

```sh
MPLBACKEND=Agg python -m examples.electromagnetism.second_harmonic.scenarios
```

The material model follows [Hardhienata et al., *Bond Model and Group Theory of Second Harmonic
Generation in GaAs(001)*](https://arxiv.org/abs/1408.1185). For propagation, see [*Optical
Properties of Solids*, chapter 11](https://web.mit.edu/6.732/www/opt.pdf).
