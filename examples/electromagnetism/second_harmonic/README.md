# A crystal that doubles the frequency

Second-harmonic generation from a tetrahedral crystal, written as a bilinear extensor with two
electric-field inputs and one polarization output. The notebook builds the response from four
directed bonds, turns the crystal as an extensor, binds a pump to obtain a linear probe map, and
grows the doubled light along a uniform, a mismatched and a periodically flipped crystal.

Open `second_harmonic.ipynb` in Jupyter, or write the figures and the animation to `plots/` from
the repository root:

```sh
MPLBACKEND=Agg python -m examples.electromagnetism.second_harmonic.scenarios
```

The material model follows [Hardhienata et al., *Bond Model and Group Theory of Second Harmonic
Generation in GaAs(001)*](https://arxiv.org/abs/1408.1185). For propagation, see [*Optical
Properties of Solids*, chapter 11](https://web.mit.edu/6.732/www/opt.pdf).
