Belongs in: Week 1 exercise session (exercise-sessions/01-introduction.qmd), as the
odds-form exercise on conditional independence.

Cut from lecture-notes/01-introduction.qmd (sections "The economic content of a
probability" and "Two signals: when may we update twice?", including the figure
fig-sequential-signal-information) because chapter 1 now carries a single running example.
The derivation these sections contained, posterior odds as prior odds times likelihood
ratios, survives in the chapter in general notation inside the result block "marginal
likelihood, posterior odds and Bayes factors", and the figure is replaced there by the
accumulating log Bayes factor on the market data.

---

### The economic content of a probability

The same issue appears when a lender interprets a credit warning:
how often does the warning appear among firms that default, and among firms
that survive? A statement about the signal conditional on default is not a
statement about default conditional on the signal.

For illustration, let default probability be 2%, sensitivity 80%, and the
false-positive rate 10%. Among 10,000 firms, 160 defaulting firms and 980
surviving firms would be flagged on average. Therefore

$$
\Pr(\text{default}\mid\text{flag})
=\frac{0.8(0.02)}{0.8(0.02)+0.1(0.98)}\approx0.140.
$$

The warning increases risk substantially, but rare defaults remain a small
part of all flagged cases. Before calculating any posterior, name the population,
the observation mechanism, and the quantity being conditioned on.

## Two signals: when may we update twice?

Suppose a lender assigns probability $p(D)=0.03$ to default. A warning is more
common among defaulting firms than among other firms: $p(W_1\mid D)=0.9$ and
$p(W_1\mid D^c)=0.2$. These are assumptions about the **data-generating process**,
not posterior probabilities. Out of 1,000 comparable firms we would expect 27
defaulting firms and 194 non-defaulting firms to trigger the warning. Therefore

$$
p(D\mid W_1)=\frac{0.9(0.03)}{0.9(0.03)+0.2(0.97)}\approx0.122.
$$

The denominator counts all warnings; the numerator counts those associated with
default. This counting argument is the source of Bayes' formula here.
For another warning, the correct next likelihood is $p(W_2\mid D,W_1)$.
Only **conditional independence given default status** lets us replace it by
$p(W_2\mid D)$. Under that assumption, with sensitivity 0.8 and false-positive
probability 0.1, posterior odds become

$$
\underbrace{\frac{p(D\mid W_1,W_2)}{p(D^c\mid W_1,W_2)}}_{\text{new odds}}
=\underbrace{\frac{0.03}{0.97}}_{\text{prior odds}}
\underbrace{\frac{0.9}{0.2}}_{\text{first likelihood ratio}}
\underbrace{\frac{0.8}{0.1}}_{\text{second likelihood ratio}}.
$$

If the second report merely copies the first, it adds no information: conditional
on the first warning it occurs with probability one under either state.
The following plot contrasts these two information structures, using exactly
the odds updates above.

```{python}
#| label: fig-sequential-signal-information
#| fig-cap: "Two independent measurements can strengthen evidence; a copied report cannot. Independence here is conditional on default status."
import numpy as np
import matplotlib.pyplot as plt

signal_odds = np.array([0.03 / 0.97, 0.03 / 0.97 * 4.5,
                        0.03 / 0.97 * 4.5 * 8])
signal_prob = signal_odds / (1 + signal_odds)
fig, ax = plt.subplots()
ax.plot(range(3), signal_prob, 'o-', color='#527a99')
ax.plot(range(3), [signal_prob[0], signal_prob[1], signal_prob[1]],
        'o--', color='#aa8066')
ax.text(2.03, signal_prob[2], 'Independent signal', va='center')
ax.text(2.03, signal_prob[1], 'Copied report', va='center')
ax.set(xticks=[0, 1, 2], xticklabels=['Prior', 'First warning', 'Second report'],
       ylabel='Probability of default', ylim=(0, 0.65), xlim=(-0.1, 3.1))
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
plt.show()
```

The large difference comes from the likelihood assumption, not from a different
Bayes rule. In the week's beta-binomial grid exercise, the analogous assumption
is that trials are independent **conditional on the common success probability**.
Sequential updating and a single batch update should then give the same grid posterior.

*Source connection:* Lopes, *Overview of Bayesian econometrics*, PDF pp. 3-7 and
12-16; SGPE Lecture 1, pp. 16-26. The credit example is an adaptation.
