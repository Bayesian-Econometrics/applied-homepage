Belongs in: Week 1 exercise session (exercise-sessions/01-introduction.qmd), as the second
conjugate lab (default intensity in a loan book).

Cut from lecture-notes/01-introduction.qmd (section "A second conjugate model: rare
events"): the figure and the numerical loan-book example move, because chapter 1 now
carries one running example. The Poisson-gamma derivation itself stays in the chapter, in
general notation, in the section "Where else conjugacy works, and where it stops".

---

### A second conjugate model: rare events

The coin is a convenient first example but a misleading one, because economics and finance
care at least as much about counts of rare events: defaults in a loan book, operational
loss events, jumps in a price series. The natural model is a Poisson process, and the
conjugate machinery works the same way, which is the point of doing it twice.

Suppose $n$ events are observed over an exposure of length $x$, so the likelihood for the
intensity $\theta$ is proportional to

$$
l(\theta \mid {\bf x}) \propto \theta^{n} e^{-x\theta}.
$$

A **gamma prior** $\theta \sim \text{Gamma}(a, b)$ has density proportional to
$\theta^{a-1} e^{-b\theta}$. Multiplying and collecting exponents,

$$
f(\theta \mid {\bf x}) \propto \theta^{n} e^{-x\theta} \cdot
\theta^{a-1} e^{-b\theta} = \theta^{(a+n)-1} e^{-(b+x)\theta},
$$

which is the kernel of a $\text{Gamma}(a + n, b + x)$ density. Again we recognised the
shape instead of integrating. The update rule reads as plainly as the beta one: add the
event count to the shape and the exposure to the rate, so $a$ is a prior number of events
and $b$ a prior exposure. Letting $a, b \to 0$ gives the improper objective prior
$f(\theta) \propto 1/\theta$.

**Reading the computation.** The Poisson model combines event counts with exposure. Gamma rate increases with firm-years, not just with the number of rows; inspect both posterior mean and units.

```{python}
a_g, b_g = 2.0, 500.0              # prior: 2 events in 500 firm-years
n_ev, exposure = 7, 800            # observed: 7 defaults in 800 firm-years

post_g = stats.gamma(a_g + n_ev, scale=1 / (b_g + exposure))
theta_g = np.linspace(0, 0.03, 400)

fig, ax = plt.subplots()
ax.plot(theta_g, stats.gamma(a_g, scale=1 / b_g).pdf(theta_g), ":", label="prior")
ax.plot(theta_g, post_g.pdf(theta_g), label="posterior")
ax.axvline(n_ev / exposure, color="grey", lw=1, label="MLE = 7/800")
ax.set_xlabel(r"default intensity $\theta$"); ax.set_ylabel("density")
ax.legend(); ax.set_title("Poisson-gamma updating for a rare event")
plt.tight_layout(); plt.show()

print(f"prior mean {a_g/b_g:.5f}   MLE {n_ev/exposure:.5f}   "
      f"posterior mean {post_g.mean():.5f}")
print(f"95% credible interval [{post_g.ppf(0.025):.5f}, {post_g.ppf(0.975):.5f}]")
```

The posterior mean again lands between the prior mean and the maximum likelihood estimate,
and it is again a weighted average, now with weights $b/(b+x)$ and $x/(b+x)$ set by prior
exposure against observed exposure.
