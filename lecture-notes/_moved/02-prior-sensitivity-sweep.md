Belongs in: Week 2 exercise session (exercise-sessions/02-linear-regression.qmd), as the
small-sample prior sweep.

Cut from lecture-notes/02-linear-regression.qmd (section "Prior sensitivity in a small
sample", two chunks). It is replaced in the chapter by a single section, "Shrinkage with a
single tightness parameter", whose one figure drives both samples from one parameter and
reports the same twenty-four month conclusions. No derivation is involved.

---

## Prior sensitivity in a small sample

The harder question is what happens when the prior is strong and the sample short. We keep
only the first twenty-four months, January 1990 to December 1991, and tighten the prior sd
on the market beta around one. This is not an artificial exercise: it is the position of an
analyst covering a newly listed sector, and the two years in question span a genuine
banking recession, so the short-sample estimate is unusually far from the full-sample one.

**Reading the computation.** Holding the shortened dataset fixed isolates the effect of prior assumptions. Record both the movement of the beta estimate and the interval width; a tighter interval need not be better calibrated if the prior is wrong.

```{python}
ys, Xs = y[:24], X[:24]
b_ols_s = np.linalg.solve(Xs.T @ Xs, Xs.T @ ys)
print(f"{panel.index[0]:%Y-%m} to {panel.index[23]:%Y-%m}: "
      f"24-month OLS market beta {b_ols_s[1]:.3f}")
gb = np.linspace(0.6, 2.1, 500)
fig, ax = plt.subplots()
for tb in [0.02, 0.05, 0.10, 0.25, 1.00]:
    Bi = np.linalg.inv(np.diag(np.array([0.002, tb, 0.500, 0.500])**2 / sigma2_bar))
    ps = nig_update(ys, Xs, b0, Bi, a0, d0)
    sc = np.sqrt(ps["d1"] / ps["a1"] * np.diag(ps["B1"]))
    d = stats.t(df=2 * ps["a1"], loc=ps["b1"][1], scale=sc[1])
    lo, hi = d.ppf([0.025, 0.975])
    print(f"prior sd {tb:4.2f}:  post mean {ps['b1'][1]:.3f}   "
          f"95% [{lo:.3f}, {hi:.3f}]   width {hi - lo:.3f}")
    ax.plot(gb, d.pdf(gb), label=rf"prior sd $= {tb:.2f}$")
ax.axvline(b_ols[1], color="black", lw=1.2, label=f"full-sample OLS {b_ols[1]:.2f}")
ax.axvline(b_ols_s[1], color="grey", ls=":", label=f"24m OLS {b_ols_s[1]:.2f}")
ax.set_xlabel(r"market beta $\beta_1$")
ax.set_ylabel("posterior density")
ax.legend(fontsize=9)
ax.set_title("Prior tightness and the posterior beta, 24 months of data")
plt.tight_layout()
plt.show()
```

With a prior standard deviation of $0.02$ the posterior mean is $1.011$, essentially the
prior value of one, and the interval $[0.960, 1.061]$ is six times narrower than the widest
one printed: the prior has replaced the sample. Loosening it lets the twenty-four
observations dominate and the mean climbs through $1.061$, $1.186$ and $1.440$ to $1.583$,
approaching the short-sample OLS estimate of $1.688$. The economic stakes are visible here
in a way they were not in the full sample. Over 1990 and 1991 banks were far more sensitive
to the market than they have been on average since, and a tight prior at one would have
told a risk officer that the sector was market neutral in exactly the two years it was not.
A narrow posterior around the wrong centre is false confidence, not precision, so such a
sweep belongs alongside any small-sample result. The next lecture replaces this closed-form
posterior with Monte Carlo integration, which is what we must do once the prior is no
longer conjugate.

The five posterior means just printed trace out a single path from the prior mean towards
the OLS estimate as the prior loosens, which is worth its own picture.

**Reading the computation.** Each point reuses a posterior mean already printed above, now plotted against the prior sd instead of alongside a density curve. A path climbing from the prior mean towards the OLS line visualises shrinkage directly.

```{python}
tbs = np.array([0.02, 0.05, 0.10, 0.25, 1.00])
post_means_tb = np.array([
    nig_update(ys, Xs, b0,
               np.linalg.inv(np.diag(
                   np.array([0.002, tb, 0.500, 0.500]) ** 2 / sigma2_bar)),
               a0, d0)["b1"][1]
    for tb in tbs])

fig, ax = plt.subplots()
ax.plot(tbs, post_means_tb, "o-", color="steelblue", label="posterior mean of beta_mkt")
ax.axhline(b_ols_s[1], color="grey", ls=":", label=f"24m OLS {b_ols_s[1]:.2f}")
ax.axhline(b0[1], color="black", lw=1, ls="--", label="prior mean 1.00")
ax.set_xlabel("prior sd on market beta")
ax.set_ylabel("posterior mean of market beta")
ax.legend(fontsize=9)
ax.set_title("Shrinkage path: posterior mean from prior mean towards OLS")
plt.tight_layout()
plt.show()
```

The path starts pinned at the prior mean of one when the prior sd is $0.02$ and climbs
towards the short-sample OLS estimate of $1.688$ as the prior loosens, flattening out at
$1.583$ once the prior is wide enough to stop restraining the data. Reading the shrinkage
as a path rather than as five separate density curves makes the direction and the
diminishing returns of loosening the prior easier to see at a glance. It also shows that
the twenty-four month sample never quite reaches its own least-squares answer even under a
prior standard deviation of one, because two years of data are simply not enough to
overwhelm any prior worth writing down.



