Belongs in: Week 2 exercise session (exercise-sessions/02-linear-regression.qmd), as the
posterior-draws lab on transforming coefficients.

Cut from lecture-notes/02-linear-regression.qmd (section "Functions of the coefficients",
including the two-panel break-even ratio figure). It opens a second inference target beside
the running banking regression and contains no derivation, so under rule 2 it moves to the
exercise session rather than staying in the chapter.

---

## Functions of the coefficients

Posterior summaries are rarely wanted for the coefficients themselves. The quantity a desk
cares about is usually a function of them, and once the posterior is available as draws any
such function is free: transform every draw and summarise the result. No new derivation is
needed, because a function of a random vector has whatever distribution the draws induce.

The natural example here is the market return at which the banking industry is expected to
break even. Setting the conditional mean to zero with the other two factors at their
neutral value of zero, $\beta_0 + \beta_1 x = 0$, gives the break-even market
return $x^\star = -\beta_0/\beta_1$. It is a ratio of two coefficients, and ratios carry a
warning that Bauwens states plainly and every applied worker should remember: a ratio whose
denominator has posterior mass near zero need not possess moments at all. The posterior mean
may fail to exist even though every individual draw is finite, in which case the sample mean
never settles as the number of draws grows and reporting it is meaningless. Quantiles always
exist, so they are the honest summary.

Whether the warning bites is a property of the sample, not of the algebra, and this regression
lets us see both cases at once. The market beta has posterior mean $1.1890$ with standard
deviation $0.0329$, so zero is thirty-six standard deviations away and the denominator is
effectively never small. The size loading has posterior mean $-0.0998$ with standard deviation
$0.0471$, so zero sits about two standard deviations away and a noticeable share of the draws
lands close to it. Dividing by the first is safe; dividing by the second is not. The
economics reinforces the point: "the market return at which banks break even" is a
sensible quantity, whereas "the size loading at which banks break even" is not a question
anyone would ask, and the posterior refuses to answer it.

**Reading the computation.** Both ratios are built from the same posterior draws, changing
only the denominator. The left panel shows the distribution of the dangerous one, trimmed at
the first and ninety-ninth percentiles so the tails do not swallow the picture. The right panel
plots the running mean of each ratio against the number of draws used, which is the diagnostic
for whether a mean exists at all.

```{python}
safe = -beta_draws[:, 0] / beta_draws[:, 1]      # denominator far from zero
risky = -beta_draws[:, 0] / beta_draws[:, 2]     # denominator close to zero
run = lambda v: np.cumsum(v) / np.arange(1, v.size + 1)
lo_r, hi_r = np.quantile(risky, [0.01, 0.99])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.hist(risky[(risky > lo_r) & (risky < hi_r)], bins=80, density=True, color="steelblue")
ax1.axvline(np.median(risky), color="black", lw=1, label="median")
ax1.set_xlabel(r"$-\beta_0/\beta_2$"); ax1.set_ylabel("density"); ax1.legend(fontsize=9)
ax1.set_title("Ratio with mass near zero in the denominator")

d = np.arange(1, safe.size + 1)
ax2.plot(d, run(safe), lw=0.9, label=r"$-\beta_0/\beta_1$  (beta far from 0)")
ax2.plot(d, run(risky), lw=0.9, label=r"$-\beta_0/\beta_2$  (loading near 0)")
ax2.set_xscale("log"); ax2.set_xlabel("draws used"); ax2.set_ylabel("running mean")
ax2.legend(fontsize=8); ax2.set_title("One running mean settles, the other does not")
plt.tight_layout(); plt.show()

for nm, v in (("safe ", safe), ("risky", risky)):
    print(f"{nm}: median {np.median(v):8.4f}   90% interval "
          f"[{np.quantile(v, 0.05):8.4f}, {np.quantile(v, 0.95):8.4f}]   "
          f"running mean 10k {run(v)[9_999]:9.4f}  200k {run(v)[-1]:9.4f}")
```

The left panel is sharply peaked with tails running far beyond the plotted range, which is what
dividing by a quantity that can be near zero produces. The right panel makes the practical
point: the safe ratio's running mean is flat from a few thousand draws onwards, while the risky
one has moved from $-0.0005$ at ten thousand draws to $-0.0148$ at two hundred thousand,
each jump caused by one draw with a size loading close to zero. Both medians and both intervals,
by contrast, are stable to the printed precision. Summarise ratios by quantiles.

The safe ratio also has an economic reading worth stating. Its posterior median is
$0.0011$, so the market must return about eleven basis points above the risk-free rate in
a month before the banking industry is expected to break even, and its ninety percent
interval runs from $-0.0005$ to $0.0027$. The break-even point is positive because the
estimated alpha is negative: with the size and value factors at zero, banks need a
slightly rising market simply to stand still. Given that the average monthly market excess
return over this sample is several times eleven basis points, this is a small drag rather
than a damning one, and the interval comfortably includes zero.

