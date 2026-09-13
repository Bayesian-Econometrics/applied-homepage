Belongs in: Week 1 exercise session (exercise-sessions/01-introduction.qmd), as the
follow-up question to the century-of-market-months lab.

Cut from lecture-notes/01-introduction.qmd (the closing computation of "The same model on
a century of market data"), because the spec reduces that section to one figure and one
paragraph. No derivation is involved; the chapter keeps the model itself and the warning
that Lecture 8 drops the constant-parameter assumption.

---

The split into prior and data assumes the two periods are informative about the same
$\theta$. If the market's monthly win rate changed after 1990, then using the earlier
period as a prior does not sharpen the estimate, it contaminates it. The assumption is
testable in a crude way: if the modern frequency lands far in the tail of the prior, the
two are hard to reconcile.

```{python}
z_pos = (y_new / n_new - prior_m.mean()) / prior_m.std()
prior_pred_sd = np.sqrt(prior_m.mean() * (1 - prior_m.mean()) / n_new)
print(f"modern frequency is {z_pos:.1f} prior standard deviations above the prior mean")
print(f"but sampling noise alone in {n_new} months is about "
      f"{prior_pred_sd:.4f}, i.e. {(y_new/n_new - prior_m.mean())/prior_pred_sd:.1f} "
      f"standard errors")
```

The modern frequency is high relative to the prior, but relative to the sampling noise in
$439$ months it is well within the range one would expect if $\theta$ had not changed at
all. The two periods are compatible, so pooling them is defensible here. In Lecture 8 we
drop the assumption entirely and let the parameter drift.
