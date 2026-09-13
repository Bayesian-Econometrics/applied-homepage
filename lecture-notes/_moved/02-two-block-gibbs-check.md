Belongs in: Week 2 exercise session (exercise-sessions/02-linear-regression.qmd), as the
sampler sanity check that anticipates Lecture 4.

Cut from lecture-notes/02-linear-regression.qmd (section "The same posterior from two
blocks"): only the code and its discussion move. The derivation it carried, the two full
conditionals of the Normal-Inverse-Gamma posterior, stays in the chapter as the visible
result block "Result: the two full conditionals".

---

### The same posterior from two blocks

The analytic result also has a sampling counterpart worth seeing once, because it is the
bridge to the computational half of the course. The two conditionals of the joint posterior
are both standard,

$$
\beta \mid \sigma^2, y \sim \mathcal{N}(b_1, \sigma^2 B_1), \qquad
\sigma^2 \mid \beta, y \sim \text{Inverse-Gamma}\!\left(\alpha_1 + \tfrac{k}{2},\;
\delta_1 + \tfrac{1}{2}(\beta - b_1)^\top B_1^{-1}(\beta - b_1)\right),
$$

the second differing from the marginal $\text{Inverse-Gamma}(\alpha_1, \delta_1)$ by exactly
the $k/2$ that integrating the coefficients out removed, plus the quadratic form that the
fixed $\beta$ contributes. Cycling between the two draws is a two-block Gibbs sampler, the
subject of Lecture 4. Here we know the answer, so the sampler is a test rather than a tool.

**Reading the computation.** Each sweep draws the coefficients given the current variance,
then the variance given those coefficients. After discarding the warm-up sweeps the sampled
marginals must reproduce the analytic ones; if they do not, the conditionals are wrong.

```{python}
n_sweep, burn = 6000, 1000
sig2_state = ssr / (n - k)
P1 = B0_inv + XtX
L1 = np.linalg.cholesky(post["B1"])
gibbs_beta = np.empty((n_sweep, k))
gibbs_sig2 = np.empty(n_sweep)

for r in range(n_sweep):
    beta_r = post["b1"] + np.sqrt(sig2_state) * (L1 @ rng.standard_normal(k))
    quad = (beta_r - post["b1"]) @ (P1 @ (beta_r - post["b1"]))
    sig2_state = stats.invgamma(a=post["a1"] + k / 2.0,
                                scale=post["d1"] + quad / 2.0).rvs(random_state=rng)
    gibbs_beta[r], gibbs_sig2[r] = beta_r, sig2_state

kept = gibbs_beta[burn:]
print(pd.DataFrame({"analytic mean": post["b1"], "Gibbs mean": kept.mean(axis=0),
                    "analytic sd": sd_beta, "Gibbs sd": kept.std(axis=0)},
                   index=names).round(4))
print(f"\nE[sigma^2|y] analytic {post['d1']/(post['a1']-1):.6f}   "
      f"Gibbs {gibbs_sig2[burn:].mean():.6f}")
```

The sampled means and standard deviations agree with the analytic ones to three or four
decimals, which is the sanity check every sampler must pass on a model with a known answer
before it is trusted on one without.

