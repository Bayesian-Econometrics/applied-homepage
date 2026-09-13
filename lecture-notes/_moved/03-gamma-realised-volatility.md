> Cut from lecture note 3 in the restructuring pass. Belongs in the **Week 3 exercise
> session**. A second, independent inference problem (gamma model for monthly realised
> volatility, prior used as the importance proposal, posterior predictive of a stressed
> month). Removed under the one-example-per-chapter rule; it is self-contained and needs
> no edits beyond a data-loading preamble.

## A gamma model with no conjugate posterior

We close with two applications built entirely from the tools above. The first models a
positive-valued financial series directly: the market's **monthly realised volatility**,
computed as the square root of the sum of squared daily excess returns within each
calendar month and expressed in percentage points. Realised volatility is positive,
right-skewed and heavily used as an input to risk models, and a **gamma distribution**,
$Y \mid \nu, \lambda \sim \text{Gamma}(\nu, 1/\lambda)$ with shape $\nu$ and rate
$\lambda$, density $f(y \mid \nu, \lambda) = \frac{\lambda^\nu}{\Gamma(\nu)}
y^{\nu-1}e^{-\lambda y}$, is the standard first model for such a series. There is no
conjugate prior for the gamma family in both parameters simultaneously, so we place a
weakly informative prior directly, $\nu \sim \text{Uniform}(0.1, 15)$ and $\lambda \sim
\text{Gamma}(2, 1)$ independent of $\nu$. The likelihood depends on the data only through
the sufficient statistics $n$, $\sum_i y_i$ and $\sum_i \log y_i$, so once those three
numbers are in hand, evaluating the likelihood at any candidate $(\nu, \lambda)$ costs
nothing further. We use **the prior itself as the importance proposal**, a common
shortcut for a first pass at a non-conjugate posterior: because the unnormalised posterior
is $\tilde p(\nu,\lambda) = l(\nu,\lambda \mid y)\, p(\nu)p(\lambda)$ and the proposal is
$q(\nu,\lambda) = p(\nu)p(\lambda)$, the weight collapses to the likelihood itself,
$w(\nu,\lambda) = \tilde p/q = l(\nu,\lambda\mid y)$, so we need only draw from the
prior and weight by how well each draw explains the data.

```{python}
rv = ((ff_d["2000-01-01":]["mkt_rf"] ** 2)
      .groupby(pd.Grouper(freq="MS")).sum().pow(0.5) * 100)
rv = rv[rv > 0]
n_obs = rv.size
sum_y, sum_logy = rv.sum(), np.log(rv).sum()

S_g = 400_000
nu_s = rng.uniform(0.1, 15, size=S_g)
lam_s = rng.gamma(shape=2.0, scale=1.0, size=S_g)

loglik = (n_obs * nu_s * np.log(lam_s) - n_obs * gammaln(nu_s)
          + (nu_s - 1) * sum_logy - lam_s * sum_y)
logw = loglik - loglik.max()                     # for numerical stability
w_g = np.exp(logw)
w_g_norm = w_g / w_g.sum()

post_mean_nu = np.sum(nu_s * w_g_norm)
post_mean_lam = np.sum(lam_s * w_g_norm)
ess_g = 1.0 / np.sum(w_g_norm ** 2)
print(f"realised volatility: n={n_obs} months, {rv.index[0]:%Y-%m} to {rv.index[-1]:%Y-%m}")
print(f"sample mean {rv.mean():.3f}, sd {rv.std():.3f}, max {rv.max():.3f} (percentage points)")
print(f"posterior mean: nu = {post_mean_nu:.3f}, lambda = {post_mean_lam:.3f}, "
      f"implied mean nu/lambda = {np.sum(nu_s/lam_s*w_g_norm):.3f}")
print(f"effective sample size: {ess_g:.0f} out of {S_g}")
```

The fitted shape of $\nu = 4.007$ with rate $\lambda = 0.837$ says that monthly realised
volatility is right-skewed but not extremely so, and the implied model mean of $4.791$
percentage points reproduces the sample mean of $4.795$ almost exactly, which is the least
a two-parameter fit should manage. The effective sample size of $861$ is far below the
$400{,}000$ draws taken, because the prior is much more diffuse than the likelihood and so
is a poor match to it as a proposal; this is the price of the shortcut, paid here in
wasted draws rather than in a wrong answer, and it is exactly the kind of situation the
next lectures address with proposals that adapt to the posterior instead of ignoring it.

The posterior predictive probability that *next month's* realised volatility exceeds some
threshold is now one line of simulation once we have weighted parameter draws: for each
$(\nu^{(s)}, \lambda^{(s)})$ draw one predictive value $Y^{(s)} \sim
\text{Gamma}(\nu^{(s)}, 1/\lambda^{(s)})$ and average the indicator that it exceeds the
threshold, using the same self-normalised weights as before. This is the discrete Monte
Carlo version of $\Pr(Y_{n+1} > c \mid y) = \int \Pr(Y_{n+1}>c \mid \nu,\lambda)\,
p(\nu,\lambda\mid y)\, d\nu\, d\lambda$. We take the threshold at eight percentage points
a month, roughly $28\%$ annualised, a level a risk committee would regard as a stressed
market.

```{python}
y_pred = rng.gamma(shape=nu_s, scale=1 / lam_s)          # one draw per (nu, lambda)
threshold = 8.0                                          # percentage points per month

pred_prob = np.sum((y_pred > threshold) * w_g_norm)
print(f"P(next month's realised volatility > {threshold} | data)  ~=  {pred_prob:.4f}")
print(f"observed frequency above {threshold} in the sample: {(rv > threshold).mean():.4f}")

fig, ax = plt.subplots()
ax.hist(rv, bins=25, density=True, alpha=0.5, label="observed realised volatility")
ax.hist(y_pred, bins=200, weights=w_g_norm, range=(0, 20), density=True,
        histtype="step", lw=2, label="posterior predictive")
ax.axvline(threshold, color="black", ls=":", label=f"threshold = {threshold}")
ax.set_xlabel("monthly realised volatility (percentage points)")
ax.set_ylabel("density"); ax.set_xlim(0, 20)
ax.legend(); ax.set_title("Posterior predictive distribution of next month's realised volatility")
plt.tight_layout(); plt.show()
```

The predictive density carries parameter uncertainty on top of the sampling variability
the gamma model already implies, exactly the effect Lecture 1's beta-binomial predictive
illustrated in the coin example. The predictive probability of a stressed month, $0.1004$, sits a little
below the observed frequency of $0.1097$, because the gamma model's smooth right tail is
thinner than the handful of extreme months the sample actually contains; whether that gap
is acceptable smoothing or misspecification is a question the exercises take up, and it is
the kind of check a risk model should always face. The
printed probability is the number a risk manager would actually want, and it required no
closed form anywhere in the calculation, only the weighted Monte Carlo average of an
indicator.
