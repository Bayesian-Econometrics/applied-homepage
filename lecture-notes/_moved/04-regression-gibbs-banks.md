> Cut from lecture note 4 in the restructuring pass. Belongs in the **Week 4 exercise
> session**, where the multiple-regression Gibbs sampler is already the paired exercise.
> The two-block regression sampler validated against Lecture 2's closed-form posterior
> for the banking industry portfolio on the three Fama-French factors, with the
> probability statements and the four marginal histograms. The general
> beta / sigma^2 conditionals survive in the rewritten chapter in weighted form
> (X'WX, X'Wy); what moves here is the banks application. Needs a data-loading preamble
> (ff_factors_monthly.parquet, ff_industry49_monthly.parquet).

## From two blocks to many: the regression model

The semi-conjugate normal model has two scalar blocks. Lecture 2 introduced Bayesian
linear regression, $y = X\beta + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0,
\sigma^2 I_n)$, as a model whose posterior is available in closed form only for
particular conjugate priors and only after an integral that gets uglier as $k$ grows.
The Gibbs sampler turns that same model into a two-block problem exactly like the one
above, except that one block is now a $k$-dimensional vector instead of a scalar, and it
is the paired exercise for this week because it is the natural bridge from the toy
two-parameter case to a model of real applied size.

We place a conjugate **Normal prior** on the coefficients and an **inverse-gamma prior**
on the variance, independent of each other exactly as $\mu$ and $\tau$ were above,

$$
\beta \sim \mathcal{N}(b_0, B_0), \qquad \sigma^2 \sim \text{Inverse-Gamma}(c_0, d_0).
$$

The same two-step logic as before, fixing $\sigma^2$ and completing the square in the
vector $\beta$, or fixing $\beta$ and collecting the exponents of $\sigma^2$, gives the
full conditionals

$$
\beta \mid \sigma^2, y \sim \mathcal{N}(b_1, B_1),
\qquad
B_1 = \left( B_0^{-1} + \tfrac{1}{\sigma^2} X^\top X \right)^{-1},
\qquad
b_1 = B_1 \left( B_0^{-1} b_0 + \tfrac{1}{\sigma^2} X^\top y \right),
$$

$$
\sigma^2 \mid \beta, y \sim \text{Inverse-Gamma}\!\left(
c_0 + \tfrac{n}{2},\;\; d_0 + \tfrac{1}{2}(y - X\beta)^\top (y - X\beta)
\right).
$$

The posterior precision matrix $B_1^{-1}$ adds the prior precision $B_0^{-1}$ to the data
precision $\tfrac{1}{\sigma^2}X^\top X$, in exact matrix analogy to $c + n\tau$ above; the
scale parameter of the inverse-gamma gains half the residual sum of squares, in exact
analogy to the rate of $\tau$'s conditional. Nothing new happens except that scalars
become vectors and matrices; the derivation is the same completing-the-square and
exponent-collecting argument carried out coordinate by coordinate at once.

Rather than invent a regression, we take the one Lecture 2 already solved with pen and
paper: the excess return on the US **banking industry** portfolio regressed on the
market, size and value factors of Fama and French, monthly from January 1990. That choice
is deliberate. Lecture 2 derived a closed-form posterior for this exact dataset, so we
know the right answer before the sampler starts, and a sampler that reproduces a known
analytic result on real data is a sampler we can then trust on a model that has no
analytic result at all.

To make the comparison exact we give the coefficients an essentially flat Normal prior
and the variance an inverse-gamma prior with shape and scale of $10^{-6}$, which is a
proper prior numerically indistinguishable from the improper $p(\beta,\sigma^2) \propto
1/\sigma^2$ limit that Lecture 2 studied under the heading of the noninformative limit.
Under that limit the marginal posterior of $\beta$ is
$t_{n-k}\big(\hat\beta_{\text{OLS}},\, s^2 (X^\top X)^{-1}\big)$, so its mean is the
least-squares estimate and its standard deviation is
$\sqrt{\mathbb{E}[\sigma^2 \mid y]\,\operatorname{diag}\left[(X^\top X)^{-1}\right]}$ with
$\mathbb{E}[\sigma^2 \mid y] = \text{SSR}/(n-k-2)$. Those are the two columns the sampler
has to hit.

**Reading the computation.** The block builds the same dependent variable and design
matrix as Lecture 2, runs least squares, and evaluates the analytic posterior moments
that the improper-prior algebra delivers. Nothing here is sampled yet.

```{python}
industry = pd.read_parquet(DATA / "ff_industry49_monthly.parquet")
panel = ff_m.join(industry[["banks"]], how="inner")["1990-01":].dropna()

y = (panel["banks"] - panel["rf"]).to_numpy()            # excess return on banks
X = np.column_stack([np.ones(len(panel)), panel["mkt_rf"],
                     panel["smb"], panel["hml"]])
n_r, k_r = X.shape
names = ["alpha", "beta_mkt", "gamma_smb", "delta_hml"]

beta_ols = np.linalg.solve(X.T @ X, X.T @ y)
ssr = float((y - X @ beta_ols) @ (y - X @ beta_ols))
sigma2_analytic = ssr / (n_r - k_r - 2)
sd_analytic = np.sqrt(sigma2_analytic * np.diag(np.linalg.inv(X.T @ X)))

print(f"{n_r} months, {panel.index[0]:%Y-%m} to {panel.index[-1]:%Y-%m}, k = {k_r}")
print(pd.DataFrame({"analytic mean": beta_ols, "analytic sd": sd_analytic},
                   index=names).round(6))
print(f"analytic E[sigma^2|y] = {sigma2_analytic:.7f}")
```

The analytic posterior puts the market beta at $1.19326$ with a standard deviation of
$0.03321$, the value loading at $0.82341$ with $0.04455$, the size loading at $-0.10040$
with $0.04741$ and the monthly alpha at $-0.00207$ with $0.00143$, and the posterior mean
of the error variance at $0.0008716$. These are Lecture 2's numbers, obtained there by
completing the matrix square; here they are simply the target.

We now set the near-flat priors and alternate the two full-conditional draws, sampling
the inverse-gamma by drawing a gamma variate and inverting it, since if $G \sim
\text{Gamma}(\text{shape}, \text{rate})$ then $1/G \sim \text{Inverse-Gamma}(\text{shape},
\text{rate})$. The chain gets its own random-number stream so that this section can be
read and rerun independently of the ones before it.

```{python}
rng_reg = np.random.default_rng(2026)

b0, B0_inv = np.zeros(k_r), np.linalg.inv(np.eye(k_r) * 100.0)
c0, d0 = 1e-6, 1e-6
n_iter_r = 12_000
beta_draws, sigma2_draws = np.empty((n_iter_r, k_r)), np.empty(n_iter_r)

sigma2, XtX, Xty = ssr / (n_r - k_r), X.T @ X, X.T @ y
for t in range(n_iter_r):
    B1 = np.linalg.inv(B0_inv + XtX / sigma2)
    b1 = B1 @ (B0_inv @ b0 + Xty / sigma2)
    beta = b1 + np.linalg.cholesky(B1) @ rng_reg.standard_normal(k_r)

    resid = y - X @ beta
    sigma2 = 1.0 / rng_reg.gamma(c0 + n_r / 2, 1.0 / (d0 + 0.5 * (resid @ resid)))
    beta_draws[t], sigma2_draws[t] = beta, sigma2

print("sampler finished:", beta_draws.shape, sigma2_draws.shape)
```

As before we discard an initial burn-in segment before summarising, this time a shorter
one relative to the run because the regression conditionals mix quickly.

```{python}
burn_r = 2000
beta_post, sigma2_post = beta_draws[burn_r:], sigma2_draws[burn_r:]

table = pd.DataFrame({
    "analytic mean": beta_ols, "Gibbs mean": beta_post.mean(axis=0),
    "analytic sd": sd_analytic, "Gibbs sd": beta_post.std(axis=0),
    "2.5%": np.quantile(beta_post, 0.025, axis=0),
    "97.5%": np.quantile(beta_post, 0.975, axis=0),
}, index=names)
print(table.round(5))
print(f"\nsigma2: analytic {sigma2_analytic:.7f}  Gibbs {sigma2_post.mean():.7f}")
print(f"largest gap in the posterior means {np.abs(beta_post.mean(axis=0) - beta_ols).max():.2e}")
```

The sampler reproduces the algebra. The largest discrepancy between a Gibbs posterior
mean and its analytic counterpart is $7.4\times10^{-4}$ of a coefficient unit, which for
the market beta is two percent of one posterior standard deviation and therefore pure
Monte Carlo error, and the posterior mean of the error variance comes out at $0.0008721$
against the analytic $0.0008716$. A loop of one Normal draw and one inverse-gamma draw,
which never touched the marginal Student-$t$ distribution or the matrix square that
produced it, has landed on the same answer. That is the validation this section exists
for, and it is the reason we are entitled to use the same loop in the next section, where
no analytic answer exists.

The economics is Lecture 2's and survives the change of method intact: banks carry a
market beta of about $1.19$, well above one, an enormous value loading of about $0.82$, a
small negative size loading, and a monthly alpha of about minus twenty-one basis points.
What the draws add is that any probability statement about these quantities is now a
counting exercise.

```{python}
print(f"P(market beta > 1 | y)     {(beta_post[:, 1] > 1).mean():.4f}")
print(f"P(value loading > 0 | y)   {(beta_post[:, 3] > 0).mean():.4f}")
print(f"P(size loading < 0 | y)    {(beta_post[:, 2] < 0).mean():.4f}")
print(f"P(alpha < 0 | y)           {(beta_post[:, 0] < 0).mean():.4f}")
```

Not one of the $10{,}000$ retained draws puts the market beta below one or the value
loading below zero, so both probabilities print as $1.0000$: after $439$ months these two
features of the banking sector are not in question. The size loading is negative in
$0.9821$ of draws and the alpha is negative in $0.9228$ of them, so both lean one way
without being settled, which is the honest description of a small effect estimated on a
few hundred observations. What the sampler adds beyond these point summaries is the full
joint posterior, visualised below for each coefficient.

```{python}
labels = [r"$\alpha$ (monthly alpha)", r"$\beta_{\rm mkt}$",
          r"$\gamma_{\rm smb}$", r"$\delta_{\rm hml}$"]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for j in range(k_r):
    axes[j].hist(beta_post[:, j], bins=40, density=True, color="steelblue", alpha=0.85)
    axes[j].axvline(beta_ols[j], color="red", lw=1.2, label="analytic mean")
    axes[j].set_title(labels[j], fontsize=10)
axes[1].axvline(1.0, color="black", ls=":", lw=1.5, label="benchmark 1")
axes[0].axvline(0.0, color="black", ls=":", lw=1.5, label="zero")
axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
fig.suptitle("Marginal posteriors from the regression Gibbs sampler, banks 1990-2026")
plt.tight_layout(); plt.show()
```

Each marginal posterior is tightly concentrated around the analytic posterior mean drawn
in red, and the two economic benchmarks tell the story at a glance: the vertical line at
one sits far to the left of the market beta's density, while the line at zero sits inside
the right shoulder of alpha's. The width of each histogram is the posterior uncertainty
the credible intervals in the table summarise, obtained here from the same two-line
structure as the semi-conjugate model above, just with a vector and a matrix in place of
two scalars.
