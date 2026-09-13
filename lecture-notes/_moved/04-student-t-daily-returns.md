> Cut from lecture note 4 in the restructuring pass. Belongs in the **Week 4 exercise
> session**. The Student-t regression by data augmentation applied to a daily AR(1) for
> the market excess return, 1980 to 2026: the griddy-Gibbs draw for nu, the comparison
> with the Gaussian-error sampler that overturns the apparent short-horizon reversal,
> and the outlier-weight scatter naming Black Monday and the 2020 crash days. The
> derivations it contained (the Normal/inverse-Gamma scale mixture, the conditional of
> one mixing variance, the non-standard conditional of nu) are all retained in the
> rewritten chapter in general form on the shared monthly target; what moves here is the
> daily application and the Gaussian-versus-Student comparison. Needs a data-loading
> preamble (ff_factors_daily.parquet).

## Regression with heavy-tailed errors: Student-t via data augmentation

Every regression sampler built so far assumed Gaussian errors, and asset returns are the textbook
case where that is wrong: extreme returns occur far more often than a Normal error would predict.
A **Student-$t$** error, with its heavier tails, is the natural replacement, but the likelihood is
not conjugate to anything, since
$\propto\big[1+(y_t-x_t^\top\beta)^2/(\nu\sigma^2)\big]^{-(\nu+1)/2}$ raises the quadratic form to
a power instead of an exponential, so completing the square no longer produces anything
recognisable, and $(\beta,\sigma^2,\nu)$ has no closed-form posterior. Data augmentation,
introduced above for the latent-class mixture, resolves exactly this kind of impasse, and this
section builds a second instance of it, following Chapter 3 of Bauwens' lecture notes on Bayesian
econometrics.

### The scale-mixture representation

Introduce, for each observation, an unobserved positive scale $\lambda_t$ and write

$$
y_t = x_t^\top\beta + \lambda_t u_t, \qquad u_t \sim \mathcal{N}(0,\sigma^2), \qquad
\lambda_t^2 \sim \text{Inverse-Gamma}\!\left(\tfrac{\nu}{2},\tfrac{\nu}{2}\right),
$$

with $u_t$ and $\lambda_t^2$ independent of each other and across $t$. Conditional on $\lambda_t$
the error $\varepsilon_t=\lambda_t u_t$ is Normal with variance $\sigma^2\lambda_t^2$; the claim
is that integrating $\lambda_t$ out gives exactly a Student-$t_\nu$ error.

::: {.callout-note collapse="true"}
## Derivation: a Normal mixed over an inverse-gamma is a Student-t

Derive it with the same
beta-integral trick Lecture 2 used for the marginal Student-$t$ posterior of $\beta$: substitute
$u=1/\sigma^2$ there, or $v=1/\lambda_t^2$ here, and use $\int_0^\infty
v^{a-1}e^{-bv}dv=\Gamma(a)/b^a$. Writing $\phi=\lambda_t^2$, the joint density of $\varepsilon_t$
and $\phi$, with its two powers of $\phi$ and its two exponential terms (both containing $1/\phi$)
collected, is

$$
p(\varepsilon_t,\phi) =
\underbrace{(2\pi\sigma^2\phi)^{-1/2}\exp\!\left(-\frac{\varepsilon_t^2}{2\sigma^2\phi}\right)}_{\varepsilon_t\mid\phi}
\times
\underbrace{\frac{(\nu/2)^{\nu/2}}{\Gamma(\nu/2)}\phi^{-\nu/2-1}\exp\!\left(-\frac{\nu}{2\phi}\right)}_{\phi}
\propto \phi^{-(\nu+3)/2}\exp\!\left[-\frac{1}{\phi}\left(
\frac{\nu}{2}+\frac{\varepsilon_t^2}{2\sigma^2}\right)\right].
$$

Substituting $v=1/\phi$, so $\phi^{-(\nu+3)/2}=v^{(\nu+3)/2}$ and $d\phi=-dv/v^2$, turns the
integral over $\phi\in(0,\infty)$ into the same gamma integral as before, with $a=(\nu+1)/2$:

$$
\int_0^\infty v^{(\nu+3)/2-2}\exp\!\left[-v\left(\frac{\nu}{2}+\frac{\varepsilon_t^2}{2\sigma^2}\right)\right]dv
=\Gamma\!\left(\frac{\nu+1}{2}\right)\left(\frac{\nu}{2}+\frac{\varepsilon_t^2}{2\sigma^2}\right)^{-(\nu+1)/2}.
$$
:::

Restoring the constants carried along the way gives
$p(\varepsilon_t)\propto\left[1+\varepsilon_t^2/(\nu\sigma^2)\right]^{-(\nu+1)/2}$, exactly the
kernel of a Student-$t_\nu$ density scaled by $\sigma$, matching the source's
$t(0,1,\nu^{-1}\sigma^{-2},\nu)$: marginalising a Normal over this Inverse-Gamma mixing variance
gives a Student-$t_\nu$ error, with $\{\varepsilon_t=\lambda_tu_t\}$ independent across $t$ since
$\lambda_t,u_t$ are independent by construction.

### From the mixture to a heteroskedastic regression

Its use is in the *conditional* model, not the marginal: fix every $\lambda_t$ and divide the
$t$-th row of the regression equation by it,

$$
\frac{y_t}{\lambda_t} = \left(\frac{x_t}{\lambda_t}\right)^{\!\top}\!\beta + u_t, \qquad
u_t \sim \mathcal{N}(0,\sigma^2),
$$

homoskedastic again: the troublesome $\lambda_t$ has been divided out of the error. Writing
$D_\lambda=\text{diag}(\lambda_1,\dots,\lambda_T)$, $y_\lambda=D_\lambda^{-1}y$ and
$X_\lambda=D_\lambda^{-1}X$, conditioning on $\lambda$ turns the Student regression into exactly
the Normal-Inverse-Gamma regression of Lecture 2 applied to $(X_\lambda,y_\lambda)$ instead of
$(X,y)$. Returning to that lecture's scaled prior, $\beta\mid\sigma^2\sim\mathcal
N(b_0,\sigma^2B_0)$, $\sigma^2\sim\text{Inverse-Gamma}(\alpha_0,\delta_0)$, in place of the
unscaled semi-conjugate prior used earlier in this chapter, the weighted updates are

$$
B_1(\lambda)^{-1} = B_0^{-1} + X_\lambda^\top X_\lambda, \qquad
b_1(\lambda) = B_1(\lambda)\left(B_0^{-1}b_0 + X_\lambda^\top y_\lambda\right),
$$
$$
\beta \mid \sigma^2,\nu,\lambda,y \sim \mathcal{N}\!\left(b_1(\lambda),\, \sigma^2 B_1(\lambda)\right),
\qquad
\sigma^2 \mid \beta,\nu,\lambda,y \sim \text{Inverse-Gamma}\!\left(\alpha_0+\tfrac{T}{2},\;
\delta_0 + \tfrac{1}{2}\sum_{t=1}^T \frac{(y_t-x_t^\top\beta)^2}{\lambda_t^2}\right),
$$

each term weighted by $1/\lambda_t^2$, matching Bauwens' $M_*(\lambda)=M_0+X_\lambda^\top
X_\lambda$ and $\beta_*(\lambda)$ under the Lecture 2 bridge; with $\lambda$ known nothing depends
on $\nu$, since $\nu$ enters only through $\lambda$'s prior and full conditional.

### The full conditional of the mixing variables

What remains is $\lambda$. Write the standardised residual $\tilde y_t =
(y_t-x_t^\top\beta)/\sigma$, so $\tilde y_t = \lambda_t\tilde u_t$ with $\tilde u_t\sim\mathcal
N(0,1)$ given $\beta,\sigma$.

::: {.callout-note collapse="true"}
## Derivation: the conditional of one mixing variance

Holding $\beta,\sigma,\nu$ fixed, the joint density of $\tilde y_t$
and $\lambda_t^2$ collects the same two powers as above, but now $\tilde y_t$ plays the role
$\varepsilon_t$ played there and $\lambda_t^2$ is the unknown rather than the variable integrated
out:

$$
f(\tilde y_t\mid\lambda_t^2)\,f(\lambda_t^2) \propto
(\lambda_t^2)^{-1/2}\exp\!\left(-\frac{\tilde y_t^2}{2\lambda_t^2}\right) \times
(\lambda_t^2)^{-\nu/2-1}\exp\!\left(-\frac{\nu}{2\lambda_t^2}\right)
= (\lambda_t^2)^{-(\nu+3)/2}\exp\!\left(-\frac{\nu+\tilde y_t^2}{2\lambda_t^2}\right).
$$

Comparing the power and the exponent with the Inverse-Gamma kernel
$(\lambda_t^2)^{-a-1}\exp(-d/\lambda_t^2)$ identifies $a=(\nu+1)/2$ and $d=(\nu+\tilde y_t^2)/2$.
:::

The $\lambda_t^2$ are therefore conditionally independent across $t$ with
$$
\lambda_t^2 \mid \beta,\sigma^2,\nu,y \sim \text{Inverse-Gamma}\!\left(\frac{\nu+1}{2},\;
\frac{\nu+\tilde y_t^2}{2}\right),
$$
matching the source's $\text{IG}_2(\nu+1,\nu+\tilde y_t^2)$ under the bridge of Lecture 2. Since
the Inverse-Gamma$(a,d)$ mean is $d/(a-1)$ for $a>1$, the posterior mean of $\lambda_t^2$ grows
roughly linearly with $\tilde y_t^2$, so the weight $1/\lambda_t^2$ that enters the previous
section's updates shrinks towards zero as $\tilde y_t^2$ grows: a large standardised residual is
automatically down-weighted next sweep, without ever declaring it an outlier or removing it, which
is the practical payoff of the whole construction.

### Degrees of freedom as an unknown

The one block left is $\nu$. Its full conditional, keeping only factors depending on it, is
$\propto\phi(\nu)\prod_t\Gamma(\nu/2)^{-1}(\nu/2)^{\nu/2}(\lambda_t^2)^{-\nu/2-1}
\exp(-\nu/(2\lambda_t^2))$, not the kernel of any standard family. This is exactly the gap
Metropolis-Hastings fills, as in the source's own **combining Gibbs and MH** example, where an MH
step replaces the direct draw for one troublesome block inside an otherwise ordinary Gibbs sweep;
the same idea applies here to $\nu$ while $\beta,\sigma^2,\lambda$ keep their direct draws. Three
routes are workable: fix $\nu$; add a Metropolis step for it; or restrict it to a finite grid and
draw it from the discrete conditional above, renormalised. We use the grid, which needs no tuning
and sidesteps the flat prior $\phi(\nu)\propto 1$ being non-integrable on $(0,\infty)$, since the
likelihood barely distinguishes large values of $\nu$; a proper prior decaying fast enough, such
as a half-Cauchy, is the alternative remedy, since $\phi(\nu)\propto 1/\nu$ does not decay quickly
enough.

### A worked example: fat tails in daily market returns

Fat tails in daily equity returns are not a modelling hypothesis that needs a simulation
to illustrate it; they are the single most robust stylised fact in empirical finance. So
we run the sampler on the thing itself: the daily excess return on the US market
portfolio from January 1980 to July 2026, a window chosen because it contains Black
Monday in October 1987, the financial crisis of 2008, the pandemic crash of March 2020
and the tariff turmoil of April 2025, four episodes that no Gaussian error term can
accommodate.

The regression is a genuine economic question rather than a wrapper for the error
distribution. We regress today's market excess return on yesterday's,

$$
r_t = \beta_0 + \beta_1 r_{t-1} + \varepsilon_t,
$$

so $\beta_0$ is the average daily premium and $\beta_1$ measures **short-horizon
predictability**: a negative value means the market reverses the previous day's move, a
positive value means it continues it, and zero means the day-to-day sign is
unforecastable. Whether that coefficient is negative turns out to depend entirely on
whether the errors are allowed to have fat tails, which is exactly the point of the
section.

**Reading the computation.** The block loads the daily factor file, cuts it at January
1980, and lines up today's excess return against yesterday's. The two printed moments
describe the tails of the dependent variable before any model is fitted.

```{python}
ff_d = pd.read_parquet(DATA / "ff_factors_daily.parquet")["1980-01-01":]
r_series = ff_d["mkt_rf"]
dates_t = r_series.index[1:]

yt = r_series.to_numpy()[1:]                     # today's excess return
Xt = np.column_stack([np.ones(yt.size), r_series.to_numpy()[:-1]])   # constant, lag
n_t, k_t = Xt.shape

print(f"{n_t} trading days, {dates_t[0]:%Y-%m-%d} to {dates_t[-1]:%Y-%m-%d}")
print(f"daily sd {yt.std(ddof=1):.6f}, annualised {np.sqrt(252)*yt.std(ddof=1):.2%}")
print(f"sample excess kurtosis {stats.kurtosis(yt):.2f}  (Gaussian = 0)")
print(f"worst day {yt.min():.4f} on {dates_t[yt.argmin()]:%Y-%m-%d}, "
      f"best day {yt.max():.4f} on {dates_t[yt.argmax()]:%Y-%m-%d}")
```

The sample is $11{,}739$ trading days. Its daily standard deviation of $1.1029\%$ is an
annualised volatility of $17.51\%$, unremarkable, but its excess kurtosis of $15.05$ is
not: a Gaussian sample of this length would produce a value near zero, and a value of
fifteen means the extremes are far too large and too frequent for the normal
distribution. The worst day is $-17.44\%$ on 19 October 1987 and the best is $11.36\%$ on
13 October 2008. Under a Gaussian error with the fitted standard deviation, a
sixteen-sigma day such as Black Monday has a probability so small that it would not be
expected once in the age of the universe. Something in the model has to give, and the
Student-$t$ error is the least drastic thing to change.

The sampler cycles over the four full conditionals derived above: $\beta$, $\sigma^2$,
each $\lambda_t^2$, and $\nu$ from the discrete grid conditional. With nearly twelve
thousand observations the grid needs to be fine, because the data are informative enough
to distinguish $\nu = 2.9$ from $\nu = 3.1$, and the grid conditional is evaluated
through the sufficient statistics $\sum_t \log \lambda_t^2$ and $\sum_t 1/\lambda_t^2$
rather than by forming a grid-by-observation matrix, which keeps one sweep an $O(T)$
operation and the whole run to a couple of seconds.

```{python}
rng_t = np.random.default_rng(24)                # a dedicated stream for this section

nu_grid = np.arange(2.1, 20.01, 0.02); a_grid = nu_grid / 2.0
b0_t, B0_inv_t = np.zeros(k_t), np.linalg.inv(np.eye(k_t) * 100.0)   # near-flat on beta
a0_t, d0_t, n_iter_t, burn_t = 1e-6, 1e-12, 2500, 500

beta_t_draws = np.empty((n_iter_t, k_t))
sigma2_t_draws, nu_t_draws = np.empty(n_iter_t), np.empty(n_iter_t)
weight_sum = np.zeros(n_t)

lam2, sigma2_t, nu_t = np.ones(n_t), yt.var(), 8.0
for t in range(n_iter_t):
    wgt = 1.0 / lam2                                     # weight 1/lambda_t^2
    Xw, yw = Xt * np.sqrt(wgt)[:, None], yt * np.sqrt(wgt)
    B1t = np.linalg.inv(B0_inv_t + Xw.T @ Xw)
    b1t = B1t @ (B0_inv_t @ b0_t + Xw.T @ yw)
    beta_t = b1t + np.sqrt(sigma2_t) * (np.linalg.cholesky(B1t) @ rng_t.standard_normal(k_t))
    resid = yt - Xt @ beta_t
    resid_w = resid * np.sqrt(wgt)
    sigma2_t = 1.0 / rng_t.gamma(a0_t + n_t / 2, 1.0 / (d0_t + 0.5 * (resid_w @ resid_w)))
    ytilde = resid / np.sqrt(sigma2_t)
    lam2 = 1.0 / rng_t.gamma((nu_t + 1) / 2, 2.0 / (nu_t + ytilde ** 2))
    S1, S2 = np.log(lam2).sum(), (1.0 / lam2).sum()      # sufficient statistics for nu
    logp_nu = (n_t * (a_grid * np.log(a_grid) - gammaln(a_grid))
               - (a_grid + 1.0) * S1 - a_grid * S2)
    p_nu = np.exp(logp_nu - logp_nu.max()); p_nu /= p_nu.sum()
    nu_t = rng_t.choice(nu_grid, p=p_nu)
    beta_t_draws[t], sigma2_t_draws[t], nu_t_draws[t] = beta_t, sigma2_t, nu_t
    if t >= burn_t:
        weight_sum += 1.0 / lam2

beta_t_post, sigma2_t_post = beta_t_draws[burn_t:], sigma2_t_draws[burn_t:]
nu_t_post = nu_t_draws[burn_t:]
w_post_mean = weight_sum / (n_iter_t - burn_t)
print(f"kept {beta_t_post.shape[0]} sweeps after discarding {burn_t} for burn-in")
```

The degrees of freedom are the headline result, because they are a direct measurement of
how fat the tails of the daily market return actually are.

```{python}
nu_hat = nu_t_post.mean()
print(f"posterior mean of nu {nu_hat:.4f}, "
      f"95% credible interval [{np.quantile(nu_t_post, 0.025):.4f}, "
      f"{np.quantile(nu_t_post, 0.975):.4f}]")
print(f"posterior scale sigma {np.sqrt(sigma2_t_post.mean()):.6f}")
print(f"implied sd of the error sigma*sqrt(nu/(nu-2)) = "
      f"{np.sqrt(sigma2_t_post.mean() * nu_hat / (nu_hat - 2)):.6f}")
print(f"P(nu < 4 | y) = {(nu_t_post < 4).mean():.4f}")
```

The posterior mean of $\nu$ is $2.9386$ with a ninety-five percent credible interval from
$2.7600$ to $3.1400$, and every retained draw is below four. That is a very strong
statement. A Student-$t$ distribution has a finite variance only for $\nu > 2$ and a
finite kurtosis only for $\nu > 4$, so the data say that the daily market return has a
variance that is only just finite and a fourth moment that is not finite at all. Every
number anyone reports about the kurtosis of daily returns, including the $15.05$ printed
above, is therefore an estimate of a population quantity that does not exist, which is
worth knowing before quoting one. The fitted scale is $\sigma = 0.006702$, and because
the scale is not the standard deviation for a $t$ distribution, the implied error
standard deviation is $\sigma\sqrt{\nu/(\nu-2)} = 0.011859$.

To see what modelling the tails buys, we fit the same $(X_t, y_t)$ with the ordinary
Gaussian-error sampler used earlier in this chapter, with no $\lambda_t$ or $\nu$ to
update.

```{python}
rng_g = np.random.default_rng(25)

b0_g, B0_inv_g = np.zeros(k_t), np.linalg.inv(np.eye(k_t) * 100.0)
beta_g_draws, sigma2_g_draws = np.empty((n_iter_t, k_t)), np.empty(n_iter_t)
sigma2_g, XtXt, Xtyt = yt.var(), Xt.T @ Xt, Xt.T @ yt
for t in range(n_iter_t):
    B1g = np.linalg.inv(B0_inv_g + XtXt / sigma2_g)
    b1g = B1g @ (B0_inv_g @ b0_g + Xtyt / sigma2_g)
    beta_g = b1g + np.linalg.cholesky(B1g) @ rng_g.standard_normal(k_t)
    resid_g = yt - Xt @ beta_g
    sigma2_g = 1.0 / rng_g.gamma(a0_t + n_t / 2, 1.0 / (d0_t + 0.5 * (resid_g @ resid_g)))
    beta_g_draws[t], sigma2_g_draws[t] = beta_g, sigma2_g
beta_g_post, sigma2_g_post = beta_g_draws[burn_t:], sigma2_g_draws[burn_t:]

print(pd.DataFrame({"Gaussian mean": beta_g_post.mean(axis=0),
                    "Gaussian sd": beta_g_post.std(axis=0),
                    "Student mean": beta_t_post.mean(axis=0),
                    "Student sd": beta_t_post.std(axis=0)},
                   index=["intercept", "lagged return"]).round(6))
print(f"Gaussian residual sd {np.sqrt(sigma2_g_post.mean()):.6f}")
print(f"P(lagged coefficient < 0 | y): Gaussian {(beta_g_post[:, 1] < 0).mean():.4f}   "
      f"Student {(beta_t_post[:, 1] < 0).mean():.4f}")
print(f"annualised intercept: Gaussian {252*beta_g_post[:, 0].mean():.2%}   "
      f"Student {252*beta_t_post[:, 0].mean():.2%}")
```

The two models disagree about economics, not merely about standard errors. Under Gaussian
errors the coefficient on the lagged return is $-0.027876$ with a posterior standard
deviation of $0.009327$, so the market appears to reverse about three percent of the
previous day's move, and the posterior probability that the coefficient is negative is
$0.9985$: on this evidence a desk would conclude that short-horizon reversal is real.
Under Student errors the same coefficient is $0.003854$ with a standard deviation of
$0.008414$, essentially zero, and the probability of a negative coefficient falls to
$0.3200$. The apparent reversal was manufactured by a handful of crash days, on which a
huge fall was followed by a large rebound; the Gaussian model, unable to treat those days
as anything other than ordinary observations, read them as a systematic pattern, while
the Student model discounts them and finds nothing left. This is the practical cost of
the wrong error distribution, and it is not a small one.

The intercepts differ for the same reason and in the same direction. The Gaussian
estimate of the average daily premium is $0.000373$, or $9.40\%$ a year, close to what an
investor actually earned; the Student estimate is $0.000662$, or $16.68\%$ a year, because
the robust fit largely ignores the crash days on which investors lost most heavily. Both
numbers are correct answers to different questions: the Gaussian intercept estimates the
sample average return, the Student intercept estimates the centre of the bulk of the
distribution. For a risk manager, the difference between the two is the premium the
crashes took away, and reporting the robust figure as an expected return would be a
serious mistake.

The mechanism behind all of this is the weight $1/\lambda_t^2$, and it can be read off
directly. The next block plots the posterior mean weight of each of the $11{,}739$ days
against its standardised residual and names the days that are discounted most heavily.

```{python}
beta_hat_t = beta_t_post.mean(axis=0)
sigma_hat_t = np.sqrt(sigma2_t_post.mean())
ytilde_hat = (yt - Xt @ beta_hat_t) / sigma_hat_t

fig, ax = plt.subplots()
ax.scatter(ytilde_hat, w_post_mean, s=6, color="steelblue", alpha=0.4)
ax.set_yscale("log")
ax.set_xlabel(r"standardised residual  $\tilde y_t$")
ax.set_ylabel(r"posterior mean of  $1/\lambda_t^2$  (log scale)")
ax.set_title("Outlier discounting on daily market returns, 1980-2026")
for i in np.argsort(-np.abs(ytilde_hat))[:3]:
    ax.annotate(f"{dates_t[i]:%d %b %Y}", (ytilde_hat[i], w_post_mean[i]),
                textcoords="offset points", xytext=(6, 6), fontsize=8)
plt.tight_layout(); plt.show()

big = np.argsort(-np.abs(ytilde_hat))[:8]
print(pd.DataFrame({"date": dates_t[big].strftime("%Y-%m-%d"),
                    "excess return": yt[big],
                    "std residual": ytilde_hat[big],
                    "weight": w_post_mean[big]}).round(4).to_string(index=False))
print(f"median weight {np.median(w_post_mean):.4f}, "
      f"smallest {w_post_mean.min():.4f}, "
      f"days with weight below 0.1: {(w_post_mean < 0.1).sum()}")
```

The scatter is an almost perfect inverted V on the logarithmic scale, which is the
inverse-gamma mean $d/(a-1) = (\nu + \tilde y_t^2)/(\nu - 1)$ doing exactly what the
derivation promised: the weight falls roughly with the square of the standardised
residual. The eight most heavily discounted days are a list of financial history. Black
Monday, 19 October 1987, has a standardised residual of $-26.09$ and receives a weight of
$0.0056$, so the model treats it as worth about one two-hundredth of a normal day. Then
come 16 March 2020, the worst day of the pandemic crash, at $-18.07$ and a weight of
$0.0120$; 13 October 2008 and 28 October 2008, the two violent rebounds inside the
financial crisis; 12 March 2020 and 24 March 2020; 9 April 2025, the day the announced
tariffs were suspended; and 1 December 2008. Nobody told the sampler that these dates
were special. It found them from the residuals alone, and it dealt with them by
down-weighting rather than by deleting, so each still contributes to the posterior in
proportion to how much a fat-tailed model thinks it should. The median weight is $1.1269$,
slightly above one because the weights must average out against the discounted extremes,
and $94$ days out of $11{,}739$ receive a weight below one tenth.

Finally, the posterior of $\nu$ itself, on the left, and the two fitted posteriors of the
lagged-return coefficient side by side, on the right.

```{python}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].hist(nu_t_post, bins=40, density=True, color="darkorange", alpha=0.85)
axes[0].axvline(4.0, color="black", ls=":", label="finite kurtosis needs $\\nu>4$")
axes[0].set_xlabel(r"$\nu$"); axes[0].set_ylabel("posterior density")
axes[0].legend(fontsize=8); axes[0].set_title("Degrees of freedom of the daily return")

axes[1].hist(beta_g_post[:, 1], bins=40, density=True, alpha=0.6, color="red",
             label="Gaussian errors")
axes[1].hist(beta_t_post[:, 1], bins=40, density=True, alpha=0.6, color="steelblue",
             label="Student errors")
axes[1].axvline(0.0, color="black", ls=":", label="no predictability")
axes[1].set_xlabel(r"coefficient on $r_{t-1}$"); axes[1].legend(fontsize=8)
axes[1].set_title("Same data, two error models")
plt.tight_layout(); plt.show()
```

The left panel shows a posterior for $\nu$ concentrated in a narrow band around three and
lying entirely to the left of the dotted line at four, so the finite-kurtosis region
receives no posterior mass whatever. The right panel shows the two coefficient posteriors
barely overlapping: the Gaussian density sits almost wholly to the left of zero and the
Student density almost exactly on it. Two analysts working with the same eleven thousand
returns and differing only in their error assumption would report opposite conclusions
about whether the equity market reverses from one day to the next, and the fatness of the
tails is not a matter of taste but something the data themselves have just measured.
