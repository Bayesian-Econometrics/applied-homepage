> Cut from lecture note 3 in the restructuring pass. Belongs in the **Week 3 exercise
> session** (`exercise-sessions/03-monte-carlo-importance-sampling.qmd`): the conjugate
> inverse-Gamma posterior for the daily market variance, the Monte Carlo demonstration
> run on it, the log-log check of the 1/sqrt(S) rate, and the exponentially tilted
> importance sampler for a crash probability. Lecture 3 now runs a single Student-t
> target on monthly data; this conjugate model is the natural warm-up exercise because
> every simulation answer can be checked against closed-form algebra.

## A posterior for the variance of the market

## A posterior for the variance of the market

The running example needs a posterior, and the simplest honest one for a risk desk is a
posterior for the variance of the daily market excess return. Write $r_t$ for the daily
excess return on the US market portfolio and assume, over a short window,
$r_t \mid \sigma^2 \sim \mathcal{N}(0, \sigma^2)$ independently. Two modelling choices
are being made and both should be said aloud. Setting the mean to zero is defensible at
the daily frequency, where the average excess return is a tiny fraction of the daily
standard deviation and estimating it costs more than it buys. Assuming a constant
variance is defensible only over a short window, which is why risk desks use a rolling
one-year lookback rather than the whole history; Lecture 12 replaces this assumption
with stochastic volatility.

With an $\text{Inverse-Gamma}(\alpha_0, \delta_0)$ prior the update is the one derived in
Lecture 2 with the coefficients removed, so the posterior is again inverse-Gamma,

$$
\sigma^2 \mid r \sim \text{Inverse-Gamma}\!\left(\alpha_0 + \frac{n}{2},\;
\delta_0 + \frac{1}{2}\sum_{t=1}^{n} r_t^2\right),
$$

with prior shape $\alpha_0 = 3$ and scale $\delta_0 = 0.0002$, whose mean is $0.0001$, a
daily standard deviation of one percent, or about sixteen percent a year. That is the
order of magnitude of equity index volatility, and the shape of three keeps the prior
loose.

**Reading the computation.** The block loads the daily factor file, takes the most recent
$250$ trading days as the estimation window, and applies the two-line inverse-Gamma
update. The posterior is reported as an annualised volatility, multiplying the daily
standard deviation by $\sqrt{252}$, because that is the unit a desk quotes.

```{python}
ff_d = pd.read_parquet(DATA / "ff_factors_daily.parquet")
window = ff_d["2000-01-01":].tail(250)              # one-year rolling lookback
r_day = window["mkt_rf"].to_numpy()

a0_v, d0_v = 3.0, 0.0002                            # Inverse-Gamma(3, 0.0002)
n_day = r_day.size
a1_v = a0_v + n_day / 2.0
d1_v = d0_v + 0.5 * (r_day ** 2).sum()
var_post = stats.invgamma(a=a1_v, scale=d1_v)

print(f"{n_day} trading days, {window.index[0]:%Y-%m-%d} to {window.index[-1]:%Y-%m-%d}")
print(f"posterior for sigma^2: IG({a1_v:.1f}, {d1_v:.6f})")
print(f"E[sigma^2 | r] = {var_post.mean():.3e}   "
      f"annualised vol at the posterior mean {np.sqrt(var_post.mean()*252):.4f}")
lo_v, hi_v = np.sqrt(var_post.ppf([0.025, 0.975]) * 252)
print(f"95% credible interval for annualised volatility: [{lo_v:.4f}, {hi_v:.4f}]")
```

The window runs from August 2025 to July 2026 and contains $250$ daily returns. The
posterior mean variance corresponds to an annualised volatility of $13.26\%$, and the
credible interval runs from $12.16\%$ to $14.46\%$. That interval is the whole subject of
this chapter. A desk that quotes $13.26\%$ and stops has thrown away the statement that
the data are equally content with $12.2\%$ or $14.5\%$, and the cost of throwing it away
is a number we compute at the end of the lecture.

## Monte Carlo on the conjugate posterior

```{python}
S = 5000
draws_v = var_post.rvs(size=S, random_state=rng)

c_vol = 0.14                                        # annualised volatility limit
c_var = c_vol ** 2 / 252                            # the same limit as a daily variance

true_mean = var_post.mean()
true_tail = 1 - var_post.cdf(c_var)

mc_mean = draws_v.mean()
mc_tail = np.mean(draws_v > c_var)

print(f"E[sigma^2|r]            exact {true_mean:.4e}   MC {mc_mean:.4e}")
print(f"P(annual vol > {c_vol:.0%} | r)   exact {true_tail:.4f}   MC {mc_tail:.4f}")
```

Both Monte Carlo estimates reproduce the values the conjugate algebra delivers exactly,
the mean variance to within two tenths of a percent and the tail probability to within a
thousandth. The tail probability is nothing but the average of an indicator
variable, so the identical averaging machine handles a mean, a probability, a variance,
or any other posterior summary without a single new idea; only the function $g$ inside
the sum changes. The number itself is worth pausing on: the posterior puts probability
$0.1034$, about one chance in ten, on the market's volatility over this window having
been above the fourteen percent limit, which is a sentence about a risk limit that no
point estimate of $13.26\%$ can produce.

## Where the error comes from: the Monte Carlo standard error

A Monte Carlo estimate is a random number computed from a finite sample, so it needs an
error bar exactly as an estimator computed from data does. The derivation above gives it:
the sample standard deviation of the $g(\theta^{(s)})$ divided by $\sqrt{S}$.

```{python}
mcse_mean = draws_v.std(ddof=1) / np.sqrt(S)
print(f"MC mean {mc_mean:.4e}  +/- {1.96*mcse_mean:.2e}  (95% MC interval)")

ind = (draws_v > c_var).astype(float)                # g(theta) = 1{sigma^2 > c}
mcse_tail = ind.std(ddof=1) / np.sqrt(S)
print(f"MC tail {mc_tail:.4f}  +/- {1.96*mcse_tail:.4f}  (95% MC interval)")
print(f"relative MC error: mean {mcse_mean/mc_mean:.4%}, tail {mcse_tail/mc_tail:.4%}")
```

The two intervals reported here describe uncertainty in the *Monte Carlo estimate*
coming from having drawn only $S$ values, not uncertainty about $\sigma^2$ itself, which
is already fully captured by the posterior. Drawing more samples narrows these
intervals arbitrarily; it changes nothing about the posterior, which is fixed once the
data and the prior are fixed. Note also that the tail-probability MCSE is far
larger relative to its estimate than the mean's MCSE is: an indicator has a Bernoulli
variance $p(1-p)$, so the relative error of a rare-event probability is much worse than
that of a well-centred mean at the same $S$. That asymmetry is not a curiosity here; it
is the reason the tail calculation later in this chapter needs importance sampling and
the mean does not.

### Watching the $1/\sqrt{S}$ rate

The clearest way to internalise the square-root rate is to watch the running average as
draws accumulate, with a band of width $\pm 1.96 \times \text{MCSE}$ around the known
posterior mean. The running estimate should wander inside a band that narrows steadily,
never settling but converging. We plot the running estimate of the annualised volatility
implied by the mean variance, so that the vertical axis carries units a desk recognises.

```{python}
running_mean = np.cumsum(draws_v) / np.arange(1, S + 1)
s_grid = np.arange(1, S + 1)
band = 1.96 * var_post.std() / np.sqrt(s_grid)

fig, ax = plt.subplots()
ax.plot(s_grid, np.sqrt(running_mean * 252), lw=1, label="running MC estimate")
ax.axhline(np.sqrt(true_mean * 252), color="black", lw=1, ls=":",
           label="exact posterior mean")
ax.fill_between(s_grid, np.sqrt(np.maximum(true_mean - band, 1e-12) * 252),
                np.sqrt((true_mean + band) * 252),
                alpha=0.2, label=r"exact $\pm\,1.96\,$MCSE")
ax.set_xlabel("number of draws  S")
ax.set_ylabel("annualised volatility at the mean variance")
ax.set_ylim(np.sqrt(true_mean * 252) - 0.02, np.sqrt(true_mean * 252) + 0.02)
ax.legend(); ax.set_title("Monte Carlo estimate converges at rate 1/sqrt(S)")
plt.tight_layout(); plt.show()
```

The band visibly narrows as $S$ grows and the running average stays inside it, which is
the derivation above made concrete. To confirm the rate rather than merely gesture at
it, we can measure the actual root-mean-squared error of the estimator over many
independent repetitions at a range of sample sizes and check that it falls in
proportion to $1/\sqrt{S}$, which on a log-log plot is a straight line of slope
$-1/2$.

```{python}
sizes = np.array([50, 200, 800, 3200, 12800])
reps = 200
rmse = np.empty(len(sizes))
for i, m in enumerate(sizes):
    ests = np.array([var_post.rvs(size=m, random_state=rng).mean()
                      for _ in range(reps)])
    rmse[i] = np.sqrt(np.mean((ests - true_mean) ** 2))

ref = rmse[0] * np.sqrt(sizes[0] / sizes)         # 1/sqrt(S) reference line

fig, ax = plt.subplots()
ax.loglog(sizes, rmse, "o-", label="empirical RMSE")
ax.loglog(sizes, ref, "--", color="grey", label=r"$\propto 1/\sqrt{S}$ reference")
ax.set_xlabel("number of draws  S"); ax.set_ylabel(r"RMSE of the estimate of $E[\sigma^2|r]$")
ax.legend(); ax.set_title("Error falls like 1/sqrt(S) on a log-log scale")
plt.tight_layout(); plt.show()
```

The empirical curve sits on top of the reference line across three orders of magnitude
in $S$, which is the $\sqrt{S}$ law made visible rather than merely quoted: each step in
the grid of sizes multiplies $S$ by exactly 4, and the RMSE drops by a factor close to
2 at every step, exactly as $1/\sqrt{4} = 1/2$ predicts.

## Importance sampling for a rare event: a crash probability

### The case importance sampling was invented for: a crash probability

Reweighting a proposal is worth the trouble even when the target *can* be sampled
directly, provided the quantity we want is rare. The natural risk-desk example is the
probability of a crash. Take the conjugate variance posterior from the daily window and
ask for the predictive probability that the market loses more than fifteen percent over
the next month, that is over $H = 21$ trading days,

$$
\Pr(R_H < -0.15 \mid r) = \int \Pr(R_H < -0.15 \mid \sigma^2)\,p(\sigma^2\mid r)\,d\sigma^2,
$$

with $R_H \mid \sigma^2 \sim \mathcal{N}(0, H\sigma^2)$. Mixing a Normal over an
inverse-Gamma gives a Student $t$, so this particular integral happens to have a closed
form and we can check every simulation answer against it, but the *shape* of the problem
is completely general: it is an expectation of an indicator that is almost always zero.

The naive estimator draws $\sigma^2$ from its posterior and a standard Normal shock, forms
$R_H$, and averages the indicator. Because the event has probability of the order of
$5\times10^{-5}$, a sample of twenty thousand draws contains on average one hit and quite
often none at all, and the estimator's relative error is therefore enormous, exactly as
the Bernoulli variance $p(1-p)$ predicted earlier. Importance sampling fixes this by
**shifting the shock distribution into the tail**: propose $Z \sim \mathcal{N}(\mu, 1)$
with $\mu = -4.5$ instead of $\mathcal{N}(0,1)$, so that most proposals land in the loss
region, and correct with the weight

$$
w(Z) = \frac{\varphi(Z)}{\varphi(Z - \mu)} = \exp\!\left(-\mu Z + \tfrac{1}{2}\mu^2\right).
$$

The variance draw is left alone, since the posterior for $\sigma^2$ is easy to sample and
is not what makes the event rare.

```{python}
H, c_crash = 21, -0.15
nu_pred, scale_pred = 2 * a1_v, np.sqrt(H * d1_v / a1_v)
exact_crash = stats.t(df=nu_pred, scale=scale_pred).cdf(c_crash)

S_tail = 200_000
sig_path = np.sqrt(H * var_post.rvs(size=S_tail, random_state=rng))

Z_naive = rng.standard_normal(S_tail)
hit_naive = (sig_path * Z_naive < c_crash).astype(float)
p_naive = hit_naive.mean()
se_naive = hit_naive.std(ddof=1) / np.sqrt(S_tail)

mu_shift = -4.5
Z_is = rng.standard_normal(S_tail) + mu_shift
w_tail = np.exp(-mu_shift * Z_is + 0.5 * mu_shift ** 2)
contrib = (sig_path * Z_is < c_crash) * w_tail
p_is = contrib.mean()
se_is = contrib.std(ddof=1) / np.sqrt(S_tail)

print(f"exact P(21-day return < {c_crash:.0%}) = {exact_crash:.3e}")
print(f"naive MC {p_naive:.3e}  se {se_naive:.2e}  "
      f"relative error {se_naive/max(p_naive,1e-12):.1%}  hits {int(hit_naive.sum())}")
print(f"tilted IS {p_is:.3e}  se {se_is:.2e}  relative error {se_is/p_is:.2%}")
print(f"variance ratio naive / IS: {(se_naive/se_is)**2:.0f}")
```

The exact answer is $5.382\times10^{-5}$, roughly one month in eighteen thousand six
hundred. The naive estimator finds nine hits in two hundred thousand draws, reports
$4.500\times10^{-5}$ and carries a relative error of $33\%$, while the tilted estimator
returns $5.391\times10^{-5}$ with a relative error of $0.69\%$ from the same number of
draws, a variance reduction of $1{,}640$ fold. Put in the terms a desk uses, the naive
answer is consistent with anything between one such month in thirteen thousand and one in
sixty thousand, while the importance-sampling answer pins the frequency down to within a
percent. This is precisely the
regime where the naive average is inefficient rather than wrong, and where the choice of
proposal is not a matter of taste but the difference between an answer and a shrug.

## Value-at-risk under the conjugate posterior

### One-month value-at-risk

The risk officer's quantity is the loss that will be exceeded with probability one percent
over a $21$ trading-day horizon. Under the plug-in convention the variance is fixed at its
posterior mean $\hat\sigma^2 = \mathbb{E}[\sigma^2\mid r]$ and the horizon return is
Normal, so

$$
\text{VaR}^{\text{plug}}_{0.99} = z_{0.99}\sqrt{H\hat\sigma^2},
\qquad z_{0.99} = 2.326 .
$$

Integrating over the posterior instead of fixing it replaces the Normal by a Student $t$,
because a Normal scale-mixed over an inverse-Gamma is exactly a $t$: with
$\sigma^2\mid r \sim \text{IG}(\alpha_1,\delta_1)$ the horizon return has
$2\alpha_1$ degrees of freedom and scale $\sqrt{H\delta_1/\alpha_1}$. The two answers
differ only through the fatter tail that parameter uncertainty creates, so the difference
grows as we move further into the tail, which is the practically important part of the
statement. We compute both analytically and confirm the second by Monte Carlo, and price
the gap on a ten million euro position.

```{python}
notional = 10_000_000.0
t_pred = stats.t(df=nu_pred, scale=scale_pred)

rows = []
for q in (0.95, 0.99, 0.995):
    v_plug = stats.norm.ppf(q) * np.sqrt(H * var_post.mean())
    v_par = t_pred.ppf(q)
    rows.append({"level": q, "plug-in (bp)": 1e4 * v_plug, "posterior (bp)": 1e4 * v_par,
                 "gap (bp)": 1e4 * (v_par - v_plug),
                 "gap (EUR on 10m)": notional * (v_par - v_plug)})
var_table = pd.DataFrame(rows).set_index("level")
print(var_table.round(2).to_string())

sig_mc = np.sqrt(H * var_post.rvs(size=200_000, random_state=rng))
R_mc = sig_mc * rng.standard_normal(200_000)
print(f"\nMonte Carlo check at 99%: {1e4*np.quantile(R_mc, 0.01):.1f} bp "
      f"(analytic {-1e4*t_pred.ppf(0.01):.1f} bp, sign flipped for a loss)")
```

At the ninety-five percent level the two conventions agree to within a fifth of a basis
point, and the plug-in figure is if anything the *larger* of the two, because a Student
$t$ with many degrees of freedom is slightly thinner than the Normal in the shoulders
before it becomes fatter in the tail. At ninety-nine percent the ordering has reversed and
the posterior figure of $892.59$ basis points exceeds the plug-in $890.49$ by $2.10$ basis
points, which is $2{,}102$ euro of additional capital on a ten million euro position, and
at ninety-nine and a half percent the gap widens again to $3.52$ basis points, or $3{,}515$
euro. The Monte Carlo check returns $892.1$ basis points against the
analytic $892.6$, agreeing to within its own simulation error and confirming that
averaging over the posterior of the variance and reading off a quantile of the mixture is
the same operation as the closed-form Student $t$.

These are not large numbers, and saying so is part of the lesson. With $250$ daily
observations the variance is well estimated and parameter uncertainty is worth a couple of
basis points at the ninety-nine percent level. The size of the correction is governed by
the degrees of freedom $2\alpha_1 = 256$, which grows with the window, so a desk using a
sixty-day lookback would face a materially larger correction and a desk using ten years of
data a negligible one. What does not shrink is the direction: beyond about the ninety-eighth
percentile the plug-in convention understates the loss every time, and it understates it by
more the deeper into the tail the limit is set.

