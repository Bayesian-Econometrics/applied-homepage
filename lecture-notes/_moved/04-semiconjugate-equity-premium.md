> Cut from lecture note 4 in the restructuring pass. Belongs in the **Week 4 exercise
> session**. The semi-conjugate Normal model in the precision parameterisation
> (mu ~ N(m, 1/c), tau ~ Gamma(a/2, b/2)) and its run on 439 months of market excess
> returns, with the Student-t(n-1) benchmark, the annualised premium summary, the
> mu-tau scatter and the running-mean figure. Removed as a second inference problem:
> the rewritten chapter derives the same two conditionals in general weighted-regression
> form on the shared Student-t target, so no derivation is lost here, only the second
> data application. Needs a data-loading preamble (ff_factors_monthly.parquet) and the
> plotting defaults.

## What does conditioning on the current variance mean?

Imagine observing a short series of returns and trying to learn both its average and its
volatility. With a large volatility, a proposed average can be quite far from the sample
mean without making the observations implausible. With a small volatility, the same
discrepancy is strong evidence against that average. The two unknowns therefore interact.

Gibbs sampling temporarily fixes one unknown **for one calculation**. Given a particular
variance, learning the mean is a one-parameter Normal update. Then, given the newly drawn
mean, squared residuals tell us about the variance. The next iteration repeats both
calculations using new values. We are not claiming that either parameter is actually
known, nor fixing either at its estimated value forever.

Below we describe volatility by its **precision** $\tau=1/\sigma^2$.
For a known precision, one observation has density proportional to
$\sqrt{\tau}\exp[-\tau(x_i-\mu)^2/2]$. Thus a large $\tau$ means a narrow density:
the observation is informative about $\mu$. For $n$ independent observations, multiplication
produces $\tau^{n/2}\exp[-\tau\sum_i(x_i-\mu)^2/2]$.
Read this expression once as a function of $\mu$, and once as a function of $\tau$.
In the first reading it is a quadratic exponential; in the second it is a power times
an exponential. That is why Normal and Gamma full conditionals will appear.

Choosing an independent Normal prior for $\mu$ and a Gamma prior for $\tau$ expresses
separate beliefs about location and precision. It differs from Week 2's scaled NIG
prior. The conditional derivations must follow the prior actually chosen; changing the
prior while keeping the old sampler would change the question without changing the answer.

## A model where the joint is hard but the pieces are easy

The cleanest setting to see why this trade works is the **semi-conjugate normal model**,
and it is worth attaching an economic question to it immediately, because the model is
not a toy. **What is the expected excess return on the US equity market, and how sure can
we be?** That single number, the **equity premium**, is the input every strategic asset
allocation starts from, and the two things we do not know about it are exactly its mean
and its variance. Write $x_1, \dots, x_n$ for the monthly excess returns on the market
portfolio and model them as independent draws from a normal distribution with unknown
mean $\mu$, the premium we are after, and unknown precision $\tau = 1/\sigma^2$, the
reciprocal of the variance that measures how noisy those returns are,

$$
x_i \mid \mu, \tau \sim \mathcal{N}\!\left(\mu, \tfrac{1}{\tau}\right), \qquad i = 1, \dots, n.
$$

Both parameters are unknown, so we need priors for both. We choose them independently of
each other,

$$
\mu \sim \mathcal{N}\!\left(m, \tfrac{1}{c}\right), \qquad
\tau \sim \text{Gamma}\!\left(\tfrac{a}{2}, \tfrac{b}{2}\right),
$$

a normal prior on the mean and a gamma prior on the precision, parameterised with the
$1/2$'s so that the algebra below lines up with the usual normal-gamma notation. This is
called **semi-conjugate**: each prior is conjugate to the likelihood if we imagine the
*other* parameter were known, but the pair together is not jointly conjugate, because
$\mu$'s prior does not know about $\tau$ and vice versa. Multiplying likelihood and
priors and keeping every term,

$$
f(\mu, \tau \mid {\bf x}) \;\propto\;
\underbrace{\tau^{n/2} \exp\!\left(-\frac{\tau}{2}\sum_{i=1}^n (x_i - \mu)^2\right)}_{\text{likelihood}}
\times
\underbrace{\exp\!\left(-\frac{c}{2}(\mu - m)^2\right)}_{\text{prior on } \mu}
\times
\underbrace{\tau^{a/2 - 1} \exp\!\left(-\frac{b}{2}\tau\right)}_{\text{prior on } \tau}.
$$

The sum of squares splits into a piece that does not involve $\mu$ and a piece that does,
$\sum_i (x_i - \mu)^2 = (n-1)s^2 + n(\bar{x} - \mu)^2$, where $\bar{x}$ is the sample mean
and $s^2$ the sample variance. Substituting this and collecting the two powers of $\tau$
gives the joint posterior kernel

$$
f(\mu, \tau \mid {\bf x}) \;\propto\;
\tau^{\frac{a+n}{2} - 1}
\exp\!\left(-\frac{c}{2}(\mu - m)^2 - \frac{\tau}{2}\Big[b + (n-1)s^2 + n(\bar{x} - \mu)^2\Big]\right).
$$

This expression is not the kernel of any distribution we have named. $\mu$ and $\tau$
appear multiplied together inside the exponential, through the $\tau (\bar x - \mu)^2$
term, so the joint density does not factor and no standard two-dimensional family
matches it. Integrating it over $\mu$ or $\tau$ to get a marginal, or to get the
normalising constant, is not something we can do by inspection. If this were the end of
the story, the semi-conjugate model would be as intractable as the Cauchy example from
Lecture 1.

### The conditional posterior of $\mu$

It is not the end of the story, because holding one parameter fixed dissolves the
problem. Fix $\tau$ at its current value and look at the posterior kernel above as a
function of $\mu$ alone; every factor that does not contain $\mu$, including the entire
$\tau^{(a+n)/2-1}\exp(-\tau b/2 - \tau(n-1)s^2/2)$ block, is a constant that gets absorbed
into the proportionality sign:

$$
f(\mu \mid \tau, {\bf x}) \;\propto\;
\exp\!\left(-\frac{c}{2}(\mu - m)^2 - \frac{n\tau}{2}(\mu - \bar{x})^2\right).
$$

Both terms in the exponent are quadratic in $\mu$, so this is a normal kernel in
disguise, and we uncover it by **completing the square**.

::: {.callout-note collapse="true"}
## Derivation: completing the square in the mean

Expand both squares and collect powers of $\mu$,

$$
-\frac{1}{2}\Big[c(\mu-m)^2 + n\tau(\mu-\bar{x})^2\Big]
= -\frac{1}{2}\Big[(c + n\tau)\mu^2 - 2(cm + n\tau\bar{x})\mu + \big(cm^2 + n\tau\bar{x}^2\big)\Big].
$$

The bracket is a quadratic in $\mu$ with leading coefficient $c + n\tau$. Factor that
coefficient out of the first two terms and add and subtract the square needed to make
them a perfect square,

$$
-\frac{c+n\tau}{2}\left[\mu^2 - 2\mu \cdot \frac{cm+n\tau\bar{x}}{c+n\tau}\right]
= -\frac{c+n\tau}{2}\left[\mu - \frac{cm+n\tau\bar{x}}{c+n\tau}\right]^2 + \text{const},
$$

where "const" collects everything that does not involve $\mu$ and is absorbed by the
proportionality sign. What remains is exactly the kernel of a normal density in $\mu$,
with the quantity in brackets as the mean and the coefficient in front as the precision.
:::

Reading off the parameters of that kernel,

$$
\mu \mid \tau, {\bf x} \sim \mathcal{N}\!\left(\frac{cm + n\tau\bar{x}}{c + n\tau},\; \frac{1}{c + n\tau}\right).
$$

The posterior precision $c + n\tau$ adds the prior precision to the data precision
$n\tau$, and the posterior mean is a precision-weighted average of the prior mean $m$
and the sample mean $\bar{x}$: exactly the same weighted-average structure that appeared
for the coin in Lecture 1, now with precisions in place of pseudo-counts. Notice that
this conditional depends on $\tau$: a different current value of $\tau$ rescales how much
weight the data get relative to the prior, which is precisely the coupling that made the
joint distribution intractable in the first place.

### The conditional posterior of $\tau$

Now fix $\mu$ instead and look at the same kernel as a function of $\tau$. Every factor
that does not contain $\tau$, here the entire $\exp(-c(\mu-m)^2/2)$ term, drops out of the
proportionality:

$$
f(\tau \mid \mu, {\bf x}) \;\propto\;
\tau^{\frac{a+n}{2}-1} \exp\!\left(-\frac{\tau}{2}\Big[b + (n-1)s^2 + n(\bar{x}-\mu)^2\Big]\right).
$$

No completing the square is needed here; the exponent is already linear in $\tau$; we
only have to **collect the exponents** correctly, which we already did when we combined
the $\tau^{n/2}$ from the likelihood with the $\tau^{a/2-1}$ from the prior into
$\tau^{(a+n)/2-1}$ above. What is left is precisely the kernel of a gamma density,
$\tau^{\text{shape}-1}\exp(-\text{rate}\cdot\tau)$, with shape and rate read directly off
the exponent and the power of $\tau$:

$$
\tau \mid \mu, {\bf x} \sim \text{Gamma}\!\left(\frac{a+n}{2},\; \frac{b + (n-1)s^2 + n(\bar{x}-\mu)^2}{2}\right).
$$

The shape parameter gains half the sample size, exactly as in the Poisson-gamma update
of Lecture 1. The rate parameter gains half the residual sum of squares evaluated at the
current $\mu$: the further the current $\mu$ sits from $\bar{x}$, the larger that sum of
squares, the larger the rate, and the smaller $\tau$ tends to be drawn, which is sensible:
a mean that fits the data badly should be paired with a larger error variance.

Both conditionals are standard distributions, even though the joint posterior we started
from is not, and this is the entire mechanism the Gibbs sampler exploits. The obstacle
that stopped us, the cross term $\tau(\bar{x}-\mu)^2$ coupling the two parameters together,
is precisely what disappears once we condition on one of them.

## Running the sampler on the equity premium

We now put the derivation to work on the question the section opened with. The data are
the monthly excess return on the US market portfolio from the Kenneth French data
library, from January 1990 to the end of the file. The starting date is the one used in
Lecture 2, so the sample is the same one the banking regression will use later in this
chapter, and thirty-six years is long enough to contain the technology boom and bust, the
financial crisis, the pandemic and the inflation shock without reaching back into the
very different regulatory and monetary regime of the 1970s.

**Reading the computation.** The block loads the monthly factor file, extracts the market
excess return, and prints the two sufficient statistics, the sample mean and the sample
variance, that enter both full conditionals along with the sample size.

```{python}
ff_m = pd.read_parquet(DATA / "ff_factors_monthly.parquet")
premium = ff_m["mkt_rf"]["1990-01":]
x = premium.to_numpy()
n = x.size

xbar, s2 = x.mean(), x.var(ddof=1)
print(f"{n} months, {premium.index[0]:%Y-%m} to {premium.index[-1]:%Y-%m}")
print(f"mean monthly excess return {xbar:.6f}  ({12*xbar:.4%} a year)")
print(f"monthly standard deviation {np.sqrt(s2):.6f}  ({np.sqrt(12*s2):.4%} a year)")
```

The sample is $439$ months, from January 1990 to July 2026. The average monthly excess
return is $0.7447\%$, which multiplied by twelve is an annualised premium of $8.9369\%$,
and the monthly standard deviation is $4.3922\%$, an annualised volatility of $15.2151\%$.
Those three numbers, and nothing else about the $439$ returns, are what the two full
conditionals need.

Both priors have to be stated before we can sample, and both admit an economic reading.
For the mean we take $\mu \sim \mathcal{N}(0, 0.02^2)$: centred at zero, so the prior
does not assume that equities are rewarded at all, with a standard deviation of two
percent a month, which is twenty-four percent a year and therefore wide enough to include
any premium anyone has seriously proposed. For the precision we take
$\tau \sim \text{Gamma}(a/2, b/2)$ with $a = 2$ and $b = 0.0032$, whose mean is
$a/b = 625$, that is a prior guess of four percent monthly volatility, and whose shape of
one makes it worth about two months of data. Both are weakly informative by design: with
$439$ observations the interesting question is what the data say, not what we assumed.

**Reading the computation.** The sampler alternates the two draws derived above, starting
$\mu$ at five percent a month, a deliberately absurd value, so that the burn-in is
visible in the trace plot rather than having to be taken on trust.

```{python}
m, c = 0.0, 1.0 / 0.02**2     # prior mean zero, prior sd 2% a month
a, b = 2.0, 0.0032            # Gamma(a/2, b/2), prior mean 625 = 1/0.04^2

burn, n_iter = 1000, 20_000
total = burn + n_iter
mu_draws = np.empty(total)
tau_draws = np.empty(total)

mu = 0.05                                  # deliberately far from anything plausible
for t in range(total):
    rate_tau = 0.5 * (b + (n - 1) * s2 + n * (mu - xbar) ** 2)
    tau = rng.gamma(0.5 * (a + n), 1.0 / rate_tau)
    mu = rng.normal((c * m + n * tau * xbar) / (c + n * tau),
                    1.0 / np.sqrt(c + n * tau))
    mu_draws[t], tau_draws[t] = mu, tau

mu_post, tau_post = mu_draws[burn:], tau_draws[burn:]
sigma_post = 1.0 / np.sqrt(tau_post)
print(f"kept {mu_post.size} draws after discarding {burn} for burn-in")
print(f"prior precision c = {c:,.0f}, data precision n/s^2 = {n/s2:,.0f}")
```

Watching the chain move is the whole point of this lecture, so before summarising anything
the next block traces both parameters against the sweep number and shades the burn-in
segment that is about to be discarded.

**Reading the computation.** Each line is one parameter's value at every sweep, in the order the sampler produced it. A trace that leaves the shaded region and settles into a stable band is the visual definition of the chain having converged.

```{python}
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
axes[0].plot(mu_draws, lw=0.5, color="steelblue")
axes[0].axvspan(0, burn, color="grey", alpha=0.25, label="burn-in")
axes[0].axhline(xbar, color="black", ls=":", label="sample mean")
axes[0].set_ylabel(r"$\mu$"); axes[0].legend(fontsize=8)
axes[1].plot(tau_draws, lw=0.5, color="darkorange")
axes[1].axvspan(0, burn, color="grey", alpha=0.25)
axes[1].axhline(1 / s2, color="black", ls=":", label="1 / sample variance")
axes[1].set_xlabel("sweep"); axes[1].set_ylabel(r"$\tau$"); axes[1].legend(fontsize=8)
fig.suptitle("Gibbs sampler trace plots, with burn-in shaded")
plt.tight_layout(); plt.show()
```

The mean trace starts at the deliberately poor value of five percent a month and
collapses towards the sample mean within a handful of sweeps, well inside the shaded
burn-in window, while the precision trace settles just as quickly into a stable band
around the reciprocal of the sample variance. Once a chain looks like this, the remaining
draws can be treated as coming from the posterior itself rather than from wherever we
happened to start it.

Sampling $\tau$ before $\mu$ in each sweep, rather than the other order used in the
derivation, changes nothing about the argument above: the two full conditionals are
still exactly what we derived, and either visiting order leaves the joint posterior
stationary. When the priors are weak relative to the sample, as ours are here because the
prior precision $c = 2{,}500$ is dwarfed by the data precision $n\tau$, which at the
sample variance is $227{,}560$, the
posterior of $\mu$ should reduce nearly to the textbook result for an unknown-variance
normal sample: a Student-$t$ distribution centred at $\bar{x}$ with $n - 1$ degrees of
freedom and scale $s / \sqrt{n}$. That classical result is exactly the closed-form
answer this problem has when priors are flat enough, so it is the natural benchmark to
check the sampler against.

```{python}
t_ref = stats.t(df=n - 1, loc=xbar, scale=np.sqrt(s2 / n))
grid = np.linspace(mu_post.min(), mu_post.max(), 400)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].hist(mu_post, bins=40, density=True, alpha=0.7, color="steelblue", label="Gibbs draws")
axes[0].plot(grid, t_ref.pdf(grid), "r-", lw=2, label="reference $t_{n-1}$")
axes[0].axvline(0.0, color="black", ls=":", label="zero premium")
axes[0].set_xlabel(r"monthly premium $\mu$"); axes[0].legend(fontsize=9)
axes[1].hist(sigma_post, bins=40, density=True, alpha=0.7, color="darkorange")
axes[1].axvline(np.sqrt(s2), color="black", ls=":", label="sample sd")
axes[1].set_xlabel(r"monthly volatility $\sigma$"); axes[1].legend(fontsize=9)
fig.suptitle("Posterior of the monthly equity premium and its volatility")
plt.tight_layout(); plt.show()

print(f"posterior mean(mu)    {mu_post.mean():.6f}   reference t mean {t_ref.mean():.6f}")
print(f"posterior mean(sigma) {sigma_post.mean():.6f}   sample sd {np.sqrt(s2):.6f}")
```

The histogram of sampled $\mu$ values overlays the reference Student-$t$ curve almost
exactly, which is the check the whole derivation was building toward: a method that
never touched the joint normalising constant has reproduced a distribution we can also
get by a completely different, purely classical route. The posterior mean of the monthly
premium is $0.007375$ against the reference value of $0.007447$, the tiny gap being the
prior at zero pulling the estimate down by about three hundredths of a standard error,
and the posterior mean of $\sigma$ is $0.043983$ against a sample standard deviation of
$0.043922$. This is precisely the ergodic theorem in action: a long, dependent sequence
of draws has produced sample averages that match the target as reliably as an
independent Monte Carlo sample would.

The point of the exercise, though, is not the sampler but the premium, and the draws
answer the question in the units an investment committee uses. Annualising each draw of
$\mu$ by multiplying it by twelve turns the posterior over a monthly mean into a
posterior over the annual equity premium, and every summary follows from the draws
without a further line of algebra.

**Reading the computation.** Each draw of $\mu$ is scaled to an annual figure and each
draw of $\sigma$ to an annual volatility; the ratio of the two posterior means is the
implied Sharpe ratio. The final line counts the draws that fall below zero, which is the
posterior probability that equities are not rewarded at all.

```{python}
premium_ann = 12.0 * mu_post
vol_ann = np.sqrt(12.0) * sigma_post

print(f"annualised equity premium: posterior mean {premium_ann.mean():.4%}")
print(f"  90% credible interval [{np.quantile(premium_ann, 0.05):.4%}, "
      f"{np.quantile(premium_ann, 0.95):.4%}]")
print(f"  95% credible interval [{np.quantile(premium_ann, 0.025):.4%}, "
      f"{np.quantile(premium_ann, 0.975):.4%}]")
print(f"annualised volatility: posterior mean {vol_ann.mean():.4%}")
print(f"implied Sharpe ratio {premium_ann.mean() / vol_ann.mean():.4f}")
print(f"P(premium <= 0 | data) = {(premium_ann <= 0).mean():.4f}")
```

The posterior mean of the annualised equity premium is $8.8501\%$, with a ninety percent
credible interval running from $4.6909\%$ to $12.9751\%$ and a ninety-five percent
interval from $3.9268\%$ to $13.7402\%$. That interval is the number worth arguing about.
After thirty-six years of monthly data, one of the longest samples any practitioner would
regard as coming from a single regime, we still cannot say whether the reward for holding
equities is four percent a year or fourteen, and the difference between those two figures
is the difference between two entirely different pension funding plans. The posterior
probability that the premium is not positive at all is $0.0001$, so the sign is not in
doubt; the magnitude very much is. Annualised volatility, by contrast, is pinned down at
$15.2360\%$, and the implied Sharpe ratio of $0.5809$ inherits all of its uncertainty from
the numerator. This asymmetry, second moments estimated well and first moments badly, is
the single most important empirical fact about financial data, and it is the reason the
rest of this course spends far more effort on variances than on means.

The two histograms above look at each parameter alone; the block below scatters the kept
draws of $\mu$ and $\tau$ against each other to show how they move together.

**Reading the computation.** Each point is one kept sweep's joint value of mu and tau, thinned for plotting speed. A cloud with a visible tilt shows the two parameters are not sampled independently of one another.

```{python}
fig, ax = plt.subplots()
ax.scatter(mu_post[::5], tau_post[::5], s=5, alpha=0.15, color="steelblue")
ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"$\tau$")
ax.set_title("Joint Gibbs draws of mu and tau")
plt.tight_layout(); plt.show()

corr_mt = np.corrcoef(mu_post, tau_post)[0, 1]
print(f"correlation between mu and tau draws: {corr_mt:.3f}")
```

The cloud is close to circular and the printed correlation of $0.016$ is negligible,
which makes sense once we recall why the joint posterior was hard in the first place: the
coupling term $\tau(\bar{x} - \mu)^2$ ties the two parameters together in the kernel, but
with $439$ months the data pin $\mu$ down so tightly around $\bar{x}$ that this coupling
has little room left to act, and the two marginals end up looking nearly independent in
the draws even though the model does not assume they are. Economically the same point
reads as follows: how volatile the market has been says almost nothing about how large
its average reward was, which is why estimating the premium accurately would require a
longer history rather than a higher-frequency one.

The ergodic theorem promises convergence of a running average, which is worth seeing
directly rather than only invoking by name.

**Reading the computation.** The running mean at sweep t averages every kept draw of mu up to that sweep. A running mean that stabilises rather than continuing to wander is the ergodic theorem holding in this particular run.

```{python}
running_mean = np.cumsum(mu_post) / np.arange(1, mu_post.size + 1)
fig, ax = plt.subplots()
ax.plot(running_mean, lw=1, color="steelblue")
ax.axhline(t_ref.mean(), color="black", ls=":", label="reference mean")
ax.set_xlabel("sweep after burn-in"); ax.set_ylabel(r"running mean of $\mu$")
ax.legend(fontsize=9)
ax.set_title("The ergodic theorem: running mean of mu converges")
plt.tight_layout(); plt.show()
```

After an initial swing the running mean settles onto the reference value and stays there,
which is exactly the almost-sure convergence the ergodic theorem states: a long enough
average along one dependent chain lands on the same number a large independent sample
would have given.
