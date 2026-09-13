> Cut from lecture note 3 in the restructuring pass. Belongs in the **Week 3 exercise
> session**. This is the one-parameter Student-t scale model with a lognormal prior and
> fixed nu = 5, superseded in the lecture by the three-parameter Student-t target
> (mu, sigma, nu) on 1990-01 to 2026-07 monthly data. It remains a good exercise because
> the single parameter makes one-dimensional quadrature an exact check. The
> small-sample-bias experiment for the self-normalised estimator and the restricted
> (stress-condition) rejection step go with it.

## The model and its quadrature ground truth

## Monthly returns with fat tails: when we cannot sample directly

Everything so far assumed we could draw straight from the posterior, which only worked
because the inverse-Gamma prior is conjugate to a Gaussian likelihood with a known mean.
That Gaussian assumption is the weak link in the model, and a risk manager knows it:
monthly equity returns have fatter tails than a normal distribution allows, and the
months that matter for a risk number are precisely the ones a normal model treats as
impossible. So replace the normal likelihood with a **Student $t$** one. Write $r_t$ for
the monthly market excess return and model

$$
r_t \mid \sigma \;\sim\; \sigma \cdot t_\nu, \qquad \nu = 5 \text{ fixed},
$$

so that $\sigma$ is a scale parameter and the return has standard deviation
$\sigma\sqrt{\nu/(\nu-2)}$. Degrees of freedom are held at five, a conventional choice for
monthly equity returns, so that the chapter's target stays one-dimensional and can be
checked against quadrature. For the prior we take $\sigma \sim \text{LogNormal}$ with
median $0.04$ and log standard deviation $0.5$, a weakly informative statement that
monthly scale is around four percent and could plausibly be half or double that.

Nothing here is conjugate. Multiplying $n$ Student $t$ densities by a lognormal prior
produces no recognisable kernel, and the posterior

$$
p(\sigma \mid r) \;\propto\; \sigma^{-n}\prod_{t=1}^{n}
\left(1 + \frac{r_t^2}{\nu\sigma^2}\right)^{-(\nu+1)/2}
\times \frac{1}{\sigma}\exp\!\left[-\frac{(\log\sigma - \log 0.04)^2}{2(0.5)^2}\right]
$$

has a shape with no name. We cannot sample it with any canned generator, and the
normalising constant is a one-dimensional integral we happen to be able to compute by
quadrature here only because $\sigma$ is scalar. We use that quadrature once, as a ground
truth to check the simulation methods against, exactly as the closed-form inverse-Gamma
posterior served as ground truth above.

**Reading the computation.** The block loads the monthly factor file, keeps the modern
window from January 2000, codes the log kernel by hand for speed, subtracts its maximum
so that the exponential does not underflow, and integrates by quadrature.

```{python}
ff_m = pd.read_parquet(DATA / "ff_factors_monthly.parquet")
r_mon = ff_m["2000-01":]["mkt_rf"].to_numpy()
n_mon, nu_t = r_mon.size, 5.0

t_const = (gammaln((nu_t + 1) / 2) - gammaln(nu_t / 2)
           - 0.5 * np.log(nu_t * np.pi))

def log_kernel(sigma):
    """Log of the unnormalised posterior of the Student-t scale, vectorised."""
    s = np.atleast_1d(np.asarray(sigma, dtype=float))
    out = np.full(s.shape, -np.inf)
    ok = s > 0
    sk = s[ok]
    q = 1.0 + (r_mon[None, :] / sk[:, None]) ** 2 / nu_t
    loglik = (n_mon * t_const - n_mon * np.log(sk)
              - 0.5 * (nu_t + 1.0) * np.log(q).sum(axis=1))
    logprior = (-np.log(sk) - np.log(0.5) - 0.5 * np.log(2 * np.pi)
                - (np.log(sk) - np.log(0.04)) ** 2 / (2 * 0.5 ** 2))
    out[ok] = loglik + logprior
    return out.reshape(np.shape(sigma))      # scalar in, scalar out

sig_grid = np.linspace(1e-4, 0.12, 4000)
log_c = log_kernel(sig_grid).max()                  # shift for numerical stability

def unnorm_post(sigma):
    return np.exp(log_kernel(sigma) - log_c)

Z, _ = quad(unnorm_post, 0.0, 0.2)                  # normalising constant
true_mean_t, _ = quad(lambda s: s * unnorm_post(s) / Z, 0.0, 0.2)

print(f"{n_mon} months, {ff_m['2000-01':].index[0]:%Y-%m} to "
      f"{ff_m['2000-01':].index[-1]:%Y-%m}")
print(f"quadrature posterior mean of sigma: {true_mean_t:.5f} per month")
print(f"implied return sd {true_mean_t*np.sqrt(nu_t/(nu_t-2)):.4f} monthly, "
      f"{true_mean_t*np.sqrt(nu_t/(nu_t-2))*np.sqrt(12):.4f} annualised")
```

The sample is $319$ months, from January 2000 to July 2026. The posterior mean scale is
$0.0382$, which under the $t_5$ model implies a monthly return standard deviation of
$4.94\%$ and an annualised figure of $17.10\%$. That is visibly above the $13.26\%$
estimated from the last year of daily data, and the gap is not a contradiction: the
monthly window covers the dot-com collapse, 2008 and 2020, while the daily window covers
one comparatively calm year. Which number a desk should use depends entirely on whether
it is asking about the current regime or about the long run.

```{python}
dens_t = unnorm_post(sig_grid) / Z

fig, ax = plt.subplots()
ax.plot(sig_grid, dens_t, label=r"posterior of $\sigma$ (Student-$t_5$ likelihood)")
ax.axvline(true_mean_t, color="grey", ls=":", label=f"mean = {true_mean_t:.4f}")
ax.set_xlim(0.03, 0.05)
ax.set_xlabel(r"monthly scale $\sigma$"); ax.set_ylabel("density")
ax.legend(); ax.set_title("A non-conjugate posterior: no sampler exists for this shape")
plt.tight_layout(); plt.show()
```

The curve is a legitimate density, sharply unimodal and concentrated between about
$0.034$ and $0.043$, but it belongs to no textbook family. Quadrature answered the
one-dimensional question, and this is exactly where Lecture 1 warned we would run out of
road: had we also treated the degrees of freedom, a drift and a leverage parameter as
unknown, this same integral would need a fine grid over each dimension, and the cost
explodes long before an economically interesting model is reached. What we need is a way
to compute posterior expectations under a density we can *evaluate* but not *sample*,
without ever computing $Z$. That is importance sampling.

## Importance sampling on this target

We put this to work on the Student-$t$ scale posterior, proposing from a $t_4$ distribution
centred near the posterior's bulk with a scale three times wider than the posterior's own,
so that the proposal is both generously spread and heavier-tailed than the target.

```{python}
S_is = 20_000
prop = stats.t(df=4, loc=0.0382, scale=0.0056)
samp = prop.rvs(size=S_is, random_state=rng)

w = unnorm_post(samp) / prop.pdf(samp)             # unnormalised weights
w_norm = w / w.sum()                               # self-normalised weights

is_mean = np.sum(samp * w_norm)
ess_good = 1.0 / np.sum(w_norm ** 2)
print(f"posterior mean of sigma   quadrature {true_mean_t:.5f}   IS {is_mean:.5f}")
print(f"effective sample size: {ess_good:.0f} out of {S_is}")
```

The importance-sampling estimate lands within a few hundred-thousandths of the quadrature
answer, obtained entirely from Student-$t$ draws reweighted by the target's shape, with no
canned sampler for a lognormal-times-$t$-likelihood posterior anywhere in sight. The printed
effective sample size is explained next.

## Self-normalised bias at small sample sizes

### Self-normalised bias at small sample sizes, numerically

The claim that $\hat{g}_S^{\,\text{IS}}$ is consistent but not unbiased is easy to state and
easy to check directly: draw many independent importance samples at a deliberately small $S$,
average the resulting estimates, and compare that average with the quadrature truth; then
repeat at a larger $S$ and watch the gap shrink. Two practical points shape the experiment.
The repetition counts are modest so that the chapter renders quickly, which leaves some
noise in the reported averages. And the proposal used here is deliberately *not* the good
one: with a proposal as well matched as the $t_4$ above, the bias at $S = 20$ is already
smaller than the noise in fifteen hundred repetitions, so we shift the proposal's centre
down to $0.033$ and narrow its scale, keeping it valid but mismatched, which is the
situation in which the finite-sample bias is worth worrying about in the first place.

```{python}
bias_prop = stats.t(df=4, loc=0.033, scale=0.004)     # valid but deliberately mismatched

def is_mean_estimate(S):
    th = bias_prop.rvs(size=S, random_state=rng)
    wt = unnorm_post(th) / bias_prop.pdf(th)
    return np.sum(th * wt) / np.sum(wt)

S_small, S_large, reps_bias = 20, 500, 1500
est_small = np.array([is_mean_estimate(S_small) for _ in range(reps_bias)])
est_large = np.array([is_mean_estimate(S_large) for _ in range(reps_bias)])

fig, ax = plt.subplots()
ax.hist(est_small, bins=40, density=True, alpha=0.6, label=f"S = {S_small}")
ax.hist(est_large, bins=40, density=True, alpha=0.6, label=f"S = {S_large}")
ax.axvline(true_mean_t, color="black", lw=1.5, label="quadrature truth")
ax.set_xlabel(r"self-normalised estimate of $E[\sigma\mid r]$"); ax.set_ylabel("density")
ax.legend(fontsize=9)
ax.set_title(f"Bias shrinks with S over {reps_bias} repetitions")
plt.tight_layout(); plt.show()

print(f"true posterior mean (quadrature): {true_mean_t:.5f}")
print(f"mean of {reps_bias} estimates at S={S_small}: {est_small.mean():.5f}  "
      f"(bias {est_small.mean()-true_mean_t:+.5f})")
print(f"mean of {reps_bias} estimates at S={S_large}: {est_large.mean():.5f}  "
      f"(bias {est_large.mean()-true_mean_t:+.5f})")
```

At `S = 20` the histogram of estimates is wide and its centre is displaced from the
quadrature truth by $-0.00017$, a small-sample bias exactly of the kind the
ratio-of-averages argument predicts, while at `S = 500` the distribution has narrowed
sharply and the average bias has fallen below $10^{-5}$, so that it rounds to zero at the
precision printed. Averaging many replications at each $S$ isolates the bias from the sampling
noise that a single run would also show, and printing the two average biases side by side is
the cleanest numerical demonstration that consistency, not unbiasedness, is what
self-normalisation buys.

## A deliberately bad proposal

```{python}
bad_prop = stats.norm(0.034, 0.0010)                # mis-centred and too narrow
samp_b = bad_prop.rvs(size=S_is, random_state=rng)
w_bad = unnorm_post(samp_b) / bad_prop.pdf(samp_b)
w_bad_norm = w_bad / w_bad.sum()

is_mean_bad = np.sum(samp_b * w_bad_norm)
ess_bad = 1.0 / np.sum(w_bad_norm ** 2)
print(f"bad-proposal mean estimate: {is_mean_bad:.5f}  (true {true_mean_t:.5f})")
print(f"effective sample size: {ess_bad:.0f} out of {S_is}")
print(f"largest single normalised weight: {w_bad_norm.max():.3f}")
```

The posterior for this scale parameter has almost all its mass between roughly $0.034$
and $0.043$, while the bad proposal concentrates around $0.034$ with a standard deviation
of only $0.001$: the proposal rarely visits the region the target cares about, the rare
draws that do land there receive very large weights to compensate, and the effective
sample size collapses from $8{,}584$ to $3$ out of twenty thousand, with a single draw
carrying $55\%$ of the total weight. The point estimate, $0.03781$ against the true
$0.03824$, is visibly biased downwards at this finite $S$ despite the theoretical guarantee of consistency, because consistency is a
statement about $S \to \infty$ and a collapsed ESS means we are effectively very far from
that limit. In risk terms the bad proposal understates monthly volatility, which is the
direction that gets a desk into trouble. Plotting the two weight distributions side by
side shows the contrast directly.

```{python}
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
for ax, w_, ess_, ttl in [(axes[0], w_norm, ess_good, "good proposal ($t_4$, wide)"),
                          (axes[1], w_bad_norm, ess_bad, "bad proposal (narrow Normal)")]:
    ax.hist(w_, bins=40, color="steelblue", alpha=0.85)
    ax.set_title(f"{ttl}\nESS = {ess_:.0f}")
    ax.set_xlabel("normalised weight")
axes[0].set_ylabel("count")
fig.suptitle("Importance weights: even vs degenerate")
plt.tight_layout(); plt.show()
```

The good proposal produces many small, comparable weights clustered together; the bad
one produces a spike of near-zero weights and a long thin tail of a few draws that
dominate the sum. The general lesson, and the reason importance sampling is taught
alongside a warning, is that a safe proposal needs **heavier tails than the target** and
must cover every region where the target has appreciable mass. A proposal that is too
narrow, or that decays faster than the target in some direction, produces a handful of
runaway weights and an estimator whose variance can be infinite even though its bias
vanishes asymptotically; the estimate looks perfectly reasonable until, on a bad draw of
the random seed, it suddenly is not. Finding such a proposal by hand becomes close to
impossible once $\theta$ has more than a few components, which is exactly the
difficulty that motivates the Markov chain samplers of the next two lectures.

## Rejection sampling on this target, and a parameter restriction

We apply it to the Student-$t$ scale posterior with the same wide $t_4$ proposal used for
importance sampling, generating candidates in batches so the loop terminates quickly.

```{python}
M = (unnorm_post(sig_grid) / prop.pdf(sig_grid)).max()   # smallest valid envelope constant

N_target, accepted = 20_000, []
n_tried = 0
while len(accepted) < N_target:
    cand = prop.rvs(size=10_000, random_state=rng)
    u = rng.uniform(size=cand.size)
    keep = u < unnorm_post(cand) / (M * prop.pdf(cand))
    accepted.extend(cand[keep])
    n_tried += cand.size

rej_draws = np.array(accepted[:N_target])
print(f"envelope constant M = {M:.4e}")
print(f"acceptance rate: {len(accepted)/n_tried:.3f}  (theory: Z/M = {Z/M:.3f})")
print(f"rejection-sample mean: {rej_draws.mean():.5f}  (quadrature {true_mean_t:.5f})")
```

The observed acceptance rate matches the theoretical $Z/M$ closely, and the mean of the
accepted draws reproduces the quadrature answer without any weight ever appearing in the
calculation: these are genuine, exchangeable draws from the posterior, usable in any
downstream calculation exactly like the direct inverse-Gamma draws from the first section.
The price is the acceptance rate itself. In one dimension an envelope constant a little
above one is usually attainable with a sensible proposal, but in $d$ dimensions the
volume between a safe envelope and the target typically grows so that $M$ grows
exponentially with $d$, and the acceptance rate $1/M$ collapses correspondingly:
rejection sampling suffers from the same curse of dimensionality as the grid, just with
a slower fuse.

### A special, simple case: rejection under a parameter restriction

One situation removes the need to search for an envelope constant at all. Suppose the target
is an unrestricted posterior with kernel $\kappa(\theta)$ truncated to a restricted region
$\Theta_r \subset \Theta$, so that the restricted kernel is
$\kappa(\theta)\mathbf{1}\{\theta \in \Theta_r\}$, and the unrestricted posterior is already
easy to simulate, whether directly or by the machinery above. Then the unrestricted posterior
itself can serve as the proposal, with envelope constant exactly $M = 1$, because
$\kappa(\theta)\mathbf{1}\{\theta \in \Theta_r\} \le 1 \cdot \kappa(\theta)$ holds trivially,
and the algorithm collapses to drawing from the unrestricted posterior and discarding any draw
that falls outside $\Theta_r$. The acceptance rate is then simply
$\Pr(\theta \in \Theta_r \mid y)$, the very quantity a marginal-probability calculation would
want in any case, and it can be small whenever the restriction removes most of the unrestricted
posterior's mass, which is the same curse-of-dimensionality warning as before in a different
guise: restricting several components of $\theta$ at once shrinks the accepted region
multiplicatively, exactly as an inequality-constrained prior such as a sign restriction on a
single regression coefficient can leave very little posterior mass behind.

The restriction with an economic reading here is a **stress condition**: a committee may want
the posterior conditional on the market's monthly scale exceeding $0.04$, which under the
$t_5$ model is a monthly return standard deviation above $5.16\%$, or about $17.9\%$ a year.
We simply filter the unrestricted rejection draws rather than building any new envelope.

```{python}
theta_r = 0.04
restricted = rej_draws[rej_draws > theta_r]
accept_rate_r = restricted.size / rej_draws.size
true_prob_r, _ = quad(lambda s: unnorm_post(s) / Z, theta_r, 0.2)

print(f"P(sigma > {theta_r} | r): quadrature {true_prob_r:.4f}   "
      f"filtered rejection-sample acceptance rate {accept_rate_r:.4f}")
print(f"restricted posterior mean: {restricted.mean():.5f}  "
      f"(unrestricted posterior mean {rej_draws.mean():.5f})")
```

The filtered acceptance rate matches the quadrature probability closely, confirming that simply
discarding draws outside $\Theta_r$ reproduces the restricted posterior exactly, with no
weighting and no envelope search beyond the one already done for the unrestricted problem; this
is the trick used for sign or inequality restrictions on regression coefficients, and it stays
cheap here only because the restriction acts on an already-tractable one-dimensional draw
rather than on a many-dimensional joint region. The economic reading of the printed
probability is direct: the data assign about a seventeen percent posterior probability, $0.1722$ by quadrature, to the
stress scenario, and conditional on it the posterior mean scale is $0.0411$ rather than
$0.0382$.

## Sampling-importance-resampling on this target

```{python}
idx = rng.choice(S_is, size=S_is, replace=True, p=w_norm)
sir_draws = samp[idx]

fig, ax = plt.subplots()
ax.hist(sir_draws, bins=40, density=True, alpha=0.6, label="SIR resample")
ax.plot(sig_grid, dens_t, lw=2, label="true posterior (quadrature)")
ax.set_xlim(0.03, 0.05)
ax.set_xlabel(r"monthly scale $\sigma$"); ax.set_ylabel("density")
ax.legend(); ax.set_title("SIR turns weighted draws into an (approximate) posterior sample")
plt.tight_layout(); plt.show()
print(f"SIR resample mean: {sir_draws.mean():.5f}  (quadrature {true_mean_t:.5f})")
```

The resampled histogram tracks the true density well because the original importance
sample here had a healthy effective sample size; with a degenerate weight distribution
like the bad proposal above, the same recipe would simply copy the same handful of
dominant points over and over, which is the resampling analogue of a collapsed ESS.
SIR is the conceptual bridge to Lectures 4 and 5: instead of reweighting a fixed
proposal once, Markov chain Monte Carlo methods let the sampler *adapt* its own
proposal as it explores the space, which is what makes them viable in high dimensions
where a single fixed $q$, however carefully chosen, cannot cover the target.
