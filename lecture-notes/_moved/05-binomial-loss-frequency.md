> Cut from lecture note 5 in the restructuring pass. Belongs in the **Week 5 exercise
> session**. The scalar non-conjugate binomial target (losing months in the last 24,
> truncated Normal prior) with four samplers on it: the uniform independence sampler, the
> logit random walk with its Jacobian, the one-dimensional slice sampler with bracket
> shrinking, and a mixed-kernel chain, plus the four-way density comparison against a
> grid. The derivations it contained survive in the rewritten chapter in general form:
> the change-of-variables Jacobian is now a visible result for an arbitrary smooth
> reparameterisation, the kernel-mixing argument is stated in the main line, and slice
> sampling appears as a result block alongside the NUTS slice selection. What moves here
> is the second inference problem and its code. Needs a data-loading preamble
> (ff_factors_monthly.parquet).

## A non-conjugate binomial model: the recent loss frequency

Before the multivariate problem, we need a scalar target small enough to check against a grid. The
natural one here is the unconditional version of the question the chapter opened with. Over the most
recent **twenty-four months** of the Kenneth French monthly file, how often was the market's excess
return negative, and what does that say about the probability $\theta$ that next month is a losing one?
Two years is the window a risk committee would actually look at, short enough to describe the current
regime and short enough that the answer is genuinely uncertain.

Lecture 1 answered a question of this shape with a conjugate $\text{Beta}$ prior. Replace that prior with
a **truncated normal** prior on $[0, 1]$, $\theta \sim \mathcal{N}(0.5, 1)$ truncated to the unit
interval, a weak statement that a coin-flip is the natural reference point, and conjugacy is gone: the
posterior kernel is
$$
f(\theta \mid y) \;\propto\; \theta^{y}(1-\theta)^{n-y} \times
\exp\!\left(-\frac{(\theta - 0.5)^2}{2}\right), \qquad 0 < \theta < 1,
$$
and no standard family has this shape, so there is no arithmetic update left to do. This is the smallest
possible setting in which to see the taxonomy of the previous section at work, because the target is a
single scalar and we can check every sampler against a numerically computed ground truth. The truncation's
own normalising constant is a fixed number that does not depend on $\theta$, so it can be folded into the
proportionality and ignored throughout; only the shape in $\theta$ matters for a Metropolis-Hastings ratio.

The next chunk counts the losing months, codes the log-posterior kernel, evaluates it on a fine grid
exactly as in Lecture 1's grid method, and reports the posterior mean this produces, which serves as the
reference every sampler below is checked against.

```{python}
ff_m = pd.read_parquet(DATA / "ff_factors_monthly.parquet")
recent = ff_m["mkt_rf"].tail(24)

n_bin = int(recent.size)
y_bin = int((recent < 0).sum())
mu_p, sigma_p = 0.5, 1.0

def log_post_binom(theta):
    theta = np.asarray(theta, dtype=float)
    out = np.full(theta.shape, -np.inf)
    ok = (theta > 0) & (theta < 1)
    out[ok] = (y_bin * np.log(theta[ok]) + (n_bin - y_bin) * np.log(1 - theta[ok])
               + stats.norm.logpdf(theta[ok], mu_p, sigma_p))
    return out

theta_grid = np.linspace(1e-4, 1 - 1e-4, 4001)
w_grid = np.exp(log_post_binom(theta_grid) - log_post_binom(theta_grid).max())
dens_grid = w_grid / np.trapezoid(w_grid, theta_grid)
true_mean = np.trapezoid(theta_grid * dens_grid, theta_grid)
print(f"{recent.index[0]:%Y-%m} to {recent.index[-1]:%Y-%m}: "
      f"{y_bin} losing months out of {n_bin}, frequency {y_bin/n_bin:.4f}")
print(f"grid posterior mean under the truncated normal prior: {true_mean:.4f}")
```

Eleven of the last twenty-four months were losing ones, a frequency of $0.4583$, and the grid posterior
mean is $0.4619$, pulled a little towards the prior's half because twenty-four observations are not many.
With $\sigma = 1$ the truncated normal is nearly flat across the whole unit interval, a weaker prior than
Lecture 1's $\text{Beta}(5,5)$, whose effective sample size of ten pseudo-observations would have pulled
harder. Having no closed form does not make the posterior any less real, only harder to summarise, which
is exactly the gap Metropolis-Hastings fills. Note also how uninformative two years are: we shall see
below that the posterior standard deviation is close to ten percentage points, so this sample cannot
distinguish a market that loses forty percent of the time from one that loses fifty-five percent of
the time.

### The independence sampler on the non-conjugate posterior

We use $q(\theta') = \mathcal{U}(0, 1)$, a fixed proposal that does not depend on where the chain currently
is. Because $q$ is uniform it evaluates to the same constant at every point, so it cancels from the ratio
of importance weights derived above and the acceptance rule needs only the posterior kernel,
$\alpha = \min\{1, \pi(\theta')/\pi(\theta)\}$. The proposal has heavier tails than the truncated normal
posterior only in the loose sense that it reaches everywhere in $(0,1)$ with equal density, which is
enough here because the target itself lives entirely on that interval and, with only two years of data,
is not sharply concentrated within it.

```{python}
def independence_sampler(log_post, theta0, n_iter, rng):
    theta = theta0; lp = log_post(np.array([theta]))[0]
    chain = np.empty(n_iter); n_accept = 0
    for t in range(n_iter):
        prop = rng.uniform()
        lp_prop = log_post(np.array([prop]))[0]
        if np.log(rng.uniform()) < lp_prop - lp:
            theta, lp = prop, lp_prop
            n_accept += 1
        chain[t] = theta
    return chain, n_accept / n_iter

burn_b, iters_b = 1000, 10000
ind_chain, ind_acc = independence_sampler(log_post_binom, 0.5, burn_b + iters_b, rng)
ind_post = ind_chain[burn_b:]
print(f"independence sampler: acceptance {ind_acc:.3f}, "
      f"posterior mean {ind_post.mean():.4f} (grid: {true_mean:.4f})")
```

The acceptance rate is moderate and the sampled mean lands close to the grid value, so a proposal that
ignores the current state entirely is doing fine here, precisely because the uniform proposal is a
reasonable match for a posterior that is not sharply peaked. Had we used the whole post-2000 sample of
three hundred and nineteen months instead of the last twenty-four, the posterior would be roughly four
times narrower and the same uniform proposal would waste most of its draws. In a higher-dimensional or
more concentrated problem a fixed proposal would rarely land near the bulk of the posterior and the
acceptance rate would collapse, which is why the independence sampler is used mostly for
low-dimensional, well-behaved targets.

### A random walk on the logit scale, with its Jacobian

A random walk directly on $\theta$ risks proposing values outside $(0,1)$. The standard fix is to walk on
the unconstrained **logit** scale $\eta = \log[\theta/(1-\theta)]$ and map back,
$\eta' \sim \mathcal{N}(\eta, s^2)$, $\theta' = 1/(1+e^{-\eta'})$. The proposal is symmetric in $\eta$, but
we still evaluate the posterior in $\theta$, and a change of variables is not free: the acceptance ratio
picks up an extra factor of $\theta'(1-\theta')/[\theta(1-\theta)]$, the Jacobian of the logit transform,
$$
\alpha = \min\left\{1,\; \frac{\pi_\theta(\theta')\,\theta'(1-\theta')}{\pi_\theta(\theta)\,\theta(1-\theta)}\right\}.
$$
Forgetting it is a common bug: the chain would still run, but it would target the wrong distribution.

::: {.callout-note collapse="true"}
## Derivation: where the logit Jacobian comes from

Since $\theta = 1/(1+e^{-\eta})$ has $d\theta/d\eta = \theta(1-\theta)$, the change-of-variable rule
gives the induced density of $\eta$ as
$$
\pi_\eta(\eta) = \pi_\theta(\theta(\eta))\,\left|\frac{d\theta}{d\eta}\right|
= \pi_\theta(\theta(\eta))\,\theta(1-\theta).
$$
A symmetric random walk in $\eta$-space has an acceptance ratio built from $\pi_\eta$ alone, because the
Gaussian proposal densities cancel as shown above,
$$
\alpha = \min\left\{1, \frac{\pi_\eta(\eta')}{\pi_\eta(\eta)}\right\}
= \min\left\{1, \frac{\pi_\theta(\theta')\,\theta'(1-\theta')}{\pi_\theta(\theta)\,\theta(1-\theta)}\right\},
$$
which is the displayed rule. The same argument applies to any smooth reparameterisation used to remove a
constraint: a log transform for a positive scale parameter contributes a factor $\sigma'/\sigma$, and a
Fisher transform for a correlation contributes $(1-\rho'^2)/(1-\rho^2)$. The Jacobian is not an
optional correction, it is part of the density being targeted.
:::

```{python}
def rw_logit_sampler(log_post, theta0, step, n_iter, rng):
    theta = theta0; lp = log_post(np.array([theta]))[0]
    chain = np.empty(n_iter); n_accept = 0
    for t in range(n_iter):
        eta = np.log(theta / (1 - theta)) + step * rng.normal()
        prop = 1.0 / (1.0 + np.exp(-eta))
        lp_prop = log_post(np.array([prop]))[0]
        log_ratio = (lp_prop - lp + np.log(prop) + np.log(1 - prop)
                     - np.log(theta) - np.log(1 - theta))
        if np.log(rng.uniform()) < log_ratio:
            theta, lp = prop, lp_prop
            n_accept += 1
        chain[t] = theta
    return chain, n_accept / n_iter

rw_chain, rw_acc = rw_logit_sampler(log_post_binom, 0.5, 1.5, burn_b + iters_b, rng)
rw_post = rw_chain[burn_b:]
print(f"logit random walk: acceptance {rw_acc:.3f}, "
      f"posterior mean {rw_post.mean():.4f} (grid: {true_mean:.4f})")
```

Both samplers recover the same mean to two decimals, which is the check every sampler must pass before
its output is trusted, echoing the check Lecture 1 ran between the analytic and grid posteriors.

### Slice sampling: an auxiliary variable and no proposal at all

Both samplers above needed a proposal density to be chosen and tuned. **Slice sampling** removes that
choice entirely by introducing an auxiliary variable $u$ and sampling uniformly from the region under the
(unnormalised) target density,
$$
\mathcal{A} = \{(\theta, u): 0 \leq u \leq \pi(\theta)\}.
$$
If $(\theta, u)$ is uniform on $\mathcal{A}$, the marginal of $\theta$ is exactly proportional to
$\pi(\theta)$, since integrating the uniform density over $u$ from $0$ to $\pi(\theta)$ returns
$\pi(\theta)$ itself. The chain alternates two conditional draws, each of which is uniform and therefore
needs no acceptance step:

1. given $\theta_t$, draw $u_{t+1} \sim \mathcal{U}(0, \pi(\theta_t))$, which is uniform on the vertical
   slice of $\mathcal{A}$ above $\theta_t$;
2. given $u_{t+1}$, draw $\theta_{t+1}$ uniformly from the horizontal slice
   $S = \{\theta : \pi(\theta) > u_{t+1}\}$.

Each step is, by construction, an exact draw from a uniform conditional distribution of the joint density
on $\mathcal{A}$, and alternating exact conditional draws is precisely the Gibbs sampling logic of Lecture
4 applied to the two coordinates $(\theta, u)$: each step leaves the joint uniform distribution on
$\mathcal{A}$ invariant, so their composition does too, and therefore the $\theta$-marginal, which is
$\pi$, is invariant as well. No proposal density, no acceptance ratio, and nothing to tune beyond how
step 2 locates the set $S$, which in general is unknown in closed form and is instead found by shrinking a
bracket: propose a point uniformly in the current bracket, and if it fails to lie in $S$, shrink the
bracket at the failing point towards the last valid value and try again, using the invariant that the
previous $\theta_t$ always lies inside $S$ and inside the current bracket.

```{python}
def slice_sampler_1d(log_post, theta0, n_iter, rng, lo=0.0, hi=1.0):
    theta = theta0
    chain = np.empty(n_iter)
    for t in range(n_iter):
        log_u = log_post(np.array([theta]))[0] + np.log(rng.uniform())
        t_min, t_max, old = lo, hi, theta
        while True:
            theta = rng.uniform(t_min, t_max)
            if log_post(np.array([theta]))[0] > log_u:
                break
            if theta < old:
                t_min = theta
            else:
                t_max = theta
        chain[t] = theta
    return chain

slice_chain = slice_sampler_1d(log_post_binom, 0.5, burn_b + iters_b, rng)
slice_post = slice_chain[burn_b:]
print(f"slice sampler: posterior mean {slice_post.mean():.4f} (grid: {true_mean:.4f}), "
      f"no acceptance rate to report")
```

There is no acceptance rate to print because there is nothing to reject: every proposed $\theta$ that
survives the shrinking loop is used. The mean again matches the grid value, and the absence of a tuning
parameter such as the step size $s$ is the practical selling point of slice sampling whenever a
one-dimensional (or one-block) update is cheap to shrink towards.

### Mixing kernels in practice

The mixing argument above is not only theoretical. The next chunk builds one chain that alternates an
independence step and a logit random-walk step at every iteration, and checks that the resulting draws
still recover the same posterior mean as either kernel used alone.

```{python}
def mixed_sampler(log_post, theta0, step, n_iter, rng):
    theta = theta0; lp = log_post(np.array([theta]))[0]
    chain = np.empty(n_iter); n_accept = 0
    for t in range(n_iter):
        if t % 2 == 0:                                  # independence half of the sweep
            prop = rng.uniform()
        else:                                            # logit random-walk half
            eta = np.log(theta / (1 - theta)) + step * rng.normal()
            prop = 1.0 / (1.0 + np.exp(-eta))
        lp_prop = log_post(np.array([prop]))[0]
        log_ratio = lp_prop - lp
        if t % 2 == 1:
            log_ratio += np.log(prop) + np.log(1 - prop) - np.log(theta) - np.log(1 - theta)
        if np.log(rng.uniform()) < log_ratio:
            theta, lp = prop, lp_prop
            n_accept += 1
        chain[t] = theta
    return chain, n_accept / n_iter

mix_chain, mix_acc = mixed_sampler(log_post_binom, 0.5, 1.5, burn_b + iters_b, rng)
mix_post = mix_chain[burn_b:]
print(f"mixed independence/logit-RW chain: overall acceptance {mix_acc:.3f}, "
      f"posterior mean {mix_post.mean():.4f} (grid: {true_mean:.4f})")
```

The mixed chain reproduces the same mean as the two pure chains, exactly as the invariance argument
promised: composing two kernels that each individually preserve the posterior preserves it whichever order
they run in, so a modeller is free to reach for whichever proposal is convenient for each part of a
problem rather than forcing one global proposal onto the entire parameter vector.

### Comparing all four samplers

The final chunk overlays the kernel density estimate of each sampler's output against the grid posterior
and reports posterior means, standard deviations and the effective sample size, using the ESS function
built later in this lecture's diagnostics section applied here first because the comparison needs it.

```{python}
def acf(x, max_lag):
    x = x - x.mean(); n = len(x); var = np.dot(x, x) / n
    return np.array([np.dot(x[:n - k], x[k:]) / n / var for k in range(max_lag + 1)])

def ess(x):
    n = len(x); a = acf(x, n - 1); s = 0.0
    for k in range(1, n - 1, 2):
        if a[k] + a[k + 1] < 0: break
        s += a[k] + a[k + 1]
    return n / (1.0 + 2.0 * s)

methods = {"independence": ind_post, "logit random walk": rw_post,
           "slice": slice_post, "mixed kernels": mix_post}
summary_b = pd.DataFrame({name: {"mean": ch.mean(), "sd": ch.std(ddof=1), "ESS": ess(ch)}
                           for name, ch in methods.items()}).T
print(summary_b.round(4))

fig, ax = plt.subplots()
grid_kde = np.linspace(0.1, 0.85, 300)
for name, ch in methods.items():
    ax.plot(grid_kde, stats.gaussian_kde(ch)(grid_kde), label=name)
ax.plot(theta_grid, dens_grid, "k--", lw=1.5, label="grid posterior")
ax.set_xlabel(r"$\theta$, probability that a month loses"); ax.set_ylabel("density")
ax.legend(fontsize=8)
ax.set_title("Four samplers targeting the same non-conjugate posterior")
plt.tight_layout(); plt.show()
```

All four curves sit on top of the dashed grid posterior and all four means agree with it to two decimals:
the samplers disagree only in how efficiently they explore, visible in the effective sample
size column, not in what they converge to. The slice sampler needs no step size and still
is in fact the most efficient of the four here, its ten thousand kept draws behaving as
though they were independent, which is the argument for reaching for it whenever a
model's conditional update is one-dimensional or otherwise cheap to shrink. The economic reading of the
shared answer is worth stating plainly: two years of data leave the loss probability somewhere around
$0.46$ with a posterior standard deviation of about $0.095$, so nothing in the recent record distinguishes the
market from a fair coin. That is the unconditional benchmark against which the conditional model below
has to prove itself.
