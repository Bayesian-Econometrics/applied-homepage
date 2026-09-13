> Cut from lecture note 5 in the restructuring pass. Belongs in the **Week 5 exercise
> session**. Bayesian logistic regression of a losing month on the term spread and the
> lagged change in unemployment, 795 months of FRED and Kenneth French data: the
> predictor-timing discussion, the three-step-size trace comparison, the ESS/MCSE/Rhat
> diagnostics, the hand-coded HMC with its gradient, and the random-walk-versus-HMC
> efficiency table and path plot. Every general result it carried (detailed balance,
> step-size tuning, ESS, Rhat, leapfrog and its unit Jacobian) is retained in the
> rewritten chapter on the shared Student-t target; what moves here is the second
> inference problem, its economics and its figures. Needs a data-loading preamble
> (fred_macro_monthly.parquet, ff_factors_monthly.parquet) and the `acf`/`ess` helpers.

## A worked example: Bayesian logistic regression on macroeconomic predictors

We now move from a single scalar to a genuinely multivariate target, applying the algorithm to a model
with no conjugate structure at all, and to the question the chapter opened with. The probability of a
binary outcome $y_i \in \{0, 1\}$ depends on covariates $x_i$ through
$\Pr(y_i = 1 \mid \beta) = 1/(1 + e^{-x_i'\beta})$, and with an independent
$\mathcal{N}(0, \tau^2)$ prior on each coefficient the log-posterior is, up to a constant,
$$
\log \pi(\beta) \;=\; \sum_{i=1}^{n}\Big[y_i\,x_i'\beta - \log\big(1 + e^{x_i'\beta}\big)\Big]
\;-\; \frac{1}{2\tau^2}\sum_j \beta_j^2.
$$
Evaluating this is trivial, which is all Metropolis-Hastings requires.

The dependent variable is an indicator that the market's excess return in month $t+1$ was negative. The
two predictors are macroeconomic and both are dated so that they were **public information before that
return was realised**, which is the part of the exercise that requires care rather than cleverness. The
**term spread**, the ten-year Treasury yield minus the federal funds rate, is built from month $t$: both
series are daily market rates averaged within the month, so a value for month $t$ is known by the end of
month $t$. The **change in the unemployment rate** is built from months $t-2$ to $t-1$ rather than from
month $t$, because the unemployment rate for a given month is published by the Bureau of Labor Statistics
in the first week of the *following* month; using the month-$t$ figure to predict a month-$t+1$ return
would be a look-ahead of a few days, and lagging it one further month removes the problem entirely. This
is a genuine out-of-sample-in-spirit alignment: every predictor value used to explain a return was
available to an investor before that return began to accrue. It is not, however, an out-of-sample
forecasting exercise, because the coefficients are estimated on the whole sample; that distinction
matters and Lecture 9 takes it up properly.

The economic hypotheses are worth writing down before the estimates arrive. A **steep yield curve** is
the classic signal of an expansionary stance and an expected recovery, so we expect the term-spread
coefficient to be negative: a wider spread should lower the probability of a losing month. A **rising
unemployment rate** signals a deteriorating economy, so a naive reading expects a positive coefficient,
though the efficient-markets counter-argument is that a published unemployment figure is in the price
within minutes and should predict nothing at all at a one-month horizon.

**Reading the computation.** The block merges the two monthly files on their common month-start index,
builds the two predictors with the lags just described, shifts the market return forward by one month to
form the outcome, and drops any month for which some ingredient is missing.

```{python}
fred = pd.read_parquet(DATA / "fred_macro_monthly.parquet").dropna()

panel = pd.DataFrame({
    "term_spread": fred["yield_10y"] - fred["fed_funds"],      # known within month t
    "d_unemp": fred["unemployment"].diff().shift(1),           # published before month t ends
})
panel["next_ret"] = ff_m["mkt_rf"].shift(-1)                   # return in month t+1
panel = panel.dropna()                                         # drops the unmatched last month

X = np.column_stack([np.ones(len(panel)), panel["term_spread"], panel["d_unemp"]])
y = (panel["next_ret"] < 0).astype(float).to_numpy()
n = len(panel)
tau = 5.0                       # prior standard deviation on each coefficient
coef_names = ["intercept", "term_spread", "d_unemp"]

print(f"{n} months of predictors, {panel.index[0]:%Y-%m} to {panel.index[-1]:%Y-%m}, "
      f"predicting returns to {panel.index[-1] + pd.DateOffset(months=1):%Y-%m}")
print(f"share of losing months: {y.mean():.4f}")
print(panel[["term_spread", "d_unemp"]].describe().loc[["mean", "std", "min", "max"]].round(3))
```

The sample is $795$ months of predictors, from March 1960 to June 2026, explaining returns from April
1960 to July 2026. A share $0.4013$ of those months were losing ones. The term spread averages
$0.977$ percentage points with a standard deviation of $1.600$ and ranges from $-6.51$ to $3.85$, so the
sample contains both the deeply inverted curve of January 1981 and the very steep curve of the 1992
recovery. The monthly change in unemployment averages essentially zero with a standard deviation of
$0.428$ percentage points and a maximum of $10.40$, that single extreme value being the jump of April
2020, which enters as a predictor from May 2020 onwards.

```{python}
def log_posterior(beta):
    eta = X @ beta
    return np.sum(y * eta - np.logaddexp(0.0, eta)) - 0.5 * np.sum(beta**2) / tau**2

def rw_metropolis(log_post, init, step, n_iter, rng):
    d = len(init)
    chain = np.empty((n_iter, d))
    current = np.asarray(init, dtype=float)
    current_lp = log_post(current)
    n_accept = 0
    for t in range(n_iter):
        proposal = current + step * rng.normal(size=d)
        proposal_lp = log_post(proposal)
        if np.log(rng.uniform()) < proposal_lp - current_lp:
            current, current_lp = proposal, proposal_lp
            n_accept += 1
        chain[t] = current
    return chain, n_accept / n_iter
```

This is the same symmetric random-walk kernel derived earlier, now applied coordinate-wise across a
three-dimensional parameter, and the same acceptance rule applies unchanged: the proposal density cancels
because it is Gaussian and centred on the current state in every coordinate simultaneously. Because the
three coefficients live on very different scales, the step is a **vector** rather than a scalar, set
proportional to a rough estimate of each coefficient's posterior standard deviation. The next
chunk runs three chains from a zero start with a too-small, a good and a too-large multiple of that
vector, discarding 1000 burn-in draws and plotting the traces of the term-spread coefficient.

```{python}
base_step = np.array([0.12, 0.06, 0.27])       # roughly 1.4 posterior sds
n_iter, burn, init = 5000, 1000, np.zeros(3)
runs = [(m,) + rw_metropolis(log_posterior, init, m * base_step, n_iter, rng)
        for m in (0.02, 1.0, 8.0)]
post = runs[1][1][burn:]
print(pd.DataFrame({"post_mean": post.mean(axis=0), "post_sd": post.std(axis=0),
                    "P(coef < 0)": (post < 0).mean(axis=0)},
                   index=coef_names).round(4))
fig, ax = plt.subplots()
for mult, ch, acc in runs:
    ax.plot(ch[:, 1], lw=0.6, label=f"step x{mult} (acceptance {acc:.2f})")
ax.axhline(0.0, color="black", ls=":", lw=1, label="no predictability")
ax.set_xlabel("iteration"); ax.set_ylabel(r"term-spread coefficient $\beta_1$")
ax.set_title("Traces of the term-spread coefficient for three step sizes")
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
plt.show()
```

The traces show the tiny step creeping downward from zero without reaching the posterior bulk despite
accepting almost every proposal, the large step freezing for hundreds of iterations at a time, and only
the middle step giving the stationary band we want. Reading the numbers from that middle chain, the
term-spread coefficient has posterior mean $-0.1116$ with standard deviation $0.0452$, and the posterior
probability that it is negative is $0.9895$. The unemployment coefficient has posterior mean $-0.1439$
with standard deviation $0.2312$, and the posterior probability that it is negative is $0.7202$, so its
sign is barely better than a coin toss.

The economic reading is the point of the exercise. A one percentage point widening of the term spread
lowers the log-odds of a losing month by $0.1116$; at the sample's base rate of $0.4013$ that
moves the probability of a loss to $0.375$, and evaluating the fitted model at the most inverted curve
in the sample and then at the steepest, a swing of $10.36$ percentage points, moves it from $0.606$ to
$0.327$. The sign is the one the expectations hypothesis and the leading-indicator
literature both predict: an upward-sloping curve accompanies expected recovery, and equities do better
during recoveries. The size is small, which is exactly what one should expect from a monthly equity
return: a coefficient large enough to be traded profitably after costs would be a standing arbitrage.
The unemployment change, by contrast, carries no information at this horizon, which is the
efficient-markets prediction rather than a failure of the model: by the time the figure is published it
is in the price.

## Diagnosing the output

Successive draws are correlated by construction, and the **autocorrelation function** measures how fast
that dependence dies out with the lag. A chain of length $n$ therefore carries less information than $n$
independent draws, and the **effective sample size**
$$
\text{ESS} \;=\; \frac{n}{1 + 2\sum_{k=1}^{\infty}\rho_k}
$$
converts it into an equivalent number of independent draws, where $\rho_k$ is the lag-$k$
autocorrelation, using the same `acf` and `ess` functions already used on the binomial example above. Its
empirical tail is noisy, so we truncate the sum by Geyer's initial positive sequence rule, adding
consecutive pairs and stopping when a pair sum turns negative. This also gives the **Monte Carlo standard
error** of a posterior mean, $\text{MCSE} = \hat\sigma/\sqrt{\text{ESS}}$, the numerical precision of the
figure we report. A single chain cannot reveal one stuck in one region of the posterior, so we also
compare chains from **dispersed starting points**. The **potential scale reduction factor** contrasts
variation *between* chains with variation *within*: with $m$ chains of length $n$, let $W$ be the average
within-chain variance and $B$ the between-chain variance scaled by $n$, so that
$$
\hat{R} \;=\; \sqrt{\frac{\frac{n-1}{n}W + \frac{1}{n}B}{W}}.
$$
Converged chains make the two variances agree, so $\hat R$ is near one. The chunk applies the three
diagnostics to the good chain and to four chains launched from separated corners of the coefficient
space, all of them many posterior standard deviations from the mode.

```{python}
def rhat(samples):                  # samples: (m chains, n draws), one parameter
    m, n_dr = samples.shape
    W = samples.var(axis=1, ddof=1).mean()
    B = n_dr * samples.mean(axis=1).var(ddof=1)
    return np.sqrt(((n_dr - 1) / n_dr * W + B / n_dr) / W)

starts = np.array([[1.0, 1.0, 1.0], [-1.0, 1.0, -1.0],
                   [1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]])
def run_chains(mult):
    return np.stack([rw_metropolis(log_posterior, s, mult * base_step, 3000, rng)[0]
                     for s in starts])[:, burn:]

diag = pd.DataFrame({"ESS": [ess(post[:, j]) for j in range(3)]}, index=coef_names)
diag["MCSE"] = post.std(axis=0, ddof=1) / np.sqrt(diag["ESS"])
diag["post_sd"] = post.std(axis=0, ddof=1)
diag["Rhat_good"] = [rhat(run_chains(1.0)[:, :, j]) for j in range(3)]
diag["Rhat_tiny"] = [rhat(run_chains(0.02)[:, :, j]) for j in range(3)]
print(diag.round(4))
```

Four thousand retained draws are worth between two hundred and fifty and three hundred independent
ones, and the Monte Carlo standard errors are about seventeen times smaller than the corresponding
posterior standard deviations, so two decimals on a coefficient are safe and a third is close to the
limit of what the chain supports. With the good step $\hat R$ is at most $1.009$ for the three
coefficients, inside the usual warning threshold, so the four chains agree and may be pooled; with
the tiny step it is far above the warning threshold of 1.01, because the chains still sit near their own
starting corners, which is an instruction not to use the output.

## Hamiltonian Monte Carlo

The random walk ignores the shape of the posterior, so most proposals point uphill in energy and are
rejected. **Hamiltonian Monte Carlo** instead borrows the mechanics of a frictionless particle sliding
over the surface $-\log \pi(\theta)$. We augment the parameter $\theta \in \mathbb{R}^d$ with an
auxiliary **momentum** $\rho \in \mathbb{R}^d$ and define the joint density
$\pi(\theta, \rho) \propto \pi(\theta)\exp(-\tfrac{1}{2}\rho^\top M^{-1}\rho)$, which factorises, so the
marginal of $\theta$ is exactly the posterior we want; the mass matrix is taken as $M = I$. With
$U(\theta) = -\log \pi(\theta)$ as potential and $K(\rho) = \tfrac{1}{2}\rho^\top \rho$ as kinetic
energy, the negative log joint density is the **Hamiltonian** $H(\theta, \rho) = U(\theta) + K(\rho)$.
Each iteration draws a fresh momentum $\rho \sim \mathcal{N}(0, I)$, an exact draw from its conditional,
and travels along a level set of $H$ by following Hamilton's equations
$d\theta/dt = \partial H/\partial \rho = \rho$ and
$d\rho/dt = -\partial H/\partial \theta = \nabla \log \pi(\theta)$. Because $H$ is conserved along an
exact solution, the endpoint has the same joint density as the start and would be accepted with
certainty however far it travelled, so a proposal can cross the posterior without becoming improbable.
For the logistic model the gradient is closed form, the logit score minus the prior shrinkage term,
$\nabla \log \pi(\beta) = X^\top(y - (1 + e^{-X\beta})^{-1}) - \beta/\tau^2$, which for this application
reads as the vector of prediction errors projected onto the term spread and the unemployment change.

Hamilton's equations cannot be solved exactly, so we discretise them with the **leapfrog** integrator,
which interleaves half-steps in the momentum with full steps in the position and applies the pair $L$
times to give a trajectory of length $L\epsilon$. Two properties of that integrator do the work: it is
volume preserving, so no Jacobian appears in the acceptance ratio, and it is reversible, so the proposal
is symmetric. Because it is nonetheless approximate, $H$ is not exactly conserved, and the
discretisation error is removed by an ordinary Metropolis correction on the augmented space, accepting
the endpoint $(\theta^\ast, \rho^\ast)$ with probability
$\min\{1, \exp(H(\theta, \rho) - H(\theta^\ast, \rho^\ast))\}$. The energy error, and with it the
rejection rate, grows with $\epsilon$, so the step is chosen to keep acceptance near 0.65 to 0.9.

::: {.callout-note collapse="true"}
## Derivation: the leapfrog integrator and why no Jacobian appears

The leapfrog integrator with step size $\epsilon$ takes one step as
$$
\rho_{t + \epsilon/2} = \rho_t + \tfrac{\epsilon}{2}\,\nabla \log \pi(\theta_t),
\qquad
\theta_{t+\epsilon} = \theta_t + \epsilon\,\rho_{t+\epsilon/2},
\qquad
\rho_{t+\epsilon} = \rho_{t+\epsilon/2} + \tfrac{\epsilon}{2}\,\nabla \log \pi(\theta_{t+\epsilon}),
$$
and applying it $L$ times gives a trajectory of length $L\epsilon$. Each of the three sub-steps is a
**shear**: it adds to one coordinate block a function of the other block only, so its Jacobian matrix is
triangular with ones on the diagonal and determinant exactly one. The composition of volume-preserving
maps is volume preserving, so the whole trajectory has unit Jacobian and no correction term enters the
acceptance ratio.

Reversibility follows from the symmetry of the same three sub-steps: negating the final momentum,
rerunning the integrator for $L$ steps and negating again returns exactly to the starting point. Since
the momentum distribution is symmetric about zero, the proposal density from $(\theta,\rho)$ to
$(\theta^\ast,\rho^\ast)$ equals the density of the reverse move, so the proposal terms cancel from the
Metropolis ratio exactly as they did for the symmetric random walk, and only the change in $H$ remains.
A final implementation point: consecutive half-steps in the momentum merge into one full step, so $L$
leapfrog steps need only $L+1$ gradient evaluations rather than $2L$.
:::

The chunk codes the gradient and the sampler by hand, counting gradients so we can price it.

```{python}
def grad_log_posterior(beta):
    return X.T @ (y - 1.0 / (1.0 + np.exp(-(X @ beta)))) - beta / tau**2

def hmc(log_post, grad_log_post, init, step, n_steps, n_iter, rng):
    d = len(init)
    chain = np.empty((n_iter, d))
    theta = np.asarray(init, dtype=float); lp = log_post(theta)
    n_accept, n_grad = 0, 0
    for t in range(n_iter):
        mom = rng.normal(size=d)
        H0 = -lp + 0.5 * np.dot(mom, mom)
        th, mo = theta.copy(), mom + 0.5 * step * grad_log_post(theta)
        for s in range(n_steps):
            th = th + step * mo
            if s < n_steps - 1: mo = mo + step * grad_log_post(th)   # merged half-steps
        mo = mo + 0.5 * step * grad_log_post(th)
        n_grad += n_steps + 1
        lp_new = log_post(th)
        H1 = -lp_new + 0.5 * np.dot(mo, mo)
        if np.log(rng.uniform()) < H0 - H1:         # Metropolis correction
            theta, lp = th, lp_new
            n_accept += 1
        chain[t] = theta
    return chain, n_accept / n_iter, n_grad

print("gradient at the origin:", np.round(grad_log_posterior(init), 2))
```

The gradient at the origin is large and points towards the posterior mode, so a trajectory from a poor
start is pushed the right way at once; its first element is negative because fewer than half of the
months are losing ones, so the intercept must fall from zero. The tuning parameters pull against each
other: $\epsilon$ sets the accuracy of the integrator, since too large a value makes the energy error
explode and the trajectory diverge, while $L\epsilon$ sets how far a proposal travels, too short a
trajectory degenerating into a random walk with expensive steps and too long a one curving back on itself
so the extra gradients buy nothing. Since each iteration costs $L+1$ gradients, the best $L$ depends on
the local geometry.

## The no-U-turn sampler

The **no-U-turn sampler** of Hoffman and Gelman (2014) removes the need to choose the trajectory length.
Rather than a fixed number of leapfrog steps, it builds the trajectory adaptively, repeatedly
**doubling** it forward or backward in time by as many steps as it holds, so the path grows as a
balanced binary tree. Doubling stops once the trajectory begins to turn back on itself, detected from
the inner product of the momentum at each end with the displacement between the ends: a sign change
means the particle is heading back, and that U-turn names the sampler. The next state is drawn from the
whole trajectory by a slice-sampling rule that keeps the transition reversible, the same auxiliary-variable
idea used earlier in this lecture now applied to selecting a point along a trajectory rather than along the
real line, and doubling aborts early if the energy error grows too large. With dual averaging, which
adapts $\epsilon$ during warm-up to a target acceptance rate, this leaves nothing to tune, which is why it
is the default engine of modern probabilistic programming. We do not implement it; it automates the choice
of $L$ without changing the dynamics. The relevance here is direct: the hand-tuned step size below is
delicate, and a moderately larger value makes the energy error explode and the acceptance rate collapse,
which is exactly the fragility dual averaging removes.

## Comparing the two samplers

Efficiency must be measured per unit of cost, not per iteration, so the comparison times each run and
counts posterior evaluations: one log density per random-walk iteration against $L+1$ gradients per
Hamiltonian iteration, here with $\epsilon = 0.05$ and $L = 7$. It reports the smallest effective
sample size across coefficients per second and per evaluation, plus the two autocorrelations.

```{python}
def efficiency(name, run, per_iter):
    t0 = time.perf_counter(); out = run(); secs = time.perf_counter() - t0
    kept = out[0][burn:]
    e = min(ess(kept[:, j]) for j in range(3))
    return kept, {"sampler": name, "acceptance": out[1], "min_ESS": e, "seconds": secs,
                  "ESS_per_sec": e / secs, "ESS_per_1000_eval": 1000 * e / (per_iter * n_iter)}

mh_post, mh_row = efficiency("RW-MH",
                             lambda: rw_metropolis(log_posterior, init, base_step, n_iter, rng), 1)
hmc_post, hmc_row = efficiency("HMC", lambda: hmc(log_posterior, grad_log_posterior, init,
                                                  0.05, 7, n_iter, rng), 8)
print(pd.DataFrame([mh_row, hmc_row]).set_index("sampler").round(2))
print("posterior means:", np.round(mh_post.mean(axis=0), 4), np.round(hmc_post.mean(axis=0), 4))
print(f"P(term-spread coefficient < 0): RW-MH {(mh_post[:,1] < 0).mean():.4f}, "
      f"HMC {(hmc_post[:,1] < 0).mean():.4f}")
fig, ax = plt.subplots()
for series, label in [(mh_post[:, 1], "RW-MH"), (hmc_post[:, 1], "HMC")]:
    ax.plot(np.arange(41), acf(series, 40), ".-", label=label)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("lag"); ax.set_ylabel("autocorrelation")
ax.set_title(r"Autocorrelation of the term-spread coefficient under the two samplers")
ax.legend()
plt.tight_layout()
plt.show()
```

The two samplers agree on the intercept and the term-spread coefficient to within about a thousandth,
they agree on the poorly identified unemployment coefficient to within its own Monte Carlo error, and
they agree on the headline probability that the term-spread coefficient is negative, $0.9902$ against
$0.9922$, which is the first thing to check. The figure shows the random-walk autocorrelation persisting
for a few dozen lags where the Hamiltonian one has all but disappeared after one or two. The table prices
the trade-off, and on this posterior it comes out the other way from the textbook warning: Hamiltonian
Monte Carlo turns four thousand kept draws into $2{,}756$ effectively independent ones against the random
walk's $275$, at an acceptance rate of $0.87$ against $0.28$. Each Hamiltonian iteration costs eight
gradient evaluations rather than one cheap density evaluation, but the gradient is a single matrix
product over $795$ observations and costs little more in wall-clock time than the density itself. Per
thousand posterior evaluations the Hamiltonian sampler is modestly ahead, $68.9$ against $55.1$; per
second of wall-clock time, which is the column that will vary from machine to machine, it is ahead by
more than a factor of two.

What makes the difference here, and did not in a spherical toy problem, is the geometry of a real design
matrix. The three coefficients have posterior standard deviations differing by a factor of about five,
because the term spread and the unemployment change are measured on different scales and are not
orthogonal, so a spherical random-walk proposal must be tuned to the narrowest direction and then
crawls along the widest. A gradient-guided trajectory follows the ridge instead and pays for extra
dimensions roughly linearly. The general lesson stands: the crossover in favour of Hamiltonian Monte
Carlo appears as soon as dimension or curvature grows, and a realistic macro-finance posterior is
already on the right side of it.

The autocorrelation plot already showed the efficiency gap in one number per lag; the same
gap is visible directly in how the two chains move through parameter space, which the next
block draws for the same three hundred kept sweeps used above.

**Reading the computation.** Both panels trace the same number of kept sweeps of the intercept and the
term-spread coefficient. A path that revisits the same neighbourhood repeatedly covers less ground per
sweep than one that keeps moving to new territory.

```{python}
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(mh_post[:300, 0], mh_post[:300, 1], lw=0.6, color="steelblue")
axes[0].set_title("Random-walk MH"); axes[0].set_xlabel(r"intercept $\beta_0$")
axes[0].set_ylabel(r"term spread $\beta_1$")
axes[1].plot(hmc_post[:300, 0], hmc_post[:300, 1], lw=0.6, color="darkorange")
axes[1].set_title("Hamiltonian Monte Carlo"); axes[1].set_xlabel(r"intercept $\beta_0$")
fig.suptitle("300 kept sweeps in the (intercept, term-spread) plane")
plt.tight_layout(); plt.show()
```

The random-walk path is a dense, tangled scribble confined to a small patch, consistent
with its slower-decaying autocorrelation, while the Hamiltonian path covers a visibly
larger area of the same two coordinates in the same number of kept sweeps, the geometric
counterpart of the larger effective sample size reported in the efficiency table above.
