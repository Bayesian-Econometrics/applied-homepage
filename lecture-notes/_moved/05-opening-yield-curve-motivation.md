> Cut from lecture note 5 in the restructuring pass. Belongs in the **Week 5 exercise
> session**. The opening vignette that motivates Metropolis-Hastings through a logistic
> risk forecast (log odds of a losing month on the term spread and the change in
> unemployment, the worked logistic-curve illustration, the gradient reading). Removed
> under the one-example-per-chapter rule: lectures 3 to 5 now share the Student-t
> market-return target. Pairs with 05-logistic-macro-regression.md below.

## Start with a risk forecast: does the yield curve predict a losing month?

A risk committee wants a forward-looking statement rather than a backward-looking
one. **Given what the macroeconomy looks like today, what is the probability that
the market's excess return next month is negative?** A linear probability model
could answer below zero or above one, so we need a model whose predictions
remain probabilities.

Let $L_{t+1}$ indicate a negative excess return in month $t+1$ and $x_t$ summarise
information available at the end of month $t$. Assume that log odds, rather than
probability itself, respond linearly: $\log[p_t/(1-p_t)]=a+b'x_t=\eta_t$.
Exponentiating and solving for $p_t$ gives

$$p_t=\frac{e^{\eta_t}}{1+e^{\eta_t}}.$$

For one observed loss, the likelihood contribution is $p_t$; for a non-loss it
is $1-p_t$. Thus its log contribution is
$L_{t+1}\eta_t-\log(1+e^{\eta_t})$. Summing observations and adding independent
zero-centred Normal priors gives the log posterior used by the sampler. The
$\log(1+e^\eta)$ term prevents the quadratic completion that worked in regression,
which is precisely why this chapter needs a sampler at all.

The two predictors we use are the **term spread**, the ten-year Treasury yield
minus the federal funds rate, and the **change in the unemployment rate**, both
taken from the FRED monthly file and both dated so that they were public before
the return they are asked to predict. A worked illustration first: coefficients
$a=-0.4$ and $b=0.8$ on a single standardised predictor give probabilities 0.23,
0.40 and 0.60 at $x=-1,0,1$. Those are hypothetical numbers chosen to show the
shape of the logistic curve, not a finding. The empirical question is whether
$795$ months of real data move the term-spread coefficient away from zero, and by
the end of this chapter we report a posterior probability of $0.9895$ that it is
negative, meaning that a steeper curve makes a losing month less likely.

How does MH turn a computable log posterior into draws? For a symmetric proposal,
accept with probability $\min(1,\exp(\ell_{new}-\ell_{old}))$.
A proposed move from log posterior $-12$ to $-13$ is accepted with probability
$e^{-1}\approx0.37$, not automatically rejected. Allowing some downhill moves
is necessary to explore a distribution rather than merely locate its maximum.

The gradient is $X^\top(L-p)-\beta/\tau^2$: prediction errors pull the
coefficients toward the data and the prior pulls them toward zero. HMC uses
this slope to make informed joint moves; NUTS selects trajectory length.
We compare efficiency for the **same** logistic posterior below. Better
sampling does not create predictive information absent from financial returns,
and the modest size of the estimated coefficients is a reminder of that.
