> Cut from lecture note 3 in the restructuring pass. Belongs in the **Week 3 exercise
> session**. The option-pricing motivation that used to open the chapter, together with
> the Black-Scholes averaging exercise and the vomma sign argument that closed it.
> Lecture 3 now closes with value-at-risk from the predictive distribution of the shared
> Student-t target instead. The vomma argument is the one piece of algebra here and it
> should be kept intact when this is pasted into the exercise session.

## The motivating question, as it used to open the chapter

## Start at a risk desk: what does uncertain volatility cost?

A risk officer runs a one-month value-at-risk number on a position in the US
market portfolio, and an options desk two floors down prices a one-month call on
the same index. Both need a volatility. Both estimate it from the same one-year
window of daily returns, and both, by long habit, plug in a single number.
**Should we use one volatility estimate, or average the answer over the
volatilities the data leave open?** Value-at-risk and an option payoff are both
non-linear in volatility, so the two operations need not agree.

The question is not rhetorical, and this chapter answers it with the real
series. We take the market excess return from the Kenneth French daily file, use
the most recent $250$ trading days as the estimation window, put a conjugate
inverse-Gamma prior on the return variance, and obtain a posterior for that
variance rather than a point estimate. Every number in the chapter is computed
from that posterior.

For a transparent illustration of what is at stake before the data arrive,
consider a stock costing 100, a one-year strike of 100, a zero risk-free rate
and no dividends. Conditional on volatility $\sigma$, the risk-neutral model is

$$S_T=100\exp(-\tfrac12\sigma^2T+\sigma\sqrt T Z),\qquad Z\sim N(0,1).$$

The exponential keeps prices positive. Its $-\sigma^2T/2$ term offsets the
lognormal mean adjustment, so $E_Q[S_T]=100$ here. This is a pricing model under
$Q$, not a forecast of the stock's actual expected return under the physical measure.
The call value is the average payoff $C(\sigma)=E_Q[(S_T-100)_+]$.

Imagine posterior probability one half on 10% volatility and one half on 30%.
The conditional values are about 3.99 and 11.92, so the posterior average is
about 7.96. At the average volatility of 20%, the value is about 7.97. The gap
is small for this contract, but there is no rule making it zero, and there is no
rule fixing its **sign** either: moneyness and maturity change the curvature of
$C(\sigma)$, and we shall see below that on the real posterior the correction is
negative at the money and positive out of the money. Always calculate rather
than presume.

The computation follows directly from two uncertainties. First draw a plausible
volatility from its posterior. Then draw future price shocks conditional on it:

$$\widehat C=\frac1S\sum_{s=1}^S e^{-rT}
\left[S_0\exp\{(r-\tfrac12\sigma_s^2)T+\sigma_s\sqrt T Z_s\}-K\right]_+.$$

This sample average approximates an integral we could not conveniently evaluate
by hand. More draws reduce simulation error; they do not remove uncertainty
about volatility. If conditional Black–Scholes values are available, average
those instead of simulated payoffs to remove one source of Monte Carlo noise.

**The financial qualification matters.** Historical return data inform physical
return parameters. Using their volatility posterior in this pricing calculation
assumes a model linking physical and risk-neutral volatility; option prices
would provide additional information. The calculation below makes that
assumption explicit and never puts an estimated historical drift in place of $r$.

## Pricing a one-month option under posterior volatility uncertainty


The options desk faces the same posterior and a different function of it. We price a
one-month European call on the index, with spot normalised to $100$, using the average
risk-free rate over the estimation window as $r$ and the posterior of the annualised
volatility as the distribution of $\sigma$. Because the Black-Scholes formula gives the
conditional price in closed form, we average that formula over the posterior draws rather
than simulating payoffs, which removes one whole source of Monte Carlo noise, exactly as
the introduction promised.

```{python}
S0, T_opt = 100.0, H / 252
r_opt = window["rf"].mean() * 252

def bs_call(sigma, K):
    d1 = (np.log(S0 / K) + (r_opt + 0.5 * sigma ** 2) * T_opt) / (sigma * np.sqrt(T_opt))
    d2 = d1 - sigma * np.sqrt(T_opt)
    return S0 * stats.norm.cdf(d1) - K * np.exp(-r_opt * T_opt) * stats.norm.cdf(d2)

sig_ann = np.sqrt(var_post.rvs(size=200_000, random_state=rng) * 252)
sig_bar = np.sqrt(var_post.mean() * 252)

opt_rows = []
for K in (100.0, 103.0, 105.0, 107.0):
    p_plug, p_full = bs_call(sig_bar, K), bs_call(sig_ann, K).mean()
    opt_rows.append({"strike": K, "plug-in": p_plug, "posterior avg": p_full,
                     "gap (bp of spot)": 100 * (p_full - p_plug),
                     "gap (%)": 100 * (p_full / p_plug - 1),
                     "gap (EUR on 10m)": notional * (p_full - p_plug) / S0})
opt_table = pd.DataFrame(opt_rows).set_index("strike")
print(f"annualised risk-free rate used: {r_opt:.4f}, "
      f"plug-in volatility {sig_bar:.4f}, horizon {T_opt:.4f} years")
print(opt_table.round(4).to_string())
```

The at-the-money call is worth $1.7055$ under the plug-in convention and $1.7041$ when the
price is averaged over the posterior of volatility, a reduction of $0.148$ basis points of
spot, or $148$ euro on a ten million euro notional. The far out-of-the-money call at a
strike of $107$ moves the other way, rising by $0.069$ basis points of spot, which is
$0.90\%$ of its own value and $69$ euro on the same notional. The sign changes between the
strikes of $103$ and $105$.

That sign change is the economic content of the section and it contradicts a claim often
made loosely in textbooks, that Jensen's inequality guarantees a higher price once
volatility is uncertain. The claim is only true where the price is **convex** in
volatility. For a European call, the second derivative of price with respect to
volatility, the quantity traders call *vomma*, is proportional to $d_1 d_2$, and at the
money $d_1$ and $d_2$ straddle zero, so the product is negative and the price is mildly
*concave* in volatility. Averaging over a spread of volatilities then lowers the value.
Far out of the money both $d_1$ and $d_2$ are negative, vomma is positive, the price is
convex, and parameter uncertainty raises the value. In other words, ignoring the posterior
of volatility systematically overprices at-the-money options and underprices wings, which
is the same statement as saying that a plug-in model produces a flatter implied volatility
smile than parameter uncertainty alone would justify. The direction of the error is
predictable; the size, as here, must be computed. Exercise 5 of this week's session
asks you to repeat the calculation on a shorter estimation window, where the posterior
of volatility is wider and both gaps grow.

Both calculations are otherwise identical vanilla Monte Carlo, exactly the machinery of
the lecture's first section, applied to a payoff function instead of a posterior summary,
and both take the same posterior as their input. The financial qualification stated at the
start still binds: the posterior comes from realised returns under the physical measure,
and using it in a pricing formula is an assumption about the relation between physical and
risk-neutral volatility, not a fact.
