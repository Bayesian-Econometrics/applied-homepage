> Cut from lecture note 4 in the restructuring pass. Belongs in the **Week 4 exercise
> session**. The opening fund-performance vignette (alpha on the three Fama-French
> factors, conditional-on-variance completing the square with illustrative numbers).
> Removed under the one-example-per-chapter rule: lectures 3 to 5 now share the
> Student-t market-return target. Self-contained; needs no edits.

## Start with fund performance: skill, factor exposure, or noise?

A fund has outperformed the market. **Is that persistent alpha, compensation for
other factor exposures, or an unusually favourable sample?** Add size and value
returns to the market regression:

$$y_t=\alpha+\beta_M M_t+\beta_S SMB_t+\beta_H HML_t+\varepsilon_t.$$

Each factor return is observed; the loadings and residual variance are unknown.
An economically meaningful prior might express beliefs about loadings
independently of beliefs about the noise variance. That breaks the particular
scaled conjugacy of Week 2, but does not make either conditional problem hard.

See the mechanism with just alpha after subtracting current factor contributions.
Call these residualised returns $z_t$. Suppose their illustrative average is
0.4 percentage points over 20 months, and use $\alpha\sim N(0,0.5^2)$.
Conditional on variance $\sigma^2$, multiplying the likelihood and prior gives

$$\log p(\alpha\mid\sigma^2,z)
=-\tfrac12\left[(20/\sigma^2+4)\alpha^2
-2(8/\sigma^2)\alpha\right]+\text{constant}.$$

Completing the square gives variance $V=(20/\sigma^2+4)^{-1}$ and mean
$m=V(8/\sigma^2)$. If $\sigma^2=1$, the mean is 0.333; if
$\sigma^2=4$, it is 0.222. Noisier returns supply weaker evidence against zero alpha.

We do not know which variance to use. Given a current coefficient draw, calculate
its residual sum of squares $RSS$. With an independent $IG(a_0,d_0)$ variance
prior, the conditional variance posterior is
$IG(a_0+n/2,d_0+RSS/2)$. Draw from it, then draw coefficients using that variance,
and repeat. **Gibbs sampling alternates these two ordinary inference problems.**

The changing values are not a succession of revised data sets: the data stay
fixed throughout. After convergence, the chain represents joint uncertainty
about exposures and noise. The exercise draws the entire coefficient vector in
one block, preserving dependence between market, size and value loadings.
Report uncertainty about alpha, not just the fraction of raw returns that were positive.
