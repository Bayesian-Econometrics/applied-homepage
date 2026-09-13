Belongs in: Week 2 exercise session (exercise-sessions/02-linear-regression.qmd), as the
reference card students keep open while checking formulas against Bauwens or Koop.

Cut from lecture-notes/02-linear-regression.qmd (section "Reading these results in the
sources' notation"). The spec asks for a one-line bridge where a source uses IG_2(nu, s)
rather than a full translation section; that one-line bridge stays in the chapter, in the
Normal-Inverse-Gamma prior section, and the full table moves here. No derivation is
involved.

---

## Reading these results in the sources' notation

The textbook treatments use a different but equivalent parameterisation, and mixing the two
silently is the most common source of confusion when checking a formula against a
reference. The sources write the variance prior as an inverted-gamma-2 density, $\sigma^2
\sim \text{IG}_2(\nu, s)$, with density proportional to
$(\sigma^2)^{-(\nu+2)/2}\exp[-s/(2\sigma^2)]$, and parameterise the coefficient prior by a
precision matrix $M_0$ rather than a covariance factor $B_0$. The multivariate Student
density appears as $t_k(\mu, s, M, \nu)$, with mean $\mu$ and variance
$\frac{s}{\nu-2}M^{-1}$.

| This course | Sources (Bauwens; Koop) | Relation |
|---|---|---|
| $B_0$, prior covariance factor | $M_0$, prior precision | $M_0 = B_0^{-1}$ |
| $\alpha_0$, inverse-Gamma shape | $\nu_0$, degrees of freedom | $\nu_0 = 2\alpha_0$ |
| $\delta_0$, inverse-Gamma scale | $s_0$, scale | $s_0 = 2\delta_0$ |
| $B_1^{-1} = B_0^{-1} + X^\top X$ | $M_* = M_0 + X^\top X$ | identical |
| $b_1 = B_1(B_0^{-1}b_0 + X^\top y)$ | $\beta_* = M_*^{-1}(M_0\beta_0 + X^\top X\hat\beta)$ | identical |
| $\alpha_1 = \alpha_0 + n/2$ | $\nu_* = \nu_0 + n$ | $\nu_* = 2\alpha_1$ |
| $\delta_1 = \delta_0 + Q/2$ | $s_* = s_0 + \text{SSR} + \text{conflict}$ | $s_* = 2\delta_1$ |
| $\mathbb{E}[\sigma^2\mid y] = \delta_1/(\alpha_1-1)$ | $s_*/(\nu_*-2)$ | identical |

: Notation bridge {.striped}

Two conventions are easy to trip over. The sources' $\hat\beta$-centred form of the
likelihood uses $s$ for the residual sum of squares, which collides with the $s$ used as an
inverted-gamma-2 scale in the prior, so the same letter means two things three lines apart.
And the noninformative prior appears there as the limit $M_0 \to 0$, $s_0 \to 0$, $\nu_0
\to -k$, which in this chapter's symbols is $B_0^{-1} \to 0$, $\delta_0 \to 0$, $\alpha_0
\to -k/2$, exactly the substitution used above. The degrees-of-freedom limit is $-k$ rather
than $0$ so that $\nu_* = n - k$ rather than $n$, which is the same degrees-of-freedom
correction that classical regression applies.

