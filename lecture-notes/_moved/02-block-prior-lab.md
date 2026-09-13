Belongs in: Week 2 exercise session (exercise-sessions/02-linear-regression.qmd), as the
elicitation lab (six-sigma range, informative on the market beta only).

Cut from lecture-notes/02-linear-regression.qmd (the worked block-prior computation at the
end of "Choosing the prior in practice"). The chapter keeps the block-prior structure
itself, b_0 and B_0^{-1} partitioned with a zero block, and the six-sigma and
moment-inversion elicitation rules; only this second numerical prior specification and its
table move.

---

A cheap and often overlooked middle course is to be informative only where beliefs exist. If
$\beta$ splits into a block $\beta_1$ we have views on and a block $\beta_2$ we do not, set

$$
b_0 = \begin{pmatrix} b_{0,1} \\ 0 \end{pmatrix}, \qquad
B_0^{-1} = \begin{pmatrix} B_{0,11}^{-1} & 0 \\ 0 & 0 \end{pmatrix},
$$

and the posterior formulas go through unchanged, because they only ever use $B_0^{-1}$. The
zero block simply contributes no prior precision. For the banking regression this is exactly
right economically: theory says something about the market beta and nothing whatsoever about
alpha or the size and value loadings, so those three should be left to the data.

**Reading the computation.** The block prior below is informative on the market beta only,
built from the range $[0.6, 1.6]$ by the six-sigma rule, and flat on the other three
coefficients. Compare its posterior with the fully informative one and with least squares.

```{python}
bL, bU = 0.6, 1.6
beta_mean_e, beta_sd_e = 0.5 * (bL + bU), (bU - bL) / 6.0

tau_block = np.array([np.inf, beta_sd_e, np.inf, np.inf])  # flat except on the beta
B0_inv_block = np.diag(np.where(np.isfinite(tau_block),
                                sigma2_bar / tau_block**2, 0.0))
block = nig_update(y, X, np.array([0.0, beta_mean_e, 0.0, 0.0]),
                   B0_inv_block, a0, d0)
sd_block = np.sqrt(block["d1"] / (block["a1"] - 1.0) * np.diag(block["B1"]))

print(f"elicited prior for the market beta: mean {beta_mean_e:.2f}, sd {beta_sd_e:.3f}")
print(pd.DataFrame({"OLS": b_ols, "full prior": post["b1"], "block prior": block["b1"],
                    "OLS se": se_ols, "block post sd": sd_block}, index=names).round(4))
```

The market beta is pulled towards the elicited $1.1$, but only from $1.1933$ to $1.1896$,
because a prior standard deviation of $0.167$ is still weak against a sample standard error
of $0.0331$: the data are roughly twenty-five times as precise as the elicited belief, so
the belief moves the answer by about a ninth of a standard error. Alpha and the two remaining
loadings sit on their least-squares values of $-0.0020$, $-0.0992$ and $0.8230$ with
standard errors to match, because the block prior added no information about them. This is
usually the defensible position: shrink where theory speaks, stay quiet elsewhere. It is
also a useful reminder that with several hundred monthly observations an honestly elicited
prior on a well-identified coefficient will change almost nothing, and that this is a
feature rather than a disappointment.

