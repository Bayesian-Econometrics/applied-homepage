> Cut from lecture note 3 in the restructuring pass. Belongs in the **Week 3 exercise
> session**. A standalone geometric illustration of rejection sampling on a
> N(0,1) target under a N(0,4) envelope with M = 2. It duplicates the rejection-sampling
> result now stated in general notation in the lecture, on a synthetic target unrelated
> to the chapter's data, so it was removed as a repeated neighbour. It is a good
> whiteboard exercise.

## Rejection sampling: probability as an area

Importance sampling keeps proposal draws and changes their weights. Rejection
sampling instead keeps selected draws so that the retained draws have the target
density. Let $p$ and $q$ be normalised target and proposal densities. If
$p(\theta)\leq Mq(\theta)$ everywhere, draw $\theta\sim q$ and
$u\sim U(0,1)$, retaining $\theta$ when $u\leq p(\theta)/(Mq(\theta))$.
Why does that work? The joint density of proposing and accepting at $\theta$ is

$$
q(\theta)\frac{p(\theta)}{Mq(\theta)}=\frac{p(\theta)}{M}.
$$

Integrating gives acceptance probability $1/M$; conditioning on acceptance
therefore leaves density $p$. In the plot, drawing a vertical coordinate
$v=uMq(\theta)$ produces points uniformly under the envelope. Points below the
target are kept. For $p=N(0,1)$ and $q=N(0,4)$, $M=2$ is a valid global bound.

```{python}
#| label: fig-rejection-envelope-area
#| fig-cap: "Retained points lie below the target. The target occupies half the envelope's area, so long-run acceptance is 50%."
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

rejection_rng = np.random.default_rng(310)
rejection_x = rejection_rng.normal(0, 2, 700)
rejection_v = rejection_rng.uniform(size=700) * 2 * norm.pdf(rejection_x, scale=2)
rejection_keep = rejection_v <= norm.pdf(rejection_x)
rejection_grid = np.linspace(-6, 6, 400)
fig, ax = plt.subplots()
ax.scatter(rejection_x[~rejection_keep], rejection_v[~rejection_keep],
           s=8, color='#aa8066', alpha=0.45, label='Rejected')
ax.scatter(rejection_x[rejection_keep], rejection_v[rejection_keep],
           s=8, color='#527a99', alpha=0.5, label='Kept')
ax.plot(rejection_grid, norm.pdf(rejection_grid), color='#345b75', label='Target p')
ax.plot(rejection_grid, 2 * norm.pdf(rejection_grid, scale=2),
        '--', color='#666666', label='Envelope Mq')
ax.set(xlabel='Proposed parameter', ylabel='Vertical coordinate', xlim=(-6, 6))
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4)
plt.show()
```

A good envelope wastes little area, but a bound must hold in the tails as well
as on a plotted grid. This is not yet Metropolis–Hastings: there is no current
state, and rejected proposals are discarded rather than recorded as repeated
chain states. The scheduled option-pricing exercise still uses Monte Carlo and
importance weights; this geometric comparison explains the alternative before MCMC.

*Source connection:* Lopes, *Bayesian computation*, PDF pp. 16–23.
