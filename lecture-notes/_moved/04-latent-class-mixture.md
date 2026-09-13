> Cut from lecture note 4 in the restructuring pass. Belongs in the **Week 4 exercise
> session**, alongside the smartphone-clustering case study. Data augmentation with
> latent class labels: the mixture likelihood, the three full conditionals (categorical
> labels, Dirichlet weights, Beta feature probabilities), the six-chain label-switching
> demonstration and the relabelling rule. Removed as a second (and simulated) example;
> data augmentation itself is now the spine of the rewritten chapter, via the scale
> mixing variables w_t. Self-contained: it simulates its own data.

## Data augmentation: when a latent variable makes the hard part easy

Not every model comes with conditionally conjugate blocks sitting in plain sight. Often
the trick is to introduce an extra unobserved quantity, a **latent variable**, that we
never actually cared about, purely because conditioning on it turns an intractable
posterior into a collection of easy ones. This idea, **data augmentation**, is exactly
what powers the smartphone-clustering case study paired with this lecture, and it is
worth building the same mechanism ourselves on a dataset small enough to see
what is happening. This questionnaire is the one simulated example in the chapter, kept
because the point being made is about identification rather than about economics: only a
known data-generating process can show that the sampler recovers two groups it was never
told about.

Consider $N$ individuals answering $M$ binary questions, and suppose each individual
belongs to one of $K$ unobserved groups, or **latent classes**, that differ in how likely
they are to answer each question positively. Writing $x_i = (x_{i1}, \dots, x_{iM})$ for
individual $i$'s answers and $z_i \in \{1, \dots, K\}$ for their unobserved class, the
model is

$$
\Pr(z_i = k) = w_k, \qquad
\Pr(x_{im} = 1 \mid z_i = k) = \theta_{km},
$$

with priors $w \sim \text{Dirichlet}(\delta_1, \dots, \delta_K)$ on the class
probabilities and $\theta_{km} \sim \text{Beta}(\alpha_{km}, \beta_{km})$ independently
for each class $k$ and question $m$. Marginalising $z_i$ out gives the actual likelihood
of the observed answers, a $K$-component mixture of products of Bernoullis,

$$
\Pr(x_i \mid w, \theta) = \sum_{k=1}^{K} w_k \prod_{m=1}^{M} \theta_{km}^{x_{im}} (1-\theta_{km})^{1-x_{im}},
$$

and this mixture is exactly the kind of joint posterior that resists a direct Gibbs
treatment: $w$ and $\theta$ are tangled together inside a sum, the same way $\mu$ and
$\tau$ were tangled inside the exponential of the normal model, except now there is no
completing-the-square move that untangles them. Augmenting the parameter vector with the
latent labels $z = (z_1, \dots, z_N)$ removes the sum: *given* $z$, we know exactly which
class each observation belongs to, and the model separates into $K$ independent
Beta-Bernoulli and Dirichlet-multinomial updates, each of which we already know how to do
in closed form.

Conditioning on the data and the other blocks gives three full conditionals, all
standard. The class label is categorical with probabilities proportional to how well
each class explains that individual's answers,

$$
\Pr(z_i = k \mid x_i, w, \theta) \;\propto\; w_k \prod_{m=1}^{M} \theta_{km}^{x_{im}} (1-\theta_{km})^{1-x_{im}},
$$

the class probabilities are Dirichlet in the counts assigned to each class,
$w \mid z \sim \text{Dirichlet}(\delta_1 + N_1, \dots, \delta_K + N_K)$ with $N_k$ the
number of individuals currently labelled $k$, and each feature probability is Beta in the
answers of individuals currently assigned to that class,
$\theta_{km} \mid x, z \sim \text{Beta}\big(\alpha_{km} + \sum_i \mathbb{1}(z_i=k)x_{im},\;
\beta_{km} + \sum_i \mathbb{1}(z_i=k)(1-x_{im})\big)$. We simulate a simplified version of
the smartphone questionnaire, two classes and six questions, with genuinely different
usage patterns.

```{python}
rng_mix = np.random.default_rng(2026)

N, M, K = 300, 6, 2
theta_true = np.array([[0.85, 0.75, 0.20, 0.30, 0.80, 0.15],   # heavy users
                       [0.20, 0.15, 0.75, 0.70, 0.25, 0.80]])  # light users
w_true = np.array([0.55, 0.45])

z_true = rng_mix.choice(K, size=N, p=w_true)
X_bin = rng_mix.binomial(1, theta_true[z_true])
print(f"simulated {N} respondents, class sizes: {np.bincount(z_true)}")
```

The Gibbs sampler cycles through the three conditionals above, initialising the labels
at random so that the chain has to find the two groups on its own rather than being told
where they are.

```{python}
delta = np.ones(K)
alpha_p, beta_p = np.ones((K, M)), np.ones((K, M))
n_iter_m = 1500
z = rng_mix.integers(0, K, size=N)
theta_draws, w_draws = np.empty((n_iter_m, K, M)), np.empty((n_iter_m, K))

for t in range(n_iter_m):
    Nk = np.array([(z == k).sum() for k in range(K)])
    ones = np.array([X_bin[z == k].sum(axis=0) if Nk[k] > 0 else np.zeros(M) for k in range(K)])
    theta = rng_mix.beta(alpha_p + ones, beta_p + Nk[:, None] - ones)
    w = rng_mix.dirichlet(delta + Nk)

    logp = (np.log(w)[None, :] + X_bin @ np.log(theta).T
            + (1 - X_bin) @ np.log(1 - theta).T)              # (N, K)
    probs = np.exp(logp - logp.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    z = np.array([rng_mix.choice(K, p=probs[i]) for i in range(N)])
    theta_draws[t], w_draws[t] = theta, w
```

### The label-switching problem

The likelihood above is completely unchanged if we relabel every class simultaneously,
swapping "class 1" and "class 2" everywhere at once: the class probabilities permute, the
$\theta$ rows permute, and $\Pr(x_i \mid w, \theta)$ is identical either way. Nothing in
the model, the prior, or the sampler breaks this symmetry, so the chain is free to flip
which physical group it calls "class 1" partway through the run. This is
**label switching**, and it is a structural feature of every mixture model, not a bug in
this particular sampler; ignoring it and simply averaging the raw draws of, say,
$\theta_{1,1}$ can average together numbers that belong to different physical groups and
produce a meaningless posterior mean. Rather than assert this, we can force it into the
open: rerun the class-assignment step of the sampler several times, each time starting
from independently randomised initial labels on the same simulated data, and look only
at where each short run ends up.

```{python}
n_chains, n_iter_c = 6, 400
final_theta0 = np.empty(n_chains)
for chain in range(n_chains):
    rng_c = np.random.default_rng(3000 + chain)
    z_c = rng_c.integers(0, K, size=N)
    for t in range(n_iter_c):
        Nk = np.array([(z_c == k).sum() for k in range(K)])
        ones = np.array([X_bin[z_c == k].sum(axis=0) if Nk[k] > 0 else np.zeros(M) for k in range(K)])
        theta_c = rng_c.beta(alpha_p + ones, beta_p + Nk[:, None] - ones)
        w_c = rng_c.dirichlet(delta + Nk)
        logp = (np.log(w_c)[None, :] + X_bin @ np.log(theta_c).T
                + (1 - X_bin) @ np.log(1 - theta_c).T)
        probs = np.exp(logp - logp.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        z_c = np.array([rng_c.choice(K, p=probs[i]) for i in range(N)])
    final_theta0[chain] = theta_c[0, 0]
print("final Pr(question 1 = yes | class 0), six independently started chains:")
print(np.round(final_theta0, 3))
```

The six chains split cleanly into two groups: four end with class $0$ between $0.825$
and $0.943$, matching the heavy users, and two end with class $0$ at $0.152$ and $0.156$,
matching the light users, purely as a function of the random labels each chain happened
to start from. The
data and the model are identical across all six runs; only the arbitrary initial guess
decided which physical group got called "class 0". This is label switching laid bare,
and it shows why a posterior mean of $\theta_{0,1}$ computed by pooling chains, or even
computed within a single chain that happens to cross between the two modes, is not
automatically meaningful.

We deal with it by a simple relabelling rule: after the chain has run, orient every draw
so that class $0$ is the one with the higher probability on question 1, swapping the two
rows of $\theta$ and the two entries of $w$ whenever they come out the other way round.
Applying the rule to our long main chain from earlier,

```{python}
burn_m = 300
theta_post, w_post = theta_draws[burn_m:], w_draws[burn_m:]
swapped = theta_post[:, 0, 0] < theta_post[:, 1, 0]
print(f"fraction of kept draws with classes swapped relative to draw 1: {swapped.mean():.3f}")

theta_aligned = theta_post.copy()
theta_aligned[swapped] = theta_post[swapped][:, ::-1, :]
w_aligned = w_post.copy()
w_aligned[swapped] = w_post[swapped][:, ::-1]

print("posterior mean of theta, class 0 (heavy users):", np.round(theta_aligned[:, 0].mean(axis=0), 3))
print("posterior mean of theta, class 1 (light users):", np.round(theta_aligned[:, 1].mean(axis=0), 3))
print("true theta, heavy users: ", theta_true[0], "  true theta, light users:", theta_true[1])
print("posterior mean of w:", np.round(w_aligned.mean(axis=0), 3), "  true w:", w_true)
```

Within this particular long chain the swapped fraction comes out at $1.000$: every
single kept draw needed flipping, because this run happened to settle on the labelling in
which the group it calls class $0$ is the light-user group, and it never crossed to the
other labelling in its length. That is not a contradiction of the demonstration above; it
is the same phenomenon viewed from the other side. Instead of one chain wandering across
both labellings, we saw six independent short chains split between them, which is
exactly what "two equally likely labellings" predicts when a single chain is unlikely to
switch. Either way the lesson is the same: nothing in the model prefers one labelling
over the other, so the fix, fixing an identifying rule after the run and reorienting
every draw or every chain by it, is required whenever the symmetry could in principle
bite, and here it is what makes the summary readable at all. Once the main
chain is reoriented, its posterior means of $\theta$ line up closely with the two true
usage profiles we simulated from: class $0$ answers questions 1, 2 and 5 positively with
probabilities $0.888$, $0.775$ and $0.755$ against the true $0.85$, $0.75$ and $0.80$,
while class $1$ answers questions 3, 4 and 6 positively with probabilities $0.698$,
$0.809$ and $0.739$ against the true $0.75$, $0.70$ and $0.80$, and the posterior mean of
$w$ is $(0.534, 0.466)$, close to the class split of 161 and 139 individuals that this
particular simulated sample happened to realise. The rule used here, pinning the
orientation to a single feature, is the simplest of several relabelling strategies and
works whenever one question cleanly separates the classes; with more classes or less
separated groups one
instead relabels by matching each draw to a fixed reference partition, but the underlying
issue and the need to address it before summarising are the same.
